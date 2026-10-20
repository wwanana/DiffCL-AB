# -*- coding: utf-8 -*-
"""
Shared Mutation-Aware DiffCL-AB Lite
====================================

Redesigned sequence-only ΔΔG model for the optimized AB645 LOCO90 + refit
protocol.

Design goals
------------
1. Frozen ESM2 residue embeddings.
2. WT and MT pass through the SAME downstream state encoder.
3. Compact downstream dimension (default 256), rather than 1280-wide modules.
4. Explicit mutation-local delta branch from WT/MT sequence differences.
5. Lightweight global interaction branch:
      residue refinement -> learned latent pooling -> bidirectional Cross-Attn.
6. Shared one-layer Mamba branch for long-range sequence context.
7. State-aware WT->MT difference fusion.
8. Optional reviewer-safe CL-I (Main<->Mamba) and CL-II (Fused<->Main).
9. No BatchNorm.
10. Frozen-ESM sliding-window encoding for proteins exceeding native ESM2
    positional context; no silent sequence truncation.

PDB IDs and homology metadata are NEVER prediction inputs.  They only enter
contrastive exclusion masks produced by dataset.py.

Expected forward inputs are exactly the redesigned mutation-aware batch contract.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel

try:
    from mamba_ssm import Mamba
    _HAS_MAMBA = True
except Exception:
    Mamba = None
    _HAS_MAMBA = False


TensorOrMaskDict = Union[torch.Tensor, Mapping[str, torch.Tensor]]


def _cfg(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse bool from {value!r}")


# =============================================================================
# Generic masked helpers
# =============================================================================

def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3 or mask.ndim != 2:
        raise ValueError("masked_mean expects x=[B,L,D], mask=[B,L]")
    if x.shape[:2] != mask.shape:
        raise ValueError("masked_mean shape mismatch")
    w = mask.to(dtype=x.dtype).unsqueeze(-1)
    denom = w.sum(dim=1).clamp_min(1.0)
    return (x * w).sum(dim=1) / denom


def _ensure_nonempty_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    bad = ~mask.any(dim=1)
    if bad.any():
        mask = mask.clone()
        mask[bad, 0] = True
    return mask


class ResidualLocalRefinement(nn.Module):
    """Small depthwise local mixer; shared by all WT/MT/partner states."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=3, padding=1, groups=dim, bias=True
        )
        self.pw = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(dtype=x.dtype).unsqueeze(-1)
        y = self.norm(x) * m
        y = self.dwconv(y.transpose(1, 2)).transpose(1, 2)
        y = F.gelu(self.pw(y))
        y = self.dropout(y)
        return (x + y) * m


class LearnedLatentPool(nn.Module):
    """Compress full residue sequence to K learned latent tokens."""

    def __init__(
        self,
        dim: int,
        num_latents: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.query_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = _ensure_nonempty_mask(mask)
        b = x.shape[0]
        q = self.latents.unsqueeze(0).expand(b, -1, -1)
        y, _ = self.attn(
            self.query_norm(q),
            self.kv_norm(x),
            self.kv_norm(x),
            key_padding_mask=~mask.bool(),
            need_weights=False,
        )
        q = q + y
        return q + self.ff(q)


class CrossInteractionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(
            self.q_norm(q),
            self.kv_norm(kv),
            self.kv_norm(kv),
            need_weights=False,
        )
        q = q + y
        return q + self.ff(q)


# =============================================================================
# Mamba branch
# =============================================================================

class FallbackMixer(nn.Module):
    """Debug-only fallback; forbidden in strict formal runs."""

    def __init__(self, dim: int, expand: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expand),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expand, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResidualMambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        allow_fallback: bool,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        if _HAS_MAMBA:
            self.mixer = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.real_mamba = True
        else:
            if not allow_fallback:
                raise ImportError(
                    "mamba_ssm is unavailable and allow_mamba_fallback=False"
                )
            self.mixer = FallbackMixer(dim, expand, dropout)
            self.real_mamba = False

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(dtype=x.dtype).unsqueeze(-1)
        y = self.norm(x) * m
        y = self.mixer(y)
        y = self.dropout(y)
        return (x + y) * m


class SharedMambaBranch(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        allow_fallback: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResidualMambaBlock(
                    dim=dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    allow_fallback=allow_fallback,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x, mask)
        x = self.out_norm(x)
        pooled = masked_mean(x, mask.bool())
        return x, pooled


# =============================================================================
# Main/fused state encoder (SAME object for WT and MT)
# =============================================================================

class SharedComplexStateEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_latents: int,
        dropout: float,
        use_mamba: bool,
        mamba_layers: int,
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_expand: int,
        mamba_dropout: float,
        allow_mamba_fallback: bool,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.use_mamba = use_mamba

        # Same residue refinement and latent pool are applied to both partner
        # roles; a small learned role vector preserves A/B identity.
        self.local_refine = ResidualLocalRefinement(dim, dropout)
        self.latent_pool = LearnedLatentPool(
            dim=dim,
            num_latents=num_latents,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.role_embedding = nn.Parameter(torch.randn(2, dim) * 0.02)

        self.cross_a_from_b = CrossInteractionBlock(
            dim, num_heads, dropout
        )
        self.cross_b_from_a = CrossInteractionBlock(
            dim, num_heads, dropout
        )

        self.main_fuse = nn.Sequential(
            nn.LayerNorm(4 * dim),
            nn.Linear(4 * dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
            nn.LayerNorm(dim),
        )

        if use_mamba:
            self.mamba = SharedMambaBranch(
                dim=dim,
                num_layers=mamba_layers,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                dropout=mamba_dropout,
                allow_fallback=allow_mamba_fallback,
            )
            self.mamba_fuse = nn.Sequential(
                nn.LayerNorm(4 * dim),
                nn.Linear(4 * dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(dim),
            )
        else:
            self.mamba = None
            self.mamba_fuse = None

        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.Sigmoid(),
        )
        self.fused_norm = nn.LayerNorm(dim)

    def _partner_latents(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        role_index: int,
    ) -> torch.Tensor:
        x = self.local_refine(x, mask)
        role = self.role_embedding[role_index].view(1, 1, -1)
        x = x + role * mask.to(x.dtype).unsqueeze(-1)
        return self.latent_pool(x, mask.bool())

    def _combine_pair(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        projector: nn.Module,
    ) -> torch.Tensor:
        return projector(
            torch.cat([a, b, torch.abs(a - b), a * b], dim=-1)
        )

    def forward(
        self,
        a: torch.Tensor,
        a_mask: torch.Tensor,
        b: torch.Tensor,
        b_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        a_mask = a_mask.bool()
        b_mask = b_mask.bool()

        a_lat = self._partner_latents(a, a_mask, 0)
        b_lat = self._partner_latents(b, b_mask, 1)

        a_ctx = self.cross_a_from_b(a_lat, b_lat)
        b_ctx = self.cross_b_from_a(b_lat, a_lat)

        main_a = a_ctx.mean(dim=1)
        main_b = b_ctx.mean(dim=1)
        main = self._combine_pair(main_a, main_b, self.main_fuse)

        if self.use_mamba:
            assert self.mamba is not None and self.mamba_fuse is not None
            _, m_a = self.mamba(a, a_mask)
            _, m_b = self.mamba(b, b_mask)
            mamba = self._combine_pair(m_a, m_b, self.mamba_fuse)
        else:
            mamba = torch.zeros_like(main)

        gate = self.fusion_gate(torch.cat([main, mamba], dim=-1))
        fused = self.fused_norm(main + gate * mamba)

        return {
            "main": main,
            "mamba": mamba,
            "fused": fused,
            "main_partner_a": main_a,
            "main_partner_b": main_b,
        }


# =============================================================================
# Mutation-local delta
# =============================================================================

class MutationLocalDeltaBranch(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.delta_token = nn.Sequential(
            nn.LayerNorm(4 * dim),
            nn.Linear(4 * dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
            nn.LayerNorm(dim),
        )
        self.partner_fuse = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.final = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dim),
        )

    @staticmethod
    def _pool_or_zero(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        w = mask.to(x.dtype).unsqueeze(-1)
        denom = w.sum(dim=1)
        pooled = (x * w).sum(dim=1) / denom.clamp_min(1.0)
        has_any = (denom > 0).to(x.dtype)
        return pooled * has_any

    def _partner(
        self,
        wt: torch.Tensor,
        mt: torch.Tensor,
        site_mask: torch.Tensor,
        window_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        token = self.delta_token(
            torch.cat(
                [wt, mt, mt - wt, torch.abs(mt - wt)],
                dim=-1,
            )
        )
        site = self._pool_or_zero(token, site_mask)
        window = self._pool_or_zero(token, window_mask)
        rep = self.partner_fuse(torch.cat([site, window], dim=-1))
        present = site_mask.bool().any(dim=1, keepdim=True)
        return rep * present.to(rep.dtype), present

    def forward(
        self,
        wt_ab: torch.Tensor,
        mt_ab: torch.Tensor,
        wt_ag: torch.Tensor,
        mt_ag: torch.Tensor,
        ab_mutation_mask: torch.Tensor,
        ag_mutation_mask: torch.Tensor,
        ab_mutation_window_mask: torch.Tensor,
        ag_mutation_window_mask: torch.Tensor,
    ) -> torch.Tensor:
        ab, ab_present = self._partner(
            wt_ab, mt_ab,
            ab_mutation_mask, ab_mutation_window_mask
        )
        ag, ag_present = self._partner(
            wt_ag, mt_ag,
            ag_mutation_mask, ag_mutation_window_mask
        )

        # Preserve partner identity even when only one side mutates.
        combined = self.final(torch.cat([ab, ag], dim=-1))
        any_present = ab_present | ag_present
        return combined * any_present.to(combined.dtype)


# =============================================================================
# Reviewer-safe masked InfoNCE
# =============================================================================

class ProjectionHead(nn.Module):
    def __init__(self, dim: int, proj_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class MaskedSymmetricInfoNCE(nn.Module):
    """
    Symmetric cross-view InfoNCE.

    Same sample is the positive.  `exclude_mask[i,j]=True` means j must NOT be
    used as a negative for anchor i.  Anchors with no valid negatives are
    skipped.  If the whole batch has no valid negatives, a graph-safe zero loss
    is returned.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be >0")
        self.temperature = float(temperature)

    @staticmethod
    def _coerce_mask(
        mask: Optional[torch.Tensor],
        b: int,
        device: torch.device,
    ) -> torch.Tensor:
        if mask is None:
            out = torch.zeros((b, b), dtype=torch.bool, device=device)
        else:
            out = torch.as_tensor(mask, device=device).bool()
            if out.shape != (b, b):
                raise ValueError(
                    f"exclude mask must be {(b,b)}, got {tuple(out.shape)}"
                )
            out = out | out.T
        out = out.clone()
        out.fill_diagonal_(False)
        return out

    def _direction(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        exclude: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        n = a.shape[0]
        logits = (a @ b.T) / self.temperature
        eye = torch.eye(n, dtype=torch.bool, device=a.device)

        valid_neg = (~exclude) & (~eye)
        valid_neg_count = valid_neg.sum(dim=1)
        valid_anchor = valid_neg_count > 0

        # Positive plus allowed negatives.
        allowed = valid_neg | eye
        masked_logits = logits.masked_fill(
            ~allowed, torch.finfo(logits.dtype).min
        )

        if valid_anchor.any():
            log_prob = F.log_softmax(masked_logits[valid_anchor], dim=1)
            target = torch.arange(n, device=a.device)[valid_anchor]
            loss = -log_prob[
                torch.arange(target.numel(), device=a.device),
                target,
            ].mean()
        else:
            loss = (a.sum() + b.sum()) * 0.0

        possible = max(n - 1, 1)
        stats = {
            "valid_anchors": valid_anchor.sum().to(a.dtype),
            "zero_negative_anchors": (~valid_anchor).sum().to(a.dtype),
            "mean_valid_negatives": valid_neg_count.float().mean().to(a.dtype),
            "masked_negative_ratio": (
                1.0 - valid_neg_count.float().mean() / float(possible)
            ).to(a.dtype),
        }
        return loss, stats

    def forward(
        self,
        view_a: torch.Tensor,
        view_b: torch.Tensor,
        exclude_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if view_a.shape != view_b.shape:
            raise ValueError("InfoNCE views must have same shape")
        b = view_a.shape[0]
        exclude = self._coerce_mask(
            exclude_mask, b, view_a.device
        )
        loss_ab, stat_ab = self._direction(
            view_a, view_b, exclude
        )
        loss_ba, stat_ba = self._direction(
            view_b, view_a, exclude.T
        )
        loss = 0.5 * (loss_ab + loss_ba)
        stats = {
            k: 0.5 * (stat_ab[k] + stat_ba[k])
            for k in stat_ab
        }
        return loss, stats


# =============================================================================
# Full model
# =============================================================================

class SeqBindModel(nn.Module):
    """Shared Mutation-Aware DiffCL-AB Lite."""

    CATEGORY_KEYS = ("wt_ab", "wt_ag", "mt_ab", "mt_ag")

    def __init__(self, args: Any) -> None:
        super().__init__()
        self.config = args

        self.model_dim = int(_cfg(args, "model_dim", 256))
        self.hidden_size = self.model_dim  # legacy compatibility only
        self.num_heads = int(_cfg(args, "num_heads", 4))
        self.num_latent_tokens = int(_cfg(args, "num_latent_tokens", 16))
        self.dropout = float(_cfg(args, "dropout", 0.10))

        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        self.freeze_backbone = _as_bool(
            _cfg(args, "freeze_backbone", True), True
        )
        self.use_mamba_branch = _as_bool(
            _cfg(args, "use_mamba_branch", True), True
        )
        self.mamba_layers = int(_cfg(args, "mamba_layers", 1))
        self.mamba_d_state = int(_cfg(args, "mamba_d_state", 16))
        self.mamba_d_conv = int(_cfg(args, "mamba_d_conv", 4))
        self.mamba_expand = int(_cfg(args, "mamba_expand", 2))
        self.mamba_dropout = float(
            _cfg(args, "mamba_dropout", self.dropout)
        )
        self.allow_mamba_fallback = _as_bool(
            _cfg(args, "allow_mamba_fallback", False), False
        )

        self.use_contrastive = _as_bool(
            _cfg(args, "use_contrastive", False), False
        )
        self.proj_dim = int(_cfg(args, "proj_dim", 128))
        self.contrastive_temperature = float(
            _cfg(args, "contrastive_temperature", 0.10)
        )
        self.require_safe_contrastive_mask = _as_bool(
            _cfg(args, "require_safe_contrastive_mask", True), True
        )

        self.long_sequence_strategy = str(
            _cfg(args, "long_sequence_strategy", "sliding_window")
        ).lower()
        self.esm_chunk_overlap = int(
            _cfg(args, "esm_chunk_overlap", 128)
        )
        if self.long_sequence_strategy not in {
            "sliding_window", "error"
        }:
            raise ValueError(
                "long_sequence_strategy must be sliding_window or error"
            )

        model_locate = str(_cfg(args, "model_locate", ""))
        if not model_locate:
            raise ValueError("args.model_locate is required")
        local_files_only = _as_bool(
            _cfg(args, "local_files_only", False), False
        )

        self.encoder = EsmModel.from_pretrained(
            model_locate,
            local_files_only=local_files_only,
        )
        encoder_dim = int(self.encoder.config.hidden_size)
        self.esm_max_positions = int(
            getattr(self.encoder.config, "max_position_embeddings", 1026)
        )
        self.esm_max_residues = self.esm_max_positions - 2
        if not (
            0 <= self.esm_chunk_overlap < self.esm_max_residues
        ):
            raise ValueError(
                "esm_chunk_overlap must be >=0 and < native residue limit"
            )

        if self.freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
            # requires_grad=False/no_grad() do not disable dropout.
            # A frozen ESM2 must remain in inference mode during training.
            self.encoder.eval()

        # Trainable compact projection, shared for all four sequence states.
        # Keep this deterministic because the explicit mutation-delta branch
        # compares WT and MT after this projection; independent projection
        # dropout would manufacture artificial WT/MT differences.
        self.input_projection = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, self.model_dim),
            nn.GELU(),
        )

        # EXACTLY ONE state encoder object is called for WT and MT.
        self.shared_state_encoder = SharedComplexStateEncoder(
            dim=self.model_dim,
            num_heads=self.num_heads,
            num_latents=self.num_latent_tokens,
            dropout=self.dropout,
            use_mamba=self.use_mamba_branch,
            mamba_layers=self.mamba_layers,
            mamba_d_state=self.mamba_d_state,
            mamba_d_conv=self.mamba_d_conv,
            mamba_expand=self.mamba_expand,
            mamba_dropout=self.mamba_dropout,
            allow_mamba_fallback=self.allow_mamba_fallback,
        )

        self.mutation_local_branch = MutationLocalDeltaBranch(
            self.model_dim, self.dropout
        )

        self.state_difference_fusion = nn.Sequential(
            nn.LayerNorm(6 * self.model_dim),
            nn.Linear(6 * self.model_dim, 2 * self.model_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(2 * self.model_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
        )

        self.prediction_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, 2 * self.model_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(2 * self.model_dim, self.model_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim // 2, 1),
        )

        # Shared contrastive heads.  They are instantiated even when CL is off
        # so later controlled ablations have exactly the expected interface.
        self.cl_main_projector = ProjectionHead(
            self.model_dim, self.proj_dim, self.dropout
        )
        self.cl_mamba_projector = ProjectionHead(
            self.model_dim, self.proj_dim, self.dropout
        )
        self.cl_fused_projector = ProjectionHead(
            self.model_dim, self.proj_dim, self.dropout
        )
        self.cl_loss = MaskedSymmetricInfoNCE(
            self.contrastive_temperature
        )

    def train(self, mode: bool = True):
        """Keep a frozen ESM2 permanently in eval mode.

        Lightning/PyTorch recursively call `.train()` on child modules at the
        start of training.  Without this override, ESM2 dropout would reactivate
        even though its parameters are frozen.
        """
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.eval()
        return self

    # -------------------------------------------------------------------------
    # Frozen ESM2: full-sequence + sliding-window path
    # -------------------------------------------------------------------------

    @staticmethod
    def _sliding_window_starts(
        n_residues: int,
        max_residues: int,
        overlap: int,
    ) -> Tuple[int, ...]:
        if n_residues <= max_residues:
            return (0,)
        stride = max_residues - overlap
        if stride <= 0:
            raise ValueError("invalid sliding-window stride")
        starts = []
        start = 0
        while True:
            starts.append(start)
            if start + max_residues >= n_residues:
                break
            start += stride
        return tuple(starts)

    def _encode_long_single(
        self,
        row_ids: torch.Tensor,
    ) -> torch.Tensor:
        valid_len = int(row_ids.numel())
        if valid_len <= self.esm_max_positions:
            mask = torch.ones_like(row_ids, dtype=torch.long)
            return self.encoder(
                input_ids=row_ids.unsqueeze(0),
                attention_mask=mask.unsqueeze(0),
            ).last_hidden_state.squeeze(0)

        if valid_len < 3:
            raise RuntimeError("malformed ESM token sequence")

        bos = row_ids[:1]
        eos = row_ids[-1:]
        residues = row_ids[1:-1]
        n_res = int(residues.numel())

        starts = self._sliding_window_starts(
            n_res,
            self.esm_max_residues,
            self.esm_chunk_overlap,
        )

        residue_sum = None
        residue_count = None
        first_bos = None
        last_eos = None

        for wi, start in enumerate(starts):
            end = min(start + self.esm_max_residues, n_res)
            ids = torch.cat(
                [bos, residues[start:end], eos], dim=0
            )
            mask = torch.ones_like(ids, dtype=torch.long)
            h = self.encoder(
                input_ids=ids.unsqueeze(0),
                attention_mask=mask.unsqueeze(0),
            ).last_hidden_state.squeeze(0)

            if residue_sum is None:
                d = h.shape[-1]
                residue_sum = h.new_zeros((n_res, d))
                residue_count = h.new_zeros((n_res, 1))

            if wi == 0:
                first_bos = h[0]
            last_eos = h[-1]
            residue_sum[start:end] += h[1:-1]
            residue_count[start:end] += 1.0

        if (
            residue_sum is None
            or residue_count is None
            or first_bos is None
            or last_eos is None
        ):
            raise RuntimeError("sliding-window produced no output")
        if (residue_count <= 0).any():
            raise RuntimeError("sliding-window failed to cover all residues")

        residues_h = residue_sum / residue_count
        return torch.cat(
            [
                first_bos.unsqueeze(0),
                residues_h,
                last_eos.unsqueeze(0),
            ],
            dim=0,
        )

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("ESM input must be [B,L]")
        if input_ids.shape != attention_mask.shape:
            raise ValueError("ESM ids/mask shape mismatch")

        b, padded_len = input_ids.shape
        valid_lengths = attention_mask.long().sum(dim=1)
        if (valid_lengths <= 0).any():
            raise RuntimeError("all-zero ESM attention mask")

        ctx = torch.no_grad() if self.freeze_backbone else nullcontext()
        with ctx:
            if padded_len <= self.esm_max_positions:
                h = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state
            else:
                if self.long_sequence_strategy == "error":
                    raise RuntimeError(
                        "Protein exceeds native ESM2 context and "
                        "long_sequence_strategy='error'"
                    )

                d = int(self.encoder.config.hidden_size)
                dtype = next(self.encoder.parameters()).dtype
                h = torch.zeros(
                    (b, padded_len, d),
                    device=input_ids.device,
                    dtype=dtype,
                )

                for row in range(b):
                    n = int(valid_lengths[row].item())
                    row_ids = input_ids[row, :n]
                    row_h = self._encode_long_single(row_ids)
                    if row_h.shape[0] != n:
                        raise RuntimeError(
                            "stitched ESM output length mismatch"
                        )
                    h[row, :n] = row_h

        # Projection is outside no_grad: compact downstream remains trainable.
        return self.input_projection(h)

    # -------------------------------------------------------------------------
    # Contrastive masks
    # -------------------------------------------------------------------------

    @staticmethod
    def _coerce_exclusion(
        value: torch.Tensor,
        b: int,
        device: torch.device,
    ) -> torch.Tensor:
        m = torch.as_tensor(value, device=device).bool()
        if m.shape != (b, b):
            raise ValueError(
                f"contrastive exclusion mask must be {(b,b)}"
            )
        m = m | m.T
        m = m.clone()
        m.fill_diagonal_(False)
        return m

    def _global_contrastive_exclusion(
        self,
        masks: Optional[TensorOrMaskDict],
        b: int,
        device: torch.device,
    ) -> torch.Tensor:
        out = torch.zeros((b, b), dtype=torch.bool, device=device)
        if masks is None:
            if self.use_contrastive and self.require_safe_contrastive_mask:
                raise RuntimeError(
                    "Contrastive learning enabled without reviewer-safe "
                    "contrastive_exclusion_masks"
                )
            return out

        if torch.is_tensor(masks):
            return self._coerce_exclusion(masks, b, device)

        if not isinstance(masks, Mapping):
            raise TypeError(
                "contrastive_exclusion_masks must be tensor/dict/None"
            )

        for key in ("all", "wt_ab", "wt_ag", "mt_ab", "mt_ag"):
            if key in masks:
                out |= self._coerce_exclusion(
                    masks[key], b, device
                )
        out.fill_diagonal_(False)
        return out

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        wt_ab_inputs_ids: torch.Tensor,
        wt_ab_inputs_mask: torch.Tensor,
        mut_ab_inputs_ids: torch.Tensor,
        mt_ab_inputs_mask: torch.Tensor,
        wt_ag_inputs_ids: torch.Tensor,
        wt_ag_inputs_mask: torch.Tensor,
        mut_ag_inputs_ids: torch.Tensor,
        mt_ag_inputs_mask: torch.Tensor,
        ab_mutation_mask: torch.Tensor,
        ag_mutation_mask: torch.Tensor,
        ab_mutation_window_mask: torch.Tensor,
        ag_mutation_window_mask: torch.Tensor,
        contrastive_exclusion_masks: Optional[TensorOrMaskDict] = None,
        return_aux: bool = False,
    ):
        # Frozen ESM2 + one shared compact projection.
        wt_ab = self._encode(
            wt_ab_inputs_ids, wt_ab_inputs_mask
        )
        wt_ag = self._encode(
            wt_ag_inputs_ids, wt_ag_inputs_mask
        )
        mt_ab = self._encode(
            mut_ab_inputs_ids, mt_ab_inputs_mask
        )
        mt_ag = self._encode(
            mut_ag_inputs_ids, mt_ag_inputs_mask
        )

        # EXACT same state encoder object for WT and MT.
        wt_state = self.shared_state_encoder(
            wt_ab, wt_ab_inputs_mask,
            wt_ag, wt_ag_inputs_mask,
        )
        mt_state = self.shared_state_encoder(
            mt_ab, mt_ab_inputs_mask,
            mt_ag, mt_ag_inputs_mask,
        )

        mutation = self.mutation_local_branch(
            wt_ab=wt_ab,
            mt_ab=mt_ab,
            wt_ag=wt_ag,
            mt_ag=mt_ag,
            ab_mutation_mask=ab_mutation_mask,
            ag_mutation_mask=ag_mutation_mask,
            ab_mutation_window_mask=ab_mutation_window_mask,
            ag_mutation_window_mask=ag_mutation_window_mask,
        )

        h_wt = wt_state["fused"]
        h_mt = mt_state["fused"]
        delta = h_mt - h_wt

        final = self.state_difference_fusion(
            torch.cat(
                [
                    h_wt,
                    h_mt,
                    delta,
                    torch.abs(delta),
                    h_wt * h_mt,
                    mutation,
                ],
                dim=-1,
            )
        )
        pred = self.prediction_head(final).squeeze(-1)

        if not return_aux:
            return pred

        zero = pred.sum() * 0.0
        cl1_loss = zero
        cl2_loss = zero
        contrastive_stats: Dict[str, torch.Tensor] = {
            "cl1_valid_anchors_mean": zero.detach(),
            "cl1_zero_negative_anchors_mean": zero.detach(),
            "cl1_mean_valid_negatives": zero.detach(),
            "cl1_masked_negative_ratio": zero.detach(),
            "cl2_valid_anchors_mean": zero.detach(),
            "cl2_zero_negative_anchors_mean": zero.detach(),
            "cl2_mean_valid_negatives": zero.detach(),
            "cl2_masked_negative_ratio": zero.detach(),
        }

        if self.use_contrastive:
            if not self.use_mamba_branch:
                raise RuntimeError(
                    "CL-I requires use_mamba_branch=True"
                )

            b = pred.shape[0]
            exclusion = self._global_contrastive_exclusion(
                contrastive_exclusion_masks,
                b,
                pred.device,
            )

            main_views = torch.cat(
                [wt_state["main"], mt_state["main"]],
                dim=0,
            )
            mamba_views = torch.cat(
                [wt_state["mamba"], mt_state["mamba"]],
                dim=0,
            )
            fused_views = torch.cat(
                [wt_state["fused"], mt_state["fused"]],
                dim=0,
            )

            # The same sample index across state type remains positive.
            # Duplicate WT/MT rows are not mixed as negatives because the
            # exclusion matrix is repeated block-diagonally/off-diagonally.
            ex2 = exclusion.repeat(2, 2)
            # WT and MT states from the SAME original mutation are semantically
            # related views and must never become negatives for one another.
            same_sample = torch.eye(
                b, dtype=torch.bool, device=pred.device
            )
            ex2[:b, b:] |= same_sample
            ex2[b:, :b] |= same_sample
            ex2.fill_diagonal_(False)

            p_main = self.cl_main_projector(main_views)
            p_mamba = self.cl_mamba_projector(mamba_views)
            p_fused = self.cl_fused_projector(fused_views)

            cl1_loss, s1 = self.cl_loss(
                p_main, p_mamba, ex2
            )
            cl2_loss, s2 = self.cl_loss(
                p_fused, p_main, ex2
            )

            contrastive_stats = {
                "cl1_valid_anchors_mean": s1["valid_anchors"],
                "cl1_zero_negative_anchors_mean": s1["zero_negative_anchors"],
                "cl1_mean_valid_negatives": s1["mean_valid_negatives"],
                "cl1_masked_negative_ratio": s1["masked_negative_ratio"],
                "cl2_valid_anchors_mean": s2["valid_anchors"],
                "cl2_zero_negative_anchors_mean": s2["zero_negative_anchors"],
                "cl2_mean_valid_negatives": s2["mean_valid_negatives"],
                "cl2_masked_negative_ratio": s2["masked_negative_ratio"],
            }

        return {
            "pred": pred,
            "cl1_loss": cl1_loss,
            "cl2_loss": cl2_loss,
            "unchanged_loss": zero,
            "contrastive_stats": contrastive_stats,
            "wt_state": wt_state,
            "mt_state": mt_state,
            "mutation_representation": mutation,
            "difference_representation": final,
        }

    # -------------------------------------------------------------------------
    # Formal audit
    # -------------------------------------------------------------------------

    def audit_formal_configuration(self) -> Dict[str, Any]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(
            p.numel() for p in self.parameters()
            if p.requires_grad
        )
        esm_total = sum(p.numel() for p in self.encoder.parameters())
        esm_trainable = sum(
            p.numel() for p in self.encoder.parameters()
            if p.requires_grad
        )
        downstream = total - esm_total
        uses_batchnorm = any(
            isinstance(
                m,
                (
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.SyncBatchNorm,
                ),
            )
            for m in self.modules()
        )

        real_mamba = _HAS_MAMBA
        if self.use_mamba_branch:
            blocks = list(
                self.shared_state_encoder.mamba.blocks  # type: ignore[union-attr]
            )
            real_mamba = bool(
                blocks and all(b.real_mamba for b in blocks)
            )

        return {
            "model_name": "SharedMutationAwareDiffCLABLite",
            "model_dim": self.model_dim,
            "num_latent_tokens": self.num_latent_tokens,
            "freeze_backbone": self.freeze_backbone,
            "frozen_encoder_eval_mode": (
                (not self.encoder.training)
                if self.freeze_backbone else None
            ),
            "wt_mt_shared_state_encoder": True,
            "uses_batchnorm": uses_batchnorm,
            "use_mamba_branch": self.use_mamba_branch,
            "mamba_layers": self.mamba_layers,
            "real_mamba_available": real_mamba,
            "allow_mamba_fallback": self.allow_mamba_fallback,
            "use_contrastive": self.use_contrastive,
            "require_safe_contrastive_mask": self.require_safe_contrastive_mask,
            "long_sequence_strategy": self.long_sequence_strategy,
            "esm_max_positions": self.esm_max_positions,
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "downstream_parameters": int(downstream),
            "esm_parameters": int(esm_total),
            "trainable_esm_parameters": int(esm_trainable),
        }

    def parameter_report(self) -> Dict[str, Any]:
        return self.audit_formal_configuration()


# Backward-friendly descriptive alias.
SharedMutationAwareDiffCLAB = SeqBindModel


__all__ = [
    "SeqBindModel",
    "SharedMutationAwareDiffCLAB",
    "MaskedSymmetricInfoNCE",
]
