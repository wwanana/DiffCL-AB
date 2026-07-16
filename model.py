import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel
from model_module.rope_attn import MutilHeadSelfAttn

# =========================================
# 0. Optional Mamba import
# =========================================
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except Exception:
    HAS_MAMBA = False
    Mamba = None


# =========================================
# 1. Basic modules
# =========================================
class MaskedConv1d(nn.Conv1d):
    """
    Input:
        x: [B, L, C]
        input_mask: [B, L] or [B, L, 1]
    Output:
        [B, L, C_out]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=padding,
        )

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            if input_mask.dim() == 2:
                input_mask = input_mask.unsqueeze(-1)
            x = x * input_mask.float()
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class AttnMean(nn.Module):
    """
    [B, L, H] -> [B, H]
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.layer = MaskedConv1d(hidden_size, 1, 1)

    def forward(self, x, input_mask=None):
        x = self.layer_norm(x)
        batch_size = x.shape[0]

        attn = self.layer(x)  # [B, L, 1]
        attn = attn.view(batch_size, -1)  # [B, L]

        if input_mask is not None:
            if input_mask.dim() == 3:
                input_mask = input_mask.squeeze(-1)
            attn = attn.masked_fill(~input_mask.bool(), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn.unsqueeze(1), x).squeeze(1)  # [B, H]
        return out


class AttnTransform(nn.Module):
    """
    [B, L, H] -> [B, L, H]
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.layer = MaskedConv1d(hidden_size, 1, 1)

    def forward(self, x, input_mask=None):
        x = self.layer_norm(x)
        batch_size = x.shape[0]

        attn = self.layer(x)
        attn = attn.view(batch_size, -1)  # [B, L]

        if input_mask is not None:
            if input_mask.dim() == 3:
                input_mask = input_mask.squeeze(-1)
            attn = attn.masked_fill(~input_mask.bool(), float("-inf"))

        attn = F.softmax(attn, dim=-1).view(batch_size, -1, 1)  # [B, L, 1]
        x = attn * x
        return x


class OutHead(nn.Module):
    """
    Final regression head
    [B, H] -> [B, 1]
    """

    def __init__(self, hidden_dim, out_dim=1, dropout=0.1):
        super().__init__()
        mid_dim = hidden_dim // 2 if hidden_dim >= 256 else hidden_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# =========================================
# 2. Mamba branch
# =========================================
class FallbackSequenceMixer(nn.Module):
    """
    Fallback when mamba_ssm is unavailable
    """

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dwconv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size
        )
        self.pwconv = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, x, mask=None):
        h = self.norm1(x)
        h = h.transpose(1, 2)
        h = self.dwconv(h)
        h = self.pwconv(h)
        h = h.transpose(1, 2)
        x = x + self.dropout(h)

        h2 = self.ffn(self.norm2(x))
        x = x + self.dropout(h2)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            x = x * mask.float()
        return x


class MambaBlock(nn.Module):
    def __init__(self, hidden_size, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        if HAS_MAMBA:
            self.mixer = Mamba(
                d_model=hidden_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.mixer = None

        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )

        self.fallback = None if HAS_MAMBA else FallbackSequenceMixer(hidden_size, dropout)

    def forward(self, x, mask=None):
        if HAS_MAMBA:
            h = self.norm1(x)
            h = self.mixer(h)
            x = x + self.dropout(h)

            h2 = self.ffn(self.norm2(x))
            x = x + self.dropout(h2)
        else:
            x = self.fallback(x, mask)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            x = x * mask.float()
        return x


class MambaEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers=2, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    hidden_size=hidden_size,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


# =========================================
# 3. Contrastive modules
# =========================================
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


class SymmetricInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        if z1.size(0) < 2:
            return z1.new_tensor(0.0)

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        logits = torch.matmul(z1, z2.transpose(0, 1)) / self.temperature
        targets = torch.arange(z1.size(0), device=z1.device)

        loss_12 = self.ce(logits, targets)
        loss_21 = self.ce(logits.transpose(0, 1), targets)
        return 0.5 * (loss_12 + loss_21)


# =========================================
# 4. Fusion helpers
# =========================================
class BilinearFeatureFusion(nn.Module):
    """
    Fuse main-branch feature and mamba-branch feature
    """

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        self.proj_main = nn.Linear(hidden_size, hidden_size)
        self.proj_mamba = nn.Linear(hidden_size, hidden_size)

        self.candidate = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.Sigmoid(),
        )

        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(self, main_feat, mamba_feat):
        h_main = self.proj_main(main_feat)
        h_mamba = self.proj_mamba(mamba_feat)
        interaction = h_main * h_mamba

        fusion_input = torch.cat([main_feat, mamba_feat, interaction], dim=-1)
        cand = self.candidate(fusion_input)
        gate = self.gate(fusion_input)

        fused = main_feat + gate * cand
        fused = self.out_norm(fused)
        return fused


class RelationEncoder(nn.Module):
    """
    Encode antibody-antigen relation into one vector
    """

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, ab_feat, ag_feat):
        x = torch.cat(
            [
                ab_feat,
                ag_feat,
                ab_feat * ag_feat,
                torch.abs(ab_feat - ag_feat),
            ],
            dim=-1,
        )
        return self.net(x)


class LearnableSharedAntigen(nn.Module):
    """
    Learnable shared antigen instead of hard average
    """

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.alpha_net = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(self, wt_ag, mt_ag):
        feat = torch.cat([wt_ag, mt_ag, torch.abs(wt_ag - mt_ag)], dim=-1)
        alpha = self.alpha_net(feat)  # [B, H]
        shared = alpha * wt_ag + (1.0 - alpha) * mt_ag
        shared = self.out_norm(shared)
        return shared, alpha


class QueryTokenAttentionPool(nn.Module):
    """
    Query-guided token attention pooling
    query: [B, H]
    tokens: [B, T, H]
    """

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(self, query, tokens):
        q = self.q_proj(query).unsqueeze(1)      # [B, 1, H]
        k = self.k_proj(tokens)                  # [B, T, H]
        v = self.v_proj(tokens)                  # [B, T, H]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_size)  # [B,1,T]
        weights = F.softmax(scores, dim=-1)     # [B,1,T]
        weights = self.dropout(weights)

        pooled = torch.matmul(weights, v).squeeze(1)  # [B,H]
        pooled = self.out_proj(pooled)
        pooled = self.out_norm(pooled + query)
        return pooled, weights.squeeze(1)


class StateAwareDifferenceFusion(nn.Module):
    """
    Final fusion:
    1. learnable shared antigen
    2. difference path: focus on mutation-induced change
    3. state path: retain wt/mt absolute binding states
    4. gated fusion of diff_rep and state_rep
    """

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        self.shared_antigen = LearnableSharedAntigen(hidden_size, dropout=dropout)
        self.relation_encoder = RelationEncoder(hidden_size, dropout=dropout)

        self.diff_query_proj = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

        self.state_query_proj = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

        self.diff_type_embed = nn.Parameter(torch.randn(1, 5, hidden_size) * 0.02)
        self.state_type_embed = nn.Parameter(torch.randn(1, 5, hidden_size) * 0.02)

        self.diff_pool = QueryTokenAttentionPool(hidden_size, dropout=dropout)
        self.state_pool = QueryTokenAttentionPool(hidden_size, dropout=dropout)

        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.Sigmoid(),
        )

        self.fusion_candidate = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, wt_ab, mt_ab, wt_ag, mt_ag):
        # 1) learnable shared antigen
        ag_shared, ag_alpha = self.shared_antigen(wt_ag, mt_ag)  # [B,H], [B,H]

        # 2) relation states
        wt_bind = self.relation_encoder(wt_ab, ag_shared)  # [B,H]
        mt_bind = self.relation_encoder(mt_ab, ag_shared)  # [B,H]

        # 3) mutation-induced differences
        diff_bind = mt_bind - wt_bind
        abs_diff_bind = torch.abs(diff_bind)

        diff_ab = mt_ab - wt_ab
        abs_diff_ab = torch.abs(diff_ab)

        bind_mean = 0.5 * (wt_bind + mt_bind)

        # -----------------------
        # difference path
        # -----------------------
        diff_tokens = torch.stack(
            [
                diff_bind,
                abs_diff_bind,
                diff_ab,
                abs_diff_ab,
                bind_mean,
            ],
            dim=1,
        )  # [B,5,H]
        diff_tokens = diff_tokens + self.diff_type_embed

        diff_query = self.diff_query_proj(torch.cat([diff_bind, diff_ab], dim=-1))  # [B,H]
        diff_rep, diff_attn_weights = self.diff_pool(diff_query, diff_tokens)        # [B,H], [B,5]

        # -----------------------
        # state path
        # -----------------------
        state_tokens = torch.stack(
            [
                wt_bind,
                mt_bind,
                wt_ab,
                mt_ab,
                ag_shared,
            ],
            dim=1,
        )  # [B,5,H]
        state_tokens = state_tokens + self.state_type_embed

        state_query = self.state_query_proj(torch.cat([wt_bind, mt_bind, ag_shared], dim=-1))
        state_rep, state_attn_weights = self.state_pool(state_query, state_tokens)

        # -----------------------
        # gated fusion
        # -----------------------
        fusion_input = torch.cat(
            [
                diff_rep,
                state_rep,
                torch.abs(diff_rep - state_rep),
            ],
            dim=-1,
        )

        gate = self.fusion_gate(fusion_input)          # [B,H]
        candidate = self.fusion_candidate(fusion_input)

        rep = gate * diff_rep + (1.0 - gate) * state_rep + candidate
        rep = self.final_norm(rep)

        aux = {
            "ag_shared": ag_shared,
            "ag_alpha": ag_alpha,
            "wt_bind": wt_bind,
            "mt_bind": mt_bind,
            "diff_bind": diff_bind,
            "diff_ab": diff_ab,
            "diff_rep": diff_rep,
            "state_rep": state_rep,
        }

        return rep, diff_attn_weights, state_attn_weights, aux


# =========================================
# 5. Main model
# =========================================
class SeqBindModel(nn.Module):
    """
    Upgraded final design:
    - keep backbone + main branch + mamba + contrastive + bilinear fusion
    - replace hard difference-only head with:
        1) learnable shared antigen
        2) difference path
        3) state path
        4) gated fusion
    """

    def __init__(self, args):
        super().__init__()
        self.config = args
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_heads
        self.dropout = args.dropout

        self.use_mamba_branch = getattr(args, "use_mamba_branch", True)
        self.use_contrastive = getattr(args, "use_contrastive", True)

        self.encoder = EsmModel.from_pretrained(args.model_locate)

        # ---------- main branch ----------
        self.wt_ab_conv_transformer = AttnTransform(hidden_size=self.hidden_size)
        self.wt_ag_conv_transformer = AttnTransform(hidden_size=self.hidden_size)
        self.mt_ab_conv_transformer = AttnTransform(hidden_size=self.hidden_size)
        self.mt_ag_conv_transformer = AttnTransform(hidden_size=self.hidden_size)

        self.wt_ab_attn = MutilHeadSelfAttn(
            num_heads=self.num_heads,
            hidden_dim=self.hidden_size,
            dropout=self.dropout,
        )
        self.mt_ab_attn = MutilHeadSelfAttn(
            num_heads=self.num_heads,
            hidden_dim=self.hidden_size,
            dropout=self.dropout,
        )
        self.wt_ag_attn = MutilHeadSelfAttn(
            num_heads=self.num_heads,
            hidden_dim=self.hidden_size,
            dropout=self.dropout,
        )
        self.mt_ag_attn = MutilHeadSelfAttn(
            num_heads=self.num_heads,
            hidden_dim=self.hidden_size,
            dropout=self.dropout,
        )

        self.wt_ab_mean = AttnMean(self.hidden_size)
        self.mt_ab_mean = AttnMean(self.hidden_size)
        self.wt_ag_mean = AttnMean(self.hidden_size)
        self.mt_ag_mean = AttnMean(self.hidden_size)

        # ---------- mamba branch ----------
        self.wt_ab_mamba = MambaEncoder(
            hidden_size=self.hidden_size,
            num_layers=getattr(args, "mamba_layers", 2),
            d_state=getattr(args, "mamba_d_state", 16),
            d_conv=getattr(args, "mamba_d_conv", 4),
            expand=getattr(args, "mamba_expand", 2),
            dropout=getattr(args, "mamba_dropout", 0.1),
        )
        self.wt_ag_mamba = MambaEncoder(
            hidden_size=self.hidden_size,
            num_layers=getattr(args, "mamba_layers", 2),
            d_state=getattr(args, "mamba_d_state", 16),
            d_conv=getattr(args, "mamba_d_conv", 4),
            expand=getattr(args, "mamba_expand", 2),
            dropout=getattr(args, "mamba_dropout", 0.1),
        )
        self.mt_ab_mamba = MambaEncoder(
            hidden_size=self.hidden_size,
            num_layers=getattr(args, "mamba_layers", 2),
            d_state=getattr(args, "mamba_d_state", 16),
            d_conv=getattr(args, "mamba_d_conv", 4),
            expand=getattr(args, "mamba_expand", 2),
            dropout=getattr(args, "mamba_dropout", 0.1),
        )
        self.mt_ag_mamba = MambaEncoder(
            hidden_size=self.hidden_size,
            num_layers=getattr(args, "mamba_layers", 2),
            d_state=getattr(args, "mamba_d_state", 16),
            d_conv=getattr(args, "mamba_d_conv", 4),
            expand=getattr(args, "mamba_expand", 2),
            dropout=getattr(args, "mamba_dropout", 0.1),
        )

        self.wt_ab_mamba_pool = AttnMean(self.hidden_size)
        self.wt_ag_mamba_pool = AttnMean(self.hidden_size)
        self.mt_ab_mamba_pool = AttnMean(self.hidden_size)
        self.mt_ag_mamba_pool = AttnMean(self.hidden_size)

        # ---------- contrastive ----------
        proj_dim = getattr(args, "proj_dim", 256)
        contrastive_temperature = getattr(args, "contrastive_temperature", 0.1)

        # 内部小权重，外部仍由 cl2_weight 控制总强度
        self.ag_consistency_inner_weight = getattr(args, "ag_consistency_inner_weight", 0.1)

        self.main_mamba_proj = ProjectionHead(
            in_dim=self.hidden_size,
            proj_dim=proj_dim,
            dropout=self.dropout,
        )
        self.fused_consistency_proj = ProjectionHead(
            in_dim=self.hidden_size,
            proj_dim=proj_dim,
            dropout=self.dropout,
        )
        self.cl_criterion = SymmetricInfoNCELoss(
            temperature=contrastive_temperature
        )

        # ---------- bilinear fusion ----------
        self.wt_ab_fusion = BilinearFeatureFusion(self.hidden_size, dropout=self.dropout)
        self.wt_ag_fusion = BilinearFeatureFusion(self.hidden_size, dropout=self.dropout)
        self.mt_ab_fusion = BilinearFeatureFusion(self.hidden_size, dropout=self.dropout)
        self.mt_ag_fusion = BilinearFeatureFusion(self.hidden_size, dropout=self.dropout)

        # ---------- final upgraded fusion ----------
        self.final_fusion = StateAwareDifferenceFusion(
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        )

        # ---------- final head ----------
        self.out_head = OutHead(
            hidden_dim=self.hidden_size,
            out_dim=1,
            dropout=self.dropout,
        )

        # freeze backbone
        if getattr(self.config, "freeze_backbone", False):
            for param in self.encoder.parameters():
                param.requires_grad = False

    def get_encoder_embeddings(self, input_ids, attention_mask):
        if getattr(self.config, "freeze_backbone", False):
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return outputs.last_hidden_state  # [B, L, H]

    def _compute_main_branch(
        self,
        wt_ab_embeddings,
        wt_ag_embeddings,
        mt_ab_embeddings,
        mt_ag_embeddings,
        wt_ab_inputs_mask,
        wt_ag_inputs_mask,
        mt_ab_inputs_mask,
        mt_ag_inputs_mask,
    ):
        wt_ab_embeddings = self.wt_ab_conv_transformer(
            wt_ab_embeddings, wt_ab_inputs_mask
        )
        wt_ag_embeddings = self.wt_ag_conv_transformer(
            wt_ag_embeddings, wt_ag_inputs_mask
        )
        mt_ab_embeddings = self.mt_ab_conv_transformer(
            mt_ab_embeddings, mt_ab_inputs_mask
        )
        mt_ag_embeddings = self.mt_ag_conv_transformer(
            mt_ag_embeddings, mt_ag_inputs_mask
        )

        wt_ag_ctx = self.wt_ag_attn(
            wt_ag_embeddings, wt_ab_embeddings, wt_ab_embeddings, mask=wt_ab_inputs_mask
        )
        wt_ab_ctx = self.wt_ab_attn(
            wt_ab_embeddings, wt_ag_embeddings, wt_ag_embeddings, mask=wt_ag_inputs_mask
        )

        mt_ag_ctx = self.mt_ag_attn(
            mt_ag_embeddings, mt_ab_embeddings, mt_ab_embeddings, mask=mt_ab_inputs_mask
        )
        mt_ab_ctx = self.mt_ab_attn(
            mt_ab_embeddings, mt_ag_embeddings, mt_ag_embeddings, mask=mt_ag_inputs_mask
        )

        main_wt_ab = self.wt_ab_mean(wt_ab_ctx, wt_ab_inputs_mask)
        main_wt_ag = self.wt_ag_mean(wt_ag_ctx, wt_ag_inputs_mask)
        main_mt_ab = self.mt_ab_mean(mt_ab_ctx, mt_ab_inputs_mask)
        main_mt_ag = self.mt_ag_mean(mt_ag_ctx, mt_ag_inputs_mask)

        return main_wt_ab, main_wt_ag, main_mt_ab, main_mt_ag

    def _compute_mamba_branch(
        self,
        wt_ab_embeddings,
        wt_ag_embeddings,
        mt_ab_embeddings,
        mt_ag_embeddings,
        wt_ab_inputs_mask,
        wt_ag_inputs_mask,
        mt_ab_inputs_mask,
        mt_ag_inputs_mask,
    ):
        if not self.use_mamba_branch:
            return None

        m_wt_ab = self.wt_ab_mamba(wt_ab_embeddings, wt_ab_inputs_mask)
        m_wt_ag = self.wt_ag_mamba(wt_ag_embeddings, wt_ag_inputs_mask)
        m_mt_ab = self.mt_ab_mamba(mt_ab_embeddings, mt_ab_inputs_mask)
        m_mt_ag = self.mt_ag_mamba(mt_ag_embeddings, mt_ag_inputs_mask)

        m_wt_ab = self.wt_ab_mamba_pool(m_wt_ab, wt_ab_inputs_mask)
        m_wt_ag = self.wt_ag_mamba_pool(m_wt_ag, wt_ag_inputs_mask)
        m_mt_ab = self.mt_ab_mamba_pool(m_mt_ab, mt_ab_inputs_mask)
        m_mt_ag = self.mt_ag_mamba_pool(m_mt_ag, mt_ag_inputs_mask)

        return m_wt_ab, m_wt_ag, m_mt_ab, m_mt_ag

    def _compute_cl1_loss(self, main_feats, mamba_feats):
        if (not self.use_contrastive) or (mamba_feats is None):
            return main_feats[0].new_tensor(0.0)

        losses = []
        for main_f, mamba_f in zip(main_feats, mamba_feats):
            z1 = self.main_mamba_proj(main_f)
            z2 = self.main_mamba_proj(mamba_f)
            losses.append(self.cl_criterion(z1, z2))

        return torch.stack(losses).mean()

    def _compute_cl2_loss(self, fused_feats, main_feats):
        """
        fused-main consistency + soft antigen consistency
        """
        if self.use_contrastive:
            losses = []
            for fused_f, main_f in zip(fused_feats, main_feats):
                z1 = self.fused_consistency_proj(fused_f)
                z2 = self.fused_consistency_proj(main_f.detach())
                losses.append(self.cl_criterion(z1, z2))
            cl_core = torch.stack(losses).mean()
        else:
            cl_core = fused_feats[0].new_tensor(0.0)

        wt_ag_fused = fused_feats[1]
        mt_ag_fused = fused_feats[3]
        ag_consistency = F.mse_loss(wt_ag_fused, mt_ag_fused)

        return cl_core + self.ag_consistency_inner_weight * ag_consistency

    def forward(
        self,
        wt_ab_inputs_ids,
        wt_ab_inputs_mask,
        mut_ab_inputs_ids,
        mt_ab_inputs_mask,
        wt_ag_inputs_ids,
        wt_ag_inputs_mask,
        mut_ag_inputs_ids,
        mt_ag_inputs_mask,
        wt_features=None,
        mt_features=None,
        labels=None,
        return_aux=False,
    ):
        # =========================================
        # 1. ESM embeddings
        # =========================================
        wt_ab_embeddings = self.get_encoder_embeddings(
            input_ids=wt_ab_inputs_ids,
            attention_mask=wt_ab_inputs_mask,
        )

        mt_ab_embeddings = self.get_encoder_embeddings(
            input_ids=mut_ab_inputs_ids,
            attention_mask=mt_ab_inputs_mask,
        )

        wt_ag_embeddings = self.get_encoder_embeddings(
            input_ids=wt_ag_inputs_ids,
            attention_mask=wt_ag_inputs_mask,
        )

        mt_ag_embeddings = self.get_encoder_embeddings(
            input_ids=mut_ag_inputs_ids,
            attention_mask=mt_ag_inputs_mask,
        )

        # =========================================
        # 2. Main branch
        # =========================================
        main_wt_ab, main_wt_ag, main_mt_ab, main_mt_ag = self._compute_main_branch(
            wt_ab_embeddings=wt_ab_embeddings,
            wt_ag_embeddings=wt_ag_embeddings,
            mt_ab_embeddings=mt_ab_embeddings,
            mt_ag_embeddings=mt_ag_embeddings,
            wt_ab_inputs_mask=wt_ab_inputs_mask,
            wt_ag_inputs_mask=wt_ag_inputs_mask,
            mt_ab_inputs_mask=mt_ab_inputs_mask,
            mt_ag_inputs_mask=mt_ag_inputs_mask,
        )

        main_feats = (main_wt_ab, main_wt_ag, main_mt_ab, main_mt_ag)

        # =========================================
        # 3. Mamba branch
        # =========================================
        mamba_feats = self._compute_mamba_branch(
            wt_ab_embeddings=wt_ab_embeddings,
            wt_ag_embeddings=wt_ag_embeddings,
            mt_ab_embeddings=mt_ab_embeddings,
            mt_ag_embeddings=mt_ag_embeddings,
            wt_ab_inputs_mask=wt_ab_inputs_mask,
            wt_ag_inputs_mask=wt_ag_inputs_mask,
            mt_ab_inputs_mask=mt_ab_inputs_mask,
            mt_ag_inputs_mask=mt_ag_inputs_mask,
        )

        if mamba_feats is None:
            m_wt_ab, m_wt_ag, m_mt_ab, m_mt_ag = main_feats
            mamba_feats = (m_wt_ab, m_wt_ag, m_mt_ab, m_mt_ag)
        else:
            m_wt_ab, m_wt_ag, m_mt_ab, m_mt_ag = mamba_feats

        # =========================================
        # 4. CL1
        # =========================================
        cl1_loss = self._compute_cl1_loss(main_feats, mamba_feats)

        # =========================================
        # 5. Bilinear fusion
        # =========================================
        final_wt_ab = self.wt_ab_fusion(main_wt_ab, m_wt_ab)
        final_wt_ag = self.wt_ag_fusion(main_wt_ag, m_wt_ag)
        final_mt_ab = self.mt_ab_fusion(main_mt_ab, m_mt_ab)
        final_mt_ag = self.mt_ag_fusion(main_mt_ag, m_mt_ag)

        fused_feats = (final_wt_ab, final_wt_ag, final_mt_ab, final_mt_ag)

        # =========================================
        # 6. CL2
        # =========================================
        cl2_loss = self._compute_cl2_loss(fused_feats, main_feats)

        # =========================================
        # 7. Final upgraded fusion
        # =========================================
        final_rep, diff_attn_weights, state_attn_weights, final_aux = self.final_fusion(
            wt_ab=final_wt_ab,
            mt_ab=final_mt_ab,
            wt_ag=final_wt_ag,
            mt_ag=final_mt_ag,
        )

        # =========================================
        # 8. Prediction
        # =========================================
        out = self.out_head(final_rep).squeeze(1)  # [B]

        if return_aux:
            return {
                "pred": out,
                "cl1_loss": cl1_loss,
                "cl2_loss": cl2_loss,
                "main_feats": {
                    "wt_ab": main_wt_ab,
                    "wt_ag": main_wt_ag,
                    "mt_ab": main_mt_ab,
                    "mt_ag": main_mt_ag,
                },
                "mamba_feats": {
                    "wt_ab": m_wt_ab,
                    "wt_ag": m_wt_ag,
                    "mt_ab": m_mt_ab,
                    "mt_ag": m_mt_ag,
                },
                "fused_feats": {
                    "wt_ab": final_wt_ab,
                    "wt_ag": final_wt_ag,
                    "mt_ab": final_mt_ab,
                    "mt_ag": final_mt_ag,
                },
                "final_aux": final_aux,
                "diff_attn_weights": diff_attn_weights,    # [B, 5]
                "state_attn_weights": state_attn_weights,  # [B, 5]
            }

        return out