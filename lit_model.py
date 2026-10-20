# -*- coding: utf-8 -*-
"""
PyTorch Lightning module for Shared Mutation-Aware DiffCL-AB Lite
=================================================================

Designed for the redesigned sequence-only DiffCL-AB pipeline:

- model.py:
    frozen ESM2;
    shared WT/MT state encoder;
    compact downstream representation;
    mutation-local delta branch;
    lightweight Cross-Attention + Mamba;
    optional reviewer-safe CL-I/CL-II.

- dataset.py:
    train-only target standardization;
    mutation-site/window masks;
    pairwise_regression_mask;
    reviewer-safe contrastive_exclusion_masks;
    raw ΔΔG labels and PDB IDs for metrics/audit only.

Primary AB645 protocol:
    Purged Homology-Aware LOCO90 + refit.

Stage 1 / selection:
    inner_train.csv -> optimize
    val.csv         -> choose best epoch / training duration
    test.csv        -> never used for optimization

Stage 2 / refit:
    REINITIALIZE model
    refit_train.csv -> train for Stage-1-selected duration
    test.csv        -> evaluate once

Initial objective:
    L = MSE(z-scored ΔΔG)

Optional later objective:
    L = L_MSE
        + λ_pair * L_pair
        + w_CL(epoch) * (λ_CL1 * L_CL1 + λ_CL2 * L_CL2)

L_pair is SmoothL1 on within-complex mutation differences:
    (ŷ_i - ŷ_j) vs (y_i - y_j)

All validation/test regression metrics are inverse-transformed and reported in
raw kcal/mol.  `val_mse` is therefore the recommended Stage-1 monitor.

The redesigned primary model intentionally does NOT optimize unchanged-partner
consistency by default.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover
    import lightning.pytorch as pl

from model import SeqBindModel


MODEL_BATCH_KEYS: Tuple[str, ...] = (
    "wt_ab_inputs_ids",
    "wt_ab_inputs_mask",
    "mut_ab_inputs_ids",
    "mt_ab_inputs_mask",
    "wt_ag_inputs_ids",
    "wt_ag_inputs_mask",
    "mut_ag_inputs_ids",
    "mt_ag_inputs_mask",
)

MUTATION_MASK_KEYS: Tuple[str, ...] = (
    "ab_mutation_mask",
    "ag_mutation_mask",
    "ab_mutation_window_mask",
    "ag_mutation_window_mask",
)

CONTRASTIVE_MASK_KEYS: Tuple[str, ...] = (
    "all",
    "wt_ab",
    "wt_ag",
    "mt_ab",
    "mt_ag",
)


# =============================================================================
# Configuration helpers
# =============================================================================

def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot interpret {value!r} as bool")


def _to_plain_config(args: Any = None, **kwargs: Any) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if args is not None:
        if isinstance(args, Mapping):
            cfg.update(dict(args))
        elif hasattr(args, "__dict__"):
            cfg.update(vars(args))
        else:
            raise TypeError(
                "LitModel expects a mapping / Namespace / SimpleNamespace, "
                f"got {type(args)!r}"
            )
    cfg.update(kwargs)
    for key, value in list(cfg.items()):
        if isinstance(value, Path):
            cfg[key] = str(value)
    return cfg


def _set_default(cfg: MutableMapping[str, Any], key: str, value: Any) -> None:
    if key not in cfg or cfg[key] is None:
        cfg[key] = value


def _safe_scalar_tensor(
    value: Any,
    *,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    t = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if t.numel() != 1:
        raise ValueError(f"{name} must be scalar, got {tuple(t.shape)}")
    t = t.reshape(())
    if not torch.isfinite(t):
        raise ValueError(f"{name} must be finite")
    return t


# =============================================================================
# Metric helpers
# =============================================================================

def _pearson_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return float("nan")

    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = math.sqrt(float(np.dot(x0, x0) * np.dot(y0, y0)))
    if denom <= 0.0 or not math.isfinite(denom):
        return float("nan")
    return float(np.dot(x0, y0) / denom)


def _average_ranks_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks_sorted = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i + 1
        while j < n and sx[j] == sx[i]:
            j += 1
        ranks_sorted[i:j] = 0.5 * ((i + 1) + j)
        i = j

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def _spearman_np(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    return _pearson_np(_average_ranks_np(x), _average_ranks_np(y))


def _direction_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    epsilon: float = 0.0,
) -> Tuple[float, float, int]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    keep = np.abs(target) > float(epsilon)
    if not np.any(keep):
        return float("nan"), float("nan"), 0

    p = pred[keep]
    y = target[keep]
    true_pos = y > 0.0
    true_neg = y < 0.0
    pred_pos = p > 0.0
    pred_neg = p < 0.0

    # Explicit sign agreement. A prediction exactly at zero is ambiguous and
    # therefore incorrect for a nonzero target; it must not be silently treated
    # as a negative prediction.
    correct = (true_pos & pred_pos) | (true_neg & pred_neg)
    acc = float(np.mean(correct))

    recalls: List[float] = []
    if np.any(true_pos):
        recalls.append(float(np.mean(pred_pos[true_pos])))
    if np.any(true_neg):
        recalls.append(float(np.mean(pred_neg[true_neg])))

    bal = float(np.mean(recalls)) if recalls else float("nan")
    return acc, bal, int(keep.sum())


def _regression_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    pdb_ids: Sequence[str],
    *,
    direction_epsilon: float,
    train_mean_raw: Optional[float],
) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)

    if pred.size != target.size or pred.size != len(pdb_ids):
        raise ValueError("prediction / target / pdb length mismatch")

    finite = np.isfinite(pred) & np.isfinite(target)
    pred = pred[finite]
    target = target[finite]
    pdb_ids = [p for p, ok in zip(pdb_ids, finite) if bool(ok)]

    n = int(pred.size)
    if n == 0:
        raise RuntimeError("No finite prediction/target pairs")

    error = pred - target
    mse = float(np.mean(error**2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(error)))

    pred_mean = float(np.mean(pred))
    target_mean = float(np.mean(target))
    pred_std = float(np.std(pred, ddof=1)) if n > 1 else 0.0
    target_std = float(np.std(target, ddof=1)) if n > 1 else 0.0

    sse = float(np.sum(error**2))
    sst = float(np.sum((target - target_mean) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0.0 else float("nan")

    pcc = _pearson_np(pred, target)
    rho = _spearman_np(pred, target)
    dacc, dbacc, ndir = _direction_metrics(
        pred, target, epsilon=direction_epsilon
    )

    # Complex-aware diagnostics.
    per_complex_errors: Dict[str, List[float]] = defaultdict(list)
    per_complex_pairs: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    for pdb, e, yp, yt in zip(pdb_ids, error, pred, target):
        key = str(pdb)
        per_complex_errors[key].append(float(e))
        per_complex_pairs[key].append((float(yp), float(yt)))

    complex_mae: List[float] = []
    complex_rmse: List[float] = []
    within_pred_std: List[float] = []
    within_target_std: List[float] = []
    within_ratio: List[float] = []

    for pdb in sorted(per_complex_errors):
        e = np.asarray(per_complex_errors[pdb], dtype=np.float64)
        complex_mae.append(float(np.mean(np.abs(e))))
        complex_rmse.append(float(np.sqrt(np.mean(e**2))))

        pairs = per_complex_pairs[pdb]
        if len(pairs) >= 2:
            yp = np.asarray([x[0] for x in pairs], dtype=np.float64)
            yt = np.asarray([x[1] for x in pairs], dtype=np.float64)
            ps = float(np.std(yp, ddof=1))
            ts = float(np.std(yt, ddof=1))
            within_pred_std.append(ps)
            within_target_std.append(ts)
            if ts > 1e-12:
                within_ratio.append(ps / ts)

    std_ratio = pred_std / target_std if target_std > 1e-12 else float("nan")

    metrics: Dict[str, float] = {
        "n": float(n),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "pearson_corr": pcc,
        "spearman_corr": rho,
        "r2": r2,
        "direction_acc": dacc,
        "direction_balanced_acc": dbacc,
        "n_direction": float(ndir),
        "pred_mean": pred_mean,
        "target_mean": target_mean,
        "pred_std": pred_std,
        "target_std": target_std,
        "pred_target_std_ratio": std_ratio,
        "complex_macro_mae": float(np.mean(complex_mae)),
        "complex_macro_rmse": float(np.mean(complex_rmse)),
        "n_complexes": float(len(per_complex_errors)),
        "within_complex_pred_std_median": (
            float(np.median(within_pred_std))
            if within_pred_std else float("nan")
        ),
        "within_complex_target_std_median": (
            float(np.median(within_target_std))
            if within_target_std else float("nan")
        ),
        "within_complex_std_ratio_median": (
            float(np.median(within_ratio))
            if within_ratio else float("nan")
        ),
    }

    if train_mean_raw is not None and math.isfinite(float(train_mean_raw)):
        baseline = np.full_like(target, float(train_mean_raw))
        baseline_rmse = float(np.sqrt(np.mean((baseline - target) ** 2)))
        metrics["train_mean_baseline_rmse"] = baseline_rmse
        metrics["rmse_gain_vs_train_mean"] = baseline_rmse - rmse

    return metrics


# =============================================================================
# Pairwise mutation-difference loss
# =============================================================================

def pairwise_mutation_difference_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pair_mask: torch.Tensor,
    *,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SmoothL1 on same-complex, distinct-input mutation differences.

    pair_mask is supplied by dataset.py and should exclude exact duplicate model
    inputs / repeated measurements.
    """
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    if pred.shape != target.shape:
        raise ValueError("pred and target must have identical 1-D shape")

    b = int(pred.numel())
    mask = torch.as_tensor(pair_mask, device=pred.device).bool()
    if mask.shape != (b, b):
        raise ValueError(
            f"pairwise_regression_mask must be {(b, b)}, got {tuple(mask.shape)}"
        )

    mask = mask | mask.T
    mask = torch.triu(mask, diagonal=1)
    n_pairs = mask.sum()

    if int(n_pairs.item()) == 0:
        return pred.sum() * 0.0, n_pairs.to(dtype=pred.dtype)

    pred_diff = pred[:, None] - pred[None, :]
    target_diff = target[:, None] - target[None, :]

    loss = F.smooth_l1_loss(
        pred_diff[mask],
        target_diff[mask],
        beta=float(beta),
        reduction="mean",
    )
    return loss, n_pairs.to(dtype=pred.dtype)


# =============================================================================
# Lightning module
# =============================================================================

class LitModel(pl.LightningModule):
    def __init__(self, args: Any = None, **kwargs: Any) -> None:
        super().__init__()

        cfg = _to_plain_config(args, **kwargs)

        # Model defaults.
        _set_default(cfg, "model_dim", 256)
        _set_default(cfg, "num_heads", 4)
        _set_default(cfg, "interaction_heads", cfg["num_heads"])
        _set_default(cfg, "num_latent_tokens", 16)
        _set_default(cfg, "dropout", 0.10)
        _set_default(cfg, "freeze_backbone", True)

        _set_default(cfg, "use_mamba_branch", True)
        _set_default(cfg, "mamba_layers", 1)
        _set_default(cfg, "mamba_d_state", 16)
        _set_default(cfg, "mamba_d_conv", 4)
        _set_default(cfg, "mamba_expand", 2)
        _set_default(cfg, "mamba_dropout", 0.10)
        _set_default(cfg, "allow_mamba_fallback", False)

        _set_default(cfg, "long_sequence_strategy", "sliding_window")
        _set_default(cfg, "esm_chunk_overlap", 128)

        # Primary regression.
        _set_default(cfg, "loss", "mse")
        _set_default(cfg, "target_standardize", True)

        # Pairwise auxiliary loss.
        _set_default(cfg, "use_pairwise_loss", False)
        _set_default(cfg, "pairwise_weight", 0.10)
        _set_default(cfg, "pairwise_beta", 1.0)

        # Contrastive learning, initially OFF.
        _set_default(cfg, "use_contrastive", False)
        _set_default(cfg, "require_safe_contrastive_mask", True)
        _set_default(cfg, "proj_dim", 128)
        _set_default(cfg, "contrastive_temperature", 0.10)
        _set_default(cfg, "cl1_weight", 0.01)
        _set_default(cfg, "cl2_weight", 0.005)
        _set_default(cfg, "cl_warmup_start_epoch", 5)
        _set_default(cfg, "cl_warmup_end_epoch", 15)

        # Explicitly disabled in redesigned primary objective.
        _set_default(cfg, "unchanged_weight", 0.0)
        _set_default(cfg, "ag_consistency_inner_weight", 0.0)

        # Optimizer / scheduler.
        _set_default(cfg, "lr", 1e-4)
        _set_default(cfg, "weight_decay", 1e-4)
        _set_default(cfg, "adam_beta1", 0.9)
        _set_default(cfg, "adam_beta2", 0.999)
        _set_default(cfg, "adam_eps", 1e-8)
        _set_default(cfg, "warmup_ratio", 0.05)
        _set_default(cfg, "min_lr", 1e-6)
        _set_default(cfg, "use_lr_scheduler", True)

        # Evaluation / formal safety.
        _set_default(cfg, "direction_epsilon", 0.0)
        _set_default(cfg, "strict_reviewer_mode", True)
        _set_default(cfg, "log_detailed_contrastive_stats", True)
        _set_default(cfg, "training_stage", "selection")

        self.cfg = cfg
        self.args = SimpleNamespace(**cfg)
        self.save_hyperparameters(cfg)

        self.loss_name = str(cfg["loss"]).strip().lower()
        if self.loss_name != "mse":
            raise ValueError(
                "Primary redesigned model currently requires loss='mse'. "
                "Huber/MAE should only be separate controlled ablations."
            )

        self.target_standardize = _as_bool(cfg["target_standardize"], True)

        self.use_pairwise_loss = _as_bool(cfg["use_pairwise_loss"], False)
        self.pairwise_weight = float(cfg["pairwise_weight"])
        self.pairwise_beta = float(cfg["pairwise_beta"])

        self.use_contrastive = _as_bool(cfg["use_contrastive"], False)
        self.cl1_weight = float(cfg["cl1_weight"])
        self.cl2_weight = float(cfg["cl2_weight"])
        self.cl_warmup_start_epoch = int(cfg["cl_warmup_start_epoch"])
        self.cl_warmup_end_epoch = int(cfg["cl_warmup_end_epoch"])

        self.lr = float(cfg["lr"])
        self.weight_decay = float(cfg["weight_decay"])
        self.adam_beta1 = float(cfg["adam_beta1"])
        self.adam_beta2 = float(cfg["adam_beta2"])
        self.adam_eps = float(cfg["adam_eps"])
        self.warmup_ratio = float(cfg["warmup_ratio"])
        self.min_lr = float(cfg["min_lr"])
        self.use_lr_scheduler = _as_bool(cfg["use_lr_scheduler"], True)

        self.direction_epsilon = float(cfg["direction_epsilon"])
        self.strict_reviewer_mode = _as_bool(cfg["strict_reviewer_mode"], True)
        self.log_detailed_contrastive_stats = _as_bool(
            cfg["log_detailed_contrastive_stats"], True
        )
        self.training_stage = str(cfg["training_stage"]).strip().lower()

        if self.training_stage not in {"selection", "refit"}:
            raise ValueError("training_stage must be selection or refit")
        if self.pairwise_weight < 0.0:
            raise ValueError("pairwise_weight must be >=0")
        if self.pairwise_beta <= 0.0:
            raise ValueError("pairwise_beta must be >0")
        if self.cl1_weight < 0.0 or self.cl2_weight < 0.0:
            raise ValueError("CL weights must be >=0")
        if self.cl_warmup_end_epoch < self.cl_warmup_start_epoch:
            raise ValueError("invalid CL warm-up interval")
        if self.lr <= 0.0:
            raise ValueError("lr must be >0")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be >=0")
        if not (0.0 <= self.warmup_ratio < 1.0):
            raise ValueError("warmup_ratio must be in [0,1)")
        if not (0.0 <= self.min_lr <= self.lr):
            raise ValueError("min_lr must satisfy 0 <= min_lr <= lr")

        self.model = SeqBindModel(self.args)

        self._val_cache: List[Dict[str, Any]] = []
        self._test_cache: List[Dict[str, Any]] = []

        self.last_val_metrics: Dict[str, float] = {}
        self.last_test_metrics: Dict[str, float] = {}
        self.last_val_predictions: List[Dict[str, Any]] = []
        self.last_test_predictions: List[Dict[str, Any]] = []

        self._formal_audit_done = False
        self._run_formal_audit()

    # -------------------------------------------------------------------------
    # Formal model/config audit
    # -------------------------------------------------------------------------

    def _run_formal_audit(self) -> None:
        if self._formal_audit_done:
            return

        if not hasattr(self.model, "audit_formal_configuration"):
            raise RuntimeError(
                "model.py/lit_model.py mismatch: model lacks "
                "audit_formal_configuration()"
            )

        report = self.model.audit_formal_configuration()
        errors: List[str] = []

        if int(report.get("trainable_esm_parameters", -1)) != 0:
            errors.append("ESM2 must be fully frozen")
        if bool(report.get("uses_batchnorm", True)):
            errors.append("BatchNorm is forbidden in redesigned small-batch model")
        if not bool(report.get("wt_mt_shared_state_encoder", False)):
            errors.append("WT/MT state encoder is not shared")
        if not bool(report.get("freeze_backbone", False)):
            errors.append("freeze_backbone must be True")
        if not bool(report.get("frozen_encoder_eval_mode", False)):
            errors.append(
                "Frozen ESM2 is not locked in eval mode; backbone dropout would "
                "inject stochastic WT/MT differences"
            )

        if bool(report.get("use_mamba_branch", False)):
            if (
                self.strict_reviewer_mode
                and not bool(report.get("real_mamba_available", False))
            ):
                errors.append("formal Mamba run requires real mamba_ssm")
            if (
                self.strict_reviewer_mode
                and bool(report.get("allow_mamba_fallback", False))
            ):
                errors.append("allow_mamba_fallback must be False")

        if bool(report.get("use_contrastive", False)) != self.use_contrastive:
            errors.append("model and LitModel disagree on use_contrastive")

        if (
            self.use_contrastive
            and not bool(report.get("require_safe_contrastive_mask", False))
        ):
            errors.append("reviewer-safe CL masks must be required")

        if float(self.cfg.get("unchanged_weight", 0.0)) != 0.0:
            errors.append("unchanged_weight must be 0 in redesigned primary model")
        if float(self.cfg.get("ag_consistency_inner_weight", 0.0)) != 0.0:
            errors.append("legacy antigen-consistency weight must be 0")

        if errors:
            raise RuntimeError(
                "DiffCL-AB formal configuration audit FAILED:\n- "
                + "\n- ".join(errors)
            )

        if not self.target_standardize:
            warnings.warn(
                "target_standardize=False: allowed only as an ablation. "
                "Recommended LOCO training uses train-only z-scoring."
            )

        if int(report.get("model_dim", 256)) > 512:
            warnings.warn(
                f"model_dim={report.get('model_dim')} is larger than recommended "
                "for current AB645 data volume."
            )

        self._formal_audit_done = True

    # -------------------------------------------------------------------------
    # Batch contract
    # -------------------------------------------------------------------------

    def _validate_batch_contract(
        self,
        batch: Mapping[str, Any],
        *,
        training: bool,
    ) -> None:
        required = list(MODEL_BATCH_KEYS) + list(MUTATION_MASK_KEYS) + ["labels"]

        if self.target_standardize:
            required += ["labels_raw", "target_mean", "target_std"]
        if training and self.use_pairwise_loss:
            required += ["pairwise_regression_mask"]
        if training and self.use_contrastive:
            required += ["contrastive_exclusion_masks"]

        missing = [k for k in required if k not in batch]
        if missing:
            raise KeyError(
                "Batch does not match redesigned mutation-aware dataset.py. "
                f"Missing keys: {missing}"
            )

        labels = batch["labels"]
        if not torch.is_tensor(labels):
            raise TypeError("labels must be a tensor")
        b = int(labels.numel())
        if b <= 0:
            raise ValueError("empty batch")

        for key in MODEL_BATCH_KEYS:
            x = batch[key]
            if not torch.is_tensor(x) or x.ndim != 2 or int(x.shape[0]) != b:
                raise ValueError(f"{key} must be [B,L] with B={b}")

        for key in MUTATION_MASK_KEYS:
            x = batch[key]
            if not torch.is_tensor(x) or x.ndim != 2 or int(x.shape[0]) != b:
                raise ValueError(f"{key} must be [B,L] with B={b}")

        if self.target_standardize:
            raw = batch["labels_raw"]
            if not torch.is_tensor(raw) or int(raw.numel()) != b:
                raise ValueError("labels_raw must match labels")

            mean = _safe_scalar_tensor(
                batch["target_mean"], reference=labels, name="target_mean"
            )
            std = _safe_scalar_tensor(
                batch["target_std"], reference=labels, name="target_std"
            )
            if float(std.detach().cpu()) <= 0.0:
                raise ValueError("target_std must be >0")

            reconstructed = labels.reshape(-1) * std + mean
            raw_here = raw.reshape(-1).to(
                device=reconstructed.device,
                dtype=reconstructed.dtype,
            )
            if not torch.allclose(
                reconstructed.detach(),
                raw_here.detach(),
                atol=2e-4,
                rtol=2e-4,
            ):
                raise RuntimeError(
                    "labels/labels_raw/target_mean/target_std are inconsistent"
                )

        if self.use_contrastive:
            masks = batch.get("contrastive_exclusion_masks")
            if not isinstance(masks, Mapping):
                raise TypeError("contrastive_exclusion_masks must be mapping")
            for key in CONTRASTIVE_MASK_KEYS:
                if key not in masks:
                    raise KeyError(f"contrastive mask missing {key!r}")
                m = masks[key]
                if not torch.is_tensor(m) or m.shape != (b, b):
                    raise ValueError(f"contrastive mask {key} must be {(b,b)}")

    def _forward_from_batch(
        self,
        batch: Mapping[str, Any],
        *,
        return_aux: bool,
    ):
        kwargs: Dict[str, Any] = {k: batch[k] for k in MODEL_BATCH_KEYS}
        kwargs.update({k: batch[k] for k in MUTATION_MASK_KEYS})
        kwargs["contrastive_exclusion_masks"] = batch.get(
            "contrastive_exclusion_masks"
        )
        kwargs["return_aux"] = bool(return_aux)
        return self.model(**kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # -------------------------------------------------------------------------
    # Target scale
    # -------------------------------------------------------------------------

    def _raw_pred_and_target(
        self,
        pred_model_scale: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = pred_model_scale.reshape(-1)
        target_model = batch["labels"].reshape(-1).to(
            device=pred.device,
            dtype=pred.dtype,
        )

        if self.target_standardize:
            mean = _safe_scalar_tensor(
                batch["target_mean"], reference=pred, name="target_mean"
            )
            std = _safe_scalar_tensor(
                batch["target_std"], reference=pred, name="target_std"
            )
            pred_raw = pred * std + mean
            target_raw = batch["labels_raw"].reshape(-1).to(
                device=pred.device,
                dtype=pred.dtype,
            )
        else:
            pred_raw = pred
            target_raw = (
                batch["labels_raw"].reshape(-1).to(
                    device=pred.device,
                    dtype=pred.dtype,
                )
                if "labels_raw" in batch else target_model
            )
            mean = pred.new_tensor(float("nan"))
            std = pred.new_tensor(1.0)

        return pred_raw, target_raw, mean, std

    # -------------------------------------------------------------------------
    # Objective
    # -------------------------------------------------------------------------

    def _contrastive_warmup_factor(self) -> float:
        if not self.use_contrastive:
            return 0.0

        start = self.cl_warmup_start_epoch
        end = self.cl_warmup_end_epoch
        epoch = int(self.current_epoch)

        if end <= start:
            return 1.0 if epoch >= start else 0.0
        if epoch <= start:
            return 0.0
        if epoch >= end:
            return 1.0
        return float(epoch - start) / float(end - start)

    def _objective(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        aux: Mapping[str, Any],
        batch: Mapping[str, Any],
    ) -> Dict[str, torch.Tensor]:
        pred = pred.reshape(-1)
        target = target.reshape(-1).to(
            device=pred.device,
            dtype=pred.dtype,
        )

        regression = F.mse_loss(pred, target)

        if self.use_pairwise_loss:
            pairwise, n_pairs = pairwise_mutation_difference_loss(
                pred,
                target,
                batch["pairwise_regression_mask"],
                beta=self.pairwise_beta,
            )
        else:
            pairwise = pred.sum() * 0.0
            n_pairs = pred.new_zeros(())

        if self.use_contrastive:
            cl1 = torch.as_tensor(
                aux["cl1_loss"],
                device=pred.device,
                dtype=pred.dtype,
            ).reshape(())
            cl2 = torch.as_tensor(
                aux["cl2_loss"],
                device=pred.device,
                dtype=pred.dtype,
            ).reshape(())
            warm = pred.new_tensor(self._contrastive_warmup_factor())
        else:
            cl1 = pred.sum() * 0.0
            cl2 = pred.sum() * 0.0
            warm = pred.new_zeros(())

        total = (
            regression
            + self.pairwise_weight * pairwise
            + warm * (self.cl1_weight * cl1 + self.cl2_weight * cl2)
        )

        if not torch.isfinite(total):
            raise FloatingPointError("non-finite total loss")

        return {
            "total": total,
            "regression": regression,
            "pairwise": pairwise,
            "pairwise_n_pairs": n_pairs,
            "cl1": cl1,
            "cl2": cl2,
            "cl_warmup_factor": warm,
            "effective_cl1_weight": warm * self.cl1_weight,
            "effective_cl2_weight": warm * self.cl2_weight,
        }

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def training_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx
        self._validate_batch_contract(batch, training=True)

        out = self._forward_from_batch(batch, return_aux=True)
        if not isinstance(out, Mapping) or "pred" not in out:
            raise TypeError("model(return_aux=True) must return mapping with pred")

        pred = out["pred"].reshape(-1)
        target = batch["labels"].reshape(-1).to(
            device=pred.device, dtype=pred.dtype
        )

        losses = self._objective(pred, target, out, batch)

        pred_raw, target_raw, _, _ = self._raw_pred_and_target(pred, batch)
        raw_mse = F.mse_loss(pred_raw, target_raw)
        raw_rmse = torch.sqrt(raw_mse.clamp_min(0.0))
        bs = int(pred.numel())

        self.log(
            "loss", losses["total"],
            on_step=True, on_epoch=True,
            prog_bar=False, batch_size=bs,
        )
        self.log(
            "train_loss", losses["total"],
            on_step=False, on_epoch=True,
            batch_size=bs,
        )
        self.log(
            "train_regression_mse_std", losses["regression"],
            on_step=False, on_epoch=True,
            batch_size=bs,
        )
        self.log(
            "train_rmse_raw", raw_rmse,
            on_step=False, on_epoch=True,
            batch_size=bs,
        )
        self.log(
            "train_pairwise_loss", losses["pairwise"],
            on_step=False, on_epoch=True,
            batch_size=bs,
        )
        self.log(
            "train_pairwise_pairs", losses["pairwise_n_pairs"],
            on_step=False, on_epoch=True,
            batch_size=bs,
        )
        self.log(
            "train_cl1_loss", losses["cl1"],
            on_step=False, on_epoch=True,
            batch_size=bs,
        )
        self.log(
            "train_cl2_loss", losses["cl2"],
            on_step=False, on_epoch=True,
            batch_size=bs,
        )
        self.log(
            "train_cl_warmup_factor", losses["cl_warmup_factor"],
            on_step=False, on_epoch=True,
            batch_size=bs,
        )
        self.log(
            "train_effective_cl1_weight", losses["effective_cl1_weight"],
            on_step=False, on_epoch=True,
            batch_size=bs,
        )
        self.log(
            "train_effective_cl2_weight", losses["effective_cl2_weight"],
            on_step=False, on_epoch=True,
            batch_size=bs,
        )

        if bs > 1:
            pstd = pred_raw.float().std(unbiased=True)
            tstd = target_raw.float().std(unbiased=True)
            self.log(
                "train_pred_std_raw_batch", pstd,
                on_step=False, on_epoch=True, batch_size=bs,
            )
            self.log(
                "train_target_std_raw_batch", tstd,
                on_step=False, on_epoch=True, batch_size=bs,
            )
            self.log(
                "train_pred_target_std_ratio_batch",
                pstd / tstd.clamp_min(1e-8),
                on_step=False, on_epoch=True, batch_size=bs,
            )

        if (
            self.use_contrastive
            and self.log_detailed_contrastive_stats
            and isinstance(out.get("contrastive_stats"), Mapping)
        ):
            for key, value in out["contrastive_stats"].items():
                if torch.is_tensor(value) and value.numel() == 1:
                    self.log(
                        f"train_{key}", value,
                        on_step=False, on_epoch=True,
                        batch_size=bs,
                    )

        if (
            self.log_detailed_contrastive_stats
            and isinstance(batch.get("contrastive_mask_stats"), Mapping)
        ):
            for key, value in batch["contrastive_mask_stats"].items():
                if torch.is_tensor(value) and value.numel() == 1:
                    self.log(
                        f"train_data_{key}",
                        value.to(device=pred.device, dtype=pred.dtype),
                        on_step=False, on_epoch=True,
                        batch_size=bs,
                    )

        return losses["total"]

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    @staticmethod
    def _batch_metadata(
        batch: Mapping[str, Any],
        n: int,
    ) -> Tuple[List[str], List[str], torch.Tensor]:
        pdbs = (
            [str(x) for x in batch["pdb_ids"]]
            if "pdb_ids" in batch
            else ["UNKNOWN"] * n
        )
        muts = (
            [str(x) for x in batch["mutations"]]
            if "mutations" in batch
            else [""] * n
        )
        if len(pdbs) != n or len(muts) != n:
            raise ValueError("metadata length mismatch")

        if "source_indices" in batch:
            idx = torch.as_tensor(batch["source_indices"]).detach().cpu().long().reshape(-1)
        elif "sample_indices" in batch:
            idx = torch.as_tensor(batch["sample_indices"]).detach().cpu().long().reshape(-1)
        else:
            idx = torch.arange(n, dtype=torch.long)

        if idx.numel() != n:
            raise ValueError("sample/source index length mismatch")
        return pdbs, muts, idx

    def _evaluation_step(
        self,
        batch: Mapping[str, Any],
        *,
        split: str,
    ) -> Dict[str, torch.Tensor]:
        self._validate_batch_contract(batch, training=False)

        pred_model = self._forward_from_batch(batch, return_aux=False)
        if isinstance(pred_model, Mapping):
            pred_model = pred_model["pred"]
        pred_model = pred_model.reshape(-1)

        target_model = batch["labels"].reshape(-1).to(
            device=pred_model.device,
            dtype=pred_model.dtype,
        )
        pred_raw, target_raw, train_mean, train_std = self._raw_pred_and_target(
            pred_model, batch
        )

        n = int(pred_model.numel())
        pdbs, muts, source_indices = self._batch_metadata(batch, n)

        item = {
            "pred_model": pred_model.detach().float().cpu(),
            "target_model": target_model.detach().float().cpu(),
            "pred_raw": pred_raw.detach().float().cpu(),
            "target_raw": target_raw.detach().float().cpu(),
            "pdb_ids": pdbs,
            "mutations": muts,
            "source_indices": source_indices,
            "train_mean_raw": (
                float(train_mean.detach().cpu())
                if torch.isfinite(train_mean) else float("nan")
            ),
            "train_std_raw": float(train_std.detach().cpu()),
        }

        if split == "val":
            self._val_cache.append(item)
        elif split == "test":
            self._test_cache.append(item)
        else:
            raise ValueError(split)

        bs = n
        self.log(
            f"{split}_mse_std_batch",
            F.mse_loss(pred_model, target_model),
            on_step=False, on_epoch=True,
            batch_size=bs, sync_dist=False,
        )
        self.log(
            f"{split}_mse_raw_batch",
            F.mse_loss(pred_raw, target_raw),
            on_step=False, on_epoch=True,
            batch_size=bs, sync_dist=False,
        )

        return {"pred": pred_raw.detach(), "target": target_raw.detach()}

    def validation_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        del batch_idx
        return self._evaluation_step(batch, split="val")

    def test_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        del batch_idx
        return self._evaluation_step(batch, split="test")

    # -------------------------------------------------------------------------
    # Epoch cache / DDP gather
    # -------------------------------------------------------------------------

    @staticmethod
    def _flatten_cache(
        cache: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        for batch in cache:
            pm = np.asarray(batch["pred_model"]).reshape(-1)
            tm = np.asarray(batch["target_model"]).reshape(-1)
            pr = np.asarray(batch["pred_raw"]).reshape(-1)
            tr = np.asarray(batch["target_raw"]).reshape(-1)
            idx = np.asarray(batch["source_indices"]).reshape(-1)
            pdbs = list(batch["pdb_ids"])
            muts = list(batch["mutations"])

            n = len(pr)
            if not (
                len(pm) == len(tm) == len(tr) == len(idx)
                == len(pdbs) == len(muts) == n
            ):
                raise RuntimeError("evaluation cache length mismatch")

            for i in range(n):
                rows.append(
                    {
                        "source_index": int(idx[i]),
                        "pdb_id": str(pdbs[i]),
                        "mutation": str(muts[i]),
                        "pred_model": float(pm[i]),
                        "target_model": float(tm[i]),
                        "pred_raw": float(pr[i]),
                        "target_raw": float(tr[i]),
                        "train_mean_raw": float(batch["train_mean_raw"]),
                        "train_std_raw": float(batch["train_std_raw"]),
                    }
                )
        return rows

    @staticmethod
    def _gather_rows_ddp(
        local_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not (dist.is_available() and dist.is_initialized()):
            return local_rows

        world = dist.get_world_size()
        gathered: List[Optional[List[Dict[str, Any]]]] = [None] * world
        dist.all_gather_object(gathered, local_rows)

        rows: List[Dict[str, Any]] = []
        for part in gathered:
            if part:
                rows.extend(part)
        return rows

    @staticmethod
    def _deduplicate_rows(
        rows: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        by_id: Dict[int, Dict[str, Any]] = {}

        for raw in rows:
            row = dict(raw)
            idx = int(row["source_index"])

            if idx not in by_id:
                by_id[idx] = row
                continue

            old = by_id[idx]
            for key in ("pred_raw", "target_raw", "target_model"):
                if not math.isclose(
                    float(old[key]), float(row[key]),
                    rel_tol=1e-5, abs_tol=1e-5,
                ):
                    raise RuntimeError(
                        f"DDP duplicate source_index={idx} disagrees on {key}"
                    )

        return [by_id[k] for k in sorted(by_id)]

    def _finalize_split(self, split: str) -> None:
        cache = self._val_cache if split == "val" else self._test_cache

        rows = self._flatten_cache(cache)
        rows = self._gather_rows_ddp(rows)
        rows = self._deduplicate_rows(rows)

        if not rows:
            raise RuntimeError(f"No {split} outputs accumulated")

        pred = np.asarray([r["pred_raw"] for r in rows], dtype=np.float64)
        target = np.asarray([r["target_raw"] for r in rows], dtype=np.float64)
        pdb_ids = [str(r["pdb_id"]) for r in rows]

        means = np.asarray(
            [r["train_mean_raw"] for r in rows],
            dtype=np.float64,
        )
        finite_means = means[np.isfinite(means)]
        train_mean_raw = (
            float(finite_means[0])
            if finite_means.size else None
        )

        if finite_means.size > 1 and not np.allclose(
            finite_means, finite_means[0],
            atol=1e-6, rtol=1e-6,
        ):
            raise RuntimeError(f"{split} batches disagree on train target mean")

        metrics = _regression_metrics(
            pred,
            target,
            pdb_ids,
            direction_epsilon=self.direction_epsilon,
            train_mean_raw=train_mean_raw,
        )

        # Also retain the full-split standardized-scale optimization metric.
        pred_model = np.asarray(
            [r["pred_model"] for r in rows], dtype=np.float64
        )
        target_model = np.asarray(
            [r["target_model"] for r in rows], dtype=np.float64
        )
        mse_std = float(np.mean((pred_model - target_model) ** 2))
        metrics["mse_std"] = mse_std
        metrics["rmse_std"] = float(math.sqrt(mse_std))

        # `val_mse` here is the full-split RAW-unit MSE.
        prog = {"rmse", "pearson_corr", "spearman_corr", "r2"}
        for key, value in metrics.items():
            if not math.isfinite(float(value)):
                continue
            self.log(
                f"{split}_{key}",
                float(value),
                on_step=False, on_epoch=True,
                prog_bar=(split == "val" and key in prog),
                sync_dist=False,
            )

        if split == "val":
            self.last_val_metrics = dict(metrics)
            self.last_val_predictions = rows
        else:
            self.last_test_metrics = dict(metrics)
            self.last_test_predictions = rows

        cache.clear()

    def on_validation_epoch_start(self) -> None:
        self._val_cache.clear()

    def on_test_epoch_start(self) -> None:
        self._test_cache.clear()

    def on_validation_epoch_end(self) -> None:
        self._finalize_split("val")

    def on_test_epoch_end(self) -> None:
        self._finalize_split("test")

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    def predict_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, Any]:
        del batch_idx, dataloader_idx
        self._validate_batch_contract(batch, training=False)

        pred_model = self._forward_from_batch(batch, return_aux=False)
        if isinstance(pred_model, Mapping):
            pred_model = pred_model["pred"]
        pred_model = pred_model.reshape(-1)

        pred_raw, target_raw, _, _ = self._raw_pred_and_target(
            pred_model, batch
        )
        n = int(pred_model.numel())
        pdbs, muts, source_indices = self._batch_metadata(batch, n)

        return {
            "pred": pred_raw.detach(),
            "pred_raw": pred_raw.detach(),
            "pred_model_scale": pred_model.detach(),
            "target_raw": target_raw.detach(),
            "source_indices": source_indices,
            "pdb_ids": pdbs,
            "mutations": muts,
        }

    # -------------------------------------------------------------------------
    # Optimizer / warmup + cosine scheduler
    # -------------------------------------------------------------------------

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters")

        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_eps,
        )

        if not self.use_lr_scheduler:
            return optimizer

        try:
            total_steps = int(self.trainer.estimated_stepping_batches)
        except Exception:
            total_steps = int(self.cfg.get("estimated_stepping_batches", 0))

        if total_steps <= 0:
            raise RuntimeError(
                "Unable to determine estimated_stepping_batches for cosine scheduler"
            )

        warmup_steps = int(round(total_steps * self.warmup_ratio))
        if self.warmup_ratio > 0.0:
            warmup_steps = max(1, warmup_steps)
        warmup_steps = min(warmup_steps, max(0, total_steps - 1))

        min_ratio = self.min_lr / self.lr

        def lr_lambda(step: int) -> float:
            step = int(step)

            if warmup_steps > 0 and step < warmup_steps:
                return max(1e-8, float(step + 1) / float(warmup_steps))

            decay_steps = max(1, total_steps - warmup_steps)
            progress = float(step - warmup_steps) / float(decay_steps)
            progress = min(max(progress, 0.0), 1.0)

            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "lr",
            },
        }

    # -------------------------------------------------------------------------
    # Human-readable audit
    # -------------------------------------------------------------------------

    def on_fit_start(self) -> None:
        self._run_formal_audit()

        if not self.trainer.is_global_zero:
            return

        r = self.model.audit_formal_configuration()

        print("=" * 100)
        print("DiffCL-AB Lite: formal training audit")
        print("=" * 100)
        print(f"training stage                : {self.training_stage}")
        print(f"loss                          : {self.loss_name}")
        print(f"target standardization        : {self.target_standardize}")
        print(f"lr / weight decay             : {self.lr} / {self.weight_decay}")
        print(f"warmup ratio / min lr         : {self.warmup_ratio} / {self.min_lr}")
        print(f"use pairwise loss             : {self.use_pairwise_loss}")
        print(f"pairwise weight / beta        : {self.pairwise_weight} / {self.pairwise_beta}")
        print(f"use contrastive               : {self.use_contrastive}")
        print(f"CL-I / CL-II base weights     : {self.cl1_weight} / {self.cl2_weight}")
        print(
            "CL warmup epochs              : "
            f"{self.cl_warmup_start_epoch} -> {self.cl_warmup_end_epoch}"
        )
        print(f"model dim                     : {r.get('model_dim')}")
        print(f"latent tokens                 : {r.get('num_latent_tokens')}")
        print(
            f"use Mamba / layers            : "
            f"{r.get('use_mamba_branch')} / {r.get('mamba_layers')}"
        )
        print(f"WT/MT encoder shared          : {r.get('wt_mt_shared_state_encoder')}")
        print(f"uses BatchNorm                : {r.get('uses_batchnorm')}")
        print(f"total parameters              : {int(r.get('total_parameters', 0)):,}")
        print(f"trainable parameters          : {int(r.get('trainable_parameters', 0)):,}")
        print(f"downstream parameters         : {int(r.get('downstream_parameters', 0)):,}")
        print(f"trainable ESM parameters      : {int(r.get('trainable_esm_parameters', 0)):,}")
        print(f"frozen ESM eval mode          : {r.get('frozen_encoder_eval_mode')}")
        print(f"long-sequence strategy        : {r.get('long_sequence_strategy')}")
        print(f"strict reviewer mode          : {self.strict_reviewer_mode}")
        print("=" * 100)

    # -------------------------------------------------------------------------
    # External LOCO trainer helpers
    # -------------------------------------------------------------------------

    def get_last_validation_metrics(self) -> Dict[str, float]:
        return dict(self.last_val_metrics)

    def get_last_test_metrics(self) -> Dict[str, float]:
        return dict(self.last_test_metrics)

    def get_last_validation_predictions(self) -> List[Dict[str, Any]]:
        return [dict(x) for x in self.last_val_predictions]

    def get_last_test_predictions(self) -> List[Dict[str, Any]]:
        return [dict(x) for x in self.last_test_predictions]


__all__ = [
    "LitModel",
    "pairwise_mutation_difference_loss",
]
