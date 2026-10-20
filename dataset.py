# -*- coding: utf-8 -*-
"""
Mutation-aware LOCO/refit dataset for Shared Mutation-Aware DiffCL-AB Lite
==========================================================================

Primary AB645 protocol:
    data/optimized_loco_splits/AB645/loco90_refit/fold_XX_<PDB>/

Stage 1:
    inner_train.csv + val.csv, test supplied only for leakage audit.

Stage 2:
    refit_train.csv, val_data=None, test supplied for final one-shot evaluation.

Safety properties
-----------------
1. No row-level random splitting is performed here.
2. PDB/homology metadata never become prediction features.
3. Train-only target z-scoring.
4. Mutation-site and ±radius local-window masks are derived from WT/MT sequences.
5. Same-complex/exact/direct-homology pairs are neutral in contrastive learning.
6. Split separation is re-audited fail-closed.
7. No silent sequence truncation; long token tensors are passed to model.py,
   which handles them by frozen-ESM sliding windows.
8. Pairwise regression supervision uses same-complex DISTINCT-input pairs only.
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoConfig, AutoTokenizer


CATEGORY_KEYS = ("wt_ab", "wt_ag", "mt_ab", "mt_ag")

AB_CATEGORY_COLUMNS: Mapping[str, Tuple[str, ...]] = {
    "wt_ab": ("antibody_light_seq", "antibody_heavy_seq"),
    "wt_ag": ("antigen_a_seq", "antigen_b_seq"),
    "mt_ab": ("antibody_light_seq_mut", "antibody_heavy_seq_mut"),
    "mt_ag": ("antigen_a_seq_mut", "antigen_b_seq_mut"),
}

S1131_CATEGORY_COLUMNS: Mapping[str, Tuple[str, ...]] = {
    "wt_ab": ("a",),
    "wt_ag": ("b",),
    "mt_ab": ("a_mut",),
    "mt_ag": ("b_mut",),
}

LABEL_CANDIDATES = ("ddG", "ddg", "label", "labels")
PDB_CANDIDATES = ("PDB", "pdb", "complex_id", "Complex", "#PDB")
MUTATION_CANDIDATES = (
    "Mutation", "mutation", "Mutation2", "mutation_clean"
)


# =============================================================================
# Utilities / schema
# =============================================================================

def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse bool from {value!r}")


def _fraction(value: Any) -> float:
    x = float(value)
    return x / 100.0 if x > 1.0 else x


def _find_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    required: bool = True,
) -> Optional[str]:
    lower = {str(c).lower(): str(c) for c in df.columns}
    for c in candidates:
        hit = lower.get(c.lower())
        if hit is not None:
            return hit
    if required:
        raise KeyError(
            f"None of columns {list(candidates)} found. "
            f"Available: {list(df.columns)}"
        )
    return None


def normalize_sequence(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(
        r"[^A-Z]", "", str(value).strip().upper()
    )


def normalize_pdb_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().upper()


def join_partner_components(parts: Sequence[str]) -> str:
    return "".join(
        normalize_sequence(x)
        for x in parts
        if normalize_sequence(x)
    )


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    category_columns: Mapping[str, Tuple[str, ...]]
    label_col: str
    pdb_col: str
    mutation_col: Optional[str]


def infer_schema(
    df: pd.DataFrame,
    dataset_name: Optional[str] = None,
) -> DatasetSchema:
    name = str(dataset_name or "").strip().upper()

    if name == "S1131" or all(
        c in df.columns for c in ("a", "b", "a_mut", "b_mut")
    ):
        category = S1131_CATEGORY_COLUMNS
        canonical_name = "S1131"
    else:
        category = AB_CATEGORY_COLUMNS
        canonical_name = (
            "AB1101" if name == "AB1101" else "AB645"
        )

    missing = [
        c
        for cols in category.values()
        for c in cols
        if c not in df.columns
    ]
    if missing:
        raise KeyError(
            f"Dataset schema missing sequence columns: {sorted(set(missing))}"
        )

    return DatasetSchema(
        name=canonical_name,
        category_columns=category,
        label_col=str(_find_column(df, LABEL_CANDIDATES, True)),
        pdb_col=str(_find_column(df, PDB_CANDIDATES, True)),
        mutation_col=_find_column(
            df, MUTATION_CANDIDATES, False
        ),
    )


# =============================================================================
# Mutation-local masks
# =============================================================================

def _component_mutation_masks(
    wt_components: Sequence[str],
    mt_components: Sequence[str],
    radius: int,
) -> Tuple[Tuple[bool, ...], Tuple[bool, ...], int]:
    if len(wt_components) != len(mt_components):
        raise RuntimeError("WT/MT component-count mismatch")

    mutation: List[bool] = []
    window: List[bool] = []
    total = 0

    for ci, (wa, ma) in enumerate(
        zip(wt_components, mt_components)
    ):
        wt = normalize_sequence(wa)
        mt = normalize_sequence(ma)

        if len(wt) != len(mt):
            raise RuntimeError(
                "WT/MT component lengths differ. Current benchmark path "
                "supports substitutions only; no silent indel alignment. "
                f"component={ci}, WT={len(wt)}, MT={len(mt)}"
            )

        local_mut = [a != b for a, b in zip(wt, mt)]
        local_win = [False] * len(wt)

        for pos, changed in enumerate(local_mut):
            if not changed:
                continue
            total += 1
            lo = max(0, pos - radius)
            hi = min(len(wt), pos + radius + 1)
            for j in range(lo, hi):
                local_win[j] = True

        mutation.extend(local_mut)
        window.extend(local_win)

    return tuple(mutation), tuple(window), total


# =============================================================================
# Direct MMseqs index
# =============================================================================

class DirectHomologyIndex:
    """
    Exact/direct homology only.  No transitive closure.
    """

    def __init__(
        self,
        fasta_path: Path,
        hits_path: Path,
        min_seq_id: float = 0.90,
        min_cov: float = 0.80,
    ) -> None:
        self.fasta_path = Path(fasta_path)
        self.hits_path = Path(hits_path)
        self.min_seq_id = _fraction(min_seq_id)
        self.min_cov = _fraction(min_cov)

        if not self.fasta_path.is_file():
            raise FileNotFoundError(self.fasta_path)
        if not self.hits_path.is_file():
            raise FileNotFoundError(self.hits_path)

        self.sid_to_seq = self._read_fasta(self.fasta_path)
        self.seq_to_sid: Dict[str, str] = {}
        for sid, seq in self.sid_to_seq.items():
            if seq in self.seq_to_sid:
                # The reviewer-grade FASTA should already contain unique input
                # sequences.  Preserve one ID if duplicate aliases exist.
                continue
            self.seq_to_sid[seq] = sid

        self.adj: Dict[str, Set[str]] = defaultdict(set)
        self.n_kept_hits = 0
        self._read_hits()

    @staticmethod
    def _read_fasta(path: Path) -> Dict[str, str]:
        out: Dict[str, str] = {}
        sid = None
        chunks: List[str] = []

        def flush() -> None:
            nonlocal sid, chunks
            if sid is None:
                return
            seq = normalize_sequence("".join(chunks))
            if not seq:
                raise ValueError(f"Empty FASTA sequence for {sid}")
            out[sid] = seq

        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    flush()
                    sid = line[1:].split()[0]
                    chunks = []
                else:
                    if sid is None:
                        raise ValueError("Malformed FASTA")
                    chunks.append(line)
            flush()

        if not out:
            raise ValueError(f"No FASTA records in {path}")
        return out

    def _read_hits(self) -> None:
        with self.hits_path.open("r", encoding="utf-8") as f:
            for raw in f:
                p = raw.rstrip("\n").split("\t")
                if len(p) < 5:
                    continue

                q, t = p[0], p[1]
                if q == t:
                    continue
                if q not in self.sid_to_seq or t not in self.sid_to_seq:
                    continue

                ident = _fraction(p[2])
                qcov = _fraction(p[3])
                tcov = _fraction(p[4])

                if (
                    ident >= self.min_seq_id
                    and qcov >= self.min_cov
                    and tcov >= self.min_cov
                ):
                    self.adj[q].add(t)
                    self.adj[t].add(q)
                    self.n_kept_hits += 1

    def are_related(self, seq_a: str, seq_b: str) -> bool:
        a = normalize_sequence(seq_a)
        b = normalize_sequence(seq_b)
        if not a or not b:
            return False
        if a == b:
            return True

        sa = self.seq_to_sid.get(a)
        sb = self.seq_to_sid.get(b)
        if sa is None or sb is None:
            return False
        return sb in self.adj.get(sa, set())

    def components_related(
        self,
        a: Sequence[str],
        b: Sequence[str],
    ) -> bool:
        aa = [normalize_sequence(x) for x in a if normalize_sequence(x)]
        bb = [normalize_sequence(x) for x in b if normalize_sequence(x)]
        return any(
            self.are_related(x, y)
            for x in aa
            for y in bb
        )

    def validate_coverage(
        self,
        sequences: Iterable[str],
        strict: bool = True,
    ) -> List[str]:
        missing: List[str] = []
        seen: Set[str] = set()
        for raw in sequences:
            seq = normalize_sequence(raw)
            if not seq or seq in seen:
                continue
            seen.add(seq)
            if seq not in self.seq_to_sid:
                missing.append(seq)

        if strict and missing:
            raise RuntimeError(
                f"{len(missing)} input component sequences are absent from "
                "the reviewer-grade MMseqs FASTA; data/artifact version mismatch."
            )
        return missing


# =============================================================================
# Row dataset
# =============================================================================

class ProteinMutationDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        schema: DatasetSchema,
        split_name: str,
        target_mean: float,
        target_std: float,
        target_standardize: bool = True,
        mutation_window_radius: int = 8,
        require_sequence_change: bool = True,
    ) -> None:
        self.df = dataframe.copy().reset_index(drop=False).rename(
            columns={"index": "_frame_index"}
        )
        self.schema = schema
        self.split_name = str(split_name)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.target_standardize = bool(target_standardize)
        self.mutation_window_radius = int(mutation_window_radius)
        self.require_sequence_change = bool(require_sequence_change)

        if self.target_std <= 0 or not math.isfinite(self.target_std):
            raise ValueError("target_std must be finite and >0")

        self.records: List[Dict[str, Any]] = []
        self._build()

    def _build(self) -> None:
        errors: List[str] = []

        for i, row in self.df.iterrows():
            pdb = normalize_pdb_id(row[self.schema.pdb_col])
            if not pdb:
                errors.append(f"row {i}: empty PDB")

            raw = float(row[self.schema.label_col])
            if not math.isfinite(raw):
                errors.append(f"row {i}: non-finite label")

            label = (
                (raw - self.target_mean) / self.target_std
                if self.target_standardize
                else raw
            )

            components: Dict[str, Tuple[str, ...]] = {}
            sequences: Dict[str, str] = {}
            for key, cols in self.schema.category_columns.items():
                comp = tuple(
                    normalize_sequence(row[c]) for c in cols
                )
                seq = join_partner_components(comp)
                if not seq:
                    errors.append(f"row {i}: empty {key}")
                components[key] = comp
                sequences[key] = seq

            ab_mut, ab_win, n_ab = _component_mutation_masks(
                components["wt_ab"],
                components["mt_ab"],
                self.mutation_window_radius,
            )
            ag_mut, ag_win, n_ag = _component_mutation_masks(
                components["wt_ag"],
                components["mt_ag"],
                self.mutation_window_radius,
            )
            n_total = n_ab + n_ag

            if self.require_sequence_change and n_total <= 0:
                errors.append(
                    f"row {i}: WT/MT inputs contain no residue change"
                )

            mutation = ""
            if self.schema.mutation_col is not None:
                v = row[self.schema.mutation_col]
                mutation = "" if pd.isna(v) else str(v)

            if "_ab645_row_id" in row.index:
                source_index = int(row["_ab645_row_id"])
            else:
                source_index = int(row["_frame_index"])

            full_key = tuple(
                sequences[k] for k in CATEGORY_KEYS
            )

            self.records.append(
                {
                    "pdb_id": pdb,
                    "mutation": mutation,
                    "label": float(label),
                    "label_raw": float(raw),
                    "sample_index": i,
                    "source_index": source_index,
                    "components": components,
                    "sequences": sequences,
                    "full_input_key": full_key,
                    "ab_mutation_residue_mask": ab_mut,
                    "ag_mutation_residue_mask": ag_mut,
                    "ab_window_residue_mask": ab_win,
                    "ag_window_residue_mask": ag_win,
                    "num_mutations_ab": n_ab,
                    "num_mutations_ag": n_ag,
                    "num_mutations_total": n_total,
                }
            )

        if errors:
            raise RuntimeError(
                f"{self.split_name} dataset validation failed:\n- "
                + "\n- ".join(errors[:30])
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.records[index]

    def all_pdbs(self) -> Set[str]:
        return {str(r["pdb_id"]) for r in self.records}

    def all_input_keys(self) -> Set[Tuple[str, ...]]:
        return {tuple(r["full_input_key"]) for r in self.records}

    def all_component_sequences(self) -> Iterator[str]:
        for r in self.records:
            for key in CATEGORY_KEYS:
                for seq in r["components"][key]:
                    if seq:
                        yield str(seq)

    def component_sequence_set(self) -> Set[str]:
        return set(self.all_component_sequences())

    def pdb_to_indices(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = defaultdict(list)
        for i, r in enumerate(self.records):
            out[str(r["pdb_id"])].append(i)
        return dict(out)


# =============================================================================
# Collator
# =============================================================================

class MutationAwareCollator:
    def __init__(
        self,
        tokenizer,
        model_max_length: int,
        homology_index: DirectHomologyIndex,
        target_mean: float,
        target_std: float,
        target_standardize: bool,
        allow_sequence_truncation: bool = False,
        long_sequence_strategy: str = "sliding_window",
    ) -> None:
        self.tokenizer = tokenizer
        self.model_max_length = int(model_max_length)
        self.homology_index = homology_index
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.target_standardize = bool(target_standardize)
        self.allow_sequence_truncation = bool(
            allow_sequence_truncation
        )
        self.long_sequence_strategy = str(
            long_sequence_strategy
        ).lower()

        self.special_ids = set(
            int(x)
            for x in getattr(
                self.tokenizer, "all_special_ids", []
            )
        )
        try:
            self.n_special = int(
                self.tokenizer.num_special_tokens_to_add(
                    pair=False
                )
            )
        except Exception:
            self.n_special = 2

    def _padding_length(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> int:
        max_res = max(
            len(str(r["sequences"][key]))
            for r in records
            for key in CATEGORY_KEYS
        )
        required = max_res + self.n_special

        if (
            required > self.model_max_length
            and not self.allow_sequence_truncation
            and self.long_sequence_strategy == "error"
        ):
            raise RuntimeError(
                f"Input requires {required} tokens > native "
                f"{self.model_max_length}; truncation is forbidden"
            )

        return (
            min(required, self.model_max_length)
            if self.allow_sequence_truncation
            else required
        )

    def _encode(
        self,
        sequences: Sequence[str],
        length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            list(sequences),
            add_special_tokens=True,
            padding="max_length",
            truncation=self.allow_sequence_truncation,
            max_length=length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return enc["input_ids"].long(), enc["attention_mask"].long()

    def _residue_to_token_mask(
        self,
        residue_masks: Sequence[Sequence[bool]],
        ids: torch.Tensor,
        attention: torch.Tensor,
    ) -> torch.Tensor:
        b, l = ids.shape
        out = torch.zeros((b, l), dtype=torch.bool)

        for i in range(b):
            residue_positions: List[int] = []
            for j in range(l):
                if int(attention[i, j]) == 0:
                    continue
                token_id = int(ids[i, j])
                if token_id in self.special_ids:
                    continue
                residue_positions.append(j)

            if len(residue_positions) != len(residue_masks[i]):
                raise RuntimeError(
                    "Tokenizer/residue alignment mismatch: "
                    f"sample={i}, residues={len(residue_masks[i])}, "
                    f"tokens={len(residue_positions)}"
                )

            for token_pos, flag in zip(
                residue_positions, residue_masks[i]
            ):
                if flag:
                    out[i, token_pos] = True

        return out

    @staticmethod
    def _same_complex(
        records: Sequence[Mapping[str, Any]],
    ) -> torch.Tensor:
        b = len(records)
        out = torch.zeros((b, b), dtype=torch.bool)
        for i in range(b):
            for j in range(i + 1, b):
                if records[i]["pdb_id"] == records[j]["pdb_id"]:
                    out[i, j] = out[j, i] = True
        return out

    @staticmethod
    def _exact_full(
        records: Sequence[Mapping[str, Any]],
    ) -> torch.Tensor:
        b = len(records)
        out = torch.zeros((b, b), dtype=torch.bool)
        for i in range(b):
            for j in range(i + 1, b):
                if (
                    records[i]["full_input_key"]
                    == records[j]["full_input_key"]
                ):
                    out[i, j] = out[j, i] = True
        return out

    def _category_homology(
        self,
        records: Sequence[Mapping[str, Any]],
        category: str,
    ) -> torch.Tensor:
        b = len(records)
        out = torch.zeros((b, b), dtype=torch.bool)
        for i in range(b):
            for j in range(i + 1, b):
                if self.homology_index.components_related(
                    records[i]["components"][category],
                    records[j]["components"][category],
                ):
                    out[i, j] = out[j, i] = True
        return out

    def __call__(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if not records:
            raise ValueError("empty batch")

        length = self._padding_length(records)

        seqs = {
            key: [
                str(r["sequences"][key]) for r in records
            ]
            for key in CATEGORY_KEYS
        }
        wt_ab_ids, wt_ab_mask = self._encode(
            seqs["wt_ab"], length
        )
        wt_ag_ids, wt_ag_mask = self._encode(
            seqs["wt_ag"], length
        )
        mt_ab_ids, mt_ab_mask = self._encode(
            seqs["mt_ab"], length
        )
        mt_ag_ids, mt_ag_mask = self._encode(
            seqs["mt_ag"], length
        )

        shapes = {
            tuple(wt_ab_ids.shape),
            tuple(wt_ag_ids.shape),
            tuple(mt_ab_ids.shape),
            tuple(mt_ag_ids.shape),
        }
        if len(shapes) != 1:
            raise RuntimeError(
                f"Four categories have inconsistent token shapes: {shapes}"
            )

        ab_mutation_mask = self._residue_to_token_mask(
            [r["ab_mutation_residue_mask"] for r in records],
            wt_ab_ids,
            wt_ab_mask,
        )
        ag_mutation_mask = self._residue_to_token_mask(
            [r["ag_mutation_residue_mask"] for r in records],
            wt_ag_ids,
            wt_ag_mask,
        )
        ab_mutation_window_mask = self._residue_to_token_mask(
            [r["ab_window_residue_mask"] for r in records],
            wt_ab_ids,
            wt_ab_mask,
        )
        ag_mutation_window_mask = self._residue_to_token_mask(
            [r["ag_window_residue_mask"] for r in records],
            wt_ag_ids,
            wt_ag_mask,
        )

        same_complex_mask = self._same_complex(records)
        exact_full = self._exact_full(records)
        global_exclusion = same_complex_mask | exact_full
        global_exclusion.fill_diagonal_(False)

        exclusion: Dict[str, torch.Tensor] = {
            "all": global_exclusion
        }
        mask_stats: Dict[str, torch.Tensor] = {
            "batch_size": torch.tensor(
                float(len(records)), dtype=torch.float32
            ),
            "same_complex_unique_pairs": torch.triu(
                same_complex_mask, diagonal=1
            ).sum().float(),
            "exact_full_input_unique_pairs": torch.triu(
                exact_full, diagonal=1
            ).sum().float(),
        }

        b = len(records)
        possible = max(b * (b - 1) // 2, 1)

        for key in CATEGORY_KEYS:
            h = self._category_homology(records, key)
            h.fill_diagonal_(False)
            exclusion[key] = h
            combined = global_exclusion | h
            valid = ~combined
            valid.fill_diagonal_(False)

            mask_stats[f"{key}_homology_unique_pairs"] = (
                torch.triu(h, diagonal=1).sum().float()
            )
            mask_stats[
                f"{key}_combined_excluded_pair_fraction"
            ] = (
                torch.triu(combined, diagonal=1).sum().float()
                / float(possible)
            )
            mask_stats[
                f"{key}_mean_candidate_negatives"
            ] = valid.sum(dim=1).float().mean()
            mask_stats[
                f"{key}_zero_candidate_negative_anchors"
            ] = (valid.sum(dim=1) == 0).sum().float()

        # Same-complex but exact duplicate inputs are excluded from pairwise
        # ΔΔG-difference auxiliary supervision.
        pairwise_regression_mask = (
            same_complex_mask & (~exact_full)
        )
        pairwise_regression_mask.fill_diagonal_(False)

        return {
            "wt_ab_inputs_ids": wt_ab_ids,
            "wt_ab_inputs_mask": wt_ab_mask,
            "mut_ab_inputs_ids": mt_ab_ids,
            "mt_ab_inputs_mask": mt_ab_mask,
            "wt_ag_inputs_ids": wt_ag_ids,
            "wt_ag_inputs_mask": wt_ag_mask,
            "mut_ag_inputs_ids": mt_ag_ids,
            "mt_ag_inputs_mask": mt_ag_mask,

            "labels": torch.tensor(
                [float(r["label"]) for r in records],
                dtype=torch.float32,
            ),
            "labels_raw": torch.tensor(
                [float(r["label_raw"]) for r in records],
                dtype=torch.float32,
            ),
            "target_mean": torch.tensor(
                self.target_mean, dtype=torch.float32
            ),
            "target_std": torch.tensor(
                self.target_std, dtype=torch.float32
            ),
            "target_standardized": torch.tensor(
                self.target_standardize,
                dtype=torch.bool,
            ),

            "ab_mutation_mask": ab_mutation_mask,
            "ag_mutation_mask": ag_mutation_mask,
            "ab_mutation_window_mask": ab_mutation_window_mask,
            "ag_mutation_window_mask": ag_mutation_window_mask,

            "contrastive_exclusion_masks": exclusion,
            "contrastive_mask_stats": mask_stats,
            "same_complex_mask": same_complex_mask,
            "pairwise_regression_mask": pairwise_regression_mask,

            "pdb_ids": [
                str(r["pdb_id"]) for r in records
            ],
            "mutations": [
                str(r["mutation"]) for r in records
            ],
            "sample_indices": torch.tensor(
                [int(r["sample_index"]) for r in records],
                dtype=torch.long,
            ),
            "source_indices": torch.tensor(
                [int(r["source_index"]) for r in records],
                dtype=torch.long,
            ),
            "num_mutations_ab": torch.tensor(
                [int(r["num_mutations_ab"]) for r in records],
                dtype=torch.long,
            ),
            "num_mutations_ag": torch.tensor(
                [int(r["num_mutations_ag"]) for r in records],
                dtype=torch.long,
            ),
            "num_mutations_total": torch.tensor(
                [int(r["num_mutations_total"]) for r in records],
                dtype=torch.long,
            ),
            "token_length": torch.tensor(
                float(length), dtype=torch.float32
            ),
        }


# =============================================================================
# Training sampler
# =============================================================================

class MutationAwareBatchSampler(Sampler[List[int]]):
    """
    Complex-tempered sampler.

    When pairwise mode is active, deliberately includes same-complex distinct
    mutations, then fills remaining batch slots with conflict-diverse examples.
    """

    def __init__(
        self,
        dataset: ProteinMutationDataset,
        homology_index: DirectHomologyIndex,
        batch_size: int,
        seed: int = 42,
        alpha: float = 0.5,
        same_complex_pairs_per_batch: int = 0,
        prefer_contrastive_diversity: bool = False,
    ) -> None:
        self.dataset = dataset
        self.homology_index = homology_index
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.alpha = float(alpha)
        self.same_complex_pairs_per_batch = int(
            same_complex_pairs_per_batch
        )
        self.prefer_contrastive_diversity = bool(
            prefer_contrastive_diversity
        )
        self.epoch = 0
        self.n = len(dataset)
        self.steps = int(math.ceil(self.n / self.batch_size))

        if self.batch_size < 2:
            raise ValueError("batch_size must be >=2")
        if 2 * self.same_complex_pairs_per_batch > self.batch_size:
            raise ValueError(
                "2*same_complex_pairs_per_batch exceeds batch_size"
            )

        self.pdb_to_indices = dataset.pdb_to_indices()
        self.index_to_pdb = [
            str(r["pdb_id"]) for r in dataset.records
        ]
        self.pair_pdbs = [
            p
            for p, ids in self.pdb_to_indices.items()
            if len(ids) >= 2
        ]

        weights = []
        for i in range(self.n):
            nc = len(
                self.pdb_to_indices[
                    self.index_to_pdb[i]
                ]
            )
            weights.append(nc ** (-self.alpha))
        self.row_probs = torch.tensor(
            weights, dtype=torch.double
        )
        self.row_probs /= self.row_probs.sum()

        self.conflicts: List[Set[int]] = [
            set() for _ in range(self.n)
        ]
        if self.prefer_contrastive_diversity:
            self._build_conflicts()

    def _records_conflict(self, i: int, j: int) -> bool:
        a = self.dataset.records[i]
        b = self.dataset.records[j]

        if a["pdb_id"] == b["pdb_id"]:
            return True
        if a["full_input_key"] == b["full_input_key"]:
            return True

        for key in CATEGORY_KEYS:
            if self.homology_index.components_related(
                a["components"][key],
                b["components"][key],
            ):
                return True
        return False

    def _build_conflicts(self) -> None:
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self._records_conflict(i, j):
                    self.conflicts[i].add(j)
                    self.conflicts[j].add(i)

    def _draw_row(
        self,
        generator: torch.Generator,
        batch: List[int],
    ) -> int:
        pool = torch.multinomial(
            self.row_probs,
            num_samples=min(max(32, 4 * self.batch_size), 256),
            replacement=True,
            generator=generator,
        ).tolist()

        for idx in pool:
            idx = int(idx)
            if idx in batch:
                continue
            if self.prefer_contrastive_diversity and any(
                idx in self.conflicts[j] for j in batch
            ):
                continue
            return idx

        for idx in pool:
            idx = int(idx)
            if idx not in batch:
                return idx
        return int(pool[0])

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        self.epoch += 1

        for _ in range(self.steps):
            batch: List[int] = []
            used_pdb: Set[str] = set()

            # Force requested same-complex pairs.
            if self.same_complex_pairs_per_batch > 0:
                order = torch.randperm(
                    len(self.pair_pdbs), generator=g
                ).tolist()

                for pos in order:
                    if (
                        len(used_pdb)
                        >= self.same_complex_pairs_per_batch
                    ):
                        break

                    pdb = self.pair_pdbs[pos]
                    ids = self.pdb_to_indices[pdb]
                    perm = torch.randperm(
                        len(ids), generator=g
                    ).tolist()

                    chosen = None
                    for a_pos in range(len(perm)):
                        for b_pos in range(
                            a_pos + 1, len(perm)
                        ):
                            a = ids[perm[a_pos]]
                            b = ids[perm[b_pos]]
                            if (
                                self.dataset.records[a]["full_input_key"]
                                != self.dataset.records[b]["full_input_key"]
                            ):
                                chosen = (a, b)
                                break
                        if chosen is not None:
                            break

                    if chosen is not None:
                        batch.extend(chosen)
                        used_pdb.add(pdb)

            while len(batch) < self.batch_size:
                batch.append(self._draw_row(g, batch))

            yield batch[: self.batch_size]

    def __len__(self) -> int:
        return self.steps


# =============================================================================
# Split audit
# =============================================================================

def _resolve_mmseqs_dir(args: Any, dataset_name: str) -> Path:
    explicit = getattr(args, "mmseqs_dir", None)
    candidates: List[Path] = []

    if explicit:
        candidates.append(Path(str(explicit)).expanduser())

    data_folder = getattr(args, "data_folder", None)
    if data_folder:
        p = Path(str(data_folder)).expanduser()
        for _ in range(7):
            candidates.append(p / "mmseqs")
            p = p.parent

    candidates.append(
        Path("./data/reviewer_grade_splits_final")
        / dataset_name
        / "purged5_id90_cov80"
        / "mmseqs"
    )

    for p in candidates:
        if (
            (p / "all_unique_input_sequences.fasta").is_file()
            and (p / "all_vs_all.tsv").is_file()
            and (p / "pdb_homology_edges.csv").is_file()
        ):
            return p.resolve()

    raise FileNotFoundError(
        "Could not resolve reviewer-grade MMseqs directory. "
        "Pass args.mmseqs_dir explicitly."
    )


def _load_relation_edges(
    mmseqs_dir: Path,
    mode: str,
    min_seq_id: float,
    min_cov: float,
) -> pd.DataFrame:
    p = mmseqs_dir / "pdb_homology_edges.csv"
    df = pd.read_csv(p)

    required = {
        "pdb_a",
        "pdb_b",
        "max_identity",
        "max_qcov",
        "max_tcov",
        "reasons",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"{p} missing columns {sorted(missing)}"
        )

    out = df.copy()
    if mode == "homology90":
        keep = (
            (pd.to_numeric(out["max_identity"]) >= min_seq_id - 1e-12)
            & (pd.to_numeric(out["max_qcov"]) >= min_cov - 1e-12)
            & (pd.to_numeric(out["max_tcov"]) >= min_cov - 1e-12)
        )
        out = out.loc[keep]
    elif mode == "exact":
        reasons = out["reasons"].fillna("").astype(str)
        exact_reason = reasons.str.contains(
            "exact_sequence", regex=False
        )
        exact_numeric = (
            (pd.to_numeric(out["max_identity"]) >= 1.0 - 1e-9)
            & (pd.to_numeric(out["max_qcov"]) >= 1.0 - 1e-9)
            & (pd.to_numeric(out["max_tcov"]) >= 1.0 - 1e-9)
        )
        out = out.loc[exact_reason | exact_numeric]
    else:
        raise ValueError(
            "split_relation_mode must be homology90 or exact"
        )

    return out.reset_index(drop=True)


def _edge_count_between(
    left: Set[str],
    right: Set[str],
    edges: pd.DataFrame,
) -> int:
    n = 0
    for _, r in edges.iterrows():
        a = normalize_pdb_id(r["pdb_a"])
        b = normalize_pdb_id(r["pdb_b"])
        if (
            (a in left and b in right)
            or (b in left and a in right)
        ):
            n += 1
    return n


def _audit_pair(
    a: ProteinMutationDataset,
    b: ProteinMutationDataset,
    na: str,
    nb: str,
    edges: pd.DataFrame,
) -> Dict[str, int]:
    return {
        f"pdb_overlap_{na}_{nb}": len(
            a.all_pdbs() & b.all_pdbs()
        ),
        f"exact_input_overlap_{na}_{nb}": len(
            a.all_input_keys() & b.all_input_keys()
        ),
        f"exact_sequence_overlap_{na}_{nb}": len(
            a.component_sequence_set()
            & b.component_sequence_set()
        ),
        f"direct_relation_edges_{na}_{nb}": _edge_count_between(
            a.all_pdbs(),
            b.all_pdbs(),
            edges,
        ),
    }


def audit_partition_separation(
    train_set: ProteinMutationDataset,
    val_set: Optional[ProteinMutationDataset],
    test_set: Optional[ProteinMutationDataset],
    relation_edges: pd.DataFrame,
    stage: str,
) -> Dict[str, int]:
    checks: Dict[str, int] = {}
    pairs = []

    if val_set is not None:
        pairs.append(
            (train_set, val_set, "train", "val")
        )
    if test_set is not None:
        pairs.append(
            (train_set, test_set, "train", "test")
        )
    if val_set is not None and test_set is not None:
        pairs.append(
            (val_set, test_set, "val", "test")
        )

    for a, b, na, nb in pairs:
        checks.update(
            _audit_pair(a, b, na, nb, relation_edges)
        )

    if stage == "selection" and val_set is None:
        raise RuntimeError(
            "selection stage requires validation data"
        )
    if test_set is None:
        raise RuntimeError(
            "formal selection/refit requires test data for leakage audit"
        )

    bad = {k: v for k, v in checks.items() if int(v) != 0}
    if bad:
        raise RuntimeError(
            "LOCO partition leakage audit FAILED: "
            + str(bad)
        )

    return checks


# =============================================================================
# Wrapper
# =============================================================================

class WrapperDataset:
    def __init__(
        self,
        args: Any,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        test_data: Optional[pd.DataFrame] = None,
        stage: Optional[str] = None,
    ) -> None:
        self.args = args
        self.stage = str(
            stage
            or getattr(args, "training_stage", "selection")
        ).lower()

        if self.stage not in {"selection", "refit"}:
            raise ValueError(
                "stage must be selection or refit"
            )
        if self.stage == "selection" and val_data is None:
            raise ValueError(
                "selection stage requires val_data"
            )
        if self.stage == "refit" and val_data is not None:
            raise ValueError(
                "refit stage requires val_data=None"
            )

        self.seed = int(getattr(args, "seed", 42))
        self.batch_size = int(
            getattr(args, "batch_size", 8)
        )
        self.eval_batch_size = int(
            getattr(args, "eval_batch_size", self.batch_size)
        )
        self.num_workers = int(
            getattr(args, "num_workers", 2)
        )
        self.pin_memory = _as_bool(
            getattr(args, "pin_memory", torch.cuda.is_available())
        )
        self.target_standardize = _as_bool(
            getattr(args, "target_standardize", True), True
        )
        self.mutation_window_radius = int(
            getattr(args, "mutation_window_radius", 8)
        )
        self.require_sequence_change = _as_bool(
            getattr(args, "require_sequence_change", True), True
        )
        self.allow_sequence_truncation = _as_bool(
            getattr(args, "allow_sequence_truncation", False), False
        )
        self.long_sequence_strategy = str(
            getattr(
                args,
                "long_sequence_strategy",
                "sliding_window",
            )
        ).lower()

        self.use_pairwise_loss = _as_bool(
            getattr(args, "use_pairwise_loss", False), False
        )
        self.use_contrastive = _as_bool(
            getattr(args, "use_contrastive", False), False
        )
        self.same_complex_pairs_per_batch = int(
            getattr(
                args,
                "same_complex_pairs_per_batch",
                2 if self.use_pairwise_loss else 0,
            )
        )
        self.complex_temper_alpha = float(
            getattr(args, "complex_temper_alpha", 0.5)
        )
        self.prefer_contrastive_diversity = _as_bool(
            getattr(
                args,
                "prefer_contrastive_diversity",
                self.use_contrastive,
            )
        )

        dataset_name = getattr(
            args,
            "data_name",
            getattr(args, "dataset_name", "AB645"),
        )
        self.schema = infer_schema(
            train_data, dataset_name
        )

        train_y = pd.to_numeric(
            train_data[self.schema.label_col],
            errors="raise",
        ).to_numpy(dtype=float)
        if not np.isfinite(train_y).all():
            raise ValueError("non-finite training labels")

        self.target_mean = float(train_y.mean())
        self.target_std = float(
            train_y.std(ddof=1)
        )
        if (
            not math.isfinite(self.target_std)
            or self.target_std <= 1e-8
        ):
            raise RuntimeError(
                "degenerate training target std"
            )

        common = dict(
            schema=self.schema,
            target_mean=self.target_mean,
            target_std=self.target_std,
            target_standardize=self.target_standardize,
            mutation_window_radius=self.mutation_window_radius,
            require_sequence_change=self.require_sequence_change,
        )

        self.train_dataset = ProteinMutationDataset(
            train_data,
            split_name="train",
            **common,
        )
        self.val_dataset = (
            ProteinMutationDataset(
                val_data,
                split_name="val",
                **common,
            )
            if val_data is not None
            else None
        )
        self.test_dataset = (
            ProteinMutationDataset(
                test_data,
                split_name="test",
                **common,
            )
            if test_data is not None
            else None
        )

        model_locate = str(
            getattr(args, "model_locate")
        )
        local_only = _as_bool(
            getattr(args, "local_files_only", False), False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_locate,
            local_files_only=local_only,
        )
        config = AutoConfig.from_pretrained(
            model_locate,
            local_files_only=local_only,
        )
        self.model_max_length = int(
            getattr(
                config,
                "max_position_embeddings",
                1026,
            )
        )

        self.mmseqs_dir = _resolve_mmseqs_dir(
            args, self.schema.name
        )
        min_id = _fraction(
            getattr(args, "contrastive_min_seq_id", 0.90)
        )
        min_cov = _fraction(
            getattr(args, "contrastive_min_cov", 0.80)
        )
        self.homology_index = DirectHomologyIndex(
            self.mmseqs_dir
            / "all_unique_input_sequences.fasta",
            self.mmseqs_dir / "all_vs_all.tsv",
            min_seq_id=min_id,
            min_cov=min_cov,
        )

        all_components = list(
            self.train_dataset.all_component_sequences()
        )
        if self.val_dataset is not None:
            all_components += list(
                self.val_dataset.all_component_sequences()
            )
        if self.test_dataset is not None:
            all_components += list(
                self.test_dataset.all_component_sequences()
            )
        self.homology_index.validate_coverage(
            all_components, strict=True
        )

        mode = str(
            getattr(args, "split_relation_mode", "auto")
        ).lower()
        if mode == "auto":
            protocol = str(
                getattr(args, "protocol", "")
            ).lower()
            folder = str(
                getattr(args, "data_folder", "")
            ).lower()
            mode = (
                "exact"
                if "loco_exact_refit" in (
                    protocol + " " + folder
                )
                else "homology90"
            )

        self.split_relation_mode = mode
        self.split_relation_edges = _load_relation_edges(
            self.mmseqs_dir,
            mode=mode,
            min_seq_id=min_id,
            min_cov=min_cov,
        )

        self.split_audit = audit_partition_separation(
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            self.split_relation_edges,
            self.stage,
        )

        self.collate_fn = MutationAwareCollator(
            tokenizer=self.tokenizer,
            model_max_length=self.model_max_length,
            homology_index=self.homology_index,
            target_mean=self.target_mean,
            target_std=self.target_std,
            target_standardize=self.target_standardize,
            allow_sequence_truncation=self.allow_sequence_truncation,
            long_sequence_strategy=self.long_sequence_strategy,
        )

        print("=" * 88)
        print("Mutation-aware LOCO WrapperDataset")
        print("=" * 88)
        print("stage                  :", self.stage)
        print(
            "train / val / test     :",
            len(self.train_dataset),
            len(self.val_dataset)
            if self.val_dataset is not None else 0,
            len(self.test_dataset)
            if self.test_dataset is not None else 0,
        )
        print(
            "train target mean/std  :",
            f"{self.target_mean:.6f}",
            f"{self.target_std:.6f}",
        )
        print("batch size             :", self.batch_size)
        print("mutation radius        :", self.mutation_window_radius)
        print("split relation mode    :", self.split_relation_mode)
        print("split audit            : PASS")
        print("silent truncation      :", self.allow_sequence_truncation)
        print("=" * 88)

    def get_train_loader(self) -> DataLoader:
        sampler = MutationAwareBatchSampler(
            dataset=self.train_dataset,
            homology_index=self.homology_index,
            batch_size=self.batch_size,
            seed=self.seed,
            alpha=self.complex_temper_alpha,
            same_complex_pairs_per_batch=(
                self.same_complex_pairs_per_batch
                if self.use_pairwise_loss
                else 0
            ),
            prefer_contrastive_diversity=(
                self.prefer_contrastive_diversity
            ),
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=(
                _seed_worker
                if self.num_workers > 0
                else None
            ),
        )

    def _eval_loader(
        self,
        ds: Dataset,
    ) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=(
                _seed_worker
                if self.num_workers > 0
                else None
            ),
        )

    def get_val_loader(self) -> Optional[DataLoader]:
        return (
            self._eval_loader(self.val_dataset)
            if self.val_dataset is not None
            else None
        )

    def get_test_loader(self) -> Optional[DataLoader]:
        return (
            self._eval_loader(self.test_dataset)
            if self.test_dataset is not None
            else None
        )

    def inverse_transform_targets(
        self,
        y: torch.Tensor,
    ) -> torch.Tensor:
        if not self.target_standardize:
            return y
        return y * self.target_std + self.target_mean


def inspect_batch(batch: Mapping[str, Any]) -> None:
    required = (
        "wt_ab_inputs_ids",
        "mut_ab_inputs_ids",
        "wt_ag_inputs_ids",
        "mut_ag_inputs_ids",
        "labels",
        "labels_raw",
        "ab_mutation_mask",
        "ag_mutation_mask",
        "ab_mutation_window_mask",
        "ag_mutation_window_mask",
        "pairwise_regression_mask",
        "same_complex_mask",
        "contrastive_exclusion_masks",
    )
    missing = [k for k in required if k not in batch]
    if missing:
        raise KeyError(f"batch missing {missing}")
    print("Batch sanity: PASS")


__all__ = [
    "WrapperDataset",
    "ProteinMutationDataset",
    "MutationAwareCollator",
    "MutationAwareBatchSampler",
    "DirectHomologyIndex",
    "DatasetSchema",
    "infer_schema",
    "normalize_sequence",
    "normalize_pdb_id",
    "audit_partition_separation",
    "inspect_batch",
]
