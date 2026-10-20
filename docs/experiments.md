# Evaluation protocols

## Fixed inputs and model

`model.py`, `dataset.py`, and `lit_model.py` preserve the deposited final implementation byte for byte. Their SHA-256 digests are in `checkpoints/source_sha256.json`. The release interfaces in `diffcl/` implement protocol orchestration and the declared ablations around this model. Use `run_experiment.py`; older project-specific trainer paths are not required.

The default network uses frozen ESM2-650M features (1,280 dimensions), a deterministic projection to 256 dimensions, 16 latent tokens, four attention heads, and dropout 0.1. The shared Mamba branch has one block with `d_state=16`, `d_conv=4`, and `expand=2`. The mutation-local radius is eight residues and windows stop at component-chain boundaries. The final six-block fusion projects 1,536 → 512 → 256; the regression head projects 256 → 512 → 128 → 1. Long inputs use overlapping ESM windows; they are not silently truncated.

## Ordinary five-fold CV

1. Use the model-valid dataset and its stable `record_id`.
2. Group records with identical concatenated WT-Ab, WT-Ag, MT-Ab, and MT-Ag inputs. Repeated measurements, including conflicting labels, remain in the same outer fold.
3. Apply shuffled five-fold partitioning to the sorted unique groups with seed 42.
4. For outer fold `f`, reserve 10% of development groups for validation using seed `42 + f`.
5. Select training duration on the inner training/validation split, initialize a new model, and refit on the complete outer development pool. Evaluate that fold's test records once.

Shared parent complexes and related sequences may cross ordinary partitions. This protocol addresses held-out variants within the available collection; it is distinct from sequence-separated evaluation.

## Low-homology five-fold CV

Outer folds group whole PDB complexes, with shuffled five-fold partitioning of sorted complex identifiers at seed 42. For each fold, remove a development complex if it shares the held-out PDB identifier, an exact component sequence, or a qualifying direct MMseqs2 relationship with any test component. Both WT and MT components from both partner roles participate in this check. A relationship qualifies at identity ≥0.30, query coverage ≥0.80, and target coverage ≥0.80; edges are treated symmetrically without transitive closure.

The remaining development complexes are divided into inner training and validation groups with a 10% validation fraction and seed `42 + f`. Inner train/validation complex overlap is prohibited; an additional inner homology purge is not applied. The 30% criterion governs separation of the entire development pool from the outer test set. Because complexes contain different numbers of variants, test and inner validation row counts need not be equal across folds.

Purged row identifiers and all four partition memberships are saved in `data/splits/*/lh30/manifest.json`. These fixed manifests enable matched comparisons, and empty or invalid partitions cause an error rather than relaxing the criterion. Smaller training pools are an intrinsic consequence of this protocol and must be reported alongside prediction metrics.

## Mutation-depth evaluation

The deposited MutD partitions retain the original 581/64 selection split, 645-record refit pool, and 455-record test set. Single and multiple substitutions are defined by the deposited mutation annotations. MutD permits shared parent complexes. MutD-LH30 uses the same test set and the same direct 30% identity / 80% bidirectional coverage exclusion rule, retaining 51/9 selection records and 60 for refit. The tests and labels remain unchanged between these settings.

The fixed partitions are provided directly as CSVs. `scripts.audit_release` independently verifies their relationship to the curated source data and recomputes the eligible LH30 refit pool. Annotation depth can differ from the number of distinct changes in the encoded sequence components; see [data.md](data.md).

## Optimization and duration selection

`configs/reference.json` specifies the common executable reference configuration. It is derived from the deposited final MutD configuration and is used unchanged for the newly reconstructed CV protocols unless a separate configuration is supplied. It should not be described as a recovery of unrecorded dataset-specific historical tuning.

| Setting | Value |
| --- | --- |
| Seed / batch size | 42 / 8 |
| AdamW learning rate / weight decay | 4 × 10⁻⁵ / 3 × 10⁻⁴ |
| LR warmup fraction / minimum LR | 0.08 / 10⁻⁶ |
| Pairwise Smooth L1 weight / beta | 0.05 / 1.0 |
| CL-I / CL-II weights | 0.005 / 0.0025 |
| Contrastive temperature | 0.10 |
| CL warmup | Epochs 5–15, using the deposited scheduler |
| Selection scheduler horizon | 40 epochs |
| Eligible selection epochs | 16–40 |
| Refit duration range | 32–40 epochs |
| Gradient norm clipping | 1.0 |

Among eligible validation epochs, use epochs with predicted/target standard-deviation ratio ≥0.30 when at least one satisfies the threshold. Retain epochs whose RMSE is within 3% of the best eligible RMSE, then maximize the mean of PCC and Spearman ρ; ties favor lower RMSE and then the earlier epoch. Clip the selected duration to 32–40 epochs for refitting. The refit LR schedule retains the 40-epoch horizon and stops at the selected duration. Target normalization uses only the current stage's training data and sample standard deviation (`ddof=1`).

This is the selection rule present in the deposited final MutD trainer. The reported archived run selected epoch 22 and refit for 32 epochs. New runs make their own validation-based selection; the runner does not force the archived epoch or inspect test metrics to choose training settings.

## Ablations

| Argument | Executed change |
| --- | --- |
| `full` | Complete model and objectives |
| `cnn` | Replace the shared Mamba sequence branch with a shared residual Conv1d branch; keep interaction, fusion, local features, and objectives |
| `no_cl1` | Set CL-I weight to zero |
| `no_cl2` | Set CL-II weight to zero |
| `no_cl` | Disable both contrastive objectives |
| `simple_diff` | Retain WT, MT, and local features; remove signed difference, absolute difference, and product blocks |

The released CNN comparator uses pre-LayerNorm, a full-channel kernel-3 convolution with padding one, GELU, a linear projection and dropout, residual addition, output LayerNorm, and masked mean pooling. Parameters are shared across partners and WT/MT. It is an explicit comparator implementation because the original CNN module was not present in the supplied final source. The comparison is not parameter-count matched. Simple Diff uses a 3d → 2d → d fusion MLP rather than 6d → 2d → d. Consequently, these ablations change representation construction and parameterization; their results should be interpreted accordingly.

All ablation runs use the same ordinary partitions and reference configuration as the complete model. No historical ablation run files are required to execute them, and no synthetic ablation predictions are included.

## Matched external comparisons

The manuscript's third-party methods are separate software packages. This repository provides the DiffCL-AB experiment family and interfaces for applying external methods to exactly the same data; it does not substitute new implementations for AttABseq, ProtAttBA, DDGPred, FoldX, or the PPI-model adaptations.

Export one evaluation partition:

```bash
python -m scripts.baseline_partitions --dataset AB645 --protocol ordinary \
  --fold 1 --output runs/baseline_inputs/AB645/fold1
```

Use the external method's official implementation and record its exact commit, dependencies, feature construction, and hyperparameters. Train and select using only `inner_train.csv` and `val.csv`; refit using `refit_train.csv` when applicable. Structure-based scoring methods should retain their own stated scoring procedure. Do not interpret unmodified binary PPI classification outputs as ΔΔG predictions.

Export external predictions with `source_index` equal to the provided `record_id` and `y_pred` in kcal mol⁻¹, using this repository's sign convention. Import and verify them with:

```bash
python -m scripts.import_predictions --test runs/baseline_inputs/AB645/fold1/test.csv \
  --predictions path/to/external_predictions.csv --method METHOD_NAME \
  --implementation 'UPSTREAM_URL@COMMIT_SHA' --fold 1 \
  --output runs/baselines/METHOD_NAME/AB645/fold1
```

Aggregate all five imported folds with `aggregate_cv.py`. This verifies complete row coverage and labels; it cannot certify that a separate external training program respected the development/test boundary. Preserve that program's configuration and logs with the comparison results.

## Interpretation and verification scope

Ordinary CV, sequence-separated CV, and the fixed mutation-depth tests answer different generalization questions. AB645 and AB1101 are not independent source collections. Repeated measurements and censored/nonbinder assignments can influence error and correlation metrics. Post hoc structural cases provide molecular context but do not independently establish predictive performance.

Release checks verify real data processing, split construction, the computational graph and gradients, small-model training/validation/test execution, inference serialization, and archived metric calculations. They do not replace a full ESM2/Mamba GPU rerun. For numerical replication, record the dataset hashes, fixed fold manifest, backbone revision, complete configuration, seed, and software/hardware environment alongside every prediction file. The original ordinary/low-homology fold assignments and original CNN comparator are not inferred from the paper's reported aggregate scores.
