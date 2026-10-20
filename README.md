# DiffCL-AB

Sequence-only prediction of mutation-induced changes in antibody–antigen binding affinity.

This repository accompanies **“DiffCL-AB: A Sequence-Only Difference-Aware Contrastive Learning Framework for Predicting Antibody–Antigen Binding Affinity Changes upon Mutation.”** It provides the model, curated inputs, evaluation partitions, training and inference commands, component ablations, and structural case-analysis code.

The prediction target is ΔΔG = ΔG<sub>MT</sub> − ΔG<sub>WT</sub>, in kcal mol⁻¹. Positive values indicate weaker binding after mutation; negative values indicate stronger binding. Coordinates are not model inputs.

## Model

DiffCL-AB uses frozen ESM2-650M residue embeddings [1] and one shared encoder for wild-type and mutant complexes. A compact interaction branch combines local convolution, learned latent tokens, and bidirectional cross-attention. A parallel Mamba branch [2] models sequence context. A residual gate integrates the two branches. Mutation-local features and explicit state comparisons—WT, MT, signed difference, absolute difference, and elementwise product—support ΔΔG regression.

Training combines standardized-target MSE, within-complex pairwise supervision, and two contrastive objectives. Same-complex, identical-input, and qualifying directly related sequence pairs are excluded from the contrastive negative pool. PDB identifiers and homology relationships are used for partitioning and loss construction; they are not regression features.

## Data and experiment coverage

| Dataset / protocol | Development records | Held-out evaluation |
| --- | ---: | --- |
| AB645 | 645 total records | Ordinary and low-homology five-fold CV |
| S1131 | 1,130 model-valid records from 1,131 source records | Ordinary and low-homology five-fold CV |
| AB1101 | 1,100 total records | Ordinary and low-homology five-fold CV |
| AB1101-MutD | 581 train / 64 validation; 645 for refit | Fixed 455-record test set |
| AB1101-MutD-LH30 | 51 train / 9 validation; 60 for refit | The same 455-record test set |

AB645 and AB1101 derive from AB-Bind [3]; AB645 is the single-substitution subset of this AB1101 release. S1131 derives from SKEMPI and the BindProfX benchmark [4,5]. The established benchmark names are retained even where curated row counts differ. See [data provenance](docs/data.md) for reconciliation, exclusions, repeated measurements, and mutation-count conventions.

All six model variants run through the same training interface: `full`, `cnn`, `no_cl1`, `no_cl2`, `no_cl`, and `simple_diff`. Ablations use the ordinary folds. The full experiment matrix contains 107 fits for one seed, each comprising selection and refitting.

The original final model source and AB1101-MutD checkpoint are deposited. The ordinary and low-homology CV manifests and the CNN comparator are explicit release implementations of the described protocols. They support new, reproducible evaluations; their outputs are not presented as recovered historical runs. The default optimization configuration is derived from the deposited final training configuration. [Protocol details](docs/experiments.md) specify these choices and how to use the same partitions for external comparison methods.

## Installation

The recorded training environment used Linux, Python 3.10.20, PyTorch 2.7.1 with CUDA 11.8, and an NVIDIA RTX 4090 D with 24 GB memory. Formal Mamba training and checkpoint inference require a compatible NVIDIA GPU. CPU checks use a small local ESM model and a debug sequence mixer to exercise the software interfaces.

```bash
git clone https://github.com/wwanana/DiffCL-AB.git
cd DiffCL-AB
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-dev.txt
python -m pip install packaging ninja wheel
python -m pip install mamba-ssm==2.3.0 --no-build-isolation
python -m pip check
python -m scripts.prepare_assets
```

Mamba installation may require the matching CUDA toolkit and a compiler; consult its [official installation instructions](https://github.com/state-spaces/mamba/tree/v2.3.0). The deposited environment did not include the optional `causal-conv1d` extension. Formal runs reject the debug fallback.

Download the backbone once:

```bash
python -m scripts.download_esm2 --output models/esm2_650m
```

The downloader resolves an immutable Hugging Face revision and records file checksums in `backbone_manifest.json`. Supply `--revision COMMIT_SHA` to reuse that revision elsewhere. The 650M backbone is downloaded separately; `checkpoints/mutd_seed42.pt` contains the downstream trainable parameters. Large deposited files are stored losslessly as compressed parts in `assets/`; `python -m scripts.prepare_assets` reconstructs their documented paths and verifies every part and final SHA-256. This step uses only the Python standard library and requires no external download.

## Check the release

```bash
python -m scripts.audit_release
python -m pytest -q
python run_suite.py --experiment all --plan-only
```

These commands check asset hashes, label and sequence consistency, split separation, checkpoint integrity, model and ablation gradients, train/validation/test execution, checkpoint loading, and metric calculations. The CPU test suite does not establish reproduction of the paper's GPU training results. GitHub Actions runs these checks on each push and pull request.

To install the CPU test environment instead of the CUDA environment, install PyTorch from `https://download.pytorch.org/whl/cpu`, then `requirements-dev.txt`; Mamba and the full ESM2 weights are unnecessary for these checks.

## Reproduce model experiments

Run all data and partition checks before starting GPU jobs:

```bash
python run_suite.py --experiment all --preflight
```

Run individual experiment families or the complete matrix:

```bash
python run_suite.py --experiment ordinary
python run_suite.py --experiment lh30
python run_suite.py --experiment ablation
python run_suite.py --experiment mutd
python run_suite.py --experiment mutd_lh30
# Alternatively, run the complete matrix:
python run_suite.py --experiment all
```

Use one family command or `all` with a fresh output directory. Existing nonempty run directories are protected from overwriting. `--datasets AB645`, `--seeds 42 43 44`, `--config PATH`, and `--output PATH` restrict or extend the matrix. Additional seeds vary training initialization while retaining the deposited evaluation partitions.

A single fold can also be launched directly:

```bash
python run_experiment.py --dataset AB645 --protocol ordinary --fold 1 \
  --variant full --output runs/AB645/ordinary/full/seed42/fold1
```

Each run writes its configuration and input hashes, validation history, selected duration, training-only target statistics, downstream checkpoint, individual test predictions, and metrics. Model selection uses only the inner development split. Refit starts from a new initialization with the same seed and uses the eligible development pool. The final test set is evaluated after refitting.

Aggregate a complete five-fold run:

```bash
python aggregate_cv.py --data data/processed/AB645.csv \
  --predictions 'runs/AB645/ordinary/full/seed42/fold*/predictions.csv' \
  --output runs/AB645/ordinary/full/seed42/summary
```

The aggregator requires every record exactly once and checks source labels and complex identifiers. It reports both pooled out-of-fold metrics and fold means with sample standard deviations. R² is computed as `1 − SSE/SST`, not as squared Pearson correlation. Undefined correlations are recorded as `null`.

## Checkpoint inference and archived predictions

```bash
python predict.py --input data/protocols/mutd/test.csv \
  --checkpoint checkpoints/mutd_seed42.pt --backbone models/esm2_650m \
  --output runs/mutd_checkpoint_predictions.csv \
  --compare results/archived/mutd_predictions.csv
```

The optional comparison checks row identity and numerical agreement, with an absolute prediction tolerance of `1e-4` kcal mol⁻¹ by default. Custom CSVs use the supplied component-sequence schema; `ddG` may be omitted for unlabeled inference. The deposited checkpoint's target statistics are provided in a companion JSON file and verified against its refit data.

Recalculate the archived results without a GPU:

```bash
python analyze_predictions.py --output runs/mutd_analysis
```

The deposited 455 predictions give RMSE **1.658515**, PCC **0.715337**, and Spearman ρ **0.468264**. RMSE rounds to **1.66** at two decimal places; the manuscript's value of 1.65 is not obtained by ordinary rounding of this file. The analysis also reports assigned-nonbinder sensitivity, represented versus unrepresented complex backgrounds, and mutation-depth subsets. These diagnostics do not determine training settings.

## Regenerate preprocessing and partitions

```bash
python -m scripts.prepare_data --output runs/rebuilt_data
python prepare_splits.py --data data/processed/AB645.csv --protocol ordinary \
  --output runs/rebuilt_splits/AB645/ordinary
python prepare_splits.py --data data/processed/AB645.csv --protocol lh30 \
  --homology data/homology/AB1101 --output runs/rebuilt_splits/AB645/lh30
```

The supplied MMseqs2 relationships [6] cover the deposited inputs. Low-homology exclusion uses ≥30% identity and ≥80% coverage of **both** component sequences. Contrastive masking uses ≥90% identity with the same coverage requirement. Both use direct relationships without transitive closure. For a changed dataset, rebuild relationships first:

```bash
python -m scripts.build_homology --data data/processed/S1131.csv \
  --output runs/new_homology/S1131 --mmseqs mmseqs --threads 4
```

The MMseqs2 version, command, input hash, and output hashes are recorded. Dataset and partition files must be regenerated together when inputs change. Sequence-separated evaluation should be interpreted alongside the reduction in training data [7].

## Structural cases

The three case examples use deposited WT PDB coordinates (1N8Z, 1DQJ, and 1VFB), archived predictions, and verified heavy-atom distances. They illustrate held-out combinations on represented complex backgrounds and were selected after examining predictions. They do not constitute independent validation or observed mutant structures.

```bash
python -m scripts.audit_cases
# Optional figure environment: install open-source PyMOL and requirements-figures.txt
python -m figures.render_structures
python -m figures.compose_figure
```

Outputs are written to `runs/figures/`. An existing case figure is provided at [figures/three_cases.png](figures/three_cases.png).

## Repository organization

| Location | Contents |
| --- | --- |
| `model.py`, `dataset.py`, `lit_model.py` | Deposited model, data contracts, and objectives |
| `diffcl/` | Experiment interfaces, ablations, data validation, and metrics |
| `configs/reference.json` | Architecture, optimization, and duration-selection settings |
| `data/source/`, `data/processed/` | Original project inputs and reproducible model-valid versions |
| `data/splits/`, `data/protocols/`, `data/homology/` | CV manifests, fixed mutation-depth partitions, and sequence relationships |
| `checkpoints/` | Final downstream checkpoint, normalization statistics, and core source hashes |
| `assets/` | Compressed data/weight parts and a manifest for lossless local reconstruction |
| `results/archived/` | Deposited individual predictions |
| `results/verification/` | Release data audits and recomputed descriptive results |
| `tests/`, `.github/workflows/` | CPU functionality tests and continuous checks |

## References

1. Lin Z, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. **Science** 379, 1123–1130 (2023). [doi:10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574).
2. Gu A, Dao T. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. **COLM** (2024). [arXiv:2312.00752](https://arxiv.org/abs/2312.00752).
3. Sirin S, Apgar JR, Bennett EM, Keating AE. AB-Bind: Antibody binding mutational database for computational affinity predictions. **Protein Science** 25, 393–409 (2016). [doi:10.1002/pro.2829](https://doi.org/10.1002/pro.2829).
4. Moal IH, Fernández-Recio J. SKEMPI: a Structural Kinetic and Energetic database of Mutant Protein Interactions and its use in empirical models. **Bioinformatics** 28, 2600–2607 (2012). [doi:10.1093/bioinformatics/bts489](https://doi.org/10.1093/bioinformatics/bts489).
5. Xiong P, Zhang C, Zheng W, Zhang Y. BindProfX: Assessing Mutation-Induced Binding Affinity Change by Protein Interface Profiles with Pseudo-Counts. **Journal of Molecular Biology** 429, 426–434 (2017). [doi:10.1016/j.jmb.2016.11.022](https://doi.org/10.1016/j.jmb.2016.11.022).
6. Steinegger M, Söding J. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. **Nature Biotechnology** 35, 1026–1028 (2017). [doi:10.1038/nbt.3988](https://doi.org/10.1038/nbt.3988).
7. Hummer AM, Schneider C, Chinery L, Deane CM. Investigating the volume and diversity of data needed for generalizable antibody–antigen ΔΔG prediction. **Nature Computational Science** 5, 635–647 (2025). [doi:10.1038/s43588-025-00823-8](https://doi.org/10.1038/s43588-025-00823-8).

For the DiffCL-AB manuscript, use [CITATION.cff](CITATION.cff). Please also cite the original data and software resources used in your analysis.
