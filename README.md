# DiffCL-AB

## Overview

Given four amino-acid sequences—wild-type antibody, wild-type antigen, mutant antibody, and mutant antigen—the model predicts the mutation-induced binding affinity change, typically denoted as ΔΔG.

The current implementation contains:

- a frozen ESM2 backbone for residue-level sequence representations;
- the original antibody–antigen interaction branch based on attention;
- a parallel Mamba branch for long-range sequence modeling;
- feature alignment and gated interaction fusion between the main and Mamba branches;
- two contrastive-learning objectives;
- difference-aware and state-aware feature fusion for final regression;
- PyTorch Lightning training, checkpointing, early stopping, resume support, and five-fold cross-validation;
- evaluation with MSE, RMSE, MAE, Pearson correlation, Spearman correlation, and R².

## Repository layout

```text
.
├── model.py                         # Main network
├── lit_model.py                     # LightningModule, losses, metrics, optimizer
├── dataset.py                       # Dataset and DataLoader logic
├── cv5_single_ab1101_resume.py      # AB1101 single-mutation 5-fold CV
├── cv5_general_resume.py            # Generic 5-fold CV for AB645 and S1131
├── debug_forward.py                 # Forward and one-epoch smoke test
├── trainer_sigmul.py                # Legacy/default training entry
├── eval.py                          # Evaluation utilities
├── model_module/                    # Attention and supporting modules
├── utils/                           # Data and callback utilities
├── scripts/
│   ├── run_cv5_ab1101.sh
│   ├── run_cv5_ab645.sh
│   ├── run_cv5_s1131.sh
│   └── download_esm2.py
├── data/
│   └── README.md                    # Expected dataset layout
├── model/
│   └── esm2_650m/                   # Local ESM2 weights; not committed
├── requirements.txt
├── environment.yml
├── .gitignore
├── CITATION.cff
└── THIRD_PARTY_NOTICES.md
```

## System requirements

Recommended:

- Linux
- NVIDIA GPU with CUDA support
- Python 3.10
- Conda or Miniconda
- PyTorch compatible with the installed CUDA runtime

The ESM2-650M backbone is memory intensive. The provided commands use a batch size of 2 and gradient accumulation of 4. Reduce `--max_length`, batch size, or Mamba depth if GPU memory is insufficient.

## Installation

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate protab_mamba
```

Because `mamba-ssm` and `causal-conv1d` are sensitive to the exact PyTorch/CUDA combination, install PyTorch first, verify CUDA, and then install Mamba:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

pip install -U packaging ninja
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation
```

For exact reproducibility, export the environment used for the reported experiments:

```bash
conda env export --from-history > environment.from-history.yml
pip freeze > requirements-lock.txt
```

### 2. Download ESM2

Do not commit pretrained weights to GitHub. Download them locally:

```bash
python scripts/download_esm2.py --output_dir ./model/esm2_650m
```

Verify the directory contains at least `config.json`, tokenizer files, and model weights.

### 3. Prepare datasets

See [`data/README.md`](data/README.md). The expected default layout is:

```text
data/
├── sigmul_data/
│   └── AB1101_single.csv
└── identity_data/
    ├── csv_AB645/
    │   ├── AB645_train.csv
    │   └── AB645_test.csv
    └── csv_S1131/
        ├── S1131_train.csv
        └── S1131_test.csv
```

Dataset files should only be redistributed when their original licenses permit it.

## Quick smoke test

Run the forward pass and a small one-epoch training check before full cross-validation:

```bash
conda activate protab_mamba
python debug_forward.py
```

A successful smoke test should complete data loading, model construction, forward propagation, backward propagation, and one small validation pass without NaN/Inf or shape errors.

## Five-fold cross-validation

All three launch scripts use the same model and training configuration:

- random seed: `42`
- folds: `5`
- maximum epochs: `120`
- early-stopping patience: `20`
- optimizer: AdamW
- learning rate: `2e-5`
- weight decay: `1e-4`
- loss: MSE
- ReduceLROnPlateau: enabled
- scheduler patience: `5`
- scheduler factor: `0.5`
- minimum learning rate: `1e-6`
- batch size: `2`
- gradient accumulation: `4`
- gradient clipping: `5.0`
- backbone: frozen ESM2-650M
- Mamba branch: enabled
- dual contrastive learning: enabled
- resume: enabled

### AB1101

```bash
bash scripts/run_cv5_ab1101.sh
```

Direct command:

```bash
python cv5_single_ab1101_resume.py \
  --data_file ./data/sigmul_data/AB1101_single.csv \
  --data_name AB1101_single \
  --output_dir ./cv5_results_AB1101_seed42 \
  --n_splits 5 \
  --val_ratio 0.1 \
  --rm_abnormal True \
  --model_locate ./model/esm2_650m \
  --num_workers 2 \
  --batch_size 2 \
  --drop_last True \
  --max_length 384 \
  --hidden_size 1280 \
  --num_heads 4 \
  --dropout 0.1 \
  --out_dim 1 \
  --use_mamba_branch True \
  --use_contrastive True \
  --mamba_layers 2 \
  --mamba_d_state 16 \
  --mamba_d_conv 4 \
  --mamba_expand 2 \
  --mamba_dropout 0.1 \
  --proj_dim 256 \
  --contrastive_temperature 0.1 \
  --cl1_weight 0.03 \
  --cl2_weight 0.01 \
  --ag_consistency_inner_weight 0.1 \
  --loss mse \
  --ckpt_monitor val_pearson_corr \
  --early_stop_monitor val_mse \
  --accelerator gpu \
  --devices 1 \
  --strategy auto \
  --max_epochs 120 \
  --patience 20 \
  --lr 2e-5 \
  --weight_decay 1e-4 \
  --use_lr_scheduler True \
  --scheduler_patience 5 \
  --scheduler_factor 0.5 \
  --min_lr 1e-6 \
  --accumulate_grad_batches 4 \
  --gradient_clip_val 5.0 \
  --gradient_clip_algorithm value \
  --precision 32 \
  --seed 42 \
  --resume True \
  --freeze_backbone
```

### AB645

```bash
bash scripts/run_cv5_ab645.sh
```

The generic CV script merges the configured train/test CSV files and performs a new shuffled five-fold split with seed 42.

### S1131

```bash
bash scripts/run_cv5_s1131.sh
```

## Resume behavior

The CV scripts are intended to support interrupted training:

- a completed fold with `fold_result.json` is skipped;
- a fold with a finished training flag and checkpoint reuses the checkpoint for testing;
- incomplete folds continue from the last valid state supported by the script.

Always reuse the same `--output_dir` when resuming. Use a new output directory for a new experimental configuration.

## Outputs

Each cross-validation output directory contains per-fold checkpoints/logs and final summaries, typically:

```text
cv5_results_*/
├── fold_1/
├── fold_2/
├── fold_3/
├── fold_4/
├── fold_5/
├── cv5_fold_results.csv
└── cv5_summary.json
```

The reported metrics are:

- MSE
- RMSE
- MAE
- Pearson correlation coefficient
- Spearman rank correlation coefficient
- R²

## Reproducibility notes

- Keep `--seed 42` and deterministic training enabled.
- Record GPU model, CUDA version, PyTorch version, and package lock file.
- Do not compare experiments that use different data preprocessing, abnormal-value filtering, fold definitions, or checkpoint-selection metrics.
- The original ProtAttBA publication used 10-fold CV for AB645/S1131 and 5-fold CV for AB1101. This repository provides five-fold launch scripts for all three datasets to match the experiments in this extension.

## Common issues

### ESM2 local path is treated as a Hugging Face repository ID

Confirm the directory exists and contains a complete saved model:

```bash
ls -lah ./model/esm2_650m
```

Use an absolute path when necessary:

```bash
--model_locate /absolute/path/to/model/esm2_650m
```

### CUDA or Mamba import failure

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.__version__, torch.version.cuda)"
python -c "from mamba_ssm import Mamba; print('mamba import ok')"
```

Reinstall `causal-conv1d` and `mamba-ssm` only after PyTorch and CUDA are confirmed compatible.

### R² fails on a one-sample final batch

Validation/test metrics should accumulate state across the entire epoch with `metric.update(...)`, then compute at epoch end. Keep `drop_last=False` for the test loader so no test samples are discarded.

### Boolean command-line arguments behave unexpectedly

Avoid `type=bool` in `argparse`, because strings such as `"False"` may still become truthy. Use a dedicated `str2bool` parser for options such as `--rm_abnormal`, `--use_mamba_branch`, and `--use_contrastive`.

## Citation

Please cite the original ProtAttBA paper:

```bibtex
@article{liu2025protattba,
  title   = {Sequence-only prediction of binding affinity changes: a robust and interpretable model for antibody engineering},
  author  = {Liu, Chen and Li, Mingchen and Tan, Yang and Gou, Wenrui and Fan, Guisheng and Zhou, Bingxin},
  journal = {Bioinformatics},
  volume  = {41},
  number  = {8},
  pages   = {btaf446},
  year    = {2025},
  doi     = {10.1093/bioinformatics/btaf446}
}
```

Add the citation for this extension after the manuscript is accepted. Update [`CITATION.cff`](CITATION.cff) with the final author list, title, year, repository URL, and DOI.

## License and third-party material

This project is derived from ProtAttBA and depends on third-party software, pretrained weights, and benchmark datasets. Before publishing:

1. verify the upstream ProtAttBA code license;
2. choose a compatible license for your modifications;
3. keep original copyright and attribution notices;
4. do not redistribute ESM2 weights or datasets unless their licenses permit it;
5. document third-party licenses in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

No license is granted merely by publishing source code without a license file.

## Acknowledgements

This work builds on the original ProtAttBA framework and the ESM2 protein language model. We thank the maintainers of PyTorch, PyTorch Lightning, Hugging Face Transformers, Mamba SSM, and the benchmark dataset providers.
