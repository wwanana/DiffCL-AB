#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SCRIPT="${CV_SCRIPT:-cv5_single_ab1101_resume.py}"
if [[ ! -f "$SCRIPT" ]]; then
  SCRIPT="cv5_single_ab1101.py"
fi
[[ -f "$SCRIPT" ]] || { echo "AB1101 CV script not found" >&2; exit 1; }

mkdir -p "${TMPDIR:-./.tmp}"
export TMPDIR="${TMPDIR:-$ROOT_DIR/.tmp}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

python "$SCRIPT" \
  --data_file ./data/sigmul_data/AB1101_single.csv \
  --data_name AB1101_single \
  --output_dir ./cv5_results_AB1101_seed42 \
  --n_splits 5 --val_ratio 0.1 --rm_abnormal True \
  --model_locate ./model/esm2_650m \
  --num_workers 2 --batch_size 2 --drop_last True --max_length 384 \
  --hidden_size 1280 --num_heads 4 --dropout 0.1 --out_dim 1 \
  --use_mamba_branch True --use_contrastive True \
  --mamba_layers 2 --mamba_d_state 16 --mamba_d_conv 4 --mamba_expand 2 --mamba_dropout 0.1 \
  --proj_dim 256 --contrastive_temperature 0.1 \
  --cl1_weight 0.03 --cl2_weight 0.01 --ag_consistency_inner_weight 0.1 \
  --loss mse --ckpt_monitor val_pearson_corr --early_stop_monitor val_mse \
  --accelerator gpu --devices 1 --strategy auto --num_nodes 1 \
  --max_epochs 120 --patience 20 --lr 2e-5 --weight_decay 1e-4 \
  --use_lr_scheduler True --scheduler_patience 5 --scheduler_factor 0.5 --min_lr 1e-6 \
  --accumulate_grad_batches 4 --gradient_clip_val 5.0 --gradient_clip_algorithm value \
  --precision 32 --seed 42 --resume True --freeze_backbone
