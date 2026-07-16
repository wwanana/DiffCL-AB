#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash fix_release_debug.sh
#   bash fix_release_debug.sh /absolute/path/to/esm2_650m

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "[1/5] Fixing persistent Conda environment variables..."
if command -v conda >/dev/null 2>&1 && [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  conda env config vars unset OMP_NUM_THREADS >/dev/null 2>&1 || true
  conda env config vars set \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false >/dev/null
fi

echo "[2/5] Checking GPU..."
nvidia-smi

echo "[3/5] Finding ESM2-650M..."
MODEL_DIR="${1:-}"
if [[ -z "$MODEL_DIR" ]]; then
  for p in \
    "${ESM2_MODEL_PATH:-}" \
    "/root/autodl-tmp/ProtAttBA-main/seq-identity_sig-mul/model/esm2_650m" \
    "/root/autodl-tmp/ProtAttBA-main/model/esm2_650m" \
    "$ROOT/model/esm2_650m"
  do
    if [[ -n "$p" && -f "$p/config.json" ]]; then
      MODEL_DIR="$p"
      break
    fi
  done
fi

if [[ -z "$MODEL_DIR" || ! -f "$MODEL_DIR/config.json" ]]; then
  CFG="$(
    find /root/autodl-tmp -type f -name config.json 2>/dev/null \
      | grep -Ei 'esm2.*650|650.*esm2' \
      | head -n 1 || true
  )"
  [[ -n "$CFG" ]] && MODEL_DIR="$(dirname "$CFG")"
fi

if [[ -z "$MODEL_DIR" || ! -f "$MODEL_DIR/config.json" ]]; then
  cat <<'EOF'
No local ESM2-650M model was found.

Download it with:
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="facebook/esm2_t33_650M_UR50D",
    local_dir="./model/esm2_650m",
)
PY

Then run:
bash fix_release_debug.sh
EOF
  exit 2
fi

MODEL_DIR="$(readlink -f "$MODEL_DIR")"
echo "Using model: $MODEL_DIR"

echo "[4/5] Creating a local ignored model link..."
mkdir -p "$ROOT/model"
if [[ "$(readlink -f "$ROOT/model/esm2_650m" 2>/dev/null || true)" != "$MODEL_DIR" ]]; then
  rm -rf "$ROOT/model/esm2_650m"
  ln -s "$MODEL_DIR" "$ROOT/model/esm2_650m"
fi

touch "$ROOT/.gitignore"
grep -qxF "model/esm2_650m" "$ROOT/.gitignore" \
  || echo "model/esm2_650m" >> "$ROOT/.gitignore"
grep -qxF "model/esm2_650m/" "$ROOT/.gitignore" \
  || echo "model/esm2_650m/" >> "$ROOT/.gitignore"

export ESM2_MODEL_PATH="$MODEL_DIR"

echo "[5/5] Running the synthetic smoke test..."
python "$ROOT/debug_forward.py" --model_locate "$MODEL_DIR"
