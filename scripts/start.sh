#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$ROOT_DIR/.venv/bin/activate"
export PYTHONPATH="$ROOT_DIR"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf}"
unset TRANSFORMERS_CACHE
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export BATCH_BUCKETS=1,2,4,8,16,32,64
export TORCH_COMPILE="${TORCH_COMPILE:-1}"
export CUDA_GRAPHS="${CUDA_GRAPHS:-1}"

exec python -m uvicorn src.server:app --host 0.0.0.0 --port 8000 --workers 1 --lifespan on --log-level debug