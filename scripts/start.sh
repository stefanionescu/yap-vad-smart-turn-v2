#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
cd "$ROOT_DIR"

source .venv/bin/activate

export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf}"
unset TRANSFORMERS_CACHE
export HUGGINGFACE_HUB_CACHE="$HF_HOME"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT_DIR"

PORT=8000

echo "[start] Launching Smart Turn server on 0.0.0.0:${PORT}"
exec uvicorn src.server:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers 1 \
  --loop uvloop \
  --http httptools \
  --log-level debug



