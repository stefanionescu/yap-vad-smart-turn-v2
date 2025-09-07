#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$ROOT_DIR/.run" "$ROOT_DIR/logs"

source "$ROOT_DIR/.venv/bin/activate"

export PYTHONPATH="$ROOT_DIR"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf}"
unset TRANSFORMERS_CACHE
export HUGGINGFACE_HUB_CACHE="$HF_HOME"

# Minimal tunables used by the app; others use defaults
export BATCH_BUCKETS="${BATCH_BUCKETS:-1,2,4,6}"
export TORCH_COMPILE="${TORCH_COMPILE:-1}"
export DTYPE="${DTYPE:-bfloat16}"
export MICRO_BATCH_WINDOW_MS="${MICRO_BATCH_WINDOW_MS:-5}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

PORT=8000
echo "[start_bg] Launching on port ${PORT}"

nohup python -m uvicorn src.server:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers 1 \
  --lifespan on \
  --log-level info \
  > "$ROOT_DIR/logs/server.log" 2>&1 &

echo $! > "$ROOT_DIR/.run/server.pid"
echo "[start_bg] PID $(cat "$ROOT_DIR/.run/server.pid") | logs: $ROOT_DIR/logs/server.log"