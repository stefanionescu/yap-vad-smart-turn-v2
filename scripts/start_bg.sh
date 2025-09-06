#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
RUN_DIR="$ROOT_DIR/.run"
LOG_DIR="$ROOT_DIR/logs"
PID_FILE="$RUN_DIR/server.pid"
cd "$ROOT_DIR"

mkdir -p "$RUN_DIR" "$LOG_DIR"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "[start_bg] Server already running with PID $(cat "$PID_FILE")"
  exit 0
fi

source .venv/bin/activate

export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf}"
unset TRANSFORMERS_CACHE
export HUGGINGFACE_HUB_CACHE="$HF_HOME"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PORT=8000

echo "[start_bg] Launching on port ${PORT}"
nohup uvicorn src.server:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers 1 \
  --loop uvloop \
  --http httptools \
  --log-level debug \
  > "$LOG_DIR/server.log" 2>&1 &

echo $! > "$PID_FILE"
echo "[start_bg] PID $(cat "$PID_FILE") | logs: $LOG_DIR/server.log"

