#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
PID_FILE="$ROOT_DIR/.run/server.pid"

PURGE=0

usage() {
  echo "Usage: $0 [--purge]"
  echo "  --purge  Stop server and remove venv, caches (.hf), logs, run files, downloaded models, and pip caches"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --purge|-p) PURGE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

stop_pid() {
  local pid="$1"
  if kill -0 "$pid" 2>/dev/null; then
    echo "[stop] Stopping PID $pid"
    kill "$pid" || true
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
      echo "[stop] Force killing PID $pid"
      kill -9 "$pid" || true
    fi
  else
    echo "[stop] Process $pid not running"
  fi
}

# Primary: PID file
if [ -f "$PID_FILE" ]; then
  PID="$(cat "$PID_FILE" || echo)"
  if [ -n "${PID}" ]; then
    stop_pid "$PID"
  fi
  rm -f "$PID_FILE"
else
  echo "[stop] No PID file at $PID_FILE"
fi

# Fallback: try pkill by pattern (in case PID file lost)
if pgrep -fa "uvicorn .*src.server:app" >/dev/null 2>&1; then
  echo "[stop] Killing stray uvicorn src.server:app processes"
  pkill -f "uvicorn .*src.server:app" || true
fi

echo "[stop] Service stopped"

if [[ $PURGE -eq 1 ]]; then
  echo "[purge] Removing venv, caches, logs, and model weights"
  rm -rf "$ROOT_DIR/venv" || true
  rm -rf "$ROOT_DIR/.hf" || true
  rm -rf "$ROOT_DIR/logs" || true
  rm -rf "$ROOT_DIR/.run" || true
  # Python bytecode caches
  find "$ROOT_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} + || true
  # Optional: user pip/hf caches (best-effort)
  rm -rf "$HOME/.cache/pip" || true
  rm -rf "$HOME/.cache/huggingface" || true
  echo "[purge] Done"
fi

echo "[stop] Done"

