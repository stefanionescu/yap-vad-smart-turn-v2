#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
PID_FILE="$ROOT_DIR/.run/server.pid"
TAIL_PID_FILE="$ROOT_DIR/.run/tail.pid"

PURGE=0
DEEP=0

usage() {
  echo "Usage: $0 [--purge] [--deep]"
  echo "  --purge  Stop server and remove venv, caches (.hf), logs, run files, downloaded models, and pip caches"
  echo "  --deep   Also clean system-level caches (apt lists/archives) if available"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --purge|-p) PURGE=1; shift ;;
    --deep|-d) DEEP=1; shift ;;
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

# Stop tailer if running
if [ -f "$TAIL_PID_FILE" ]; then
  TAILPID="$(cat "$TAIL_PID_FILE" || echo)"
  if [ -n "$TAILPID" ] && kill -0 "$TAILPID" 2>/dev/null; then
    echo "[stop] Stopping tail logs PID $TAILPID"
    kill "$TAILPID" || true
  fi
  rm -f "$TAIL_PID_FILE"
fi

# Fallback: try pkill by pattern (in case PID file lost)
if pgrep -fa "uvicorn .*src.server:app" >/dev/null 2>&1; then
  echo "[stop] Killing stray uvicorn src.server:app processes"
  pkill -f "uvicorn .*src.server:app" || true
fi

echo "[stop] Service stopped"

if [[ $PURGE -eq 1 ]]; then
  echo "[purge] Removing venvs, caches, logs, and model weights"
  rm -rf "$ROOT_DIR/venv" "$ROOT_DIR/.venv" || true
  rm -rf "$ROOT_DIR/.hf" || true
  rm -rf "$ROOT_DIR/logs" "$ROOT_DIR/.run" || true
  rm -f  "$ROOT_DIR/nohup.out" || true
  # Python bytecode caches within repo
  find "$ROOT_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} + || true
  # Honor env-defined HF caches (avoid -u errors for unset vars)
  HF_HOME_DIR="${HF_HOME:-}"
  TRANSFORMERS_CACHE_DIR="${TRANSFORMERS_CACHE:-}"
  HUGGINGFACE_HUB_CACHE_DIR="${HUGGINGFACE_HUB_CACHE:-}"
  for d in "$HF_HOME_DIR" "$TRANSFORMERS_CACHE_DIR" "$HUGGINGFACE_HUB_CACHE_DIR"; do
    if [ -n "$d" ] && [ -d "$d" ]; then
      echo "[purge] Removing cache: $d"
      rm -rf "$d" || true
    fi
  done
  # User-level caches (best-effort)
  rm -rf "$HOME/.cache/pip" || true
  rm -rf "$HOME/.cache/huggingface" "$HOME/.cache/huggingface_hub" || true
  rm -rf "$HOME/.cache/torch" || true
  # XDG cache
  if [ -n "${XDG_CACHE_HOME:-}" ]; then
    rm -rf "$XDG_CACHE_HOME/huggingface" || true
    rm -rf "$XDG_CACHE_HOME/torch" || true
  fi
  # Historical model dir from earlier runs (if exists)
  [ -d "/models/hf" ] && rm -rf "/models/hf" || true
  echo "[purge] Repo disk usage now:"; du -sh "$ROOT_DIR" 2>/dev/null | awk '{print $1 " used in repo"}' || true
  if [[ $DEEP -eq 1 ]]; then
    if command -v apt-get >/dev/null 2>&1; then
      echo "[deep] Cleaning apt caches"
      apt-get clean || true
      rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* || true
    fi
  fi
  echo "[purge] Done"
fi

echo "[stop] Done"

