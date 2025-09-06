#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
LOG_FILE="$ROOT_DIR/logs/server.log"

if [ ! -f "$LOG_FILE" ]; then
  echo "[logs] No log file yet at $LOG_FILE"
  exit 0
fi

exec tail -n 200 -F "$LOG_FILE"



