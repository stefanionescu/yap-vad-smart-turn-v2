#!/usr/bin/env bash
set -euo pipefail

# Orchestrates: setup → start (bg) → tail logs → wait ready → warmup

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
cd "$ROOT_DIR"
RUN_DIR="$ROOT_DIR/.run"
mkdir -p "$RUN_DIR"

DO_WARMUP=1
SAMPLE="mid.wav"
SECONDS_PAD=8

usage() {
  echo "Usage: $0 [--no-warmup] [--sample <file>] [--seconds <n>]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-warmup) DO_WARMUP=0; shift ;;
    --sample) SAMPLE=${2:-mid.wav}; shift 2 ;;
    --seconds) SECONDS_PAD=${2:-8}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

echo "[main] Running setup..."
bash "$SCRIPT_DIR/setup.sh"

echo "[main] Starting server in background..."
bash "$SCRIPT_DIR/start_bg.sh"

# Wait a moment before tailing logs to ensure log file is created
sleep 2
# Start tailing logs (in background) so we can monitor startup and record PID
bash "$SCRIPT_DIR/tail_bg_logs.sh" &
echo $! > "$RUN_DIR/tail.pid"

# Wait for readiness
echo -n "[main] Waiting for health on http://127.0.0.1:8000/health"
ATTEMPTS=60
for i in $(seq 1 $ATTEMPTS); do
  if python - <<'PY'
import urllib.request
try:
    with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=1) as r:
        import sys; sys.exit(0 if r.status==200 else 1)
except Exception:
    import sys; sys.exit(1)
PY
  then
    echo " ✔"
    break
  else
    echo -n "."; sleep 1
  fi
  if [[ $i -eq $ATTEMPTS ]]; then
    echo "\n[main] Server did not become ready in time"; exit 3
  fi
done

if [[ $DO_WARMUP -eq 1 ]]; then
  echo "[main] Warmup using sample=$SAMPLE seconds=$SECONDS_PAD"
  source venv/bin/activate
  python -m test.warmup --sample "$SAMPLE" --seconds "$SECONDS_PAD"
fi

echo "[main] Done. Logs: $ROOT_DIR/logs/server.log"


