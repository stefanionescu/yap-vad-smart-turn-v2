#!/usr/bin/env bash
set -euo pipefail

# Always execute from repo root
SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
cd "$ROOT_DIR"

echo "[setup] Preparing Smart Turn v2 environment..."

if command -v apt-get >/dev/null 2>&1; then
  echo "[setup] Updating apt cache (if available)"
  apt-get update -y || true
  echo "[setup] Installing system libs for audio I/O (ffmpeg, libsndfile1)"
  apt-get install -y --no-install-recommends ffmpeg libsndfile1 || true
fi

echo "[setup] Creating Python virtual environment at $ROOT_DIR/venv"
python3 -m venv venv
source venv/bin/activate

echo "[setup] Upgrading pip tooling"
python -m pip install --upgrade pip wheel setuptools

# Install PyTorch built for CUDA 12.4 (compatible with Runpod CUDA 12.8 drivers)
TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"
TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA}"

echo "[setup] Installing PyTorch ${TORCH_VERSION}+${TORCH_CUDA}"
pip install --index-url "$TORCH_INDEX_URL" \
  "torch==${TORCH_VERSION}+${TORCH_CUDA}" \
  "torchaudio==${TORCH_VERSION}+${TORCH_CUDA}"

echo "[setup] Installing Python dependencies from requirements.txt"
pip install -r requirements.txt

mkdir -p logs .hf .run samples

export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"

MODEL_ID="${MODEL_ID:-pipecat-ai/smart-turn-v2}"
python - <<PY
from huggingface_hub import snapshot_download
mid = "${MODEL_ID}"
print(f"[setup] Pre-downloading model: {mid}")
snapshot_download(mid)
print("[setup] Done.")
PY

echo "[setup] Complete. Activate with: source venv/bin/activate"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$ROOT_DIR/.run" "$ROOT_DIR/logs" "$ROOT_DIR/.hf"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -d "$ROOT_DIR/.venv" ]; then
  echo "[setup] Creating virtualenv at $ROOT_DIR/.venv"
  "$PYTHON_BIN" -m venv "$ROOT_DIR/.venv"
fi

source "$ROOT_DIR/.venv/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# Pick Torch build; defaults target CUDA 12.4 wheels which work on CUDA 12.8 drivers
TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"
TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA}"

echo "[setup] Installing PyTorch ${TORCH_VERSION}+${TORCH_CUDA} from ${TORCH_INDEX_URL}"
pip install --index-url "$TORCH_INDEX_URL" \
  "torch==${TORCH_VERSION}+${TORCH_CUDA}" \
  "torchaudio==${TORCH_VERSION}+${TORCH_CUDA}"

echo "[setup] Installing app dependencies"
pip install -r "$ROOT_DIR/requirements.txt"

# Pre-download model weights into local HF cache to avoid first-hit latency
export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"

MODEL_ID="${MODEL_ID:-pipecat-ai/smart-turn-v2}"
python - <<PY
from huggingface_hub import snapshot_download
mid = "${MODEL_ID}"
print(f"[setup] Pre-downloading model: {mid}")
snapshot_download(mid)
print("[setup] Done.")
PY

echo "[setup] Complete. Activate with: source $ROOT_DIR/.venv/bin/activate"


