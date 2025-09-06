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
  echo "[setup] Installing system libs for audio I/O (libsndfile1)"
  apt-get install -y --no-install-recommends libsndfile1 || true
fi

echo "[setup] Creating Python virtual environment at $ROOT_DIR/.venv"
python3 -m venv .venv
source .venv/bin/activate

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
unset TRANSFORMERS_CACHE
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export BATCH_BUCKETS="${BATCH_BUCKETS:-1,2,4,8}"
export TORCH_COMPILE="${TORCH_COMPILE:-1}"
export CUDA_GRAPHS="${CUDA_GRAPHS:-0}"
export DTYPE="${DTYPE:-bfloat16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MICRO_BATCH_WINDOW_MS="${MICRO_BATCH_WINDOW_MS:-5}"

MODEL_ID="${MODEL_ID:-pipecat-ai/smart-turn-v2}"
python - <<PY
from huggingface_hub import snapshot_download
mid = "${MODEL_ID}"
print(f"[setup] Pre-downloading model: {mid}")
snapshot_download(mid)
print("[setup] Done.")
PY

echo "[setup] Ensuring default sample at samples/mid.wav"
python - <<'PY'
import os, torch, torchaudio
root = os.path.dirname(os.path.dirname(__file__))
samples_dir = os.path.join(root, 'samples')
os.makedirs(samples_dir, exist_ok=True)
out = os.path.join(samples_dir, 'mid.wav')
if not os.path.exists(out):
    sr = 16000
    sec = 8
    wav = torch.zeros(1, sr*sec, dtype=torch.float32)
    torchaudio.save(out, wav, sample_rate=sr, format='wav')
    print(f"[setup] Wrote {out}")
else:
    print(f"[setup] Sample exists: {out}")
PY

echo "[setup] Complete. Activate with: source .venv/bin/activate"
echo "[setup] Defaults: BATCH_BUCKETS=$BATCH_BUCKETS, TORCH_COMPILE=$TORCH_COMPILE, CUDA_GRAPHS=$CUDA_GRAPHS"
echo "[setup]           DTYPE=$DTYPE, MICRO_BATCH_WINDOW_MS=$MICRO_BATCH_WINDOW_MS"



