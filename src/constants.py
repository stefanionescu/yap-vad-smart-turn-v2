import os
import torch

# Fixed server port
PORT = 8000

# Audio
SAMPLE_RATE = 16000
MAX_SECS = float(os.environ.get("MAX_SECS", "16"))
MAX_SAMPLES = int(SAMPLE_RATE * MAX_SECS)

# Model and precision
DTYPE_STR = os.environ.get("DTYPE", "bfloat16").lower()
DTYPE = torch.bfloat16 if DTYPE_STR == "bfloat16" else torch.float32
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))
MODEL_ID = os.environ.get("MODEL_ID", "pipecat-ai/smart-turn-v2")

# Batching
def _parse_buckets(env_value: str):
    try:
        return [int(x) for x in env_value.split(",") if x.strip()]
    except Exception:
        return [16, 32, 64]

BATCH_BUCKETS = _parse_buckets(os.environ.get("BATCH_BUCKETS", "16,32,64"))
MICRO_BATCH_WINDOW_MS = int(os.environ.get("MICRO_BATCH_WINDOW_MS", "2"))

# Auth
AUTH_KEY = os.environ.get("AUTH_KEY", "")

# Optimizations
USE_TORCH_COMPILE = os.environ.get("TORCH_COMPILE", "0") == "1"
USE_CUDA_GRAPHS = os.environ.get("CUDA_GRAPHS", "0") == "1"


