"""Runtime configuration and model lifecycle utilities.

Centralizes device/precision settings, batching buckets, logger setup, and
model build/compile helpers used by the server.
"""

from __future__ import annotations

import os
import torch
from loguru import logger

from ..constants import SAMPLE_RATE, MAX_SECS, DTYPE, MODEL_ID, LOG_LEVEL

try:
    from ..model import Wav2Vec2ForEndpointing
except ImportError as e:
    raise RuntimeError(
        "smart-turn-v2 needs the custom Wav2Vec2ForEndpointing from model.py"
    ) from e

# Be resilient: never crash the server if compile fails
import torch._dynamo as dynamo  # noqa: E402

dynamo.config.suppress_errors = True


# -------- Logger ----------
logger.remove()  # Remove default handler
logger.add(
    lambda msg: print(msg, end=""),
    level=LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)


# -------- Configurable buckets (1..64 by default) ----------
def parse_buckets(val: str) -> list[int]:
    """Parse a comma-separated string of integers into a list of batch sizes."""
    try:
        return [int(x) for x in val.split(",") if x.strip()]
    except Exception:
        return [1, 2, 4, 8]  # Default to smaller buckets to avoid OOM


BATCH_BUCKETS = parse_buckets(os.environ.get("BATCH_BUCKETS", "1,2,4,6"))
USE_TORCH_COMPILE = os.environ.get("TORCH_COMPILE", "1") == "1"


# -------- Device / precision ----------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
torch.set_num_threads(2)
try:
    torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass
torch.set_float32_matmul_precision("high")


MAX_SAMPLES = SAMPLE_RATE * int(float(os.environ.get("MAX_SECS", str(MAX_SECS))))


# -------- Model state ----------
_ACTIVE_MODEL: torch.nn.Module | None = None      # serves requests (eager first, then compiled)
_COMPILED_MODEL: torch.nn.Module | None = None
_COMPILED_READY = False

_EAGER_MODEL: torch.nn.Module | None = None       # plain eager for serving until compiled swaps in


def make_inputs_gpu(x: torch.Tensor, cast_to: torch.dtype | None = None) -> dict[str, torch.Tensor]:
    """Prepare model inputs. Cast and move to target DEVICE, build attention mask."""
    if x.device != DEVICE:
        x = x.to(DEVICE, non_blocking=True)
    if cast_to is not None and x.dtype != cast_to:
        x = x.to(cast_to)

    attn = torch.ones((x.shape[0], x.shape[1]), dtype=torch.long, device=DEVICE)
    return {"input_values": x, "attention_mask": attn}


def build_eager_if_needed() -> None:
    """Instantiate the eager model once and mark it as the active model."""
    global _EAGER_MODEL, _ACTIVE_MODEL
    if _EAGER_MODEL is None:
        m = Wav2Vec2ForEndpointing.from_pretrained(MODEL_ID).to(DEVICE)
        if DTYPE != torch.float32:
            m = m.to(dtype=DTYPE)
        m.eval()
        _EAGER_MODEL = m
        _ACTIVE_MODEL = _EAGER_MODEL
        logger.info("Eager model built and set ACTIVE.")


def compile_sync() -> None:
    """Synchronous compile that runs in a background thread."""
    global _COMPILED_MODEL, _ACTIVE_MODEL, _COMPILED_READY
    if _COMPILED_READY or DEVICE.type != "cuda" or not USE_TORCH_COMPILE:
        return
    try:
        logger.info("Compile(thread): building compiled model...")
        m = Wav2Vec2ForEndpointing.from_pretrained(MODEL_ID).to(DEVICE)
        if DTYPE != torch.float32:
            m = m.to(dtype=DTYPE)
        m.eval()
        m = torch.compile(m, mode="reduce-overhead")

        # Pre-warm the exact buckets we'll use
        for b in sorted(set(BATCH_BUCKETS)):
            warm = torch.zeros((b, MAX_SAMPLES), dtype=torch.float32, device=DEVICE)
            inp = make_inputs_gpu(
                warm, cast_to=(DTYPE if DTYPE != torch.float32 else None)
            )
            with torch.no_grad():
                if DEVICE.type == "cuda" and DTYPE != torch.float32:
                    with torch.autocast("cuda", dtype=DTYPE):
                        _ = m(**inp)
                else:
                    _ = m(**inp)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        _COMPILED_MODEL = m
        _COMPILED_READY = True
        _ACTIVE_MODEL = _COMPILED_MODEL
        logger.info("Compile(thread): READY and ACTIVE.")
    except Exception as e:
        logger.exception(f"Compile(thread) failed: {e}")


__all__ = [
    "logger",
    "DEVICE",
    "DTYPE",
    "BATCH_BUCKETS",
    "USE_TORCH_COMPILE",
    "MAX_SAMPLES",
    "_ACTIVE_MODEL",
    "_COMPILED_READY",
    "build_eager_if_needed",
    "compile_sync",
    "make_inputs_gpu",
]


