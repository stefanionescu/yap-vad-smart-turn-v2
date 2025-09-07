"""Runtime package: device, dtype, buckets, and model lifecycle."""

from .runtime import (
    logger,
    DEVICE,
    DTYPE,
    BATCH_BUCKETS,
    USE_TORCH_COMPILE,
    MAX_SAMPLES,
    _ACTIVE_MODEL,
    _COMPILED_READY,
    build_eager_if_needed,
    compile_sync,
    make_inputs_gpu,
)

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


