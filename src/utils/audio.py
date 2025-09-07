"""Audio preprocessing utilities."""

from __future__ import annotations

import numpy as np

from ..runtime.runtime import MAX_SAMPLES


def ensure_16k(arr: np.ndarray) -> np.ndarray:
    """Ensure mono float32 waveform at 16kHz padded/clipped to MAX_SAMPLES."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    arr = np.ravel(arr)
    if arr.size > MAX_SAMPLES:
        arr = arr[:MAX_SAMPLES]
    elif arr.size < MAX_SAMPLES:
        arr = np.pad(arr, (0, MAX_SAMPLES - arr.size))
    return arr


__all__ = ["ensure_16k"]


