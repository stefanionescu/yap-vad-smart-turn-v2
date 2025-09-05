import io
import os
from typing import Tuple

import numpy as np
import torch
import torchaudio

SAMPLE_RATE = 16000


def _load_from_npy(fpath: str) -> np.ndarray:
    arr = np.load(fpath)
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def _load_from_wav(fpath: str) -> np.ndarray:
    wav, sr = torchaudio.load(fpath)  # shape: (channels, num_frames)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=SAMPLE_RATE)
    wav = wav.squeeze(0)
    return wav.to(torch.float32).cpu().numpy()


def load_audio_from_samples(path: str, seconds_pad_to: int = 8) -> Tuple[np.ndarray, bytes]:
    """Load mono float32 PCM from samples/<path> (supports .wav and .npy),
    resample to 16 kHz if needed, pad/truncate to seconds_pad_to, and return (array, np.save bytes).
    """
    root = os.path.dirname(os.path.dirname(__file__))
    fpath = os.path.join(root, "samples", path)
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Sample not found: {fpath}")

    ext = os.path.splitext(fpath)[1].lower()
    if ext == ".npy":
        arr = _load_from_npy(fpath)
    elif ext in (".wav", ".wave"):
        arr = _load_from_wav(fpath)
    else:
        raise ValueError(f"Unsupported sample extension: {ext}. Use .wav or .npy")

    target = SAMPLE_RATE * seconds_pad_to
    if arr.size > target:
        arr = arr[-target:]
    elif arr.size < target:
        arr = np.pad(arr, (0, target - arr.size), mode="constant")

    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32, copy=False))
    return arr, buf.getvalue()


