"""Micro-batching queue and batch execution."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch

from ..constants import THRESHOLD, MICRO_BATCH_WINDOW_MS
from ..runtime import runtime as rt


@dataclass(slots=True)
class Item:
    """Queue item representing one inference request."""

    arr: np.ndarray
    fut: asyncio.Future = field(init=False)
    t_start_total: float = field(init=False)

    def __post_init__(self):
        self.fut = asyncio.get_running_loop().create_future()
        self.t_start_total = time.perf_counter()


QUEUE: asyncio.Queue[Item] = asyncio.Queue()


def _select_bucket(depth: int) -> int:
    """Return the smallest configured bucket >= depth, defaulting to the largest."""
    return min([b for b in rt.BATCH_BUCKETS if b >= depth] or [rt.BATCH_BUCKETS[-1]])


async def batcher() -> None:
    """Periodic micro-batcher that drains the queue and performs a single forward pass."""
    await asyncio.sleep(0)

    while True:
        await asyncio.sleep(MICRO_BATCH_WINDOW_MS / 1000.0)

        items: List[Item] = []
        while not QUEUE.empty() and len(items) < max(rt.BATCH_BUCKETS):
            items.append(QUEUE.get_nowait())
        if not items:
            continue

        depth = len(items)
        bucket = _select_bucket(depth)
        rt.logger.debug(f"batcher: depth={depth} bucket={bucket} queue_empty={QUEUE.empty()}")

        # Build batch tensor on GPU
        try:
            np_stack = np.stack([it.arr for it in items], axis=0).astype(np.float32, copy=False)
            batch = torch.from_numpy(np_stack)
            if rt.DEVICE.type == "cuda":
                batch = batch.pin_memory().to(rt.DEVICE, non_blocking=True)
            else:
                batch = batch.to(rt.DEVICE)
        except Exception as e:
            rt.logger.exception(f"batcher: failed to build batch tensor: {e}")
            now = time.perf_counter()
            for it in items:
                if not it.fut.done():
                    it.fut.set_result({
                        "prediction": 0,
                        "probability": 0.0,
                        "metrics": {"inference_time": 0.0, "total_time": now - it.t_start_total},
                    })
            continue

        t0 = time.perf_counter()

        try:
            # compiled or eager
            if rt._ACTIVE_MODEL is None:
                rt.build_eager_if_needed()

            m = rt._ACTIVE_MODEL
            rt.logger.debug(
                f"batcher: eager/compiled forward bucket={bucket} (compiled_ready={rt._COMPILED_READY})"
            )

            # Padding-to-bucket optimization: pad to bucket size for compiled models
            use_compiled = rt.USE_TORCH_COMPILE and rt._COMPILED_READY
            if use_compiled and batch.shape[0] < bucket:
                pad = torch.zeros((bucket, batch.shape[1]), dtype=batch.dtype, device=batch.device)
                pad[:depth].copy_(batch)
                batch_for_model = pad
                rt.logger.debug(f"batcher: padded batch from {depth} to {bucket} for compiled model")
            else:
                batch_for_model = batch

            inp = rt.make_inputs_gpu(
                batch_for_model, cast_to=(rt.DTYPE if rt.DTYPE != torch.float32 else None)
            )
            with torch.no_grad():
                if rt.DEVICE.type == "cuda" and rt.DTYPE != torch.float32:
                    with torch.autocast("cuda", dtype=rt.DTYPE):
                        out = m(**inp)
                else:
                    out = m(**inp)

            # Slice logits back to original depth
            logits = (out["logits"] if isinstance(out, dict) else out.logits)[:depth]
            probs = logits.detach().float().cpu().numpy().reshape(-1)

            t1 = time.perf_counter()
            rt.logger.debug(f"batcher: forward done in {(t1-t0)*1000:.1f} ms")

            for it, p in zip(items, probs):
                pred = 1 if p > THRESHOLD else 0
                if not it.fut.done():
                    it.fut.set_result({
                        "prediction": int(pred),
                        "probability": float(p),
                        "metrics": {"inference_time": (t1 - t0), "total_time": (t1 - it.t_start_total)},
                    })

        except Exception as e:
            rt.logger.exception(f"batcher: inference failed: {e}")
            now = time.perf_counter()
            for it in items:
                if not it.fut.done():
                    it.fut.set_result({
                        "prediction": 0,
                        "probability": 0.0,
                        "metrics": {"inference_time": 0.0, "total_time": now - it.t_start_total},
                    })


__all__ = ["Item", "QUEUE", "batcher"]


