"""Route handlers for the Smart Turn v2 server."""

from __future__ import annotations

import io
import numpy as np
from fastapi import Request, HTTPException

from ..utils.auth import auth_ok
from ..utils.audio import ensure_16k
from ..serving.batcher import Item, QUEUE
from ..runtime import runtime as rt


async def health() -> dict:
    """Simple liveness endpoint."""
    return {"ok": True}


async def status() -> dict:
    """Return runtime status and queue depth for quick diagnostics."""
    return {
        "device": str(rt.DEVICE),
        "dtype": str(rt.DTYPE),
        "compiled_ready": rt._COMPILED_READY,
        "buckets": rt.BATCH_BUCKETS,
        "queue_depth": QUEUE.qsize(),
    }


async def raw(request: Request) -> dict:
    """Accept a NumPy .npy mono float32 PCM payload and return prediction.

    Authorization: optional header "Authorization: Key <AUTH_KEY>" when AUTH_KEY is set.
    """
    if not auth_ok(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    body = await request.body()
    rt.logger.debug(f"POST /raw: received {len(body)} bytes")
    try:
        arr = np.load(io.BytesIO(body), allow_pickle=False)
        rt.logger.debug(f"POST /raw: parsed npy -> shape={arr.shape} dtype={arr.dtype}")
        arr = ensure_16k(arr)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid npy payload: {e}")

    item = Item(arr)
    await QUEUE.put(item)
    rt.logger.debug("POST /raw: queued item")
    return await item.fut


__all__ = ["health", "status", "raw"]


