import os
import io
import time
import asyncio
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, Request, HTTPException
from loguru import logger
from transformers import AutoProcessor
from pathlib import Path

# Import custom Wav2Vec2ForEndpointing from local model.py
try:
    from .model import Wav2Vec2ForEndpointing
except ImportError as e:
    raise RuntimeError("smart-turn-v2 needs the custom Wav2Vec2ForEndpointing from model.py") from e

from .constants import (
    SAMPLE_RATE,
    MAX_SECS,
    MAX_SAMPLES,
    DTYPE,
    THRESHOLD,
    BATCH_BUCKETS,
    MICRO_BATCH_WINDOW_MS,
    AUTH_KEY,
    USE_TORCH_COMPILE,
    USE_CUDA_GRAPHS,
    MODEL_ID,
)


# -----------------------------
# Device
# -----------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
torch.set_num_threads(2)


# -----------------------------
# Model & Processor Load
# -----------------------------
logger.info(f"Loading model and processor: {MODEL_ID}")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForEndpointing.from_pretrained(MODEL_ID)
model.eval().to(DEVICE)
if DTYPE != torch.float32 and DEVICE.type == "cuda":
    model.to(dtype=DTYPE)

if USE_TORCH_COMPILE and DEVICE.type == "cuda":
    logger.info("Compiling model with torch.compile(mode='reduce-overhead')")
    model = torch.compile(model, mode="reduce-overhead")


# -----------------------------
# Performance Optimizations
# -----------------------------
# TF32 for L40S performance
try:
    torch.backends.cuda.matmul.allow_tf32 = True  # ok on L40S
except Exception:
    pass
torch.set_float32_matmul_precision("high")

# CUDA graphs registry
GRAPHS: dict[int, tuple] = {}      # bucket -> (graph, logits_ref, static_inputs)
CAPTURED: set[int] = set()
CAPTURING: set[int] = set()

async def _capture_bucket(bucket: int):
    """Capture a CUDA graph for this bucket in the background."""
    try:
        x = torch.zeros((bucket, MAX_SAMPLES), dtype=torch.float32, device=DEVICE)
        inp = processor(
            x, sampling_rate=SAMPLE_RATE, padding="max_length", truncation=True,
            max_length=MAX_SAMPLES, return_attention_mask=True, return_tensors="pt",
        )
        inp = {k: v.to(DEVICE, non_blocking=True) for k, v in inp.items()}

        # eager warmup so torch.compile specializes before capture
        with torch.no_grad():
            if DEVICE.type == "cuda" and DTYPE != torch.float32:
                with torch.autocast("cuda", dtype=DTYPE):
                    _ = model(**inp)
            else:
                _ = model(**inp)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = model(**inp)
            logits_ref = out.logits

        GRAPHS[bucket] = (g, logits_ref, inp)
        CAPTURED.add(bucket)
        logger.info(f"CUDA graph captured for bucket={bucket}")
    except Exception as e:
        logger.exception(f"CUDA graph capture FAILED for bucket={bucket}: {e}")
    finally:
        CAPTURING.discard(bucket)


# -----------------------------
# App & Batching
# -----------------------------
app = FastAPI()


class Item:
    __slots__ = ("arr", "fut", "t_start_total")

    def __init__(self, arr: np.ndarray):
        self.arr = arr
        self.fut = asyncio.get_event_loop().create_future()
        self.t_start_total = time.perf_counter()


QUEUE: asyncio.Queue[Item] = asyncio.Queue()


def _ensure_16k_float32_1d(arr: np.ndarray) -> np.ndarray:
    # Expect float32 mono PCM @ 16k from Pipecat. Clamp/pad to MAX_SAMPLES.
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    arr = np.ravel(arr)
    if arr.size > MAX_SAMPLES:
        arr = arr[-MAX_SAMPLES:]
    elif arr.size < MAX_SAMPLES:
        arr = np.pad(arr, (0, MAX_SAMPLES - arr.size), mode="constant")
    return arr


def _auth_ok(request: Request) -> bool:
    if not AUTH_KEY:
        return True
    header = request.headers.get("authorization") or request.headers.get("Authorization")
    return isinstance(header, str) and header.strip() == f"Key {AUTH_KEY}"


async def _batcher():
    await asyncio.sleep(0)  # let uvicorn bind

    while True:
        await asyncio.sleep(MICRO_BATCH_WINDOW_MS / 1000.0)

        # Drain up to largest bucket
        items: List[Item] = []
        while not QUEUE.empty() and len(items) < max(BATCH_BUCKETS):
            items.append(QUEUE.get_nowait())

        if not items:
            continue

        depth = len(items)
        bucket = min([b for b in BATCH_BUCKETS if b >= depth] or [BATCH_BUCKETS[-1]])
        logger.debug(f"batcher: depth={depth} bucket={bucket} queue_empty={QUEUE.empty()}")

        # Build (B, T) tensor
        try:
            np_stack = np.stack([it.arr for it in items], axis=0).astype(np.float32, copy=False)
            batch = torch.from_numpy(np_stack)
            if DEVICE.type == "cuda":
                batch = batch.pin_memory().to(DEVICE, non_blocking=True)
            else:
                batch = batch.to(DEVICE)
        except Exception as e:
            logger.exception(f"batcher: failed to build batch tensor: {e}")
            # Fail-safe: resolve everything so clients don't hang
            now = time.perf_counter()
            for it in items:
                if not it.fut.done():
                    it.fut.set_result({
                        "prediction": 0,
                        "probability": 0.0,
                        "metrics": {"inference_time": 0.0, "total_time": now - it.t_start_total},
                    })
            continue

        t0_inf = time.perf_counter()
        logits = None

        try:
            if USE_CUDA_GRAPHS and DEVICE.type == "cuda":
                # Pad to bucket for graph replay
                if batch.shape[0] < bucket:
                    pad = torch.zeros((bucket, batch.shape[1]), dtype=batch.dtype, device=batch.device)
                    pad[: batch.shape[0]].copy_(batch)
                    batch_b = pad
                else:
                    batch_b = batch

                if bucket in GRAPHS:
                    # FAST PATH: replay
                    g, logits_ref, inp = GRAPHS[bucket]
                    inp["input_values"].copy_(batch_b)
                    g.replay()
                    logits = logits_ref
                else:
                    # Serve eagerly NOW, capture asynchronously in background
                    if bucket not in CAPTURING:
                        CAPTURING.add(bucket)
                        asyncio.create_task(_capture_bucket(bucket))

                    inp = processor(
                        batch, sampling_rate=SAMPLE_RATE, padding="max_length", truncation=True,
                        max_length=MAX_SAMPLES, return_attention_mask=True, return_tensors="pt",
                    )
                    inp = {k: v.to(DEVICE, non_blocking=True) for k, v in inp.items()}
                    with torch.no_grad():
                        if DTYPE != torch.float32:
                            with torch.autocast("cuda", dtype=DTYPE):
                                out = model(**inp)
                        else:
                            out = model(**inp)
                    logits = out.logits
            else:
                # Pure eager path
                inp = processor(
                    batch, sampling_rate=SAMPLE_RATE, padding="max_length", truncation=True,
                    max_length=MAX_SAMPLES, return_attention_mask=True, return_tensors="pt",
                )
                inp = {k: v.to(DEVICE, non_blocking=True) for k, v in inp.items()}
                with torch.no_grad():
                    if DEVICE.type == "cuda" and DTYPE != torch.float32:
                        with torch.autocast("cuda", dtype=DTYPE):
                            out = model(**inp)
                    else:
                        out = model(**inp)
                logits = out.logits

            # logits -> probs for the REAL depth (ignore padded rows)
            probs = logits.detach().float().cpu().numpy().reshape(-1)[:depth]
            t1_inf = time.perf_counter()

            for it, p in zip(items, probs):
                pred = 1 if p > THRESHOLD else 0
                total_s = (t1_inf - it.t_start_total)
                inf_s = (t1_inf - t0_inf)
                if not it.fut.done():
                    it.fut.set_result({
                        "prediction": int(pred),
                        "probability": float(p),
                        "metrics": {"inference_time": inf_s, "total_time": total_s},
                    })

        except Exception as e:
            logger.exception(f"batcher: inference failed: {e}")
            # Fail-safe: resolve everything so clients don't hang
            now = time.perf_counter()
            for it in items:
                if not it.fut.done():
                    it.fut.set_result({
                        "prediction": 0,
                        "probability": 0.0,
                        "metrics": {"inference_time": 0.0, "total_time": now - it.t_start_total},
                    })
            continue


@app.on_event("startup")
async def _on_start():
    asyncio.create_task(_batcher())
    logger.info(f"Smart Turn v2 server ready on {DEVICE} | buckets={BATCH_BUCKETS} window_ms={MICRO_BATCH_WINDOW_MS}")
    # Write readiness file for orchestrators that prefer file-based health
    try:
        root = Path(__file__).resolve().parents[1]
        run_dir = root / ".run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "ready").write_text("ok", encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to write readiness file: {e}")


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/raw")
async def raw(request: Request):
    if not _auth_ok(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    body = await request.body()
    logger.debug(f"POST /raw: received {len(body)} bytes")
    try:
        arr = np.load(io.BytesIO(body), allow_pickle=False)
        logger.debug(f"POST /raw: parsed npy -> shape={arr.shape} dtype={arr.dtype}")
        arr = _ensure_16k_float32_1d(arr)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid npy payload: {e}")

    item = Item(arr)
    await QUEUE.put(item)
    logger.debug("POST /raw: queued item")
    return await item.fut


