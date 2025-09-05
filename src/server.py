import os
import io
import time
import asyncio
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, Request, HTTPException
from loguru import logger
from transformers import AutoProcessor, AutoModelForAudioClassification

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
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
model.eval().to(DEVICE)
if DTYPE != torch.float32 and DEVICE.type == "cuda":
    model.to(dtype=DTYPE)

if USE_TORCH_COMPILE and DEVICE.type == "cuda":
    logger.info("Compiling model with torch.compile(mode='reduce-overhead')")
    model = torch.compile(model, mode="reduce-overhead")


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
    """Micro-batch loop. Every MICRO_BATCH_WINDOW_MS, drain queue up to max bucket,
    run one forward pass, and set each future with result.
    """
    if DEVICE.type == "cuda":
        # Warmup with a fixed shape. This improves first-request latency and stabilizes kernels.
        warm = torch.zeros((1, MAX_SAMPLES), dtype=torch.float32, device=DEVICE)
        warm_inputs = processor(
            warm,
            sampling_rate=SAMPLE_RATE,
            padding="max_length",
            truncation=True,
            max_length=MAX_SAMPLES,
            return_attention_mask=True,
            return_tensors="pt",
        )
        warm_inputs = {k: v.to(DEVICE) for k, v in warm_inputs.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE, enabled=(DTYPE != torch.float32)):
            _ = model(**warm_inputs)

    graphs = {}
    static_inputs = {}
    if USE_CUDA_GRAPHS and DEVICE.type == "cuda":
        torch.cuda.synchronize()
        for b in BATCH_BUCKETS:
            x = torch.zeros((b, MAX_SAMPLES), dtype=torch.float32, device=DEVICE)
            inp = processor(
                x,
                sampling_rate=SAMPLE_RATE,
                padding="max_length",
                truncation=True,
                max_length=MAX_SAMPLES,
                return_attention_mask=True,
                return_tensors="pt",
            )
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            g = torch.cuda.CUDAGraph()
            out_logits_ref = None
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                out = model(**inp)
                out_logits_ref = out.logits
            graphs[b] = (g, out_logits_ref)
            static_inputs[b] = inp
        torch.cuda.synchronize()
        logger.info(f"CUDA Graphs captured for batch sizes: {BATCH_BUCKETS}")

    while True:
        await asyncio.sleep(MICRO_BATCH_WINDOW_MS / 1000.0)

        items: List[Item] = []
        while not QUEUE.empty() and len(items) < max(BATCH_BUCKETS):
            items.append(QUEUE.get_nowait())
        if not items:
            continue

        depth = len(items)
        bucket = min([b for b in BATCH_BUCKETS if b >= depth] or [BATCH_BUCKETS[-1]])

        # Build CPU tensor → pinned → HtoD
        np_stack = np.stack([it.arr for it in items], axis=0)  # (B, T) float32
        batch = torch.from_numpy(np_stack)
        if DEVICE.type == "cuda":
            batch = batch.pin_memory().to(DEVICE, non_blocking=True)
        else:
            batch = batch.to(DEVICE)

        t0_inf = time.perf_counter()

        if USE_CUDA_GRAPHS and DEVICE.type == "cuda" and bucket in graphs and batch.shape[0] == bucket:
            # Replace input values inside static graph inputs and replay
            inp = static_inputs[bucket]
            # For Wav2Vec-style processors, input tensor key is usually "input_values"
            if "input_values" in inp:
                inp["input_values"].copy_(batch)
            else:
                # Fallback: re-tokenize if processor layout unexpected
                inp = processor(
                    batch,
                    sampling_rate=SAMPLE_RATE,
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_SAMPLES,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                inp = {k: v.to(DEVICE, non_blocking=True) for k, v in inp.items()}
            g, out_logits_ref = graphs[bucket]
            g.replay()
            probs = out_logits_ref.detach().float().cpu().numpy().reshape(-1)
        else:
            # Tokenize per batch (CPU-bound but lightweight)
            inp = processor(
                batch,
                sampling_rate=SAMPLE_RATE,
                padding="max_length",
                truncation=True,
                max_length=MAX_SAMPLES,
                return_attention_mask=True,
                return_tensors="pt",
            )
            inp = {k: v.to(DEVICE, non_blocking=True) for k, v in inp.items()}

            if DEVICE.type == "cuda":
                with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE, enabled=(DTYPE != torch.float32)):
                    out = model(**inp)
            else:
                with torch.no_grad():
                    out = model(**inp)
            # Smart Turn v2 README: logits already are sigmoid probabilities (0..1)
            logits = out.logits
            probs = logits.detach().float().cpu().numpy().reshape(-1)

        t1_inf = time.perf_counter()

        for it, p in zip(items, probs):
            pred = 1 if p > THRESHOLD else 0
            total_s = (t1_inf - it.t_start_total)
            inf_s = (t1_inf - t0_inf)
            result = {
                "prediction": int(pred),
                "probability": float(p),
                "metrics": {
                    "inference_time": inf_s,
                    "total_time": total_s,
                },
            }
            if not it.fut.done():
                it.fut.set_result(result)


@app.on_event("startup")
async def _on_start():
    asyncio.create_task(_batcher())
    logger.info(f"Smart Turn v2 server ready on {DEVICE} | buckets={BATCH_BUCKETS} window_ms={MICRO_BATCH_WINDOW_MS}")


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/raw")
async def raw(request: Request):
    if not _auth_ok(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        body = await request.body()
        arr = np.load(io.BytesIO(body), allow_pickle=False)
        arr = _ensure_16k_float32_1d(arr)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid npy payload: {e}")

    item = Item(arr)
    await QUEUE.put(item)
    return await item.fut


