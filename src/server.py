import os, io, time, asyncio
from typing import List, Dict, Tuple

import numpy as np
import torch
from fastapi import FastAPI, Request, HTTPException
from loguru import logger
from pathlib import Path

try:
    from .model import Wav2Vec2ForEndpointing
except ImportError as e:
    raise RuntimeError("smart-turn-v2 needs the custom Wav2Vec2ForEndpointing from model.py") from e

from .constants import (
    SAMPLE_RATE, MAX_SECS, DTYPE, THRESHOLD,
    MICRO_BATCH_WINDOW_MS, AUTH_KEY, MODEL_ID,
)

# -------- Configurable buckets (1..64 by default) ----------
def _parse_buckets(val: str) -> list[int]:
    try: return [int(x) for x in val.split(",") if x.strip()]
    except: return [1,2,4,8,16,32,64]

BATCH_BUCKETS = _parse_buckets(os.environ.get("BATCH_BUCKETS", "1,2,4,8,16,32,64"))
USE_TORCH_COMPILE = os.environ.get("TORCH_COMPILE", "1") == "1"
USE_CUDA_GRAPHS  = os.environ.get("CUDA_GRAPHS", "1") == "1"
CAPTURE_ALL      = os.environ.get("CAPTURE_ALL", "1") == "1"
CAPTURE_CONCURRENCY = int(os.environ.get("CAPTURE_CONCURRENCY", "1"))

# -------- Device / precision ----------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
torch.set_num_threads(2)
try:
    # L40/L4: TF32 helps a lot
    torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass
torch.set_float32_matmul_precision("high")

MAX_SAMPLES = SAMPLE_RATE * int(float(os.environ.get("MAX_SECS", str(MAX_SECS))))

# -------- State ----------
# We start with an eager model to guarantee the first request returns.
# In the background we compile another instance, then atomically swap.
_ACTIVE_MODEL: torch.nn.Module | None = None      # always ready to serve
_COMPILED_MODEL: torch.nn.Module | None = None    # becomes active when warm
_COMPILED_READY = False

# CUDA-graphs per bucket: bucket -> (graph, logits_ref, static_inputs)
GRAPHS: Dict[int, Tuple[torch.cuda.CUDAGraph, torch.Tensor, Dict[str, torch.Tensor]]] = {}
CAPTURED: set[int] = set()
CAPTURING: set[int] = set()

# -------- Helpers ----------
def _ensure_16k(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.float32: arr = arr.astype(np.float32, copy=False)
    arr = np.ravel(arr)
    if arr.size > MAX_SAMPLES: arr = arr[:MAX_SAMPLES]  # first N seconds
    elif arr.size < MAX_SAMPLES: arr = np.pad(arr, (0, MAX_SAMPLES - arr.size))
    return arr

def _auth_ok(request: Request) -> bool:
    if not AUTH_KEY: return True
    header = request.headers.get("authorization") or request.headers.get("Authorization")
    return isinstance(header, str) and header.strip() == f"Key {AUTH_KEY}"

def _make_inputs_gpu(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    # x: (B, T) float32 on DEVICE, already clamped/padded
    attn = torch.ones((x.shape[0], x.shape[1]), dtype=torch.long, device=x.device)
    return {"input_values": x, "attention_mask": attn}

async def _compile_in_background():
    """Compile a separate model instance and warm it once so it's truly ready.
       Then atomically swap in as ACTIVE and free the eager model."""
    global _COMPILED_MODEL, _ACTIVE_MODEL, _COMPILED_READY
    if not USE_TORCH_COMPILE or DEVICE.type != "cuda":
        logger.info("torch.compile disabled or no CUDA; skipping compile.")
        return
    try:
        logger.info("Background: building compiled model...")
        m = Wav2Vec2ForEndpointing.from_pretrained(MODEL_ID).to(DEVICE)
        # cast dtype if requested
        if DTYPE != torch.float32: m = m.to(dtype=DTYPE)
        m.eval()
        m = torch.compile(m, mode="reduce-overhead")
        # warm once to trigger actual compile
        warm = torch.zeros((1, MAX_SAMPLES), dtype=torch.float32, device=DEVICE)
        inp = _make_inputs_gpu(warm)
        with torch.no_grad():
            if DTYPE != torch.float32:
                with torch.autocast("cuda", dtype=DTYPE):
                    _ = m(**inp)
            else:
                _ = m(**inp)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        _COMPILED_MODEL = m
        _COMPILED_READY = True
        _ACTIVE_MODEL = _COMPILED_MODEL
        logger.info("Background: compiled model is READY and ACTIVE.")
    except Exception as e:
        logger.exception(f"Background compile failed: {e}")

async def _capture_bucket(bucket: int):
    """Capture CUDA graph for a bucket AFTER compiled model is ready."""
    global GRAPHS
    if not (USE_CUDA_GRAPHS and _COMPILED_READY and DEVICE.type == "cuda"): return
    if bucket in CAPTURED or bucket in CAPTURING: return
    CAPTURING.add(bucket)
    try:
        logger.info(f"CUDA graph capture: bucket={bucket} starting...")
        x = torch.zeros((bucket, MAX_SAMPLES), dtype=torch.float32, device=DEVICE)
        inp = _make_inputs_gpu(x)
        m = _ACTIVE_MODEL
        # eager warmup for graph specialization
        with torch.no_grad():
            if DTYPE != torch.float32:
                with torch.autocast("cuda", dtype=DTYPE):
                    _ = m(**inp)
            else:
                _ = m(**inp)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = m(**inp)
            logits_ref = out["logits"] if isinstance(out, dict) else out.logits
        GRAPHS[bucket] = (g, logits_ref, inp)
        CAPTURED.add(bucket)
        logger.info(f"CUDA graph capture: bucket={bucket} DONE.")
    except Exception as e:
        logger.exception(f"CUDA graph capture failed for bucket={bucket}: {e}")
    finally:
        CAPTURING.discard(bucket)

async def _capture_all_buckets():
    """Schedule graph capture for all buckets sequentially or with limited concurrency."""
    if not (USE_CUDA_GRAPHS and DEVICE.type == "cuda"): return
    # wait until compiled model is active
    while not _COMPILED_READY:
        await asyncio.sleep(0.05)
    logger.info(f"Starting background graph capture for buckets={BATCH_BUCKETS} (concurrency={CAPTURE_CONCURRENCY})")
    sem = asyncio.Semaphore(CAPTURE_CONCURRENCY)
    async def _task(b):
        async with sem:
            await _capture_bucket(b)
    await asyncio.gather(*[_task(b) for b in BATCH_BUCKETS])

# -------- FastAPI app & queue ----------
app = FastAPI()

class Item:
    __slots__ = ("arr", "fut", "t_start_total")
    def __init__(self, arr: np.ndarray):
        self.arr = arr
        self.fut = asyncio.get_event_loop().create_future()
        self.t_start_total = time.perf_counter()

QUEUE: asyncio.Queue[Item] = asyncio.Queue()

async def _batcher():
    await asyncio.sleep(0)  # let uvicorn bind logs

    while True:
        await asyncio.sleep(MICRO_BATCH_WINDOW_MS / 1000.0)

        items: List[Item] = []
        while not QUEUE.empty() and len(items) < max(BATCH_BUCKETS):
            items.append(QUEUE.get_nowait())
        if not items:
            continue

        depth = len(items)
        # choose smallest bucket >= depth
        bucket = min([b for b in BATCH_BUCKETS if b >= depth] or [BATCH_BUCKETS[-1]])
        logger.debug(f"batcher: depth={depth} bucket={bucket} queue_empty={QUEUE.empty()}")

        # Build batch tensor on GPU
        try:
            np_stack = np.stack([it.arr for it in items], axis=0).astype(np.float32, copy=False)
            batch = torch.from_numpy(np_stack)
            if DEVICE.type == "cuda":
                batch = batch.pin_memory().to(DEVICE, non_blocking=True)
            else:
                batch = batch.to(DEVICE)
        except Exception as e:
            logger.exception(f"batcher: failed to build batch tensor: {e}")
            now = time.perf_counter()
            for it in items:
                if not it.fut.done():
                    it.fut.set_result({
                        "prediction": 0, "probability": 0.0,
                        "metrics": {"inference_time": 0.0, "total_time": now - it.t_start_total},
                    })
            continue

        t0 = time.perf_counter()
        logits = None

        try:
            # If compiled ready AND graph captured for this bucket => replay
            if USE_CUDA_GRAPHS and _COMPILED_READY and bucket in GRAPHS and DEVICE.type == "cuda":
                logger.debug(f"batcher: replay graph bucket={bucket}")
                g, logits_ref, inp = GRAPHS[bucket]
                # pad to bucket for static graph
                if batch.shape[0] < bucket:
                    pad = torch.zeros((bucket, batch.shape[1]), dtype=batch.dtype, device=batch.device)
                    pad[: batch.shape[0]].copy_(batch)
                    batch_b = pad
                else:
                    batch_b = batch
                inp["input_values"].copy_(batch_b)
                inp["attention_mask"].fill_(1)
                g.replay()
                logits = logits_ref[:depth]  # trim to real depth

            else:
                # Serve with ACTIVE model (eager initially, compiled later)
                m = _ACTIVE_MODEL
                if m is None:
                    # First-time eager model build (synchronously; cheap)
                    logger.info("Building eager model for immediate serving...")
                    m = Wav2Vec2ForEndpointing.from_pretrained(MODEL_ID).to(DEVICE)
                    if DTYPE != torch.float32: m = m.to(dtype=DTYPE)
                    m.eval()
                    _ACTIVE_MODEL = m
                    # kick background compile + capture
                    asyncio.create_task(_compile_in_background())
                    if CAPTURE_ALL:
                        asyncio.create_task(_capture_all_buckets())

                logger.debug(f"batcher: eager forward bucket={bucket} (compiled_ready={_COMPILED_READY})")
                inp = _make_inputs_gpu(batch)
                with torch.no_grad():
                    if DEVICE.type == "cuda" and DTYPE != torch.float32:
                        with torch.autocast("cuda", dtype=DTYPE):
                            out = m(**inp)
                    else:
                        out = m(**inp)
                logits = out["logits"] if isinstance(out, dict) else out.logits

            probs = logits.detach().float().cpu().numpy().reshape(-1)
            t1 = time.perf_counter()
            logger.debug(f"batcher: forward done in {(t1-t0)*1000:.1f} ms")
            
            for it, p in zip(items, probs):
                pred = 1 if p > THRESHOLD else 0
                if not it.fut.done():
                    it.fut.set_result({
                        "prediction": int(pred),
                        "probability": float(p),
                        "metrics": {"inference_time": (t1 - t0), "total_time": (t1 - it.t_start_total)},
                    })

        except Exception as e:
            logger.exception(f"batcher: inference failed: {e}")
            now = time.perf_counter()
            for it in items:
                if not it.fut.done():
                    it.fut.set_result({
                        "prediction": 0, "probability": 0.0,
                        "metrics": {"inference_time": 0.0, "total_time": now - it.t_start_total},
                    })

# -------- FastAPI lifecycle & routes ----------
@app.on_event("startup")
async def _on_start():
    logger.info(f"Loading model (eager) + processorless path | device={DEVICE} dtype={DTYPE} buckets={BATCH_BUCKETS}")
    # NOTE: we don't pre-load here; first request builds eager model once.
    asyncio.create_task(_batcher())
    # Also kick compile/capture pre-emptively so first requests are quick
    asyncio.create_task(_compile_in_background())
    if CAPTURE_ALL:
        asyncio.create_task(_capture_all_buckets())
    # mark ready file
    try:
        root = Path(__file__).resolve().parents[1]
        run_dir = root / ".run"; run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "ready").write_text("ok", encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to write readiness file: {e}")
    logger.info("Smart Turn v2 server ready (startup tasks scheduled).")

@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/status")
async def status():
    return {
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "compiled_ready": _COMPILED_READY,
        "captured": sorted(list(CAPTURED)),
        "capturing": sorted(list(CAPTURING)),
        "buckets": BATCH_BUCKETS,
        "queue_depth": QUEUE.qsize(),
    }

@app.post("/raw")
async def raw(request: Request):
    if not _auth_ok(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    body = await request.body()
    logger.debug(f"POST /raw: received {len(body)} bytes")
    try:
        arr = np.load(io.BytesIO(body), allow_pickle=False)
        logger.debug(f"POST /raw: parsed npy -> shape={arr.shape} dtype={arr.dtype}")
        arr = _ensure_16k(arr)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid npy payload: {e}")
    item = Item(arr)
    await QUEUE.put(item)
    logger.debug("POST /raw: queued item")
    return await item.fut