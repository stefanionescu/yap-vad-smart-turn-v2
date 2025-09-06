# src/server.py
import os, io, time, asyncio
from typing import List

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
    MICRO_BATCH_WINDOW_MS, AUTH_KEY, MODEL_ID, LOG_LEVEL,
)

# Be resilient: never crash the server if compile fails
import torch._dynamo as dynamo
dynamo.config.suppress_errors = True

# Configure logging level
logger.remove()  # Remove default handler
logger.add(
    lambda msg: print(msg, end=''),
    level=LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)


# -------- Configurable buckets (1..64 by default) ----------
def _parse_buckets(val: str) -> list[int]:
    try:
        return [int(x) for x in val.split(",") if x.strip()]
    except Exception:
        return [1, 2, 4, 8]  # Default to smaller buckets to avoid OOM

BATCH_BUCKETS = _parse_buckets(os.environ.get("BATCH_BUCKETS", "1,2,4,8"))
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

# -------- State ----------
_ACTIVE_MODEL: torch.nn.Module | None = None      # serves requests (eager first, then compiled)
_COMPILED_MODEL: torch.nn.Module | None = None
_COMPILED_READY = False

_EAGER_MODEL: torch.nn.Module | None = None       # plain eager for serving until compiled swaps in

# -------- Helpers ----------
def _ensure_16k(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    arr = np.ravel(arr)
    if arr.size > MAX_SAMPLES:
        arr = arr[:MAX_SAMPLES]
    elif arr.size < MAX_SAMPLES:
        arr = np.pad(arr, (0, MAX_SAMPLES - arr.size))
    return arr

def _auth_ok(request: Request) -> bool:
    if not AUTH_KEY:
        return True
    header = request.headers.get("authorization") or request.headers.get("Authorization")
    return isinstance(header, str) and header.strip() == f"Key {AUTH_KEY}"

def _make_inputs_gpu(x: torch.Tensor, cast_to: torch.dtype | None = None) -> dict[str, torch.Tensor]:
    """
    Processorless path: we already feed normalized mono PCM @ 16k, padded to MAX_SAMPLES.
    Create attention_mask of ones and (optionally) cast input_values to match model dtype.
    """
    if x.device != DEVICE:
        x = x.to(DEVICE, non_blocking=True)
    if cast_to is not None and x.dtype != cast_to:
        x = x.to(cast_to)

    attn = torch.ones((x.shape[0], x.shape[1]), dtype=torch.long, device=DEVICE)
    return {"input_values": x, "attention_mask": attn}

def _build_eager_if_needed():
    global _EAGER_MODEL, _ACTIVE_MODEL
    if _EAGER_MODEL is None:
        m = Wav2Vec2ForEndpointing.from_pretrained(MODEL_ID).to(DEVICE)
        if DTYPE != torch.float32:
            m = m.to(dtype=DTYPE)
        m.eval()
        _EAGER_MODEL = m
        _ACTIVE_MODEL = _EAGER_MODEL
        logger.info("Eager model built and set ACTIVE.")

async def _compile_in_background():
    """Compile a separate model and warm it once, then atomically swap it in."""
    global _COMPILED_MODEL, _ACTIVE_MODEL, _COMPILED_READY
    if not USE_TORCH_COMPILE or DEVICE.type != "cuda":
        logger.info("torch.compile disabled or no CUDA; skipping compile.")
        return
    try:
        logger.info("Background: building compiled model...")
        m = Wav2Vec2ForEndpointing.from_pretrained(MODEL_ID).to(DEVICE)
        if DTYPE != torch.float32:
            m = m.to(dtype=DTYPE)
        m.eval()
        m = torch.compile(m, mode="reduce-overhead")

        # 🔥 Pre-warm all buckets so no runtime recompiles
        for b in sorted(set(BATCH_BUCKETS)):
            logger.info(f"Background: warm compile for batch={b}")
            warm = torch.zeros((b, MAX_SAMPLES), dtype=torch.float32, device=DEVICE)
            inp = _make_inputs_gpu(warm, cast_to=(DTYPE if DTYPE != torch.float32 else None))
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
        logger.info("Background: compiled model is READY and ACTIVE.")
    except Exception as e:
        logger.exception(f"Background compile failed: {e}")


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
            # compiled or eager
            if _ACTIVE_MODEL is None:
                _build_eager_if_needed()
                asyncio.create_task(_compile_in_background())

            m = _ACTIVE_MODEL
            logger.debug(f"batcher: eager/compiled forward bucket={bucket} (compiled_ready={_COMPILED_READY})")
            inp = _make_inputs_gpu(batch, cast_to=(DTYPE if DTYPE != torch.float32 else None))
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
    logger.info(f"Smart Turn v2 server | device={DEVICE} dtype={DTYPE} buckets={BATCH_BUCKETS}")
    asyncio.create_task(_batcher())

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