# Yap VAD Smart Turn

FastAPI server for `pipecat-ai/smart-turn-v2` with CUDA-efficient micro-batching on NVIDIA L40S. Same wire format as Pipecat's `HttpSmartTurnAnalyzer`: POST `/raw` with `np.save` bytes → JSON `{prediction, probability, metrics}`.

### Why this setup

- **Fast**: micro-batches with smart bucket sizes (1,2,4,6) and 5ms window, uses pinned memory and `torch.compile`. Designed for 8–16s mono 16kHz PCM.
- **Efficient**: Async queue-based batching with eager→compiled model swapping. No blocking waits.
- **Simple**: no Docker required. Works well on Runpod dedicated L40S (CUDA 12.8 drivers are compatible with PyTorch cu124 wheels).
- **Compatible**: exact Pipecat HTTP analyzer format.
- **Stable**: CUDA graphs removed, smaller batch buckets prevent OOM, expandable memory segments.

---

## One command install & run

```bash
bash scripts/main.sh
```

What it does:
- Installs/updates venv and Python deps
- Prefetches the model weights
- Creates a default sample `samples/mid.wav`
- Starts the server in the background on port 8000
- Tails logs (brief delay)
- Sends a one-shot warmup using `samples/mid.wav`

Defaults are set in scripts; no env needed for basic run. Health: `GET /health`.

---

## API

- `POST /raw`
  - Headers: `Content-Type: application/octet-stream`, optionally `Authorization: Key <AUTH_KEY>`
  - Body: raw `np.save` bytes of a `np.float32` mono PCM array at 16 kHz, duration 8–16 s.
  - Response JSON:
   ```json
   {
     "prediction": 0,
     "probability": 0.123,
     "metrics": {
       "inference_time": 0.0123,
       "total_time": 0.0189
     }
   }
   ```

- `GET /health` → `{ "ok": true }`
- `GET /status` → Server diagnostics:
  ```json
  {
    "device": "cuda:0",
    "dtype": "torch.bfloat16", 
    "compiled_ready": true,
    "buckets": [1, 2, 4, 8],
    "queue_depth": 0
  }
  ```

---

## Environment variables

- `AUTH_KEY` (optional): If set, server requires `Authorization: Key <AUTH_KEY>`.
- `BATCH_BUCKETS` (default `1,2,4,6`): Comma-separated micro-batch sizes.
- `MICRO_BATCH_WINDOW_MS` (default `5`): Batching window in milliseconds.
- `DTYPE` (`bfloat16` or `float32`, default `bfloat16`): Compute dtype.
- `THRESHOLD` (default `0.5`): Decision threshold over probability.
- `TORCH_COMPILE` (`0|1`, default `1`): Use `torch.compile` reduce-overhead mode.
- `MODEL_ID` (default `pipecat-ai/smart-turn-v2`).
- `MAX_SECS` (default `16`): Maximum audio duration in seconds.
- `PYTORCH_CUDA_ALLOC_CONF` (default `expandable_segments:True`): CUDA memory allocator.
- `LOG_LEVEL` (default `INFO`): Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
- Port is fixed to `8000` (see `src/constants.py`).

**Note**: CUDA graphs have been removed from the codebase for stability.

---

## Tests

Ensure venv is active or run after `scripts/main.sh`.

- If you already ran `bash scripts/main.sh`, you're set (venv and deps are installed).
- Otherwise, do:

```bash
bash scripts/setup.sh
bash scripts/start_bg.sh
sleep 120
source .venv/bin/activate
```

- Warmup (single request, pads/truncates to `--seconds`):

```bash
# Quiet mode (default - no debug logs)
python3 ./test/warmup.py --sample mid.wav --seconds 8

# With debug logs
LOG_LEVEL=DEBUG python3 ./test/warmup.py --sample mid.wav --seconds 8
```

- Bench (total requests and concurrency):

```bash
python3 ./test/bench.py --sample mid.wav --seconds 8 --requests 6 --concurrency 6
```

- Throughput (sustained RPS over duration; reports p50/p95):

```bash
python3 ./test/throughput.py --duration 60 --concurrency 4 --sample mid.wav --seconds 8

# With AUTH_KEY
python3 ./test/throughput.py --key dev
```

- Client (reads `.env` for `RUNPOD_TCP_HOST`, `RUNPOD_TCP_PORT`, `AUTH_KEY`):

```bash
python3 test/client --sample mid.wav --seconds 8
```

`.env` example (repo root):

```ini
RUNPOD_TCP_HOST=localhost
RUNPOD_TCP_PORT=8000
AUTH_KEY=dev
```

---

## Pipecat client example

```python
from pipecat.audio.turn.smart_turn.http_smart_turn import HttpSmartTurnAnalyzer, SmartTurnParams

analyzer = HttpSmartTurnAnalyzer(
    url="http://<host>:8000/raw",
    headers={"Authorization": "Key dev"},
    params=SmartTurnParams(stop_secs=3, max_duration_secs=8),
)
```

---

## Local poke

```python
import numpy as np, aiohttp, asyncio, io

async def main():
    arr = np.zeros(16000*8, dtype=np.float32)
    buf = io.BytesIO(); np.save(buf, arr)
    async with aiohttp.ClientSession() as s:
        async with s.post("http://localhost:8000/raw",
                          data=buf.getvalue(),
                          headers={"Content-Type":"application/octet-stream","Authorization":"Key dev"}) as r:
            print(await r.json())

asyncio.run(main())
```

---

## How it works

### Architecture Overview

The server uses an **async queue-based micro-batching architecture**:

1. **Request Flow**: `POST /raw` → parse numpy → queue item → await result
2. **Batching Loop**: Runs every `MICRO_BATCH_WINDOW_MS`, drains queue up to max bucket size
3. **Smart Bucketing**: Chooses smallest bucket size ≥ actual batch size (e.g., 3 requests → bucket=4)
4. **GPU Inference**: Batch processing with pinned memory and non-blocking transfers
5. **Model Management**: Starts with eager model, compiles in background, swaps atomically

### Key Components

- **`_batcher()`**: Main async loop that collects requests and runs inference
- **`Item`**: Request wrapper with numpy array, future, and timing
- **`QUEUE`**: Async queue holding incoming requests
- **`_ACTIVE_MODEL`**: Current serving model (eager → compiled transition)
- **`_ensure_16k()`**: Normalizes input audio to 16kHz mono, pads/truncates to MAX_SAMPLES

### Batching Algorithm

```python
# Every MICRO_BATCH_WINDOW_MS milliseconds:
while not QUEUE.empty() and len(items) < max(BATCH_BUCKETS):
    items.append(QUEUE.get_nowait())

# Choose smallest bucket >= actual batch size
bucket = min([b for b in BATCH_BUCKETS if b >= len(items)])

# Build GPU batch tensor with pinned memory
batch = torch.from_numpy(np.stack([item.arr for item in items]))
batch = batch.pin_memory().to(DEVICE, non_blocking=True)
```

### Model Loading & Compilation

1. **Eager Model**: Loads immediately on first request for fast startup
2. **Background Compilation**: `torch.compile()` runs in parallel with serving  
3. **Pre-warming**: All batch sizes (1,2,4,6) are compiled upfront to prevent runtime JIT stalls
4. **Atomic Swap**: Once compiled model is warmed up, switches `_ACTIVE_MODEL`
5. **Resilient**: Compilation failures don't crash server, falls back to eager

### What's Different from Typical Setups

- **No CUDA Graphs**: Removed for stability - torch.compile provides sufficient optimization
- **Small Batch Buckets**: (1,2,4,6) instead of (16,32,64) to prevent GPU OOM on L40S  
- **Async Queue Architecture**: Non-blocking requests with futures, no threading complexity
- **Atomic Model Swapping**: Smooth eager→compiled transition without request interruption
- **Expandable Memory Segments**: Better CUDA memory management vs default PyTorch allocator

---

## Tuning

- Default configuration uses `MICRO_BATCH_WINDOW_MS=5`, `BATCH_BUCKETS=1,2,4,6` (stable, smaller buckets to avoid OOM).
- Prefer `DTYPE=bfloat16` on L40S; switch to `float32` for accuracy validation.
- Keep a single worker process; model is loaded once.
- See "Production Deployment" section below for detailed performance tuning options.

---

## Performance

Example results from a recent run with the default configuration:

| Scenario | Settings | Results |
|---|---|---|
| Throughput (sustained) | duration=60s, concurrency=4, sample=mid.wav, seconds=8 | success=11486, fail=0, 191.43 rps, p50=20.6ms, p95=22.9ms |
| Bench (short burst) | total=4, concurrency=4 | p50=31.8ms, p95=31.8ms |
| Warmup (single request) | seconds=8 | inference_time≈10.1ms, total_time≈15.8ms |

Notes:
- Numbers will vary by hardware, driver, and load. Use `test/throughput.py` and `test/bench.py` to reproduce on your setup. In our case, we ran the tests on an L40S.
- The batcher is event-driven; tuning `MICRO_BATCH_WINDOW_MS` and `BATCH_BUCKETS` affects latency vs throughput trade-offs.

## Code Structure

```
src/
├── server.py              # Thin entrypoint (uvicorn src.server:app)
├── app/
│   └── factory.py         # FastAPI app factory + startup hook
├── api/
│   └── routes.py          # /health, /status, /raw
├── serving/
│   └── batcher.py         # Item, QUEUE, batcher() loop
├── runtime/
│   └── runtime.py         # Logger, device/dtype, buckets, compile, inputs
├── utils/
│   ├── audio.py           # ensure_16k()
│   └── auth.py            # auth_ok()
├── constants.py           # Config constants and env parsing
└── model.py               # Custom Wav2Vec2ForEndpointing model class

scripts/
├── main.sh                # One-command setup → start(bg) → warmup
├── setup.sh               # Install venv, PyTorch, model weights
├── start_bg.sh            # Background server start
├── stop.sh                # Stop server, optional cleanup
└── tail_bg_logs.sh        # Follow server logs

test/
├── warmup.py              # Single request warmup utility
├── bench.py               # Load testing with concurrency
├── client.py              # Test client with .env config
└── utils.py               # Shared test utilities
```

### Key Files Explained

- **`server.py`**: Thin entrypoint exposing `app` from `app.factory.create_app()` for `uvicorn src.server:app`.
- **`app/factory.py`**: FastAPI app factory, startup hook (starts batcher, triggers compile, writes readiness).
- **`api/routes.py`**: Route handlers for `/raw`, `/health`, `/status`.
- **`serving/batcher.py`**: Async queue (`QUEUE`), request `Item`, and `batcher()` loop.
- **`runtime/runtime.py`**: Logger, device/dtype, buckets; `build_eager_if_needed`, `compile_sync`, `make_inputs_gpu`, `_ACTIVE_MODEL` state.
- **`utils/audio.py`**: `ensure_16k()` input normalization.
- **`utils/auth.py`**: `auth_ok()` header-based auth.
- **`constants.py`**: Centralized configuration and environment parsing.
- **`model.py`**: Custom Wav2Vec2 model class that adds the binary classification head for turn-taking detection.

- **Scripts**: Deployment scripts for setup, background start, log tailing, and stop/cleanup.

---

## Production Deployment

For production use, follow this hygiene checklist:

### Security
- **Set `AUTH_KEY`** before exposing the port. Generate a strong key and use `Authorization: Key <AUTH_KEY>` header in requests.

### Resource Configuration
- **Keep `--workers 1`** per GPU. Each worker loads the full model into GPU memory.
- **Monitor `/status`** endpoint for `queue_depth` to make autoscaling decisions.
- **Leave warmup enabled** - the startup scripts trigger compilation so the first real call is fast.

### Stable Environment Variables
These are the tested, stable defaults (already set in the scripts):

```bash
export BATCH_BUCKETS="1,2,4,6"           # Small buckets to avoid OOM
export TORCH_COMPILE=1                   # Enables torch.compile optimization
export CUDA_GRAPHS=0                     # CUDA graphs disabled (cleaned up)
export DTYPE=bfloat16                    # Optimal for L40S/modern GPUs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Better memory management
export MICRO_BATCH_WINDOW_MS=5           # 5ms window (2-10ms range)
export LOG_LEVEL=DEBUG                   # Enable detailed logging for production
```

### Performance Tuning
- **For more throughput**: Increase `MICRO_BATCH_WINDOW_MS=4-8` (trades a few ms latency for better batching)
- **torch.compile mode**: `reduce-overhead` is the default. Only try `max-autotune` if you benchmark an improvement.
- **Shape specialization**: All bucket sizes are pre-warmed during compilation to avoid runtime JIT stalls
- **Alternative**: Use `dynamic=True` in torch.compile for fewer specializations (slightly slower but more flexible)
- **Hardware**: Designed for NVIDIA L40S. Works on other modern CUDA GPUs with sufficient VRAM.

### Deployment Notes
- Server binds to `0.0.0.0:8000` by default
- Logs to `logs/server.log` in background mode
- Ready signal written to `.run/ready` on startup
- Use reverse proxy (nginx/caddy) for TLS termination

---

## Notes

- Input must be 16 kHz mono float32. Server pads/truncates to ≤16 s.
- Smart Turn v2 returns sigmoid probabilities in `logits`; threshold is configurable.

---

## Stop / Purge

- Stop background server:

```bash
bash scripts/stop.sh
```

- Full purge (remove venv, caches, HF weights, logs; keep repo/system deps):

```bash
bash scripts/stop.sh --purge --deep
```



