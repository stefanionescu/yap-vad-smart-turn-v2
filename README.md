## Yap Smart Turn v2 — L40S CUDA micro-batched server (Pipecat wire format)

FastAPI server for `pipecat-ai/smart-turn-v2` with CUDA-efficient micro-batching on NVIDIA L40S. Same wire format as Pipecat's `HttpSmartTurnAnalyzer`: POST `/raw` with `np.save` bytes → JSON `{prediction, probability, metrics}`.

### Why this setup

- **Fast**: micro-batches 16/32/64 with a 1–2 ms window, uses pinned memory and optional `torch.compile`. Designed for 8–16 s mono 16 kHz PCM.
- **Simple**: no Docker required. Works well on Runpod dedicated L40S (CUDA 12.8 drivers are compatible with PyTorch cu124 wheels).
- **Compatible**: exact Pipecat HTTP analyzer format.

---

## Install

```bash
bash scripts/setup.sh
```

## Start

- Foreground (fixed port 8000):

```bash
export AUTH_KEY=dev
export BATCH_BUCKETS=16,32,64
export MICRO_BATCH_WINDOW_MS=2
export DTYPE=bfloat16
export THRESHOLD=0.5
export TORCH_COMPILE=1   # optional
export CUDA_GRAPHS=0     # optional
bash scripts/start.sh
```

- Background + logs:

```bash
bash scripts/start_bg.sh
bash scripts/tail_bg_logs.sh
```

- One-shot orchestration (always runs setup → start_bg → wait → warmup → optional tail):

```bash
bash scripts/main.sh --sample mid.wav --seconds 8 --tail
# Options: --no-warmup, --sample <file>, --seconds <n>, --tail
```

Server listens on port `8000`. Health: `GET /health`.

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

---

## Environment variables

- `AUTH_KEY` (optional): If set, server requires `Authorization: Key <AUTH_KEY>`.
- `BATCH_BUCKETS` (default `16,32,64`): Comma-separated micro-batch sizes.
- `MICRO_BATCH_WINDOW_MS` (default `2`): Batching window in milliseconds.
- `DTYPE` (`bfloat16` or `float32`, default `bfloat16` on CUDA): Compute dtype.
- `THRESHOLD` (default `0.5`): Decision threshold over probability.
- `TORCH_COMPILE` (`0|1`): Use `torch.compile` reduce-overhead mode.
- `CUDA_GRAPHS` (`0|1`): Capture CUDA graphs per bucket (advanced).
- `MODEL_ID` (default `pipecat-ai/smart-turn-v2`).
- Port is fixed to `8000` (see `src/constants.py`).

---

## Tests

Ensure venv is active (`source venv/bin/activate`) or run after `scripts/main.sh`.

- Warmup (single request, pads/truncates to `--seconds`):

```bash
python -m test.warmup --sample mid.wav --seconds 8
```

- Bench (total requests and concurrency):

```bash
python -m test.bench --sample mid.wav --seconds 8 --requests 256 --concurrency 64
```

- Client (reads `.env` for `RUNPOD_TCP_HOST`, `RUNPOD_TCP_PORT`, `AUTH_KEY`):

```bash
python -m test.client --sample mid.wav --seconds 8
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

## Tuning

- Start with `MICRO_BATCH_WINDOW_MS=2`, `BATCH_BUCKETS=16,32,64`.
- Prefer `DTYPE=bfloat16` on L40S; switch to `float32` for accuracy validation.
- Keep a single worker process; model is loaded once.

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
bash scripts/stop.sh --purge
```



