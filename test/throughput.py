#!/usr/bin/env python3
import argparse
import asyncio
import time
from typing import Optional

import aiohttp
import numpy as np

try:
    # When invoked as a module: python -m test.throughput
    from .utils import load_audio_from_samples  # type: ignore
except ImportError:
    # When invoked as a script: python test/throughput.py
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from test.utils import load_audio_from_samples


async def _worker(name: int, url: str, headers: dict, payload: bytes, stop_at: float, success: list[int], fail: list[int], latencies_s: list[float]) -> None:
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=5, sock_read=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            now = time.perf_counter()
            if now >= stop_at:
                return
            try:
                t0 = time.perf_counter()
                async with session.post(url, data=payload, headers=headers) as resp:
                    if 200 <= resp.status < 300:
                        # Validate response body; if malformed, count as failure
                        try:
                            _ = await resp.json()
                            latencies_s.append(time.perf_counter() - t0)
                            success[0] += 1
                        except Exception:
                            fail[0] += 1
                    else:
                        _ = await resp.text()
                        fail[0] += 1
            except Exception:
                # best-effort: continue; count only successes
                fail[0] += 1
                await asyncio.sleep(0)


async def run(duration_secs: int, concurrency: int, sample: str, seconds: int, url: str, key: Optional[str]) -> None:
    _, payload = load_audio_from_samples(sample, seconds_pad_to=seconds)
    headers = {"Content-Type": "application/octet-stream"}
    if key:
        headers["Authorization"] = f"Key {key}"

    stop_at = time.perf_counter() + duration_secs
    success = [0]
    fail = [0]
    latencies_s: list[float] = []

    tasks = [
        asyncio.create_task(_worker(i, url, headers, payload, stop_at, success, fail, latencies_s))
        for i in range(concurrency)
    ]
    await asyncio.gather(*tasks)

    total = success[0]
    rps = total / duration_secs if duration_secs > 0 else 0.0
    if latencies_s:
        p50_ms = float(np.percentile(latencies_s, 50)) * 1000.0
        p95_ms = float(np.percentile(latencies_s, 95)) * 1000.0
        print(f"throughput: success={success[0]} fail={fail[0]} in {duration_secs}s | {rps:.2f} rps | conc={concurrency} | p50={p50_ms:.1f}ms p95={p95_ms:.1f}ms")
    else:
        print(f"throughput: success={success[0]} fail={fail[0]} in {duration_secs}s | {rps:.2f} rps | conc={concurrency} | p50=n/a p95=n/a")


def main() -> None:
    p = argparse.ArgumentParser(description="Measure max transactions over a time window with fixed concurrency.")
    p.add_argument("--duration", type=int, default=60, help="Test duration in seconds (default: 60)")
    p.add_argument("--concurrency", type=int, default=4, choices=range(1, 7), metavar="[1-6]",
                   help="Concurrent workers (1-6, default: 4)")
    p.add_argument("--sample", type=str, default="mid.wav", help="Sample file name in samples/ (wav or npy)")
    p.add_argument("--seconds", type=int, default=8, help="Pad/truncate length in seconds (default: 8)")
    p.add_argument("--url", type=str, default="http://127.0.0.1:8000/raw", help="Target /raw URL")
    p.add_argument("--key", type=str, default="", help="Optional AUTH_KEY value")
    args = p.parse_args()

    asyncio.run(run(args.duration, args.concurrency, args.sample, args.seconds, args.url, args.key or None))


if __name__ == "__main__":
    main()


