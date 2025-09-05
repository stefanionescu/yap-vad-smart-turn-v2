import argparse
import asyncio
import aiohttp
import io
import numpy as np
import os
import time
from .utils import load_audio_from_samples


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Smart Turn server")
    p.add_argument("--auth", default=os.environ.get("AUTH_KEY", "dev"))
    p.add_argument("--sample", default="mid.wav", help="sample file name under samples/ (.wav/.npy)")
    p.add_argument("--seconds", type=int, default=8)
    p.add_argument("--requests", type=int, default=128, help="total number of requests")
    p.add_argument("--concurrency", type=int, default=32, help="concurrent in-flight requests")
    return p.parse_args()


def build_payload(args) -> bytes:
    # Supports .wav and .npy via utils
    _, body = load_audio_from_samples(args.sample, seconds_pad_to=args.seconds)
    return body


async def one(sess, url: str, headers: dict, body: bytes):
    t0 = time.perf_counter()
    async with sess.post(url, data=body, headers=headers, timeout=60) as r:
        js = await r.json()
    return (time.perf_counter() - t0) * 1000.0, js


async def worker(num: int, url: str, headers: dict, body: bytes):
    results = []
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        for _ in range(num):
            results.append(await one(s, url, headers, body))
    return results


async def main_async(args):
    # Build URL from .env style RUNPOD host/port
    host = os.getenv("RUNPOD_TCP_HOST", "localhost")
    port = int(os.getenv("RUNPOD_TCP_PORT", "8000"))
    url = f"http://{host}:{port}/raw"
    headers = {"Content-Type": "application/octet-stream", "Authorization": f"Key {args.auth}"}
    body = build_payload(args)

    # Split total requests across workers (concurrency)
    workers = max(1, min(args.concurrency, args.requests))
    base = args.requests // workers
    rem = args.requests % workers
    counts = [base + (1 if i < rem else 0) for i in range(workers)]

    tasks = [asyncio.create_task(worker(counts[i], url, headers, body)) for i in range(workers)]
    nested = await asyncio.gather(*tasks)
    all_results = [item for sub in nested for item in sub]

    lat = [x[0] for x in all_results]
    lat.sort()
    n = len(lat)
    p50 = lat[int(0.5 * n)] if n else 0.0
    p95 = lat[max(int(0.95 * n) - 1, 0)] if n else 0.0
    print(f"host={host}:{port} total={args.requests} conc={args.concurrency} p50={p50:.1f}ms p95={p95:.1f}ms")
    if all_results:
        print(all_results[0][1])


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()


