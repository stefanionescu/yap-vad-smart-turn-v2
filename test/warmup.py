import asyncio
import aiohttp
import argparse
import os
from .utils import load_audio_from_samples


def parse_args():
    p = argparse.ArgumentParser(description="Warmup requests to Smart Turn server")
    p.add_argument("--auth", default=None, help="auth key (defaults to AUTH_KEY env var)")
    p.add_argument("--sample", default="mid.wav", help="sample file name under samples/")
    p.add_argument("--n", type=int, default=1, help="number of warmup requests (default 1)")
    p.add_argument("--seconds", type=int, default=8, help="pad/truncate to this many seconds")
    return p.parse_args()


async def main():
    args = parse_args()
    # Honor AUTH_KEY conditionally - only send if set
    auth_key = args.auth or os.environ.get("AUTH_KEY")
    headers = {"Content-Type": "application/octet-stream"}
    if auth_key:
        headers["Authorization"] = f"Key {auth_key}"
        
    url = "http://localhost:8000/raw"
    _, payload = load_audio_from_samples(args.sample, seconds_pad_to=args.seconds)
    async with aiohttp.ClientSession() as s:
        for i in range(args.n):
            async with s.post(url, data=payload, headers=headers, timeout=30) as r:
                js = await r.json()
                print(f"warmup[{i}] => {js}")


if __name__ == "__main__":
    asyncio.run(main())


