import argparse, io, os, sys, asyncio, numpy as np
import aiohttp

# Allow importing utils from same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_audio_from_samples

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/raw")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--key", default="")
    ap.add_argument("--timeout", type=float, default=10)  # allow first-hit compile+capture
    args = ap.parse_args()

    # Use utils.py for consistent audio loading
    try:
        arr, body = load_audio_from_samples(args.sample, seconds_pad_to=int(args.seconds))
    except FileNotFoundError:
        print(f"Error: Sample '{args.sample}' not found in samples/ directory")
        return

    headers = {"Content-Type": "application/octet-stream"}
    if args.key:
        headers["Authorization"] = f"Key {args.key}"

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.post(args.url, data=body, headers=headers) as r:
            r.raise_for_status()
            print(await r.json())

if __name__ == "__main__":
    asyncio.run(main())