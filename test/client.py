import argparse
import asyncio
import aiohttp
import io
import numpy as np
import os
from dotenv import load_dotenv
from .utils import load_audio_from_samples


load_dotenv(override=True)


def parse_args():
    p = argparse.ArgumentParser(description="HTTP client for Smart Turn server")
    p.add_argument("--sample", default="mid.wav", help="sample file under samples/ (.wav/.npy)")
    p.add_argument("--seconds", type=int, default=8)
    return p.parse_args()


def build_payload(args) -> bytes:
    _, body = load_audio_from_samples(args.sample, seconds_pad_to=args.seconds)
    return body


async def main():
    args = parse_args()
    host = os.getenv("RUNPOD_TCP_HOST", "localhost")
    port = int(os.getenv("RUNPOD_TCP_PORT", "8000"))
    auth = os.getenv("AUTH_KEY", os.getenv("API_KEY", "dev"))
    url = f"http://{host}:{port}/raw"
    headers = {"Content-Type": "application/octet-stream", "Authorization": f"Key {auth}"}

    body = build_payload(args)
    async with aiohttp.ClientSession() as s:
        async with s.post(url, data=body, headers=headers, timeout=30) as r:
            js = await r.json()
            print(js)


if __name__ == "__main__":
    asyncio.run(main())


