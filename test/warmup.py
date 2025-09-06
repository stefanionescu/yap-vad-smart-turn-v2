import argparse, io, asyncio, numpy as np
import aiohttp, torch, torchaudio

SR = 16000

def load_first_seconds(path: str, seconds: float) -> np.ndarray:
    wav, sr = torchaudio.load(path)          # (C, T), float32/float64
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # mono
    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)
    T = int(SR * seconds)
    wav = wav[:, :T]
    if wav.shape[1] < T:
        pad = torch.zeros(1, T - wav.shape[1], dtype=wav.dtype)
        wav = torch.cat([wav, pad], dim=1)
    return wav.squeeze(0).to(torch.float32).numpy()

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/raw")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--key", default="")
    ap.add_argument("--timeout", type=float, default=180)  # allow first-hit compile+capture
    args = ap.parse_args()

    arr = load_first_seconds(args.sample, args.seconds)
    buf = io.BytesIO(); np.save(buf, arr)

    headers = {"Content-Type": "application/octet-stream"}
    if args.key:
        headers["Authorization"] = f"Key {args.key}"

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.post(args.url, data=buf.getvalue(), headers=headers) as r:
            r.raise_for_status()
            print(await r.json())

if __name__ == "__main__":
    asyncio.run(main())