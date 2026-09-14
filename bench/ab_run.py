"""One arm of the A/B: repeated runs against a running gateway.

Sends x-kvroute-strategy on every request, so the gateway does not need to
be restarted between arms (app.py keeps both routers live and picks per
request). Run both arms back to back against the same running gateway:

    uvicorn app:app --port 8000 &
    python bench/ab_run.py --strategy round_robin --repeats 3 --n 30
    python bench/ab_run.py --strategy prefix_aware --repeats 3 --n 30
    python bench/ab_compare.py

Two runs still measured minutes apart like this can drift with whatever
changed on the backend in between (GPU/cache state) — see
bench/ab_interleaved.py for a run that alternates strategy per request
within one continuous sequence, which controls for that.
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

GATEWAY = "http://localhost:8000/v1/chat/completions"
SHARED_SYSTEM_PROMPT = "You are a helpful assistant. " * 40


async def one_request(client: httpx.AsyncClient, question: str, strategy: str) -> float:
    started = time.perf_counter()
    async with client.stream(
        "POST",
        GATEWAY,
        json={
            "stream": True,
            "messages": [
                {"role": "system", "content": SHARED_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        },
        headers={"x-kvroute-strategy": strategy},
    ) as response:
        async for _ in response.aiter_bytes():
            return time.perf_counter() - started
    raise RuntimeError("no bytes received from gateway")


async def one_run(n: int, strategy: str) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        ttfts = [await one_request(client, f"question {i}", strategy) for i in range(n)]
    return {"mean": statistics.mean(ttfts), "stdev": statistics.stdev(ttfts) if n > 1 else 0.0, "n": n}


async def main(strategy: str, repeats: int, n: int, out: Path) -> None:
    runs = [await one_run(n, strategy) for _ in range(repeats)]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"strategy": strategy, "runs": runs}, indent=2))
    print(json.dumps(runs, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, choices=["round_robin", "prefix_aware"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()
    out_path = Path(f"bench/results/ab-{args.strategy}.json")
    asyncio.run(main(args.strategy, args.repeats, args.n, out_path))
