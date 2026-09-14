"""Step 10 — one arm of the A/B: repeated runs against a running gateway.

The strategy itself is fixed at gateway startup (KVROUTE_STRATEGY env var),
not by this script, since the router is chosen once at import time in
app.py. Run this once per arm:

    KVROUTE_STRATEGY=round_robin  uvicorn app:app --port 8000 &
    python bench/ab_run.py --strategy round_robin --repeats 3 --n 30
    # kill the gateway, restart with the other strategy
    KVROUTE_STRATEGY=prefix_aware uvicorn app:app --port 8000 &
    python bench/ab_run.py --strategy prefix_aware --repeats 3 --n 30
    python bench/ab_compare.py

Workload is shared-prefix heavy (one long system prompt reused across
requests) since that's the pattern prefix routing is meant to help.

Note: against tests/mock_upstream.py, which doesn't model prefix-cache
speedup at all, expect no real delta — that's Step 6's job (a real vLLM
with --enable-prefix-caching). This script is the harness; the effect it's
measuring only exists once there's a backend capable of producing it.
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


async def one_request(client: httpx.AsyncClient, question: str) -> float:
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
    ) as response:
        async for _ in response.aiter_bytes():
            return time.perf_counter() - started
    raise RuntimeError("no bytes received from gateway")


async def one_run(n: int) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        ttfts = [await one_request(client, f"question {i}") for i in range(n)]
    return {"mean": statistics.mean(ttfts), "stdev": statistics.stdev(ttfts) if n > 1 else 0.0, "n": n}


async def main(strategy: str, repeats: int, n: int, out: Path) -> None:
    runs = [await one_run(n) for _ in range(repeats)]
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
