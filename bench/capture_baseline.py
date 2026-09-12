"""Fixed workload against a running gateway; dumps TTFT quantiles to bench/results/.

Client-side TTFT (time to first byte) is measured independently of the
gateway's own histogram so the two can be cross-checked against each other.

Usage:
    python bench/capture_baseline.py --n 50 --out bench/results/baseline-mock.json
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

GATEWAY = "http://localhost:8000/v1/chat/completions"


async def one_request(client: httpx.AsyncClient) -> float:
    started = time.perf_counter()
    async with client.stream(
        "POST",
        GATEWAY,
        json={"stream": True, "messages": [{"role": "user", "content": "hi"}]},
    ) as response:
        async for _ in response.aiter_bytes():
            return time.perf_counter() - started
    raise RuntimeError("no bytes received from gateway")


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(q * len(ordered)))
    return ordered[index]


async def main(n: int, out: Path) -> None:
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Sequential, not concurrent: this is a baseline of single-request
        # latency, not a load test (that's Step 12's job).
        ttfts = [await one_request(client) for _ in range(n)]

    result = {
        "n": n,
        "ttft_seconds": {
            "p50": quantile(ttfts, 0.50),
            "p95": quantile(ttfts, 0.95),
            "p99": quantile(ttfts, 0.99),
            "mean": statistics.mean(ttfts),
            "raw": ttfts,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["ttft_seconds"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("bench/results/baseline-mock.json"))
    args = parser.parse_args()
    asyncio.run(main(args.n, args.out))
