"""Interleaved A/B: alternates round_robin/prefix_aware per request in one
continuous sequence, instead of two separate timed runs.

bench/ab_run.py's two arms are each internally consistent but can drift
against each other if GPU/cache state changes between when arm 1 finishes
and arm 2 starts minutes later — a run-order confound, not a strategy
effect. Interleaving controls for that: both strategies see the same
backend state at (approximately) the same wall-clock time throughout.

Order is randomized per pair (not always round_robin-then-prefix_aware) so
a systematic first-vs-second-in-pair bias can't leak in either.

Usage:
    uvicorn app:app --port 8000 &
    python bench/ab_interleaved.py --pairs 30
"""
import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx

GATEWAY = "http://localhost:8000/v1/chat/completions"
SHARED_SYSTEM_PROMPT = "You are a helpful assistant. " * 40
STRATEGIES = ["round_robin", "prefix_aware"]


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


def ci95(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    margin = 1.96 * stdev / len(values) ** 0.5
    return mean - margin, mean + margin


async def main(pairs: int, out: Path) -> None:
    ttfts: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    order_log = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(pairs):
            order = STRATEGIES[:]
            random.shuffle(order)
            order_log.append(order)
            for strategy in order:
                ttft = await one_request(client, f"question {i}-{strategy}", strategy)
                ttfts[strategy].append(ttft)

    summary = {}
    for strategy in STRATEGIES:
        lo, hi = ci95(ttfts[strategy])
        summary[strategy] = {
            "mean": statistics.mean(ttfts[strategy]),
            "stdev": statistics.stdev(ttfts[strategy]) if len(ttfts[strategy]) > 1 else 0.0,
            "n": len(ttfts[strategy]),
            "ci95": [lo, hi],
        }

    lo_rr, hi_rr = summary["round_robin"]["ci95"]
    lo_pa, hi_pa = summary["prefix_aware"]["ci95"]
    overlap = lo_rr <= hi_pa and lo_pa <= hi_rr

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "overlap": overlap, "raw": ttfts, "order_log": order_log}, indent=2)
    )
    print(json.dumps(summary, indent=2))
    print(f"\nCIs overlap: {overlap}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=30, help="one request per strategy per pair")
    args = parser.parse_args()
    out_path = Path("bench/results/ab-interleaved.json")
    asyncio.run(main(args.pairs, out_path))
