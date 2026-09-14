"""Step 12 — open-loop load harness with three workload profiles.

Open-loop, not closed-loop: requests are sent at a fixed rate regardless of
when earlier requests finish. A closed-loop generator (wait for a response
before sending the next) caps offered load at the gateway's own latency and
hides saturation, which is the naive-and-wrong way to load-test a server.

Usage:
    python bench/load_harness.py --profile shared_prefix --rate 5 --duration 10
    python bench/load_harness.py --profile unique_prompt --rate 5 --duration 10
    python bench/load_harness.py --profile mixed_tenant --rate 5 --duration 10
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

random.seed(0)  # fixed seed: reproducible workload across runs


def build_request(profile: str, i: int) -> tuple[dict, dict]:
    """Returns (json_payload, extra_headers)."""
    if profile == "shared_prefix":
        messages = [
            {"role": "system", "content": SHARED_SYSTEM_PROMPT},
            {"role": "user", "content": f"question {i}"},
        ]
        return {"stream": True, "messages": messages}, {}

    if profile == "unique_prompt":
        messages = [{"role": "user", "content": f"unique prompt {i} {random.random()}"}]
        return {"stream": True, "messages": messages}, {}

    if profile == "mixed_tenant":
        tenant = "tenant-a" if i % 5 == 0 else "tenant-b"
        priority = "interactive" if tenant == "tenant-b" else "batch"
        messages = [{"role": "user", "content": f"mixed prompt {i}"}]
        return {"stream": True, "messages": messages}, {"x-tenant-id": tenant, "x-priority": priority}

    raise ValueError(f"unknown profile: {profile}")


async def send_one(client: httpx.AsyncClient, payload: dict, headers: dict, results: list) -> None:
    started = time.perf_counter()
    try:
        async with client.stream("POST", GATEWAY, json=payload, headers=headers) as response:
            if response.status_code == 429:
                results.append({"shed": True})
                return
            first = True
            last = started
            ttft = None
            itls = []
            async for _ in response.aiter_bytes():
                now = time.perf_counter()
                if first:
                    ttft = now - started
                    first = False
                else:
                    itls.append(now - last)
                last = now
            if ttft is None:
                results.append({"shed": True})
                return
            results.append({"shed": False, "ttft": ttft, "itl_mean": statistics.mean(itls) if itls else None})
    except httpx.HTTPError:
        results.append({"shed": True})


async def main(profile: str, rate: float, duration: float, out: Path) -> None:
    interval = 1.0 / rate
    results: list = []
    tasks = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        start = time.perf_counter()
        i = 0
        while time.perf_counter() - start < duration:
            payload, headers = build_request(profile, i)
            tasks.append(asyncio.create_task(send_one(client, payload, headers, results)))
            i += 1
            await asyncio.sleep(interval)  # fixed send rate, independent of response time
        await asyncio.gather(*tasks)

    admitted = [r for r in results if not r["shed"]]
    shed = [r for r in results if r["shed"]]
    ttfts = [r["ttft"] for r in admitted]

    summary = {
        "profile": profile,
        "rate": rate,
        "duration": duration,
        "sent": len(results),
        "admitted": len(admitted),
        "shed": len(shed),
        "ttft_p50": statistics.median(ttfts) if ttfts else None,
        "ttft_p95": sorted(ttfts)[int(0.95 * len(ttfts))] if ttfts else None,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "raw": results}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=["shared_prefix", "unique_prompt", "mixed_tenant"])
    parser.add_argument("--rate", type=float, default=5.0, help="requests per second")
    parser.add_argument("--duration", type=float, default=10.0, help="seconds")
    args = parser.parse_args()
    out_path = Path(f"bench/results/load-{args.profile}.json")
    asyncio.run(main(args.profile, args.rate, args.duration, out_path))
