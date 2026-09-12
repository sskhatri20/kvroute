"""Step 2 acceptance check: client disconnect must cancel the upstream generation.

Usage:
    python -m uvicorn tests.mock_upstream:app --port 8001 &
    python -m uvicorn tests.mock_upstream:app --port 8002 &
    uvicorn app:app --port 8000 &
    python tests/run_smoke.py
"""
import asyncio
import sys

import httpx

GATEWAY = "http://localhost:8000/v1/chat/completions"
MOCK_COUNTER_URLS = ["http://localhost:8001/counters", "http://localhost:8002/counters"]


async def total_counters() -> dict[str, int]:
    totals = {"started": 0, "cancelled": 0, "completed": 0}
    async with httpx.AsyncClient() as client:
        for url in MOCK_COUNTER_URLS:
            data = (await client.get(url)).json()
            for key in totals:
                totals[key] += data[key]
    return totals


async def send_and_abandon() -> None:
    async with httpx.AsyncClient(timeout=5.0) as client:
        async with client.stream(
            "POST",
            GATEWAY,
            json={"stream": True, "messages": [{"role": "user", "content": "hi"}]},
        ) as response:
            async for _ in response.aiter_bytes():
                break  # one chunk in hand, now abandon the stream mid-flight


async def main() -> None:
    before = await total_counters()
    await send_and_abandon()
    await asyncio.sleep(1.0)
    after = await total_counters()

    delta = {key: after[key] - before[key] for key in before}
    print(f"started: {delta['started']}, cancelled: {delta['cancelled']}, completed: {delta['completed']}")

    if delta != {"started": 1, "cancelled": 1, "completed": 0}:
        print("FAIL: expected started=1 cancelled=1 completed=0")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    asyncio.run(main())
