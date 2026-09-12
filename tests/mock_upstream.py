"""Instrumented stand-in for a vLLM backend.

Streams a fixed number of SSE-style frames slowly enough that a client abort
lands mid-stream, and counts started/cancelled/completed so tests/run_smoke.py
can assert on cancellation propagation instead of eyeballing logs.
"""
import asyncio

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

counters = {"started": 0, "cancelled": 0, "completed": 0}

TOKEN_COUNT = 20
TOKEN_INTERVAL_SECONDS = 0.2


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/counters")
async def get_counters():
    return counters


@app.post("/v1/chat/completions")
async def chat_completions():
    counters["started"] += 1

    async def stream():
        try:
            for i in range(TOKEN_COUNT):
                yield f'data: {{"choices":[{{"delta":{{"content":"tok{i} "}}}}]}}\n\n'.encode()
                await asyncio.sleep(TOKEN_INTERVAL_SECONDS)
            yield b"data: [DONE]\n\n"
            counters["completed"] += 1
        except asyncio.CancelledError:
            counters["cancelled"] += 1
            raise

    return StreamingResponse(stream(), media_type="text/event-stream")
