import asyncio
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from fastapi.responses import StreamingResponse

from metrics import backend_errors_total, cancelled_total, inflight, itl_seconds, requests_total, ttft_seconds
from router import Backend, RoundRobinRouter

STRATEGY = os.environ.get("KVROUTE_STRATEGY", "round_robin")

backends = [
    Backend(url="http://localhost:8001/v1/chat/completions"),
    Backend(url="http://localhost:8002/v1/chat/completions"),
]
router = RoundRobinRouter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient()
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    client: httpx.AsyncClient = request.app.state.client
    backend = router.pick(backends)

    # Open the upstream connection before returning StreamingResponse so a dead
    # backend surfaces as a real HTTP error instead of a 200 with a truncated body.
    req = client.build_request("POST", backend.url, json=payload)
    try:
        upstream = await client.send(req, stream=True)
    except httpx.HTTPError as exc:
        backend_errors_total.labels(backend=backend.name).inc()
        backend.record_failure()
        raise HTTPException(status_code=502, detail=f"backend {backend.name} unreachable") from exc

    backend.record_success()
    backend.inflight += 1
    inflight.labels(backend=backend.name).set(backend.inflight)
    requests_total.labels(backend=backend.name, strategy=STRATEGY).inc()
    started = time.perf_counter()

    async def stream():
        first_token = True
        last_chunk_at = started
        try:
            async for chunk in upstream.aiter_bytes():
                now = time.perf_counter()
                if first_token:
                    ttft_seconds.labels(backend=backend.name, strategy=STRATEGY, cache_state="miss").observe(
                        now - started
                    )
                    first_token = False
                else:
                    itl_seconds.labels(backend=backend.name, strategy=STRATEGY, cache_state="miss").observe(
                        now - last_chunk_at
                    )
                last_chunk_at = now
                yield chunk
        except asyncio.CancelledError:
            cancelled_total.labels(backend=backend.name).inc()
            raise
        finally:
            backend.inflight -= 1
            inflight.labels(backend=backend.name).set(backend.inflight)
            await upstream.aclose()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"x-kvroute-backend": backend.name},
    )
