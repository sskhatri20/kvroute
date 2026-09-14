import asyncio
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis

from admission import AdmissionController
from cache import ExactCache, normalize_key
from metrics import (
    admission_total,
    backend_errors_total,
    cache_lookups_total,
    cancelled_total,
    inflight,
    itl_seconds,
    prefix_routing_total,
    requests_total,
    ttft_seconds,
)
from router import Backend, PrefixAwareRouter, RoundRobinRouter
from semantic_cache import SemanticCache

STRATEGY = os.environ.get("KVROUTE_STRATEGY", "round_robin")  # round_robin | prefix_aware
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

backends = [
    Backend(url="http://localhost:8001/v1/chat/completions"),
    Backend(url="http://localhost:8002/v1/chat/completions"),
]
router = PrefixAwareRouter() if STRATEGY == "prefix_aware" else RoundRobinRouter()
redis = Redis.from_url(REDIS_URL)
exact_cache = ExactCache(redis)
semantic_cache = SemanticCache(redis)
admission = AdmissionController()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient()
    try:
        yield
    finally:
        await app.state.client.aclose()
        await redis.aclose()


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

    tenant = request.headers.get("x-tenant-id", "default")
    priority = request.headers.get("x-priority", "interactive")
    if priority not in ("interactive", "batch"):
        priority = "interactive"

    shed_reason = admission.admit(tenant, priority, payload)
    if shed_reason:
        admission_total.labels(priority=priority, outcome=shed_reason).inc()
        raise HTTPException(status_code=429, detail=shed_reason)
    admission_total.labels(priority=priority, outcome="admitted").inc()

    messages = payload.get("messages", [])
    exact_key = normalize_key(tenant, payload)
    # Semantic similarity is keyed on the non-system turns only. A shared
    # system prompt (exactly what prefix routing targets) would otherwise
    # dominate the embedding and make every request on that prefix look
    # like a semantic-cache hit regardless of what was actually asked.
    semantic_text = " ".join(str(m.get("content", "")) for m in messages if m.get("role") != "system")

    cached_chunks = await exact_cache.get(exact_key)
    if cached_chunks is not None:
        cache_lookups_total.labels(layer="exact", outcome="hit").inc()
        cache_state = "exact_hit"
    else:
        cache_lookups_total.labels(layer="exact", outcome="miss").inc()
        cached_chunks = await semantic_cache.get(tenant, semantic_text)
        if cached_chunks is not None:
            cache_lookups_total.labels(layer="semantic", outcome="hit").inc()
            cache_state = "semantic_hit"
        else:
            cache_lookups_total.labels(layer="semantic", outcome="miss").inc()
            cache_state = "miss"

    if cached_chunks is not None:
        started = time.perf_counter()

        async def replay():
            try:
                first = True
                last = started
                for chunk in cached_chunks:
                    now = time.perf_counter()
                    if first:
                        ttft_seconds.labels(backend="cache", strategy=STRATEGY, cache_state=cache_state).observe(
                            now - started
                        )
                        first = False
                    else:
                        itl_seconds.labels(backend="cache", strategy=STRATEGY, cache_state=cache_state).observe(
                            now - last
                        )
                    last = now
                    yield chunk
            finally:
                admission.release(priority)

        return StreamingResponse(
            replay(), media_type="text/event-stream", headers={"x-kvroute-cache": cache_state}
        )

    route_outcome = None
    if isinstance(router, PrefixAwareRouter):
        backend, route_outcome = router.pick(backends, messages)
        prefix_routing_total.labels(outcome=route_outcome).inc()
    else:
        backend = router.pick(backends)

    # Open the upstream connection before returning StreamingResponse so a dead
    # backend surfaces as a real HTTP error instead of a 200 with a truncated body.
    req = client.build_request("POST", backend.url, json=payload)
    try:
        upstream = await client.send(req, stream=True)
    except httpx.HTTPError as exc:
        admission.release(priority)
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
        collected: list[bytes] = []
        completed = False
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
                collected.append(chunk)
                yield chunk
            completed = True
        except asyncio.CancelledError:
            cancelled_total.labels(backend=backend.name).inc()
            raise
        finally:
            backend.inflight -= 1
            inflight.labels(backend=backend.name).set(backend.inflight)
            admission.release(priority)
            await upstream.aclose()
            if completed and collected:
                await exact_cache.put(exact_key, collected)
                await semantic_cache.put(tenant, semantic_text, collected)

    headers = {"x-kvroute-backend": backend.name}
    if route_outcome is not None:
        headers["x-kvroute-route"] = route_outcome
    return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)
