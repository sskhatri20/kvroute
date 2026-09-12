# LLM Inference Gateway — Project Spec

**Working name:** `shard` (rename later — avoid "llm-gateway", it's taken a hundred times over)

**One-liner:** A latency-aware serving layer between clients and vLLM replicas that reuses KV cache through prefix-aware routing, cuts redundant inference with a two-tier cache, and protects GPUs with token-budget admission control.

**Duration:** 6 weeks
**Primary deliverable:** a published benchmark writeup. The code is supporting evidence.

---

## 1. Why this project

The resume bullet you are buying is *not* "built an LLM gateway." It is:

> Reduced TTFT by X% under concurrent load by routing requests to replicas holding matching KV prefix blocks, measured across N requests on a reproducible harness.

Almost nobody building gateways does prefix-aware routing, because it requires understanding PagedAttention block management rather than just wiring an HTTP proxy. That is the differentiator. Everything else in this spec exists to make that number measurable and believable.

Secondary payoff: every metric here (p99, throughput, cache hit rate, queue depth) is in the same units as your Tata 1mg work, so the resume reads as one engineer rather than a backend person with an AI hobby.

---

## 2. Scope

### In scope

| Area | What's included |
|---|---|
| Protocol | OpenAI-compatible `/v1/chat/completions`, streaming and non-streaming |
| Transport | SSE token passthrough, backpressure, client-disconnect cancellation |
| Caching | Exact-match (hash) + semantic (vector similarity) in Redis |
| Routing | Health/latency-based baseline + prefix-cache-aware routing |
| Admission | Per-tenant token-bucket on **tokens**, priority queue, load shedding |
| Observability | Prometheus metrics, Grafana dashboard, structured logs |
| Backends | 2× vLLM replicas + 1 hosted API as fallback |
| Deploy | Docker Compose (required), EKS + Terraform (stretch) |
| Evidence | Reproducible load harness + published benchmark writeup |

### Out of scope — do not build these

- **Any UI beyond a 50-line HTML page** that proves streaming works. No React, no chat interface, no dashboard beyond Grafana.
- **More than 3 backends.** Two vLLM replicas plus one hosted fallback proves the abstraction. Six providers proves nothing extra.
- **Fine-tuning, training, or model work of any kind.** This is a serving project.
- **RAG / document ingestion.** Tempting, off-thesis, and it would eat the weeks the benchmark needs.
- **Auth beyond a static API key → tenant map.** Multi-tenancy here means isolation of *budgets and queues*, not identity management.
- **Custom inference kernels.** You are routing around vLLM, not into it.
- **Agent governance, policy engines, audit trails.** Different project. Noted here so it stays out.

### Kill criteria

If by end of week 4 prefix-aware routing is not showing a measurable TTFT delta, cut it, ship the caching + admission control benchmark instead, and say so in the writeup. A negative result honestly reported still beats an unfinished repo. Do not let this slip into week 6.

---

## 3. Architecture

```
                    ┌──────────────┐
   client ──SSE──▶  │   shard      │
                    │              │
                    │  1. admit    │──▶ token bucket / priority queue
                    │  2. cache    │──▶ Redis (exact hash → vector sim)
                    │  3. route    │──▶ prefix index + health/EWMA scoring
                    │  4. stream   │──▶ SSE passthrough + cancellation
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          vLLM-A       vLLM-B      hosted API
        (prefix        (prefix      (fallback)
         caching on)    caching on)
```

**Stack:** Python 3.11+, FastAPI + Uvicorn, `httpx` (async, streaming) for upstream calls, Redis 7 for cache and prefix index, `sentence-transformers` for embeddings, Prometheus + Grafana.

**FastAPI notes that will bite you:**
- Stream with `StreamingResponse` over an async generator. Detect client disconnect via `await request.is_disconnected()` inside the generator loop, or catch `asyncio.CancelledError` — either way, the `finally` block must close the upstream `httpx` stream, or you leak GPU work (see D1).
- Use one long-lived `httpx.AsyncClient` created in the lifespan handler with tuned connection limits. Creating a client per request will dominate your latency numbers and quietly invalidate the benchmark.
- Keep the embedding call for the semantic cache off the event loop (`run_in_threadpool` or a separate worker), otherwise it blocks every concurrent stream and your p99 will lie to you.
- Run a single Uvicorn worker for benchmarking. Multiple workers means multiple in-process prefix indexes and non-reproducible routing.

**Hardware:** one rented GPU (L4 or A10, RunPod / Vast.ai) running two vLLM instances on a small model (Qwen2.5-1.5B or similar) with `--enable-prefix-caching`. Budget roughly $50–100 total for benchmark runs. You do not need a big model; you need *measurable cache behavior*.

---

## 4. Deliverables

Each has an acceptance test. Do not mark one done without it.

### D1 — Streaming proxy with correct cancellation
OpenAI-compatible endpoint that streams tokens through to the client.

**Accept when:** killing the client mid-stream causes the upstream vLLM request to abort within 100ms, verified by watching vLLM's running-request count drop. A proxy that leaks GPU work on disconnect is the single most common bug in this category — getting it right is an interview story.

### D2 — Metrics and instrumentation
TTFT, inter-token latency, end-to-end latency, tokens/sec, queue depth, cache hit/miss, per-tenant token spend. Prometheus endpoint + Grafana dashboard.

**Accept when:** the dashboard renders live during a load run and you can read p50/p95/p99 TTFT off it.

**Build this before any optimization.** Without a baseline captured in week 2, every later number is unfalsifiable.

### D3 — Two-tier cache
Exact-match hash lookup first, semantic vector similarity second, both in Redis, namespaced per tenant, configurable similarity threshold and TTL.

**Accept when:** you can produce a threshold-vs-hit-rate curve *and* a false-hit rate measured against a labelled set of ~100 query pairs you write yourself. The false-hit number is what separates this from every other semantic cache repo — most publish hit rate and quietly ignore correctness.

### D4 — Admission control
Per-tenant token-bucket limiting on tokens (not requests), two-level priority queue so interactive preempts batch, 429 shedding before the GPU saturates.

**Accept when:** under 3× overload, p99 for the high-priority tenant stays within 20% of its unloaded value while low-priority sheds.

### D5 — Prefix-aware routing ★ centerpiece
Maintain an index of which replica recently served which system-prompt prefix; route new requests sharing that prefix to the same replica so vLLM's prefix cache hits instead of recomputing.

**Accept when:** an A/B run (round-robin vs prefix-aware) over the same workload shows a TTFT delta with non-overlapping confidence intervals across ≥3 repeated runs.

Design notes to work through: prefix granularity (full system prompt vs first N tokens), index eviction policy, and the load-imbalance tradeoff — naive prefix affinity will overload one replica, so you need a fallback when the preferred replica's queue exceeds a threshold. That tension *is* the interesting part of the writeup.

### D6 — Load harness + benchmark writeup
Reproducible harness generating realistic workload mixes (shared-prefix heavy, unique-prompt heavy, mixed), plus a written report.

**Accept when:** someone else can `docker compose up` and reproduce your headline numbers within 10%.

The writeup must include: methodology, hardware, the workload generator, results with variance, **and what didn't work**. The failure section is the credibility signal.

### D7 — Deploy (stretch)
EKS via Terraform, HPA scaling on queue depth rather than CPU.

**Accept when:** HPA scales on a custom metric under synthetic load. Cut this if week 6 is tight — it's a resume line, not a differentiator.

---

## 5. Milestones

| Week | Focus | Exit condition |
|---|---|---|
| 1 | D1 — streaming proxy, cancellation | Tokens stream; disconnect aborts upstream |
| 2 | D2 — metrics, Grafana, **baseline captured** | Baseline numbers committed to the repo |
| 3 | D3 — two-tier cache | Threshold curve + false-hit rate measured |
| 4 | D5 begins — prefix index and routing | First A/B signal, or invoke kill criteria |
| 5 | D5 finishes + D4 admission control | A/B result stable across repeated runs |
| 6 | D6 — full benchmark, writeup, publish | Post live on r/LocalLLaMA + HN |

If a week slips, cut D4 before cutting D5 or D6.

---

## 6. Benchmark plan

This is the actual product. Design it in week 1, not week 6.

**Workload mixes:**
1. *Shared-prefix heavy* — many requests, one long system prompt (the RAG / agent-tooling pattern). This is where prefix routing should shine.
2. *Unique-prompt heavy* — little overlap. Establishes that routing doesn't hurt when there's nothing to reuse.
3. *Mixed, tenant-tiered* — two tenants at different priorities, one bursting. Exercises admission control.

**Conditions to compare:** no cache + round-robin (baseline) → cache only → cache + prefix routing → full stack with admission control.

**For every run report:** p50/p95/p99 TTFT, inter-token latency, throughput at saturation, GPU utilization, cache hit rate, false-hit rate, cost per 1k requests.

**Rules:** ≥3 runs per condition, report variance, discard the first run as warmup, fix the random seed on the workload generator, commit the raw data.

---

## 7. Resume bullets (fill with real numbers)

Draft these now so you know which numbers to chase. Template only — do not ship a number you haven't measured.

- Cut TTFT __% under concurrent load by routing requests to vLLM replicas holding matching KV-cache prefix blocks, validated over __ requests across three workload profiles
- Reduced inference cost __% via two-tier exact + semantic caching (__% hit rate at __ similarity threshold, __% false-hit rate on a labelled eval set)
- Sustained __ req/s with graceful degradation using per-tenant token-budget admission control and priority queueing; high-priority p99 held within __% under 3× overload
- Built OpenAI-compatible streaming proxy (FastAPI/httpx) with upstream cancellation on client disconnect, eliminating orphaned GPU work
- Instrumented with Prometheus/Grafana (TTFT, ITL, queue depth, per-tenant spend); published reproducible benchmark harness

---

## 8. Interview talking points to prepare

- Why prefix affinity conflicts with load balancing, and how you resolved it
- How you picked the semantic cache threshold, and what a false hit costs a user
- Why token-budget limiting beats request-rate limiting for LLM traffic
- What you'd change to make this multi-tenant in a real org (isolation, noisy neighbours, fairness)
- Where the design breaks at 10× scale

---

## 9. Distribution

The repo alone gets ignored. Plan for:

- README leading with the headline result and a chart, not an install section
- Writeup as a standalone post (title should name the finding, e.g. "What prefix-aware routing actually saves on vLLM")
- Posted to r/LocalLLaMA and Show HN in week 6
- Link in your LinkedIn headline and attached to cold applications

---

## 10. Open decisions

- [ ] GPU provider and model choice (affects how visible prefix-cache effects are — smaller model, tighter margins)
- [ ] Embedding model for semantic cache (local MiniLM vs hosted — latency of the cache lookup itself matters and belongs in the benchmark)
- [ ] Prefix index granularity
- [ ] Whether D7 (EKS) is in or out — decide at end of week 5