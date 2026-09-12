# kvroute — Step-by-Step Build Guide

You write the implementation. Claude Code reviews and verifies. That division is
the point: the value of this project is the interview conversation, and you can
only have that conversation about code you wrote.

---

## How to use this with Claude Code

Drop this file in the repo root and add the block below as `CLAUDE.md`. It sets
the ground rules so the agent doesn't helpfully do your project for you.

```markdown
# Working agreement

This is a learning project. The human writes all implementation code.

## You MUST NOT
- Write or rewrite implementation files in `src/kvroute/`
- Paste "here's how it should look" code blocks for the current step
- Implement a fix yourself when a test fails
- Work ahead to a later step

## You SHOULD
- Run the acceptance checks for the current step and report results
- Point at what is wrong and why, in prose, without writing the fix
- Answer conceptual questions in full — explanation is not implementation
- Push back when a design choice will contaminate the benchmark
- Write throwaway diagnostic scripts in `/tmp` when investigating

## Exception
If the human explicitly says "just write it" or "show me the code", do it.
Their call, not yours.
```

Then work step by step:

```
> I'm on Step 3. Here's what I wrote. Review it and run the acceptance check.
```

---

## The two tracks

Concepts and code interleave. Don't batch the reading — each concept lands right
before the step that forces you to use it, which is why it will stick.

| Step | Build | Concept to learn first |
|---|---|---|
| 1 | Streaming passthrough | SSE frame format |
| 2 | Cancellation | asyncio cancellation semantics |
| 3 | TTFT instrumentation | **Prefill vs decode** |
| 4 | Baseline capture | Histograms and quantile error |
| 5 | Multi-backend routing | Continuous batching |
| 6 | Real vLLM baseline | **KV cache memory math** |
| 7 | Exact-match cache | — |
| 8 | Semantic cache | Embeddings, ANN, cosine similarity |
| 9 | Prefix-aware routing | **PagedAttention, automatic prefix caching** |
| 10 | Imbalance handling + A/B | Scheduling / queueing theory basics |
| 11 | Admission control | Token budgets vs request rates |
| 12 | Load harness | Benchmark methodology, variance |
| 13 | Writeup | — |

The three bold ones are the concepts you will be asked about in interviews.
Don't skim those.

---

# Step 1 — Streaming passthrough

**Goal:** proxy a streaming completion from one hardcoded upstream to the client.
One backend. No routing, no cache, no metrics.

## Learn first: SSE frame format

Server-Sent Events is a text protocol. A frame is `data: <payload>\n\n`. The
double newline is the delimiter. OpenAI-compatible servers send one JSON object
per frame and terminate with `data: [DONE]`.

The thing to internalise: **TCP gives you a byte stream, not a frame stream.**
One `aiter_bytes()` chunk may contain half a frame, or three frames. For pure
passthrough that doesn't matter. From Step 3 onward it does.

## Build

- FastAPI app, `POST /v1/chat/completions` and `GET /healthz`
- One `httpx.AsyncClient` created in the lifespan handler
- `StreamingResponse` over an async generator forwarding upstream bytes
- Media type `text/event-stream`

## Questions to answer before writing

1. Why does `client.stream()` need `async with` rather than a plain `await`?
   What breaks if you build the generator and return it without holding the
   context open?
2. If you create the `AsyncClient` per request instead of in lifespan, what
   specifically gets slower and roughly by how much?
3. Will you forward opaque bytes or parse frames? Decide now — Step 3 needs one
   of these and switching later is annoying.

## Accept when

```bash
python -m uvicorn tests.mock_upstream:app --port 8001 &
uvicorn kvroute.app:app --port 8000 &

curl -N -X POST localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"stream":true,"messages":[{"role":"user","content":"hi"}]}'
```

Tokens arrive **incrementally**. If they land all at once, something is
buffering. Finding what is part of the exercise.

**Ask Claude Code:** "Verify tokens arrive incrementally and that only one
AsyncClient is constructed for the process lifetime."

Budget: one evening, ~60 lines.

---

# Step 2 — Cancellation

**Goal:** when the client disconnects mid-stream, the upstream generation stops.

This is the deliverable that separates your gateway from a tutorial. A proxy that
keeps generating into a closed socket burns GPU time nobody reads.

## Learn first: asyncio cancellation

- `CancelledError` is raised *at the await point*, not returned
- Starlette cancels the task driving the response generator on client disconnect
- `async with` `__aexit__` runs during unwinding — this is what closes the
  upstream connection
- Catching bare `Exception` swallows `CancelledError` in Python 3.7 and earlier;
  since 3.8 it inherits from `BaseException`, but `except Exception` in a
  *library you call* can still break the chain

## Build

**First, break it deliberately.** Wrap your yield in `try/except Exception` and
watch the mock upstream report `completed` instead of `cancelled` for an
abandoned request. Seeing the failure is worth more than avoiding it.

Then fix it:
- catch `CancelledError` explicitly, record it, **re-raise**
- `finally` block for cleanup that must run either way
- never catch bare `Exception` around the yield

## Questions to answer

1. Trace the exact chain: client TCP close → ? → ? → vLLM frees the sequence.
   Name every hop.
2. If `finally` contains `await something_slow()`, what happens during
   cancellation?
3. Your mock upstream counts `cancelled`. What's the equivalent signal on a real
   vLLM, and how would you observe it in production?

## Accept when

```bash
python tests/run_smoke.py
```

Reports `started: 1, cancelled: 1, completed: 0` for the abandoned request.

**Ask Claude Code:** "Grep for any bare `except Exception` in the streaming path
and tell me if any of them can swallow cancellation."

---

# Step 3 — TTFT instrumentation

**Goal:** measure time-to-first-token and inter-token latency, exposed as
Prometheus metrics.

## Learn first: prefill vs decode ★

The concept your entire project rests on.

**Prefill** processes the whole prompt at once. Every token attends to every
prior token in parallel — one big matrix multiply. It is **compute-bound**.
Cost scales with prompt length.

**Decode** generates one token per forward pass. Tiny matmuls, but the entire KV
cache must be read from HBM every step. It is **memory-bandwidth-bound**. Cost
scales with the number of output tokens and is nearly independent of prompt
length.

They share a GPU but are different workloads. Which gives you:

- **TTFT is a prefill number.** Prompt length dominates it.
- **ITL is a decode number.** Prompt length barely touches it.
- **Prefix caching skips prefill work only.**

So your project moves TTFT and should leave ITL roughly flat. If your benchmark
shows ITL improving from prefix routing, you have a measurement bug — that
falsifiable prediction is the strongest thing in your writeup.

Work through: for a 2000-token prompt generating 100 tokens, roughly what
fraction of total latency is prefill? Redo it for a 200-token prompt. The ratio
flip is the intuition.

## Build

- `prometheus_client` histograms for TTFT and ITL
- `GET /metrics`
- TTFT = accept time → first byte reaching the client
- ITL = gap between consecutive chunks
- Labels: backend, strategy, cache state

## Questions to answer

1. Default Prometheus buckets top out around 10s and are sparse below 100ms.
   Why is that wrong here, and what buckets do you want?
2. You measure TTFT at the gateway. How much of what you measure is *your*
   overhead vs upstream prefill? How would you separate them?
3. If a chunk contains two SSE frames, is that one ITL sample or two? What does
   your answer do to the numbers?

## Accept when

`/metrics` exposes TTFT and ITL histograms with populated buckets, and a
hand-computed TTFT from a single curl matches the histogram within 10ms.

---

# Step 4 — Baseline capture

**Goal:** a committed baseline you can compare everything against.

**Do not skip this and do not do it later.** Every number in your writeup is
unfalsifiable without a baseline captured before any optimization exists.

## Learn first: histogram quantile error

Prometheus quantiles are interpolated from buckets, so p99 accuracy is bounded by
bucket width in that region. If your p99 lands in a bucket spanning 1s to 2s,
your p99 has 1s of slop. Check where yours falls and add buckets if needed.

## Build

- `docker-compose.yml` with Prometheus (scrape interval 1s, not 15s — TTFT
  effects are sub-second) and Grafana
- Dashboard: TTFT p50/p95/p99, ITL p50, throughput, inflight per backend
- A script that runs a fixed workload and dumps results to `bench/results/`
- Commit the baseline JSON

## Accept when

Dashboard renders live during a run, and `bench/results/baseline-mock.json`
exists in git.

---

# Step 5 — Multi-backend routing

**Goal:** two backends, round-robin, health tracking, inflight depth per backend.

## Learn first: continuous batching

vLLM's scheduler admits requests per forward-pass step rather than per batch. A
finished sequence leaves the batch and a queued one joins immediately, so batch
composition changes every step.

Why you care: **you are queueing in front of a system that already queues.** If
your gateway holds requests back, vLLM's batch runs smaller than it could and you
lose throughput. If you flood it, its queue grows and TTFT rises. Your admission
control in Step 11 has to respect that, and "where should the queue live" is a
great interview answer.

## Build

- A `Router` abstract base with `pick(messages, depths) -> Backend`
- `RoundRobinRouter` as the control arm
- Inflight depth tracked per backend, incremented and decremented around the
  stream
- Health tracking: mark a backend unhealthy after N consecutive failures, retry
  after a cooldown
- Response header naming the chosen backend (makes debugging and testing trivial)

**Design constraint that matters:** both strategies must go through identical
code paths apart from the routing decision. If the arms differ anywhere else,
your A/B is contaminated and you won't know it.

## Questions to answer

1. Is inflight depth the right load signal, or should it be queued tokens? What
   would change?
2. A backend fails mid-stream, after the client already received tokens. Can you
   retry? What are the options?

## Accept when

Requests alternate across backends, killing one backend routes everything to the
other within the cooldown window, and the header reports the choice.

---

# Step 6 — Real vLLM baseline

**Goal:** get off the mock and capture a real baseline.

## Learn first: KV cache memory math ★

Per token, KV cache bytes ≈

```
2 (K and V) × layers × kv_heads × head_dim × dtype_bytes
```

Work it out for the model you're serving. Then compute how many tokens fit in
your GPU's spare VRAM after weights. That number is your real concurrency
ceiling, and knowing it is what separates people who've served models from people
who've called APIs.

Also learn why GQA/MQA exist: shrinking `kv_heads` shrinks the cache
proportionally, which is a serving decision as much as a modelling one.

## Build

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8001 --enable-prefix-caching
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8002 --enable-prefix-caching
```

`--enable-prefix-caching` is mandatory. Without it there is nothing to route
toward.

Rent one L4 or A10 (RunPod, Vast.ai). Budget $50–100 total for the project. You
don't need a big model, you need measurable cache behavior.

## Accept when

`tests/run_smoke.py` passes against real vLLM, and
`bench/results/baseline-real.json` is committed.

**Watch for:** cold-start effects on the first run. Discard it as warmup, and
say so in the methodology.

---

# Step 7 — Exact-match cache

**Goal:** identical prompts skip the model entirely.

## Build

- Hash the full normalized request (messages, model, temperature, max_tokens)
- Redis, per-tenant key namespace, configurable TTL
- Cache hits must still *stream* to the client, or your TTFT distribution gets a
  bimodal artifact that misrepresents what a user experiences

## Questions to answer

1. Does `temperature` belong in the cache key? Argue both sides.
2. On a hit, do you replay tokens at the original pacing or dump them instantly?
   What does each choice do to your TTFT and ITL numbers?

## Accept when

Second identical request returns with TTFT under 10ms and the hit rate metric
increments.

---

# Step 8 — Semantic cache

**Goal:** near-duplicate prompts hit cache, and you can state the false-hit rate.

## Learn first: embeddings and ANN

- Embeddings map text to vectors; cosine similarity measures direction, not
  magnitude
- Exact nearest-neighbour search is O(n). ANN trades recall for latency
- **HNSW**: navigable small-world graph, fast, memory-hungry, tunable via `ef`
- **IVF**: partition into cells, search a few. Cheaper memory, needs training
- Recall@k is the quality metric, and it is a *tradeoff dial*, not a bug

Doubles as your vector search interview prep.

## Build

- Local embedding model (MiniLM class). Keep it off the event loop —
  `run_in_threadpool` or a separate worker, or it blocks every concurrent stream
  and your p99 will lie to you
- Redis vector similarity, configurable threshold
- **A labelled eval set.** ~100 prompt pairs you write by hand, marked
  should-hit or should-not-hit. Tedious. This is exactly why nobody else
  publishes a false-hit number, and why yours is the credibility differentiator

## Accept when

You can produce a threshold-vs-hit-rate curve **and** a threshold-vs-false-hit
curve from your labelled set, both committed as data.

## Questions to answer

1. The embedding call is itself latency in the cache path. At what hit rate does
   semantic caching stop paying for itself?
2. Two prompts differ only by a negation. How close are they in embedding space?
   Try it. What does that imply for your threshold?

---

# Step 9 — Prefix-aware routing ★

**Goal:** route requests sharing a prompt prefix to the replica that last served
it.

The centerpiece. Give it the time.

## Learn first: PagedAttention and automatic prefix caching ★

From the vLLM paper: KV cache stored in fixed-size **blocks**, with a **block
table** mapping logical positions to physical blocks. Indirection means blocks
can be shared across sequences without copying, with copy-on-write when they
diverge. That's what makes prefix sharing cheap.

Automatic prefix caching sits on top: blocks are content-hashed, so a new request
whose prefix hashes to already-resident blocks reuses them and skips that portion
of prefill.

Two things to nail down, because they're what interviewers probe:
- **Block granularity.** Sharing happens at block boundaries, typically 16
  tokens. A prefix that diverges mid-block doesn't share that block.
- **Eviction.** Cached blocks are evicted under memory pressure, so affinity
  has a lifetime that isn't in your control.

This tells you why a character-based prefix key is an approximation — and
whether that approximation costs hit rate is a question for your benchmark, not
something to hand-wave.

## Build

- Prefix key derived from the leading portion of the prompt
- In-memory index: prefix → backend, with a TTL and eviction above some size
- `PrefixAwareRouter` behind the same interface as round-robin
- Metric counting hit / miss

**Single uvicorn worker, non-negotiable.** The index is in-process; multiple
workers means multiple partial indexes and non-reproducible routing. Moving the
index to Redis is your documented next step and a good writeup section.

## Questions to answer

1. Characters or tokens for the key? What does each cost you?
2. How long should a binding live? What does vLLM's eviction do to your answer?
3. Two requests share a system prompt but differ from token 50. Should they route
   together?

## Accept when

Six identical-prefix requests all land on one backend, verified via the response
header.

---

# Step 10 — Imbalance handling and the A/B

**Goal:** affinity that doesn't melt one replica, and a measured A/B result.

## Learn first: the tension

Naive affinity is a load-balancing failure. One popular system prompt sends every
request to one replica while the others idle. **This tradeoff is what your
writeup is actually about** — everything else is plumbing.

## Build

- Abandon affinity when the preferred backend's depth exceeds the shallowest by
  a threshold
- Metric counting `shed_imbalance` separately from plain misses
- A/B harness: same workload, `KVROUTE_STRATEGY` flipped, nothing else different

## Accept when

A/B over identical workload shows a TTFT delta with **non-overlapping confidence
intervals across at least 3 repeated runs**.

## Kill criterion

No measurable delta by the end of your Week 4 equivalent? Cut it, ship the
caching and admission control benchmark, and **report the negative result
honestly**. An honest null beats an unfinished repo, and it's a better interview
story than most positive ones.

---

# Step 11 — Admission control

**Goal:** protect the GPU, keep high-priority latency stable under overload.

## Learn first: why token budgets beat request rates

An LLM request is not a unit of work. A 50-token request and a 5000-token request
differ by two orders of magnitude in GPU cost. Rate-limiting requests lets one
tenant with long prompts starve everyone while technically staying under quota.

So budget on tokens. Which raises the real problem: **output length is unknown at
admission time.** Options are to budget on input tokens only, use `max_tokens` as
a pessimistic reservation, or estimate and reconcile after. Pick one, justify it.

## Build

- Per-tenant token bucket on tokens
- Two-level priority queue, interactive preempts batch
- Shed with 429 before the GPU saturates

## Accept when

At 3× overload, high-priority p99 stays within 20% of its unloaded value while
low-priority sheds.

---

# Step 12 — Load harness

**Goal:** the thing other people will actually run. This is the product.

## Learn first: benchmark methodology

- ≥3 runs per condition, **report variance**, not just means
- Discard the first run as warmup, say so
- Fix the random seed on the workload generator
- Commit raw data, not just charts
- **Open vs closed loop:** a closed-loop generator (wait for response, then send
  next) caps offered load at your own latency and hides saturation. Open-loop
  (send at a fixed rate regardless) is what you want. Most naive harnesses are
  closed-loop and silently wrong

## Build

Three workload profiles:

1. **Shared-prefix heavy** — many requests, one long system prompt. The RAG and
   agent-tooling pattern. Where prefix routing should shine
2. **Unique-prompt heavy** — little overlap. Proves routing doesn't *hurt* when
   there's nothing to reuse
3. **Mixed, tenant-tiered** — two tenants, different priorities, one bursting

Four conditions: baseline → cache only → cache + prefix routing → full stack.

Report per run: TTFT p50/p95/p99, ITL p50, throughput at saturation, GPU
utilization, cache hit rate, false-hit rate, cost per 1k requests.

## Accept when

Someone else can `docker compose up`, run the harness, and reproduce your
headline numbers within 10%.

---

# Step 13 — Writeup and publish

The repo is not the deliverable. The writeup is.

## Structure

1. The question, in one paragraph
2. Methodology: hardware, models, workload generator, what you controlled
3. Results, with variance and confidence intervals
4. **What didn't work.** Non-negotiable — it's the credibility signal
5. What you'd do differently at 10× scale

## Publish

- README leads with the headline result and a chart, not an install section
- Post title names the *finding*, not the project: "What prefix-aware routing
  actually saves on vLLM" pulls far more readers than "I built kvroute"
- r/LocalLLaMA and Show HN
- Link it in your LinkedIn headline and attach it to cold applications

---

# Step 14 — EKS (stretch, cut freely)

Terraform, one worker per pod, HPA on queue depth rather than CPU. Kubernetes
already does supervision and scaling, so multiple workers inside a pod means two
schedulers fighting and muddier per-replica metrics.

A resume line, not a differentiator. Cut it if Step 12 or 13 is at risk.

---

# Interview questions to have answers for

Prepare these as you go, not at the end.

- Why does prefix affinity conflict with load balancing, and how did you resolve
  it?
- Why does your project improve TTFT but not ITL?
- How did you choose the semantic cache threshold, and what does a false hit cost
  a user?
- Why token budgets instead of request rates?
- Your prefix index is in-process. What breaks at multi-worker, and what does
  fixing it cost?
- Where does this design break at 10× scale?
- What did you measure that surprised you?

The last one is the one that lands. Have a real answer.

---

# Definition of done

- [ ] Benchmark writeup published with reproducible numbers
- [ ] Harness runs on someone else's cluster
- [ ] Raw data committed
- [ ] Negative results included
- [ ] Resume bullets filled with measured numbers only