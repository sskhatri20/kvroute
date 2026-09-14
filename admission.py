"""Step 11 — admission control.

Budgets on tokens, not requests, per the build guide: a 50-token and a
5000-token request cost two orders of magnitude apart, so a request-rate
limit lets one long-prompt tenant starve everyone else while under quota.

Simplification from the spec: instead of a real preemptive priority queue
(which only makes sense once a request can be paused mid-flight, and these
are already-open streaming HTTP responses), interactive traffic gets a
reserved slice of concurrency that batch traffic can never take. Batch is
shed with 429 the moment it would eat into that reserve. Same effect —
interactive latency stays flat under batch load — without building a
scheduler around a preemption point that doesn't really exist here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

DEFAULT_CAPACITY = 10_000.0  # tokens
DEFAULT_REFILL_PER_SECOND = 2_000.0
BATCH_CONCURRENCY_SHARE = 0.5  # batch may use at most this fraction of total inflight capacity
TOTAL_CONCURRENCY = 8


@dataclass
class TokenBucket:
    capacity: float = DEFAULT_CAPACITY
    refill_per_second: float = DEFAULT_REFILL_PER_SECOND
    tokens: float = field(init=False)
    _last_refill: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = self.capacity
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
        self._last_refill = now

    def try_consume(self, amount: float) -> bool:
        self._refill()
        if self.tokens < amount:
            return False
        self.tokens -= amount
        return True


def estimate_tokens(payload: dict) -> float:
    if payload.get("max_tokens"):
        return float(payload["max_tokens"])
    prompt_chars = sum(len(str(m.get("content", ""))) for m in payload.get("messages", []))
    return prompt_chars / 4  # ~4 chars/token, plus an assumed similarly sized completion


@dataclass
class AdmissionController:
    total_concurrency: int = TOTAL_CONCURRENCY
    batch_concurrency_share: float = BATCH_CONCURRENCY_SHARE
    _buckets: dict[str, TokenBucket] = field(default_factory=dict)
    interactive_inflight: int = 0
    batch_inflight: int = 0

    def _bucket_for(self, tenant: str) -> TokenBucket:
        if tenant not in self._buckets:
            self._buckets[tenant] = TokenBucket()
        return self._buckets[tenant]

    def admit(self, tenant: str, priority: str, payload: dict) -> str | None:
        """Returns None if admitted, else a shed reason."""
        if not self._bucket_for(tenant).try_consume(estimate_tokens(payload)):
            return "shed_budget"

        batch_limit = int(self.total_concurrency * self.batch_concurrency_share)
        if priority == "batch" and self.batch_inflight >= batch_limit:
            return "shed_capacity"
        if self.interactive_inflight + self.batch_inflight >= self.total_concurrency:
            return "shed_capacity"

        if priority == "batch":
            self.batch_inflight += 1
        else:
            self.interactive_inflight += 1
        return None

    def release(self, priority: str) -> None:
        if priority == "batch":
            self.batch_inflight -= 1
        else:
            self.interactive_inflight -= 1
