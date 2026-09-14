from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

FAILURE_THRESHOLD = 3
COOLDOWN_SECONDS = 10.0


@dataclass
class Backend:
    url: str
    inflight: int = 0
    consecutive_failures: int = 0
    unhealthy_until: float = 0.0

    @property
    def name(self) -> str:
        return self.url.split("//", 1)[1].split("/", 1)[0]

    @property
    def healthy(self) -> bool:
        return time.monotonic() >= self.unhealthy_until

    def record_success(self) -> None:
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= FAILURE_THRESHOLD:
            self.unhealthy_until = time.monotonic() + COOLDOWN_SECONDS


class Router(ABC):
    @abstractmethod
    def pick(self, backends: list[Backend]) -> Backend: ...


class RoundRobinRouter(Router):
    def __init__(self) -> None:
        self._next = 0

    def pick(self, backends: list[Backend]) -> Backend:
        candidates = [b for b in backends if b.healthy] or backends
        backend = candidates[self._next % len(candidates)]
        self._next += 1
        return backend


PREFIX_KEY_CHARS = 200  # ~ a few dozen tokens; block-boundary precision isn't
                         # worth chasing for an approximation we already know is one
PREFIX_TTL_SECONDS = 300.0
PREFIX_MAX_ENTRIES = 1000
IMBALANCE_THRESHOLD = 4  # shed affinity once the preferred backend is this much deeper than the shallowest


@dataclass
class PrefixBinding:
    backend_name: str
    expires_at: float


class PrefixAwareRouter:
    """Step 9/10 — route by prompt prefix, with imbalance shedding.

    Not a `Router` subclass: `pick` here needs the prompt messages and
    returns the outcome alongside the backend, which doesn't fit the
    depth-only `Router.pick(backends) -> Backend` interface the other
    strategies share.

    In-process index: this only produces reproducible routing with a single
    uvicorn worker. Moving it to Redis is the documented next step if that
    constraint ever needs lifting.
    """

    def __init__(self) -> None:
        self._bindings: dict[str, PrefixBinding] = {}
        self._round_robin = RoundRobinRouter()

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [key for key, binding in self._bindings.items() if binding.expires_at < now]
        for key in expired:
            del self._bindings[key]

    def prefix_key(self, messages: list[dict]) -> str:
        text = " ".join(str(m.get("content", "")) for m in messages)
        return text[:PREFIX_KEY_CHARS]

    def pick(self, backends: list[Backend], messages: list[dict] | None = None) -> tuple[Backend, str]:
        """Returns (backend, outcome) where outcome is hit|miss|shed_imbalance."""
        self._evict_expired()
        healthy = {b.name: b for b in backends if b.healthy} or {b.name: b for b in backends}
        key = self.prefix_key(messages or [])

        binding = self._bindings.get(key)
        if binding is not None and binding.backend_name in healthy:
            preferred = healthy[binding.backend_name]
            shallowest = min(healthy.values(), key=lambda b: b.inflight)
            if preferred.inflight - shallowest.inflight >= IMBALANCE_THRESHOLD:
                chosen = shallowest
                outcome = "shed_imbalance"
            else:
                chosen = preferred
                outcome = "hit"
        else:
            chosen = self._round_robin.pick(list(healthy.values()))
            outcome = "miss"

        if len(self._bindings) >= PREFIX_MAX_ENTRIES and key not in self._bindings:
            oldest_key = min(self._bindings, key=lambda k: self._bindings[k].expires_at)
            del self._bindings[oldest_key]
        self._bindings[key] = PrefixBinding(backend_name=chosen.name, expires_at=time.monotonic() + PREFIX_TTL_SECONDS)

        return chosen, outcome
