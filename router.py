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
