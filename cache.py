from __future__ import annotations

import base64
import hashlib
import json

from redis.asyncio import Redis

KEY_PREFIX = "kvroute:exact"


def normalize_key(tenant: str, payload: dict) -> str:
    normalized = {
        "model": payload.get("model"),
        "messages": payload.get("messages"),
        "temperature": payload.get("temperature"),
        "max_tokens": payload.get("max_tokens"),
    }
    blob = json.dumps(normalized, sort_keys=True)
    digest = hashlib.sha256(blob.encode()).hexdigest()
    return f"{KEY_PREFIX}:{tenant}:{digest}"


class ExactCache:
    def __init__(self, redis: Redis, ttl_seconds: float = 60.0) -> None:
        self._redis = redis
        self._ttl_seconds = ttl_seconds

    async def get(self, key: str) -> list[bytes] | None:
        raw = await self._redis.get(key)
        if raw is None:
            return None
        encoded_chunks: list[str] = json.loads(raw)
        return [base64.b64decode(c) for c in encoded_chunks]

    async def put(self, key: str, chunks: list[bytes]) -> None:
        encoded_chunks = [base64.b64encode(c).decode() for c in chunks]
        await self._redis.set(key, json.dumps(encoded_chunks), ex=int(self._ttl_seconds))
