"""Step 8 — semantic cache, backed by Redis.

Plain Redis, not RediSearch: candidates are pulled back with SCAN and
compared client-side. That's an O(n) linear scan same as an in-memory list
would be — the win from moving to Redis here is per-tenant namespacing and
TTL-based eviction shared with the rest of the gateway's state, not search
speed. An HNSW/IVF index (RediSearch, or a dedicated vector DB) is the
real fix once n stops being small; not worth adding for a toy embedding.
"""
from __future__ import annotations

import base64
import json

from redis.asyncio import Redis

from embeddings import cosine_similarity, embed

KEY_PREFIX = "kvroute:semantic"


class SemanticCache:
    def __init__(self, redis: Redis, threshold: float = 0.7, ttl_seconds: float = 60.0) -> None:
        self._redis = redis
        self.threshold = threshold
        self._ttl_seconds = ttl_seconds

    async def get(self, tenant: str, text: str) -> list[bytes] | None:
        query = embed(text)
        best_chunks: list[bytes] | None = None
        best_similarity = -1.0

        async for key in self._redis.scan_iter(match=f"{KEY_PREFIX}:{tenant}:*"):
            raw = await self._redis.get(key)
            if raw is None:  # expired between SCAN and GET
                continue
            entry = json.loads(raw)
            similarity = cosine_similarity(query, entry["embedding"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_chunks = [base64.b64decode(c) for c in entry["chunks"]]

        if best_chunks is not None and best_similarity >= self.threshold:
            return best_chunks
        return None

    async def put(self, tenant: str, text: str, chunks: list[bytes]) -> None:
        key = f"{KEY_PREFIX}:{tenant}:{hash(text) & 0xFFFFFFFF:x}"
        entry = {
            "embedding": embed(text),
            "chunks": [base64.b64encode(c).decode() for c in chunks],
        }
        await self._redis.set(key, json.dumps(entry), ex=int(self._ttl_seconds))
