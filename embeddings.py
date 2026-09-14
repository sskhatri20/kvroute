from __future__ import annotations

import hashlib
import math
import re

DIMENSIONS = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function words carry no topical signal and dominate every prompt equally
# ("what", "is", "the" show up whether two prompts are duplicates or not).
# Dropping them is a cheap stand-in for the IDF weighting a real embedding
# model learns implicitly.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am",
    "what", "which", "who", "whom", "how", "do", "does", "did", "can",
    "could", "would", "will", "shall", "should", "of", "in", "on", "at",
    "to", "for", "and", "or", "me", "my", "i", "you", "your", "it",
    "this", "that", "with", "into", "using", "please", "give", "tell",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def embed(text: str) -> list[float]:
    vector = [0.0] * DIMENSIONS
    for token in _tokenize(text):
        digest = hashlib.sha256(token.encode()).digest()
        bucket = int.from_bytes(digest[:4], "big") % DIMENSIONS
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
