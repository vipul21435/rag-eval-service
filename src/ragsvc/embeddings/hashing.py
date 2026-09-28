"""A deterministic embedder built on feature hashing; no model, no download.

Each text becomes a sparse bag of word unigrams and bigrams, every feature
is hashed to a slot of a fixed-size vector with a hashed sign, and the
result is L2-normalised. Two texts are close when they share words and
word pairs, which is enough for tests, demos and evaluation fixtures to
exercise the whole retrieval path with vectors that are identical on every
machine and every run. It knows nothing about meaning: synonyms do not
match, so it is not a substitute for a trained model in production.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from hashlib import blake2b
from itertools import pairwise

import numpy as np

from ragsvc.core.bm25_index import tokenize
from ragsvc.embeddings.base import Matrix, Vector

DEFAULT_DIMENSION = 256
MIN_DIMENSION = 2


@lru_cache(maxsize=65536)
def _slot(feature: str, dimension: int) -> tuple[int, float]:
    """The (index, sign) a feature hashes to; keyed hashing keeps it stable across processes."""
    digest = blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return (value >> 1) % dimension, 1.0 if value & 1 else -1.0


def features(text: str) -> list[str]:
    """Word unigrams followed by adjacent-word bigrams, lowercased."""
    tokens = tokenize(text)
    return tokens + [f"{left} {right}" for left, right in pairwise(tokens)]


class HashEmbedder:
    """Feature-hashing embedder with a fixed ``dimension``; the default in tests and demos."""

    name = "hash"

    def __init__(self, dimension: int = DEFAULT_DIMENSION) -> None:
        if dimension < MIN_DIMENSION:
            raise ValueError(f"dimension must be at least {MIN_DIMENSION}, got {dimension}")
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return f"hash-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        rows = [self.embed_query(text) for text in texts]
        if not rows:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return np.vstack(rows)

    def embed_query(self, text: str) -> Vector:
        """Hash the text's features into a unit vector; an empty text gives the zero vector."""
        vector = np.zeros(self._dimension, dtype=np.float32)
        for feature in features(text):
            index, sign = _slot(feature, self._dimension)
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector
