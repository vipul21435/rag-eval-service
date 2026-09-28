"""An on-disk embedding cache in SQLite, and the wrapper that uses it.

Embedding is the slow part of ingestion, and the same chunks come back on
every re-index of an unchanged document. :class:`EmbeddingCache` stores one
vector per ``(provider, model, sha256(text))`` so a vector computed by one
model is never served for another, and counts hits and misses so
``/health`` can show whether the cache is doing any work.
:class:`CachedEmbedder` puts the cache in front of any
:class:`~ragsvc.embeddings.base.EmbeddingProvider` and only calls the
provider for the texts it has not seen.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np

from ragsvc.embeddings.base import EmbeddingProvider, Matrix, Vector

logger = logging.getLogger(__name__)

IN_MEMORY = ":memory:"
# Rows fetched per SELECT; comfortably below SQLite's bound-parameter limit.
_LOOKUP_BATCH = 500
_ITEM_SIZE = np.dtype(np.float32).itemsize

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    provider    TEXT    NOT NULL,
    model       TEXT    NOT NULL,
    text_sha256 TEXT    NOT NULL,
    dimension   INTEGER NOT NULL,
    vector      BLOB    NOT NULL,
    created_at  REAL    NOT NULL,
    PRIMARY KEY (provider, model, text_sha256)
)
"""


def text_key(text: str) -> str:
    """The cache key of ``text``: the hex SHA-256 of its UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheStats:
    """Lookup counters since the cache was opened, plus the rows on disk."""

    hits: int
    misses: int
    entries: int

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups served from the cache; 0.0 before the first lookup."""
        return self.hits / self.lookups if self.lookups else 0.0


class EmbeddingCache:
    """SQLite-backed store of vectors keyed by provider, model and text hash.

    ``path`` is a file (its parent directory is created) or ``":memory:"``
    for a cache that lives as long as the object. One connection is shared
    by all threads under a lock, so the ingestion worker threads and the API
    can use the same cache.
    """

    def __init__(self, path: str | os.PathLike[str] = IN_MEMORY) -> None:
        self._path = str(path)
        if self._path != IN_MEMORY:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self._path, check_same_thread=False)
        self._hits = 0
        self._misses = 0
        with self._lock, self._connection:
            self._connection.execute(_SCHEMA)
        logger.debug("Embedding cache opened at %s", self._path)

    @property
    def path(self) -> str:
        return self._path

    @property
    def stats(self) -> CacheStats:
        with self._lock:
            (entries,) = self._connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return CacheStats(hits=self._hits, misses=self._misses, entries=int(entries))

    def get_many(self, provider: str, model: str, texts: Sequence[str]) -> list[Vector | None]:
        """The cached vector for each text, or ``None`` where there is none.

        Every position counts as one hit or miss, including repeated texts.
        A stored row whose payload does not match its recorded dimension is
        treated as a miss and overwritten by the next ``put_many``.
        """
        keys = [text_key(text) for text in texts]
        found: dict[str, Vector] = {}
        with self._lock:
            for batch in _batches(sorted(set(keys))):
                placeholders = ",".join("?" * len(batch))
                rows = self._connection.execute(
                    "SELECT text_sha256, dimension, vector FROM embeddings"
                    f" WHERE provider = ? AND model = ? AND text_sha256 IN ({placeholders})",
                    (provider, model, *batch),
                ).fetchall()
                for key, dimension, blob in rows:
                    if len(blob) == dimension * _ITEM_SIZE:
                        found[key] = np.frombuffer(blob, dtype=np.float32)
                    else:
                        logger.warning("Ignoring corrupt cache row for %s/%s", provider, model)
            results = [found.get(key) for key in keys]
            hits = sum(result is not None for result in results)
            self._hits += hits
            self._misses += len(results) - hits
        return results

    def put_many(self, provider: str, model: str, texts: Sequence[str], vectors: Matrix) -> None:
        """Store one row per text; an existing row for the same key is replaced."""
        if len(texts) != int(vectors.shape[0]):
            raise ValueError(f"{len(texts)} texts but {int(vectors.shape[0])} vectors")
        now = time.time()
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        rows = [
            (provider, model, text_key(text), int(matrix.shape[1]), matrix[i].tobytes(), now)
            for i, text in enumerate(texts)
        ]
        with self._lock, self._connection:
            self._connection.executemany(
                "INSERT OR REPLACE INTO embeddings"
                " (provider, model, text_sha256, dimension, vector, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )

    def clear(self) -> None:
        """Delete every row; the hit and miss counters keep counting."""
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM embeddings")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()


def _batches(keys: Sequence[str]) -> Iterator[Sequence[str]]:
    for start in range(0, len(keys), _LOOKUP_BATCH):
        yield keys[start : start + _LOOKUP_BATCH]


class CachedEmbedder:
    """An :class:`EmbeddingProvider` that answers from ``cache`` before calling ``inner``."""

    def __init__(self, inner: EmbeddingProvider, cache: EmbeddingCache) -> None:
        self._inner = inner
        self._cache = cache

    @property
    def inner(self) -> EmbeddingProvider:
        return self._inner

    @property
    def cache(self) -> EmbeddingCache:
        return self._cache

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        """Embed only the texts the cache does not hold, in one call to the provider."""
        texts = list(texts)
        if not texts:
            return self._inner.embed_documents([])
        cached = self._cache.get_many(self.name, self.model_name, texts)
        missing = list(
            dict.fromkeys(text for text, vector in zip(texts, cached, strict=True) if vector is None)
        )
        computed: dict[str, Vector] = {}
        if missing:
            vectors = self._inner.embed_documents(missing)
            self._cache.put_many(self.name, self.model_name, missing, vectors)
            computed = dict(zip(missing, vectors, strict=True))
            logger.debug(
                "Embedded %d new text(s); %d served from cache", len(missing), len(texts) - len(missing)
            )
        rows = [
            vector if vector is not None else computed[text]
            for text, vector in zip(texts, cached, strict=True)
        ]
        return np.asarray(np.vstack(rows), dtype=np.float32)

    def embed_query(self, text: str) -> Vector:
        vector: Vector = self.embed_documents([text])[0]
        return vector
