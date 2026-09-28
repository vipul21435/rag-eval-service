"""The SQLite embedding cache and the cached provider wrapper."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from ragsvc.embeddings import CachedEmbedder, EmbeddingCache, HashEmbedder, text_key
from ragsvc.embeddings.cache import CacheStats


class CountingEmbedder:
    """A tiny provider that records every batch it is asked to embed."""

    name = "counting"
    model_name = "counting-4"
    dimension = 4

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.batches.append(list(texts))
        rows = [[len(text), 1.0, 2.0, 3.0] for text in texts]
        return np.array(rows, dtype=np.float32).reshape(len(rows), self.dimension)

    def embed_query(self, text: str) -> np.ndarray:
        vector: np.ndarray = self.embed_documents([text])[0]
        return vector


@pytest.fixture
def cache():
    with EmbeddingCache() as memory_cache:
        yield memory_cache


def vectors(*lengths: int) -> np.ndarray:
    return np.array([[length, 1.0, 2.0, 3.0] for length in lengths], dtype=np.float32)


# --- EmbeddingCache ---------------------------------------------------------


def test_cache_round_trips_vectors_and_counts_hits_and_misses(cache):
    assert cache.stats == CacheStats(hits=0, misses=0, entries=0)
    assert cache.get_many("p", "m", ["alpha", "beta"]) == [None, None]

    cache.put_many("p", "m", ["alpha", "beta"], vectors(5, 4))
    alpha, beta, gamma = cache.get_many("p", "m", ["alpha", "beta", "gamma"])

    assert alpha is not None and alpha.tolist() == [5.0, 1.0, 2.0, 3.0] and alpha.dtype == np.float32
    assert beta is not None and beta.tolist() == [4.0, 1.0, 2.0, 3.0]
    assert gamma is None
    stats = cache.stats
    assert (stats.hits, stats.misses, stats.entries, stats.lookups) == (2, 3, 2, 5)
    assert stats.hit_rate == pytest.approx(0.4)


def test_cache_keys_on_provider_model_and_text_hash(cache):
    cache.put_many("p", "m", ["alpha"], vectors(5))

    assert cache.get_many("p", "other-model", ["alpha"]) == [None]
    assert cache.get_many("other-provider", "m", ["alpha"]) == [None]
    assert cache.get_many("p", "m", ["Alpha"]) == [None]
    assert cache.get_many("p", "m", ["alpha"])[0] is not None
    assert text_key("alpha") == "8ed3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8"


def test_cache_replaces_existing_rows_and_counts_repeated_texts(cache):
    cache.put_many("p", "m", ["alpha"], vectors(5))
    cache.put_many("p", "m", ["alpha"], vectors(9))

    first, second = cache.get_many("p", "m", ["alpha", "alpha"])

    assert first is not None and second is not None
    assert first.tolist()[0] == 9.0 and second.tolist()[0] == 9.0
    assert cache.stats == CacheStats(hits=2, misses=0, entries=1)


def test_cache_persists_across_reopen_and_creates_parent_directories(tmp_path: Path):
    path = tmp_path / "nested" / "dir" / "embeddings.sqlite3"
    with EmbeddingCache(path) as first:
        first.put_many("p", "m", ["alpha"], vectors(5))
        assert first.path == str(path)

    with EmbeddingCache(path) as second:
        (found,) = second.get_many("p", "m", ["alpha"])
        assert found is not None and found.tolist()[0] == 5.0
        assert second.stats == CacheStats(hits=1, misses=0, entries=1)


def test_cache_treats_a_corrupt_row_as_a_miss(tmp_path: Path):
    path = tmp_path / "embeddings.sqlite3"
    with EmbeddingCache(path) as cache:
        cache.put_many("p", "m", ["alpha"], vectors(5))
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE embeddings SET vector = ?", (b"\x00\x00",))

    with EmbeddingCache(path) as cache:
        assert cache.get_many("p", "m", ["alpha"]) == [None]
        cache.put_many("p", "m", ["alpha"], vectors(7))
        (found,) = cache.get_many("p", "m", ["alpha"])
        assert found is not None and found.tolist()[0] == 7.0


def test_cache_looks_up_more_texts_than_one_batch(cache):
    texts = [f"text {i}" for i in range(1203)]
    cache.put_many("p", "m", texts, vectors(*range(1203)))

    found = cache.get_many("p", "m", [*texts, "absent"])

    assert all(vector is not None for vector in found[:-1]) and found[-1] is None
    assert cache.stats == CacheStats(hits=1203, misses=1, entries=1203)


def test_cache_rejects_mismatched_texts_and_vectors_and_clears(cache):
    with pytest.raises(ValueError, match="2 texts but 1 vectors"):
        cache.put_many("p", "m", ["a", "b"], vectors(1))

    cache.put_many("p", "m", ["a"], vectors(1))
    cache.clear()

    assert cache.stats.entries == 0
    assert cache.get_many("p", "m", ["a"]) == [None]


# --- CachedEmbedder ---------------------------------------------------------


def test_cached_embedder_embeds_only_misses_and_preserves_order(cache):
    inner = CountingEmbedder()
    embedder = CachedEmbedder(inner, cache)
    assert (embedder.name, embedder.model_name, embedder.dimension) == ("counting", "counting-4", 4)
    assert embedder.inner is inner and embedder.cache is cache

    first = embedder.embed_documents(["aa", "b", "aa", "cccc"])
    second = embedder.embed_documents(["cccc", "dd", "aa"])

    assert first[:, 0].tolist() == [2.0, 1.0, 2.0, 4.0]
    assert second[:, 0].tolist() == [4.0, 2.0, 2.0]
    assert first.dtype == np.float32 and second.dtype == np.float32
    assert inner.batches == [["aa", "b", "cccc"], ["dd"]]
    assert cache.stats == CacheStats(hits=2, misses=5, entries=4)


def test_cached_embedder_caches_queries_and_passes_empty_batches_through(cache):
    inner = CountingEmbedder()
    embedder = CachedEmbedder(inner, cache)

    assert embedder.embed_query("query").tolist() == [5.0, 1.0, 2.0, 3.0]
    assert embedder.embed_query("query").tolist() == [5.0, 1.0, 2.0, 3.0]
    assert embedder.embed_documents([]).shape == (0, 4)

    assert inner.batches == [["query"], []]
    assert cache.stats == CacheStats(hits=1, misses=1, entries=1)


def test_cached_embedder_matches_the_uncached_provider(cache):
    plain = HashEmbedder(dimension=32)
    cached = CachedEmbedder(HashEmbedder(dimension=32), cache)
    texts = ["dense retrieval", "sparse retrieval", "dense retrieval"]

    assert np.array_equal(cached.embed_documents(texts), plain.embed_documents(texts))
    assert np.array_equal(cached.embed_documents(texts), plain.embed_documents(texts))
    assert np.array_equal(cached.embed_query("dense retrieval"), plain.embed_query("dense retrieval"))


def test_cache_is_safe_to_share_between_threads(tmp_path: Path):
    with EmbeddingCache(tmp_path / "shared.sqlite3") as cache:
        embedder = CachedEmbedder(HashEmbedder(dimension=16), cache)
        texts = [f"shared text {i}" for i in range(50)]
        results: list[np.ndarray] = []
        errors: list[BaseException] = []

        def work() -> None:
            try:
                results.append(embedder.embed_documents(texts))
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=work) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)

        assert errors == []
        assert len(results) == 4
        assert all(np.array_equal(results[0], other) for other in results[1:])
        assert cache.stats.entries == 50
