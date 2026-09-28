"""Embedding providers: the hash embedder, the lazy sentence-transformers wrapper and the builder."""

from __future__ import annotations

import sys
import types
from typing import ClassVar

import numpy as np
import pytest

from ragsvc.core import embeddings as core_embeddings
from ragsvc.core.embeddings import encode_query, encode_texts, get_embedder, set_embedder
from ragsvc.embeddings import (
    CachedEmbedder,
    EmbeddingError,
    EmbeddingProvider,
    HashEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
)
from ragsvc.embeddings.hashing import features

# --- HashEmbedder -----------------------------------------------------------


def test_hash_embedder_is_deterministic_across_instances_and_calls():
    first = HashEmbedder(dimension=64)
    second = HashEmbedder(dimension=64)
    text = "FAISS supports dense vector retrieval."

    vector = first.embed_query(text)

    assert np.array_equal(vector, first.embed_query(text))
    assert np.array_equal(vector, second.embed_query(text))
    assert np.array_equal(vector, first.embed_documents([text, "other"])[0])


def test_hash_embedder_has_the_configured_dimension_and_unit_norm():
    embedder = HashEmbedder(dimension=32)

    matrix = embedder.embed_documents(["short", "a much longer sentence with several words in it"])

    assert embedder.dimension == 32
    assert embedder.name == "hash"
    assert embedder.model_name == "hash-32"
    assert matrix.shape == (2, 32)
    assert matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-6)
    assert embedder.embed_query("short").shape == (32,)


def test_hash_embedder_maps_empty_text_to_the_zero_vector_and_no_texts_to_an_empty_matrix():
    embedder = HashEmbedder(dimension=16)

    assert not embedder.embed_query("").any()
    assert not embedder.embed_query("   \n").any()
    assert embedder.embed_documents([]).shape == (0, 16)


def test_hash_embedder_ranks_lexically_similar_text_closer():
    embedder = HashEmbedder(dimension=256)
    query = embedder.embed_query("dense vector retrieval with FAISS")
    similar = embedder.embed_query("FAISS supports dense vector retrieval.")
    unrelated = embedder.embed_query("The recipe calls for two eggs and butter.")

    assert float(query @ similar) > float(query @ unrelated)
    assert float(query @ similar) > 0.5


def test_hash_embedder_bigrams_make_word_order_matter():
    embedder = HashEmbedder(dimension=256)

    forward = embedder.embed_query("man bites dog")
    backward = embedder.embed_query("dog bites man")

    assert not np.array_equal(forward, backward)
    assert 0.3 < float(forward @ backward) < 1.0


def test_hash_features_are_lowercased_unigrams_then_bigrams():
    assert features("Hybrid Search works") == ["hybrid", "search", "works", "hybrid search", "search works"]
    assert features("single") == ["single"]
    assert features("") == []


def test_hash_embedder_rejects_a_degenerate_dimension():
    with pytest.raises(ValueError, match="dimension"):
        HashEmbedder(dimension=1)


def test_hash_embedder_satisfies_the_provider_protocol():
    assert isinstance(HashEmbedder(), EmbeddingProvider)


# --- SentenceTransformerEmbedder --------------------------------------------


class FakeSentenceTransformer:
    """Stands in for ``sentence_transformers.SentenceTransformer``; records construction and calls."""

    instances: ClassVar[list[FakeSentenceTransformer]] = []

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self.encode_calls: list[list[str]] = []
        FakeSentenceTransformer.instances.append(self)

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
        self.encode_calls.append(list(texts))
        return np.array([[len(text), 1.0, 0.0] for text in texts], dtype=np.float64)


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    FakeSentenceTransformer.instances = []
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return FakeSentenceTransformer


def test_sentence_transformer_embedder_loads_lazily_once_and_returns_float32(fake_sentence_transformers):
    embedder = SentenceTransformerEmbedder("some/model", device="cpu")

    assert embedder.name == "sentence-transformers"
    assert embedder.model_name == "some/model"
    loaded_before = embedder.loaded
    assert fake_sentence_transformers.instances == []

    matrix = embedder.embed_documents(["ab", "abcd"])
    vector = embedder.embed_query("abc")

    assert (loaded_before, embedder.loaded) == (False, True)
    assert embedder.dimension == 3
    assert len(fake_sentence_transformers.instances) == 1
    model = fake_sentence_transformers.instances[0]
    assert (model.model_name, model.device) == ("some/model", "cpu")
    assert matrix.dtype == np.float32 and matrix.shape == (2, 3)
    assert matrix[:, 0].tolist() == [2.0, 4.0]
    assert vector.dtype == np.float32 and vector.tolist() == [3.0, 1.0, 0.0]
    assert model.encode_calls == [["ab", "abcd"], ["abc"]]
    assert embedder.embed_documents([]).shape == (0, 3)


def test_sentence_transformer_embedder_reports_missing_library_as_embedding_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(EmbeddingError, match="RAG_EMBEDDING_PROVIDER=hash"):
        SentenceTransformerEmbedder("some/model").embed_query("hello")


def test_sentence_transformer_embedder_wraps_load_failures(monkeypatch):
    module = types.ModuleType("sentence_transformers")

    def failing_constructor(model_name, device=None):
        raise OSError("no such model on the Hub")

    module.SentenceTransformer = failing_constructor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    with pytest.raises(EmbeddingError, match="could not load embedding model 'missing/model'"):
        SentenceTransformerEmbedder("missing/model").dimension  # noqa: B018 - loads the model


# --- build_embedder and the process-wide provider ---------------------------


def test_build_embedder_follows_the_settings_without_loading_a_model(settings, tmp_path):
    hashed = build_embedder(settings(embedding_provider="hash", hash_embedding_dimension=16))
    assert isinstance(hashed, HashEmbedder) and hashed.dimension == 16

    neural = build_embedder(settings(embedding_provider="sentence-transformers", embed_model_name="x/y"))
    assert isinstance(neural, SentenceTransformerEmbedder)
    assert neural.model_name == "x/y" and not neural.loaded

    cached = build_embedder(
        settings(embedding_cache_enabled=True, embedding_cache_path=tmp_path / "cache" / "emb.sqlite3")
    )
    assert isinstance(cached, CachedEmbedder)
    assert isinstance(cached.inner, HashEmbedder)
    assert (tmp_path / "cache" / "emb.sqlite3").exists()
    cached.cache.close()


def test_process_wide_embedder_follows_the_installed_settings(settings):
    first = get_embedder()
    assert get_embedder() is first
    assert isinstance(first, HashEmbedder) and first.dimension == 64

    settings(hash_embedding_dimension=32)
    second = get_embedder()

    assert second is not first
    assert isinstance(second, HashEmbedder) and second.dimension == 32


def test_set_embedder_installs_a_replacement_until_reset():
    replacement = HashEmbedder(dimension=8)

    set_embedder(replacement)
    assert get_embedder() is replacement
    assert encode_texts(["a b"]).shape == (1, 8)

    set_embedder(None)
    assert get_embedder() is not replacement
    assert get_embedder().dimension == 64


def test_core_encode_functions_delegate_to_the_provider():
    provider = get_embedder()

    matrix = encode_texts(["dense retrieval", "sparse retrieval"], show_progress=True)
    query = encode_query("dense retrieval")

    assert matrix.shape == (2, 64) and query.shape == (1, 64)
    assert np.array_equal(matrix[0], provider.embed_query("dense retrieval"))
    assert np.array_equal(query[0], matrix[0])
    assert core_embeddings.get_embedder() is provider
