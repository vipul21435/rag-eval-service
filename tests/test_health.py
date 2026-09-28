"""GET /health and GET /ready."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragsvc import __version__, api
from ragsvc.config import get_settings
from ragsvc.core.bm25_index import bm25_manager
from ragsvc.core.embeddings import get_embedder
from ragsvc.core.ingest import SourceFile, ingest_files
from ragsvc.core.vector_store import vector_store
from ragsvc.embeddings import CachedEmbedder, SentenceTransformerEmbedder
from tests.conftest import make_settings


@pytest.fixture(autouse=True)
def fixed_provider_and_empty_indexes(monkeypatch):
    monkeypatch.setattr(api, "detect_default_provider", lambda: "ollama")
    vector_store.clear()
    bm25_manager.clear()
    yield
    vector_store.clear()
    bm25_manager.clear()


def index_one_document(tmp_path: Path, text: str = "BM25 supports exact keyword retrieval.") -> None:
    path = tmp_path / "doc.md"
    path.write_text(text, encoding="utf-8")
    report = ingest_files([SourceFile.from_path(path)])
    assert report.succeeded


def test_health_reports_version_providers_index_and_cache_without_loading_a_model(settings, tmp_path):
    installed = settings(
        embedding_provider="sentence-transformers",
        embed_model_name="org/model",
        embedding_cache_enabled=True,
        embedding_cache_path=tmp_path / "emb.sqlite3",
    )
    client = TestClient(api.create_app(installed))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": __version__,
        "providers": {
            "llm": "ollama",
            "embedding": "sentence-transformers",
            "embedding_model": "org/model",
            "reranker": "cross_encoder",
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        },
        "index": {"ready": False, "chunks": 0, "type": None},
        "embedding_cache": {"enabled": True, "entries": 0, "hits": 0, "misses": 0},
    }
    embedder = get_embedder()
    assert isinstance(embedder, CachedEmbedder)
    assert isinstance(embedder.inner, SentenceTransformerEmbedder) and not embedder.inner.loaded


def test_health_shows_the_index_and_cache_counters_moving_with_ingestion(settings, tmp_path):
    settings(
        embedding_cache_enabled=True, embedding_cache_path=tmp_path / "emb.sqlite3", rerank_method="none"
    )
    client = TestClient(api.create_app())

    index_one_document(tmp_path)
    first = client.get("/health").json()
    index_one_document(tmp_path)
    second = client.get("/health").json()

    assert first["providers"] == {
        "llm": "ollama",
        "embedding": "hash",
        "embedding_model": "hash-64",
        "reranker": "none",
        "reranker_model": None,
    }
    assert first["index"] == {"ready": True, "chunks": 1, "type": "FlatL2"}
    assert first["embedding_cache"] == {"enabled": True, "entries": 1, "hits": 0, "misses": 1}
    assert second["embedding_cache"] == {"enabled": True, "entries": 1, "hits": 1, "misses": 1}


def test_health_reports_a_disabled_cache():
    client = TestClient(api.create_app())

    assert client.get("/health").json()["embedding_cache"] == {
        "enabled": False,
        "entries": 0,
        "hits": 0,
        "misses": 0,
    }


def test_ready_is_503_until_documents_are_indexed(tmp_path):
    client = TestClient(api.create_app())

    before = client.get("/ready")
    index_one_document(tmp_path)
    after = client.get("/ready")

    assert before.status_code == 503
    assert before.json() == {
        "status": "not_ready",
        "chunks": 0,
        "reason": "knowledge base is empty; upload documents first",
    }
    assert after.status_code == 200
    assert after.json() == {"status": "ready", "chunks": 1, "reason": None}


def test_health_and_ready_need_no_api_token(settings):
    settings(api_token="s3cret")
    client = TestClient(api.create_app())

    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert client.get("/api/status").status_code == 401


def test_create_app_installs_explicit_settings_process_wide():
    custom = make_settings(hash_embedding_dimension=16)
    assert get_settings() is not custom

    api.create_app(custom)

    assert get_settings() is custom
    assert get_embedder().dimension == 16
