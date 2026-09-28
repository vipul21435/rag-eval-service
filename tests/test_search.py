"""``search_chunks``: one hybrid round plus reranking, with scores, no LLM."""

from pathlib import Path

import pytest

import ragsvc.core.retriever as retriever
from ragsvc.core.bm25_index import bm25_manager
from ragsvc.core.ingest import SourceFile, ingest_files
from ragsvc.core.retriever import search_chunks
from ragsvc.core.vector_store import vector_store


@pytest.fixture(autouse=True)
def isolated_indexes():
    vector_store.clear()
    bm25_manager.clear()
    yield
    vector_store.clear()
    bm25_manager.clear()


def seed(tmp_path: Path) -> None:
    texts = {
        "faiss.md": "FAISS supports dense vector retrieval.",
        "bm25.md": "BM25 supports exact keyword retrieval.",
        "rag.md": "RAG combines retrieval with generation.",
    }
    sources = []
    for name, text in texts.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        sources.append(SourceFile.from_path(path))
    assert ingest_files(sources).succeeded


def test_search_chunks_returns_scored_chunks_best_first(tmp_path: Path, settings):
    settings(rerank_method="none")
    seed(tmp_path)

    results = search_chunks("exact keyword retrieval with BM25", top_k=2)

    assert len(results) == 2
    chunk_id, best = results[0]
    assert chunk_id == "doc_2_chunk_0"
    assert best["content"] == "BM25 supports exact keyword retrieval."
    assert best["metadata"] == {"source": "bm25.md", "doc_id": "doc_2"}
    assert best["score"] >= results[1][1]["score"] > 0.0


def test_search_chunks_defaults_top_k_to_the_rerank_setting(tmp_path: Path, settings):
    settings(rerank_method="none", rerank_top_k=1)
    seed(tmp_path)

    assert len(search_chunks("retrieval")) == 1


def test_search_chunks_on_an_empty_index_returns_nothing(settings):
    settings(rerank_method="none")

    assert search_chunks("anything") == []


def test_search_chunks_reranks_the_hybrid_candidates(tmp_path: Path, settings, monkeypatch):
    settings(rerank_method="cross_encoder", rerank_top_k=2)
    seed(tmp_path)
    calls: list[tuple[str, int, int]] = []

    def fake_rerank(query, docs, doc_ids, metas, method=None, top_k=5):
        calls.append((query, len(docs), top_k))
        ranked = [
            (doc_id, {"content": doc, "metadata": meta, "score": 0.5})
            for doc_id, doc, meta in zip(doc_ids, docs, metas, strict=True)
        ]
        return ranked[::-1]

    monkeypatch.setattr(retriever, "rerank_results", fake_rerank)

    results = search_chunks("dense vector retrieval")

    assert calls == [("dense vector retrieval", 3, 2)]
    assert len(results) == 2
    assert all(doc["score"] == 0.5 for _, doc in results)
