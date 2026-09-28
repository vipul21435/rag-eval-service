from pathlib import Path

import numpy as np
import pytest

import core.ingest as ingest
from core.bm25_index import bm25_manager
from core.ingest import SourceFile, ingest_files
from core.vector_store import vector_store


def fake_encode_texts(texts, show_progress=False):
    rng = np.random.default_rng(len(texts))
    return rng.random((len(texts), 8), dtype=np.float32)


@pytest.fixture(autouse=True)
def isolated_indexes(monkeypatch):
    monkeypatch.setattr(ingest, "encode_texts", fake_encode_texts)
    vector_store.clear()
    bm25_manager.clear()
    yield
    vector_store.clear()
    bm25_manager.clear()


def write_text(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_ingest_builds_both_indexes_with_source_metadata(tmp_path: Path):
    faiss_doc = write_text(tmp_path, "faiss.txt", "FAISS supports dense vector retrieval.")
    bm25_doc = write_text(tmp_path, "bm25.md", "BM25 supports exact keyword retrieval.")
    rag_doc = write_text(tmp_path, "rag.txt", "RAG combines retrieval with generation.")

    report = ingest_files(
        [
            SourceFile.from_path(faiss_doc),
            SourceFile(bm25_doc, "notes.md"),
            SourceFile.from_path(rag_doc),
        ]
    )

    assert report.succeeded
    assert [(r.name, r.chunks, r.ok) for r in report.files] == [
        ("faiss.txt", 1, True),
        ("notes.md", 1, True),
        ("rag.txt", 1, True),
    ]
    assert report.total_chunks == 3
    assert vector_store.is_ready
    assert vector_store.total_chunks == 3
    assert vector_store.metadatas_map["doc_1_chunk_0"] == {"source": "faiss.txt", "doc_id": "doc_1"}
    assert vector_store.metadatas_map["doc_2_chunk_0"] == {"source": "notes.md", "doc_id": "doc_2"}

    hits = bm25_manager.search("BM25 keyword", top_k=1)
    assert hits and hits[0]["id"] == "doc_2_chunk_0"


def test_ingest_reports_failures_without_aborting_the_batch(tmp_path: Path):
    good = write_text(tmp_path, "good.txt", "Hybrid retrieval merges dense and sparse scores.")
    empty = write_text(tmp_path, "empty.txt", "   \n")
    missing = tmp_path / "missing.txt"

    report = ingest_files(
        [SourceFile.from_path(good), SourceFile.from_path(empty), SourceFile.from_path(missing)]
    )

    assert not report.succeeded
    assert report.total_chunks == 1
    good_result, empty_result, missing_result = report.files
    assert good_result.ok and good_result.chunks == 1
    assert not empty_result.ok and "no extractable text" in (empty_result.error or "")
    assert not missing_result.ok and missing_result.error
    assert vector_store.total_chunks == 1


def test_ingest_replaces_previous_indexes_and_reports_progress(tmp_path: Path):
    first = write_text(tmp_path, "first.txt", "First knowledge base.")
    second = write_text(tmp_path, "second.txt", "Second knowledge base.")
    ingest_files([SourceFile.from_path(first)])
    assert vector_store.total_chunks == 1

    events: list[tuple[float, str]] = []
    report = ingest_files([SourceFile.from_path(second)], progress=lambda f, d: events.append((f, d)))

    assert report.total_chunks == 1
    assert list(vector_store.id_order) == ["doc_1_chunk_0"]
    assert vector_store.contents_map["doc_1_chunk_0"] == "Second knowledge base."
    fractions = [fraction for fraction, _ in events]
    assert fractions == sorted(fractions)
    assert events[-1] == (1.0, "Done")


def test_ingest_with_no_sources_leaves_empty_indexes():
    report = ingest_files([])

    assert report.files == []
    assert report.total_chunks == 0
    assert not report.succeeded
    assert not vector_store.is_ready
