import threading
from pathlib import Path

import numpy as np
import pytest

import ragsvc.core.ingest as ingest
from ragsvc.core.bm25_index import bm25_manager
from ragsvc.core.ingest import SourceFile, ingest_files
from ragsvc.core.vector_store import vector_store


def fake_encode_texts(texts, show_progress=False):
    rng = np.random.default_rng(len(texts))
    return rng.random((len(texts), 8), dtype=np.float32)


def unit_vectors(count: int, dimension: int = 8) -> np.ndarray:
    vectors = np.zeros((count, dimension), dtype=np.float32)
    for i in range(count):
        vectors[i, i % dimension] = 1.0
    return vectors


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


def test_concurrent_ingests_are_serialized_and_leave_a_consistent_store(tmp_path: Path, monkeypatch):
    first_encoding = threading.Event()
    release_first = threading.Event()
    encode_calls: list[int] = []

    def blocking_encode(texts, show_progress=False):
        encode_calls.append(len(texts))
        if len(encode_calls) == 1:
            first_encoding.set()
            assert release_first.wait(5), "test did not release the first ingestion"
        return unit_vectors(len(texts))

    monkeypatch.setattr(ingest, "encode_texts", blocking_encode)
    first_run = [SourceFile.from_path(write_text(tmp_path, "a.txt", "alpha"))]
    second_run = [
        SourceFile.from_path(write_text(tmp_path, f"b{i}.txt", f"bravo {word}"))
        for i, word in enumerate(("one", "two", "three"))
    ]
    second_events: list[str] = []

    first = threading.Thread(target=ingest_files, args=(first_run,))
    first.start()
    assert first_encoding.wait(5)
    second = threading.Thread(
        target=ingest_files, args=(second_run,), kwargs={"progress": lambda _f, d: second_events.append(d)}
    )
    second.start()
    second.join(0.2)

    # The second run must not have touched the indexes while the first is mid-flight.
    assert second.is_alive()
    assert second_events == []

    release_first.set()
    first.join(5)
    second.join(5)
    assert not first.is_alive() and not second.is_alive()

    assert encode_calls == [1, 3]
    assert list(vector_store.id_order) == ["doc_1_chunk_0", "doc_2_chunk_0", "doc_3_chunk_0"]
    assert vector_store.total_chunks == 3
    for position, chunk_id in enumerate(vector_store.id_order):
        docs, ids, _ = vector_store.search(unit_vectors(3)[position : position + 1], k=1)
        assert ids == [chunk_id]
        assert docs == [vector_store.contents_map[chunk_id]]
    assert bm25_manager.search("bravo three", top_k=1)[0]["id"] == "doc_3_chunk_0"


def test_ingest_with_no_sources_leaves_indexes_untouched():
    report = ingest_files([])

    assert report.files == []
    assert report.total_chunks == 0
    assert not report.succeeded
    assert not vector_store.is_ready


@pytest.mark.parametrize(
    ("name", "content", "expected_error"),
    [
        ("report.csv", b"a,b\n1,2\n", "unsupported file format '.csv'"),
        ("latin1.txt", "caf\u00e9".encode("latin-1"), "not UTF-8"),
        ("blank.md", b"   \n", "no extractable text"),
    ],
)
def test_failed_run_keeps_the_previous_knowledge_base(tmp_path: Path, name, content, expected_error):
    seed_knowledge_base(tmp_path)
    bad = tmp_path / name
    bad.write_bytes(content)

    report = ingest_files([SourceFile.from_path(bad)])

    assert not report.succeeded
    assert report.total_chunks == 0
    assert expected_error in (report.files[0].error or "")
    assert_seeded_knowledge_base_intact()


def test_embedding_failure_keeps_the_previous_knowledge_base(tmp_path: Path, monkeypatch):
    seed_knowledge_base(tmp_path)
    replacement = write_text(tmp_path, "new.md", "Replacement knowledge base.")

    def failing_encode(texts, show_progress=False):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(ingest, "encode_texts", failing_encode)

    with pytest.raises(RuntimeError, match="model download failed"):
        ingest_files([SourceFile.from_path(replacement)])

    assert_seeded_knowledge_base_intact()


def seed_knowledge_base(tmp_path: Path) -> None:
    """Index three documents; BM25 needs at least three for a positive IDF."""
    report = ingest_files(
        [
            SourceFile.from_path(write_text(tmp_path, "faiss.md", "FAISS supports dense vector retrieval.")),
            SourceFile.from_path(write_text(tmp_path, "bm25.md", "BM25 supports exact keyword retrieval.")),
            SourceFile.from_path(write_text(tmp_path, "rag.md", "RAG combines retrieval with generation.")),
        ]
    )
    assert report.succeeded


def assert_seeded_knowledge_base_intact() -> None:
    assert vector_store.is_ready and vector_store.total_chunks == 3
    assert vector_store.contents_map["doc_2_chunk_0"] == "BM25 supports exact keyword retrieval."
    assert bm25_manager.search("BM25 keyword", top_k=1)[0]["id"] == "doc_2_chunk_0"
