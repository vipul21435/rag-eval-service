"""Document ingestion: extract text, chunk it, embed it and build both indexes.

This is the write path of the knowledge base. It is shared by the REST API
and any other entry point, so callers get a structured report instead of a
human-readable status string.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from core.bm25_index import bm25_manager
from core.document_loader import extract_text
from core.embeddings import encode_texts
from core.text_splitter import split_text
from core.vector_store import vector_store

logger = logging.getLogger(__name__)

# Callback invoked with (fraction_complete, description).
ProgressCallback = Callable[[float, str], object]

# Ingestion runs replace the whole knowledge base, so two of them must not
# interleave: the API runs uploads in worker threads, and without this lock
# both could clear the indexes and then each build on top of the other's work.
_INGEST_LOCK = threading.Lock()


@dataclass(frozen=True)
class SourceFile:
    """A file to ingest plus the name recorded in chunk metadata."""

    path: Path
    name: str

    @classmethod
    def from_path(cls, path: str | os.PathLike[str]) -> SourceFile:
        resolved = Path(path)
        return cls(path=resolved, name=resolved.name)


@dataclass(frozen=True)
class FileResult:
    """Outcome of ingesting one file."""

    name: str
    chunks: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class IngestReport:
    """Outcome of one ingestion run."""

    files: list[FileResult]
    total_chunks: int

    @property
    def succeeded(self) -> bool:
        """True when at least one file produced chunks and none failed."""
        return self.total_chunks > 0 and all(result.ok for result in self.files)


def _notify(progress: ProgressCallback | None, fraction: float, description: str) -> None:
    if progress is not None:
        progress(fraction, description)


def ingest_files(
    sources: Sequence[SourceFile],
    progress: ProgressCallback | None = None,
) -> IngestReport:
    """Rebuild the vector and BM25 indexes from ``sources``.

    The existing indexes are cleared first: an ingestion run replaces the
    whole knowledge base rather than appending to it. Files that fail to
    parse are reported individually and do not abort the run. Runs are
    serialized process-wide; a concurrent call waits for the current one.
    """
    with _INGEST_LOCK:
        return _ingest_files_locked(sources, progress)


def _ingest_files_locked(sources: Sequence[SourceFile], progress: ProgressCallback | None) -> IngestReport:
    _notify(progress, 0.0, "Clearing existing indexes")
    vector_store.clear()
    bm25_manager.clear()

    results: list[FileResult] = []
    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_metadatas: list[dict[str, str]] = []

    total = len(sources)
    for index, source in enumerate(sources, start=1):
        _notify(progress, (index - 1) / max(total, 1), f"Processing {index}/{total}: {source.name}")
        try:
            text = extract_text(str(source.path))
            if not text.strip():
                raise ValueError("document is empty or has no extractable text")
            chunks = split_text(text)
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the batch
            logger.error("Failed to process %s: %s", source.name, exc)
            results.append(FileResult(name=source.name, chunks=0, error=str(exc)))
            continue

        doc_id = f"doc_{index}"
        all_chunks.extend(chunks)
        all_ids.extend(f"{doc_id}_chunk_{i}" for i in range(len(chunks)))
        all_metadatas.extend({"source": source.name, "doc_id": doc_id} for _ in chunks)
        results.append(FileResult(name=source.name, chunks=len(chunks)))

    if all_chunks:
        _notify(progress, 0.8, "Encoding chunks")
        embeddings = encode_texts(all_chunks)
        _notify(progress, 0.9, "Building FAISS index")
        vector_store.build_index(all_chunks, all_ids, all_metadatas, embeddings)
        _notify(progress, 0.95, "Building BM25 index")
        bm25_manager.build_index(all_chunks, all_ids)

    _notify(progress, 1.0, "Done")
    logger.info("Ingested %d file(s) into %d chunk(s)", total, len(all_chunks))
    return IngestReport(files=results, total_chunks=len(all_chunks))
