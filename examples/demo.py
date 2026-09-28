"""Offline demo: ingest the sample documents, then run three queries and time them.

Runs with the deterministic hash embedder and no reranker, so it needs no
network access, no model download and no LLM. It prints the ingestion
time, the embedding cache counters before and after a re-index, the top
chunks of each query with their hybrid scores, and per-query latency
percentiles. ``make demo`` runs it; ``RAG_EMBEDDING_PROVIDER`` and the
other ``RAG_`` settings are honoured except the ones the demo pins.
"""

from __future__ import annotations

import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from ragsvc.api import create_app
from ragsvc.config import Settings
from ragsvc.core.ingest import SourceFile, ingest_files
from ragsvc.core.retriever import search_chunks
from ragsvc.logging_setup import configure_logging

DOCS_DIR = Path(__file__).resolve().parent / "docs"
QUERIES = (
    "How are dense and BM25 scores combined?",
    "What key does the embedding cache use?",
    "What does GET /ready return before documents are indexed?",
)
RUNS_PER_QUERY = 20
TOP_K = 3
SNIPPET_CHARS = 90


def snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= SNIPPET_CHARS else flat[: SNIPPET_CHARS - 3] + "..."


def cache_line(client: TestClient) -> str:
    health: dict[str, Any] = client.get("/health").json()
    cache = health["embedding_cache"]
    return f"entries={cache['entries']} hits={cache['hits']} misses={cache['misses']}"


def timed_ingest(sources: list[SourceFile]) -> tuple[float, int, list[tuple[str, int]]]:
    started = time.perf_counter()
    report = ingest_files(sources)
    elapsed = time.perf_counter() - started
    return elapsed, report.total_chunks, [(result.name, result.chunks) for result in report.files]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="recallmcp-demo-") as tmp:
        settings = Settings(
            llm_provider="ollama",  # never probe for a server: the demo does not generate
            rerank_method="none",  # the cross-encoder would need a model download
            max_retrieval_iterations=1,
            embedding_cache_enabled=True,
            embedding_cache_path=Path(tmp) / "embeddings.sqlite3",
            log_level="WARNING",
            log_format="text",
        )
        configure_logging(settings.log_level, settings.log_format)
        client = TestClient(create_app(settings))

        print(f"RecallMCP demo: embedder={settings.embedding_provider}, reranker={settings.rerank_method}")
        sources = [SourceFile.from_path(path) for path in sorted(DOCS_DIR.glob("*.md"))]
        if not sources:
            print(f"no sample documents under {DOCS_DIR}", file=sys.stderr)
            return 1

        elapsed, total, per_file = timed_ingest(sources)
        print(f"\nIngest: {len(sources)} files -> {total} chunks in {elapsed * 1000:.0f} ms")
        for name, chunks in per_file:
            print(f"  {name}: {chunks} chunks")
        print(f"  embedding cache after first ingest: {cache_line(client)}")
        elapsed, total, _ = timed_ingest(sources)
        print(f"Re-ingest (unchanged files): {total} chunks in {elapsed * 1000:.0f} ms")
        print(f"  embedding cache after re-ingest:    {cache_line(client)}")
        print(f"  GET /ready -> {client.get('/ready').status_code}")

        # The first run of a query embeds it (a cache miss); later runs find
        # the query vector in the cache, so cold and warm latency are reported apart.
        cold_latencies: list[float] = []
        warm_latencies: list[float] = []
        for number, query in enumerate(QUERIES, start=1):
            latencies: list[float] = []
            for _ in range(RUNS_PER_QUERY):
                started = time.perf_counter()
                results = search_chunks(query, top_k=TOP_K)
                latencies.append((time.perf_counter() - started) * 1000)
            cold_latencies.append(latencies[0])
            warm_latencies.extend(latencies[1:])
            p50 = statistics.median(latencies[1:])
            print(
                f"\nQuery {number}: {query!r}  (cold {latencies[0]:.2f} ms,"
                f" warm p50 {p50:.2f} ms over {RUNS_PER_QUERY - 1} runs)"
            )
            for rank, (chunk_id, doc) in enumerate(results, start=1):
                source = doc["metadata"].get("source", "?")
                print(f"  {rank}. score={doc['score']:.3f} source={source} id={chunk_id}")
                print(f"     {snippet(doc['content'])}")

        ordered = sorted(warm_latencies)
        p50 = statistics.median(ordered)
        p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
        print(
            f"\nQuery latency: cold p50 {statistics.median(cold_latencies):.2f} ms"
            f" ({len(cold_latencies)} first runs); warm p50 {p50:.2f} ms, p95 {p95:.2f} ms"
            f" ({len(ordered)} runs)"
        )
        print(f"Embedding cache at exit: {cache_line(client)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
