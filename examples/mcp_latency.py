"""Measure MCP tool-call latency: in-process session and the real stdio transport.

Pins the hash embedder and no reranker, so it is offline. The in-process
part times ``ingest_document`` once and ``search``, ``list_documents`` and
``health`` over ``RUNS`` calls each through a client session on memory
streams; the stdio part starts ``python -m ragsvc.mcp_server`` as a child
process and times the same calls end to end, protocol framing included.
"""

from __future__ import annotations

import asyncio
import functools
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ragsvc.config import Settings
from ragsvc.mcp_server import build_server, in_process_session

DOCS_DIR = Path(__file__).resolve().parent / "docs"
DOCUMENT = "hybrid-retrieval.md"
QUERY = "How are dense and BM25 scores combined?"
RUNS = 50
STDIO_RUNS = 20


def latency_settings(cache_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        embedding_provider="hash",
        rerank_method="none",
        llm_provider="ollama",
        embedding_cache_path=cache_path,
        mcp_document_root=DOCS_DIR,
    )


async def timed(call: Callable[[], Awaitable[Any]], runs: int) -> tuple[float, float, float]:
    """(first, p50, p95) in milliseconds over ``runs`` calls of ``call``."""
    samples: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        result = await call()
        samples.append((time.perf_counter() - started) * 1000)
        if getattr(result, "is_error", False):
            raise RuntimeError(f"tool call failed: {result.content}")
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * len(ordered)) - 1)]
    return samples[0], statistics.median(samples), p95


async def report(session: ClientSession, label: str, runs: int) -> None:
    first, _, _ = await timed(lambda: session.call_tool("ingest_document", {"path": DOCUMENT}), 1)
    print(f"{label}: ingest_document({DOCUMENT}) {first:.2f} ms")
    for name, arguments in (
        ("search", {"query": QUERY, "top_k": 3}),
        ("list_documents", {}),
        ("health", {}),
    ):
        call = functools.partial(session.call_tool, name, arguments)
        first, p50, p95 = await timed(call, runs)
        print(f"{label}: {name} first {first:.2f} ms, p50 {p50:.2f} ms, p95 {p95:.2f} ms over {runs} calls")


async def in_process(cache_path: Path) -> None:
    server = build_server(latency_settings(cache_path))
    started = time.perf_counter()
    async with in_process_session(server) as session:
        print(f"in-process: initialize {(time.perf_counter() - started) * 1000:.2f} ms")
        await report(session, "in-process", RUNS)


async def over_stdio(cache_path: Path) -> None:
    env = {
        **os.environ,
        "RAG_EMBEDDING_PROVIDER": "hash",
        "RAG_RERANK_METHOD": "none",
        "RAG_LLM_PROVIDER": "ollama",
        "RAG_EMBEDDING_CACHE_PATH": str(cache_path),
        "RAG_MCP_DOCUMENT_ROOT": str(DOCS_DIR),
        "RAG_LOG_LEVEL": "WARNING",
    }
    params = StdioServerParameters(command=sys.executable, args=["-m", "ragsvc.mcp_server"], env=env)
    started = time.perf_counter()
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        print(f"stdio: spawn + initialize {(time.perf_counter() - started) * 1000:.0f} ms")
        await report(session, "stdio", STDIO_RUNS)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "embeddings.sqlite3"
        asyncio.run(in_process(cache_path))
        asyncio.run(over_stdio(cache_path))


if __name__ == "__main__":
    main()
