"""MCP server: the knowledge base as tools for agents, over stdio.

``recallmcp-mcp`` starts a Model Context Protocol server on standard input
and output. It exposes the same ``ragsvc`` core the HTTP API is built on,
read with the same ``RAG_`` settings: ``ingest_document`` runs the ingestion
pipeline, ``search`` is ``search_chunks`` with scores, ``list_documents``
reads the index metadata and ``health`` is the ``GET /health`` snapshot.
Logs go to stderr in the configured format; stdout carries only protocol
messages, which is what a stdio MCP client requires.

Documents can only be ingested from under ``RAG_MCP_DOCUMENT_ROOT`` (the
working directory by default): an agent driving the server should not be
able to index arbitrary files on the machine.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
from mcp import ClientSession
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.memory import create_client_server_memory_streams
from mcp.types import ToolAnnotations

from ragsvc import __version__
from ragsvc.api import health_snapshot
from ragsvc.config import Settings, get_settings, set_settings
from ragsvc.core.document_loader import SUPPORTED_EXTENSIONS, describe_supported_formats
from ragsvc.core.ingest import SourceFile, ingest_files
from ragsvc.core.retriever import search_chunks
from ragsvc.core.vector_store import vector_store
from ragsvc.logging_setup import configure_logging

logger = logging.getLogger("ragsvc.mcp")

SERVER_NAME = "recallmcp"
MAX_TOP_K = 50
# Clients that gate on annotations can auto-approve the read-only tools;
# ingest_document replaces the knowledge base, so it is flagged destructive.
READ_ONLY = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)
REPLACES_INDEX = ToolAnnotations(
    read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=False
)
INSTRUCTIONS = (
    "Local document retrieval. Call ingest_document with a path under the document root "
    "to (re)build the knowledge base from that file, then search for scored chunks. "
    "list_documents shows what is indexed and health reports the providers and index size."
)


def resolve_document(path: str, settings: Settings) -> Path:
    """The file ``path`` names, checked to be a readable supported document under the root.

    Relative paths are taken from ``RAG_MCP_DOCUMENT_ROOT``; absolute paths
    must lie under it. Raises ``ToolError`` with a message the agent can act
    on.
    """
    root = settings.mcp_document_root.resolve()
    target = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not target.is_relative_to(root):
        raise ToolError(f"{path!r} is outside the document root {root}")
    if not target.is_file():
        raise ToolError(f"{path!r} is not a file under {root}")
    suffix = target.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ToolError(
            f"unsupported file format {suffix or '(none)'!r}; supported: {describe_supported_formats()}"
        )
    size = target.stat().st_size
    if size > settings.max_upload_bytes:
        raise ToolError(f"{path!r} is {size} bytes, over the limit of {settings.max_upload_bytes}")
    return target


def indexed_documents() -> list[dict[str, Any]]:
    """One entry per indexed document, in index order, with its chunk count."""
    documents: dict[str, dict[str, Any]] = {}
    for chunk_id in vector_store.id_order:
        metadata = vector_store.metadatas_map.get(chunk_id, {})
        doc_id = str(metadata.get("doc_id", ""))
        entry = documents.setdefault(
            doc_id, {"doc_id": doc_id, "source": metadata.get("source", ""), "chunks": 0}
        )
        entry["chunks"] += 1
    return list(documents.values())


def build_server(settings: Settings | None = None) -> MCPServer[None]:
    """Build the MCP server from ``settings`` (default: the process-wide settings).

    Like ``create_app``, explicit settings become the process-wide settings,
    because the pipeline modules read those.
    """
    if settings is None:
        settings = get_settings()
    elif settings is not get_settings():
        set_settings(settings)
    active = settings
    server: MCPServer[None] = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS, version=__version__)

    @server.tool(
        description="Index one document under the document root, replacing the knowledge base.",
        annotations=REPLACES_INDEX,
    )
    async def ingest_document(path: str) -> dict[str, Any]:
        target = resolve_document(path, active)
        report = await anyio.to_thread.run_sync(ingest_files, [SourceFile(path=target, name=target.name)])
        result = report.files[0]
        if not result.ok:
            raise ToolError(f"{target.name}: {result.error}")
        logger.info("MCP ingest of %s: %d chunk(s)", target.name, result.chunks)
        return {
            "status": "success",
            "file": target.name,
            "chunks": result.chunks,
            "total_chunks": report.total_chunks,
        }

    @server.tool(
        description="The indexed chunks that best match the query, best first, with scores.",
        annotations=READ_ONLY,
    )
    async def search(query: str, top_k: int | None = None) -> dict[str, Any]:
        if not query.strip():
            raise ToolError("query must not be empty")
        if top_k is not None and not 1 <= top_k <= MAX_TOP_K:
            raise ToolError(f"top_k must be between 1 and {MAX_TOP_K}")
        if vector_store.total_chunks == 0:
            raise ToolError("knowledge base is empty; call ingest_document first")
        ranked = await anyio.to_thread.run_sync(search_chunks, query, top_k)
        results = [
            {
                "id": chunk_id,
                "score": round(float(data["score"]), 4),
                "content": data["content"],
                "source": data["metadata"].get("source"),
                "doc_id": data["metadata"].get("doc_id"),
            }
            for chunk_id, data in ranked
        ]
        return {"query": query, "results": results}

    @server.tool(description="The indexed documents with their chunk counts.", annotations=READ_ONLY)
    async def list_documents() -> dict[str, Any]:
        documents = indexed_documents()
        return {"documents": documents, "total_chunks": vector_store.total_chunks}

    @server.tool(
        description="Version, provider names, index size and embedding cache counters.", annotations=READ_ONLY
    )
    async def health() -> dict[str, Any]:
        return health_snapshot(active).model_dump()

    return server


@asynccontextmanager
async def in_process_session(server: MCPServer[None]) -> AsyncIterator[ClientSession]:
    """An initialized client session talking to ``server`` over in-memory streams.

    No subprocess and no network: the tests and the latency example use it.
    ``MCPServer`` exposes no public handle on its protocol server, so this
    reaches for the attribute ``run_stdio_async`` uses.
    """
    protocol_server = server._lowlevel_server
    async with (
        create_client_server_memory_streams() as (client_streams, server_streams),
        anyio.create_task_group() as task_group,
    ):
        task_group.start_soon(
            protocol_server.run,
            server_streams[0],
            server_streams[1],
            protocol_server.create_initialization_options(),
        )
        async with ClientSession(client_streams[0], client_streams[1]) as session:
            await session.initialize()
            yield session
        task_group.cancel_scope.cancel()


def main() -> None:
    """``recallmcp-mcp``: serve the tools on stdio."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    logger.info("Starting MCP server on stdio; document root %s", settings.mcp_document_root.resolve())
    build_server(settings).run("stdio")


if __name__ == "__main__":
    main()
