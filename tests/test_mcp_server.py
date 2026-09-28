"""The MCP server: tools over an in-process client session, hash embedder, no subprocess."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent

from ragsvc import __version__, api
from ragsvc.config import get_settings
from ragsvc.core.bm25_index import bm25_manager
from ragsvc.core.vector_store import vector_store
from ragsvc.mcp_server import build_server, in_process_session, indexed_documents, resolve_document
from tests.conftest import make_settings

pytestmark = pytest.mark.anyio

TOOL_NAMES = {"ingest_document", "search", "list_documents", "health"}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def fixed_provider_and_empty_indexes(monkeypatch):
    monkeypatch.setattr(api, "detect_default_provider", lambda: "ollama")
    vector_store.clear()
    bm25_manager.clear()
    yield
    vector_store.clear()
    bm25_manager.clear()


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    (tmp_path / "bm25.md").write_text("BM25 supports exact keyword retrieval.", encoding="utf-8")
    (tmp_path / "faiss.md").write_text("FAISS supports dense vector retrieval.", encoding="utf-8")
    (tmp_path / "empty.md").write_text("   \n", encoding="utf-8")
    (tmp_path / "notes.csv").write_text("a,b\n", encoding="utf-8")
    return tmp_path


def payload(result: CallToolResult) -> dict[str, Any]:
    assert not result.is_error, result.content
    assert result.structured_content is not None
    text = result.content[0]
    assert isinstance(text, TextContent)
    assert json.loads(text.text) == result.structured_content
    return dict(result.structured_content)


def error_text(result: CallToolResult) -> str:
    """The tool's message; the SDK prefixes it with ``Error executing tool <name>: ``."""
    assert result.is_error
    text = result.content[0]
    assert isinstance(text, TextContent)
    prefix, _, message = text.text.partition(": ")
    assert prefix.startswith("Error executing tool ")
    return message


async def test_server_lists_the_four_tools_with_schemas(docs: Path):
    server = build_server(make_settings(mcp_document_root=docs))
    async with in_process_session(server) as session:
        listed = await session.list_tools()

    tools = {tool.name: tool for tool in listed.tools}
    assert set(tools) == TOOL_NAMES
    assert tools["ingest_document"].input_schema["required"] == ["path"]
    assert set(tools["search"].input_schema["properties"]) == {"query", "top_k"}
    assert all(tool.description for tool in tools.values())


async def test_ingest_then_search_and_list_over_the_session(docs: Path):
    server = build_server(make_settings(mcp_document_root=docs, rerank_method="none"))
    async with in_process_session(server) as session:
        ingested = payload(await session.call_tool("ingest_document", {"path": "bm25.md"}))
        found = payload(await session.call_tool("search", {"query": "exact keyword retrieval", "top_k": 1}))
        listed = payload(await session.call_tool("list_documents"))

    assert ingested == {"status": "success", "file": "bm25.md", "chunks": 1, "total_chunks": 1}
    assert found["query"] == "exact keyword retrieval"
    (hit,) = found["results"]
    assert hit["id"] == "doc_1_chunk_0"
    assert hit["content"] == "BM25 supports exact keyword retrieval."
    assert hit == {**hit, "source": "bm25.md", "doc_id": "doc_1"}
    # One chunk: BM25 has no idf signal, so the score is the dense weight alone.
    assert hit["score"] == 0.7
    assert listed == {"documents": [{"doc_id": "doc_1", "source": "bm25.md", "chunks": 1}], "total_chunks": 1}


async def test_ingest_replaces_the_knowledge_base(docs: Path):
    server = build_server(make_settings(mcp_document_root=docs, rerank_method="none"))
    async with in_process_session(server) as session:
        payload(await session.call_tool("ingest_document", {"path": "bm25.md"}))
        payload(await session.call_tool("ingest_document", {"path": str(docs / "faiss.md")}))
        listed = payload(await session.call_tool("list_documents"))

    assert [doc["source"] for doc in listed["documents"]] == ["faiss.md"]


async def test_health_tool_matches_the_http_health_endpoint(docs: Path):
    server = build_server(make_settings(mcp_document_root=docs))
    async with in_process_session(server) as session:
        health = payload(await session.call_tool("health"))

    assert health["version"] == __version__
    assert health["providers"]["embedding"] == "hash"
    assert health["index"] == {"ready": False, "chunks": 0, "type": None}
    assert health == api.health_snapshot(get_settings()).model_dump()


async def test_search_errors_are_reported_as_tool_errors(docs: Path):
    server = build_server(make_settings(mcp_document_root=docs, rerank_method="none"))
    async with in_process_session(server) as session:
        empty_index = await session.call_tool("search", {"query": "anything"})
        payload(await session.call_tool("ingest_document", {"path": "bm25.md"}))
        blank = await session.call_tool("search", {"query": "   "})
        too_many = await session.call_tool("search", {"query": "x", "top_k": 51})
        unknown = await session.call_tool("search", {"quer": "x"})

    assert error_text(empty_index) == "knowledge base is empty; call ingest_document first"
    assert error_text(blank) == "query must not be empty"
    assert error_text(too_many) == "top_k must be between 1 and 50"
    assert "query" in error_text(unknown)


async def test_ingest_rejects_files_outside_the_root_unsupported_and_empty(docs: Path, tmp_path: Path):
    outside = tmp_path.parent / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    server = build_server(make_settings(mcp_document_root=docs, rerank_method="none"))
    async with in_process_session(server) as session:
        escaped = await session.call_tool("ingest_document", {"path": str(outside)})
        traversal = await session.call_tool("ingest_document", {"path": "../outside.md"})
        missing = await session.call_tool("ingest_document", {"path": "nope.md"})
        unsupported = await session.call_tool("ingest_document", {"path": "notes.csv"})
        empty = await session.call_tool("ingest_document", {"path": "empty.md"})
        listed = payload(await session.call_tool("list_documents"))

    assert "outside the document root" in error_text(escaped)
    assert "outside the document root" in error_text(traversal)
    assert "is not a file under" in error_text(missing)
    assert error_text(unsupported).startswith("unsupported file format '.csv'")
    assert error_text(empty) == "empty.md: document is empty or has no extractable text"
    assert listed == {"documents": [], "total_chunks": 0}


def test_resolve_document_enforces_the_upload_size_cap(docs: Path):
    settings = make_settings(mcp_document_root=docs, max_upload_mb=1)
    big = docs / "big.txt"
    big.write_text("x" * (1024 * 1024 + 1), encoding="utf-8")

    assert resolve_document("bm25.md", settings) == (docs / "bm25.md").resolve()
    with pytest.raises(ToolError, match="over the limit"):
        resolve_document("big.txt", settings)


def test_indexed_documents_is_empty_before_ingestion():
    assert indexed_documents() == []


def test_build_server_installs_explicit_settings_process_wide(docs: Path):
    custom = make_settings(mcp_document_root=docs, hash_embedding_dimension=16)
    assert get_settings() is not custom

    server = build_server(custom)

    assert get_settings() is custom
    assert server.name == "recallmcp"
    assert server.version == __version__


def test_main_runs_the_stdio_transport(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        "ragsvc.mcp_server.MCPServer.run", lambda self, transport="stdio": calls.append(transport)
    )
    from ragsvc.mcp_server import main

    main()

    assert calls == ["stdio"]
