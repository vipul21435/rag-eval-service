"""REST API: FastAPI application exposing upload, ask and status endpoints."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import (
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    Provider,
    detect_default_provider,
    is_configured_api_key,
    resolve_provider,
)
from core.generator import KnowledgeBaseEmptyError, ProviderError, answer_question
from core.ingest import SourceFile, ingest_files
from core.vector_store import vector_store
from features.web_search import check_serpapi_key
from utils.network import is_port_available
from version import __version__

logger = logging.getLogger("rag-api")

CANDIDATE_PORTS = (17995, 17996, 17997, 17998, 17999)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("API starting; default LLM provider: %s", detect_default_provider())
    yield
    logger.info("API stopped")


app = FastAPI(
    title="rag-eval-service",
    description="Document question answering over local FAISS + BM25 hybrid retrieval",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)
    enable_web_search: bool = False
    provider: Provider | None = Field(default=None, description="ollama or openai; default is auto-detected")


class AnswerResponse(BaseModel):
    answer: str
    reasoning: str | None = Field(default=None, description="Reasoning emitted by thinking models, if any")
    sources: list[dict[str, Any]]
    metadata: dict[str, Any]


class FileProcessResult(BaseModel):
    status: str
    message: str
    file_info: dict[str, Any] | None = None


@app.post("/api/upload", response_model=FileProcessResult)
async def upload_file(file: Annotated[UploadFile, File(...)]) -> dict[str, Any]:
    """Index one document, replacing the current knowledge base."""
    filename = file.filename or "upload"
    suffix = os.path.splitext(filename)[1]
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        report = await asyncio.to_thread(ingest_files, [SourceFile(path=Path(tmp_path), name=filename)])
        result = report.files[0]
        message = (
            f"{filename}: indexed {result.chunks} chunk(s)" if result.ok else f"{filename}: {result.error}"
        )
        return {
            "status": "success" if result.ok else "error",
            "message": message,
            "file_info": {"filename": filename, "chunks": result.chunks},
        }
    except Exception as exc:
        logger.error("Document processing failed: %s", exc)
        raise HTTPException(500, f"Document processing failed: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest) -> dict[str, Any]:
    """Answer a question from the indexed documents (and optionally the web)."""
    provider = resolve_provider(req.provider)
    try:
        answer = await asyncio.to_thread(answer_question, req.question, req.enable_web_search, provider)
    except KnowledgeBaseEmptyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ProviderError as exc:
        logger.error("LLM provider failed: %s", exc)
        raise HTTPException(502, str(exc)) from exc
    except Exception as exc:
        logger.error("Question answering failed: %s", exc)
        raise HTTPException(500, f"Question answering failed: {exc}") from exc

    return {
        "answer": answer.text,
        "reasoning": answer.reasoning,
        "sources": answer.sources,
        "metadata": {
            "enable_web_search": req.enable_web_search,
            "provider": answer.provider,
            "conflict_detected": answer.conflict_detected,
        },
    }


@app.get("/api/status")
async def check_status() -> dict[str, Any]:
    return {
        "status": "healthy",
        "version": __version__,
        "default_provider": detect_default_provider(),
        "ollama_model": OLLAMA_MODEL,
        "openai_configured": is_configured_api_key(OPENAI_API_KEY),
        "openai_model": OPENAI_MODEL,
        "serpapi_configured": check_serpapi_key(),
        "vector_store_ready": vector_store.is_ready,
        "total_chunks": vector_store.total_chunks,
    }


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    port = next((p for p in CANDIDATE_PORTS if is_port_available(p)), CANDIDATE_PORTS[0])
    logger.info("Starting API on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
