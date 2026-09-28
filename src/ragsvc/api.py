"""REST API: FastAPI application exposing upload, ask and status endpoints.

Security defaults are local-first: the server binds the loopback interface,
sends no CORS headers, caps upload size and, when ``RAG_API_TOKEN`` is set,
requires a bearer token on every ``/api`` request. See ``ragsvc.config``.

``create_app`` is the application factory; there is no module-level app so
that settings are read when the application is built, not when the module
is imported (``uvicorn --factory ragsvc.api:create_app``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ragsvc import __version__
from ragsvc.config import Provider, Settings, detect_default_provider, get_settings, resolve_provider
from ragsvc.core.document_loader import SUPPORTED_EXTENSIONS, describe_supported_formats
from ragsvc.core.generator import KnowledgeBaseEmptyError, ProviderError, answer_question
from ragsvc.core.ingest import SourceFile, ingest_files
from ragsvc.core.vector_store import vector_store

logger = logging.getLogger("rag-api")

CANDIDATE_PORTS = (17995, 17996, 17997, 17998, 17999)
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


# --- Dependencies -------------------------------------------------------------


def app_settings(request: Request) -> Settings:
    """The settings the application was built with."""
    settings: Settings = request.app.state.settings
    return settings


async def require_api_token(
    settings: Annotated[Settings, Depends(app_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Reject the request unless it carries the configured bearer token.

    A no-op when no API token is configured (the local-first default).
    """
    if settings.api_token is None:
        return
    scheme, _, token = (authorization or "").partition(" ")
    expected = settings.api_token.get_secret_value()
    if scheme.lower() != "bearer" or not secrets.compare_digest(token.strip(), expected):
        raise HTTPException(401, "missing or invalid API token", headers={"WWW-Authenticate": "Bearer"})


# --- Schemas ----------------------------------------------------------------


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


# --- Routes -----------------------------------------------------------------

router = APIRouter(prefix="/api", dependencies=[Depends(require_api_token)])


async def _spool_upload(file: UploadFile, suffix: str, limit: int) -> str:
    """Copy ``file`` to a temp file in chunks; 413 once it exceeds ``limit`` bytes."""
    written = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        try:
            while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"file exceeds the upload limit of {limit} bytes")
                tmp.write(chunk)
        except BaseException:
            tmp.close()
            os.unlink(tmp_path)
            raise
    return tmp_path


@router.post("/upload", response_model=FileProcessResult)
async def upload_file(
    file: Annotated[UploadFile, File(...)],
    settings: Annotated[Settings, Depends(app_settings)],
) -> dict[str, Any]:
    """Index one document, replacing the current knowledge base.

    A document that yields no text leaves the previous knowledge base in
    place and is reported with ``status: error``.
    """
    filename = file.filename or "upload"
    suffix = os.path.splitext(filename)[1]
    if suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            415, f"unsupported file format {suffix or '(none)'!r}; supported: {describe_supported_formats()}"
        )

    tmp_path = await _spool_upload(file, suffix, settings.max_upload_bytes)
    try:
        report = await asyncio.to_thread(ingest_files, [SourceFile(path=Path(tmp_path), name=filename)])
    except Exception as exc:
        logger.error("Document processing failed: %s", exc)
        raise HTTPException(500, f"Document processing failed: {exc}") from exc
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    result = report.files[0]
    message = f"{filename}: indexed {result.chunks} chunk(s)" if result.ok else f"{filename}: {result.error}"
    return {
        "status": "success" if result.ok else "error",
        "message": message,
        "file_info": {"filename": filename, "chunks": result.chunks},
    }


@router.post("/ask", response_model=AnswerResponse)
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


@router.get("/status")
async def check_status(settings: Annotated[Settings, Depends(app_settings)]) -> dict[str, Any]:
    return {
        "status": "healthy",
        "version": __version__,
        "default_provider": detect_default_provider(),
        "ollama_model": settings.ollama_model,
        "openai_configured": settings.openai_configured,
        "openai_model": settings.openai_model,
        "serpapi_configured": settings.serpapi_configured,
        "vector_store_ready": vector_store.is_ready,
        "total_chunks": vector_store.total_chunks,
    }


# --- Application ------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    logger.info("API starting; default LLM provider: %s", detect_default_provider())
    if settings.api_token is None and settings.api_host not in _LOOPBACK_HOSTS:
        logger.warning(
            "RAG_API_HOST=%s without RAG_API_TOKEN: the API is reachable without authentication",
            settings.api_host,
        )
    yield
    logger.info("API stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application from ``settings`` (default: the process-wide settings).

    ``settings.cors_allow_origins`` lists the browser origins allowed to call
    the API. With no origins, no CORS middleware is installed and browsers
    block cross-origin reads. Credentials are never allowed, so an allowed
    origin cannot ride on the operator's cookies.
    """
    if settings is None:
        settings = get_settings()
    application = FastAPI(
        title="rag-eval-service",
        description="Document question answering over local FAISS + BM25 hybrid retrieval",
        version=__version__,
        lifespan=lifespan,
    )
    application.state.settings = settings
    if settings.cors_allow_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_allow_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )
    application.include_router(router)
    return application
