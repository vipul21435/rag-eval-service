"""REST API: FastAPI application exposing upload, ask and status endpoints.

Security defaults are local-first: the server binds the loopback interface,
sends no CORS headers, caps upload size and, when ``API_TOKEN`` is set,
requires a bearer token on every ``/api`` request. See ``config.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import tempfile
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import (
    API_HOST,
    API_PORT,
    API_TOKEN,
    CORS_ALLOW_ORIGINS,
    MAX_UPLOAD_MB,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    Provider,
    detect_default_provider,
    is_configured_api_key,
    resolve_provider,
)
from core.document_loader import SUPPORTED_EXTENSIONS, describe_supported_formats
from core.generator import KnowledgeBaseEmptyError, ProviderError, answer_question
from core.ingest import SourceFile, ingest_files
from core.vector_store import vector_store
from features.web_search import check_serpapi_key
from utils.network import is_port_available
from version import __version__

logger = logging.getLogger("rag-api")

CANDIDATE_PORTS = (17995, 17996, 17997, 17998, 17999)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
_UPLOAD_CHUNK_BYTES = 1024 * 1024


# --- Authentication ---------------------------------------------------------


async def require_api_token(authorization: Annotated[str | None, Header()] = None) -> None:
    """Reject the request unless it carries the configured bearer token.

    A no-op when ``API_TOKEN`` is unset (the local-first default).
    """
    expected = API_TOKEN
    if expected is None:
        return
    scheme, _, token = (authorization or "").partition(" ")
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


async def _spool_upload(file: UploadFile, suffix: str) -> str:
    """Copy ``file`` to a temp file in chunks; 413 once it exceeds the limit."""
    written = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        try:
            while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"file exceeds the upload limit of {MAX_UPLOAD_BYTES} bytes")
                tmp.write(chunk)
        except BaseException:
            tmp.close()
            os.unlink(tmp_path)
            raise
    return tmp_path


@router.post("/upload", response_model=FileProcessResult)
async def upload_file(file: Annotated[UploadFile, File(...)]) -> dict[str, Any]:
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

    tmp_path = await _spool_upload(file, suffix)
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


# --- Application ------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("API starting; default LLM provider: %s", detect_default_provider())
    if API_TOKEN is None and API_HOST not in ("127.0.0.1", "localhost", "::1"):
        logger.warning("API_HOST=%s without API_TOKEN: the API is reachable without authentication", API_HOST)
    yield
    logger.info("API stopped")


def create_app(cors_origins: Sequence[str] | None = None) -> FastAPI:
    """Build the application.

    ``cors_origins`` lists the browser origins allowed to call the API; it
    defaults to ``CORS_ALLOW_ORIGINS``. With no origins, no CORS middleware is
    installed and browsers block cross-origin reads. Credentials are never
    allowed, so an allowed origin cannot ride on the operator's cookies.
    """
    application = FastAPI(
        title="rag-eval-service",
        description="Document question answering over local FAISS + BM25 hybrid retrieval",
        version=__version__,
        lifespan=lifespan,
    )
    origins = list(CORS_ALLOW_ORIGINS if cors_origins is None else cors_origins)
    if origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )
    application.include_router(router)
    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    port = API_PORT or next((p for p in CANDIDATE_PORTS if is_port_available(p)), CANDIDATE_PORTS[0])
    logger.info("Starting API on %s:%d", API_HOST, port)
    uvicorn.run(app, host=API_HOST, port=port)
