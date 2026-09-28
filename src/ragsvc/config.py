"""Configuration: environment loading, provider settings and retrieval hyperparameters.

Every setting has a local default, so the service runs with no ``.env`` file,
no API keys and no network access beyond the first model download. Values
come from the process environment, then from ``.env`` next to this file.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, get_args

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Relative to the working directory, like the rest of the service's file paths.
ENV_PATH = Path(".env")
# Existing environment variables take precedence over .env values.
load_dotenv(ENV_PATH)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


# --- LLM providers -----------------------------------------------------------
# "ollama": a local Ollama server. "openai": any OpenAI-compatible Chat
# Completions endpoint (OpenAI, vLLM, LM Studio, hosted providers).
Provider = Literal["ollama", "openai"]
PROVIDER_CHOICES: tuple[Provider, ...] = get_args(Provider)

LLM_PROVIDER = os.getenv("LLM_PROVIDER") or None

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- Retrieval models (downloaded from the Hugging Face Hub on first use) ----
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
RERANK_METHOD = os.getenv("RERANK_METHOD", "cross_encoder")  # cross_encoder | llm | none
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Retrieval hyperparameters ----------------------------------------------
CHUNK_SIZE = _env_int("CHUNK_SIZE", 400)  # characters per chunk
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 40)  # characters shared by adjacent chunks
HYBRID_ALPHA = _env_float("HYBRID_ALPHA", 0.7)  # weight of dense vs. BM25 scores (0-1)
RETRIEVAL_TOP_K = _env_int("RETRIEVAL_TOP_K", 10)  # candidates per retriever
RERANK_TOP_K = _env_int("RERANK_TOP_K", 5)  # candidates kept after reranking
MAX_RETRIEVAL_ITERATIONS = _env_int("MAX_RETRIEVAL_ITERATIONS", 3)  # recursive retrieval rounds

# --- Optional web search -----------------------------------------------------
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SEARCH_ENGINE = "google"

# --- HTTP API ---------------------------------------------------------------
# The API has no authentication unless API_TOKEN is set, so it listens on the
# loopback interface by default. Set API_HOST=0.0.0.0 to expose it, ideally
# together with API_TOKEN.
API_HOST = os.getenv("API_HOST", "127.0.0.1")
# Unset: the first free port in api.CANDIDATE_PORTS is used.
API_PORT = int(os.environ["API_PORT"]) if os.getenv("API_PORT") else None


def _env_csv(name: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in os.getenv(name, "").split(",") if item.strip())


# Browser origins allowed to call the API (comma-separated). Empty means no
# CORS headers at all: pages on other origins cannot read responses.
CORS_ALLOW_ORIGINS = _env_csv("CORS_ALLOW_ORIGINS")
# Largest document /api/upload accepts.
MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 50)
# When set, every /api request must carry "Authorization: Bearer <token>".
API_TOKEN = os.getenv("API_TOKEN") or None


def is_configured_api_key(api_key: str | None) -> bool:
    """True when ``api_key`` is a real value rather than empty or a ``Your...`` placeholder."""
    return bool(api_key and api_key.strip() and not api_key.strip().startswith("Your"))


def ollama_available(base_url: str = OLLAMA_BASE_URL, timeout: float = 2.0) -> bool:
    """True when an Ollama server answers at ``base_url``."""
    try:
        return requests.get(f"{base_url}/api/tags", timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def choose_default_provider(
    explicit: str | None,
    ollama_reachable: bool,
    openai_key: str | None,
) -> Provider:
    """Pick the provider to use when a request does not name one.

    An explicit ``LLM_PROVIDER`` wins. Otherwise local Ollama is preferred,
    then an OpenAI-compatible endpoint with a configured key. With nothing
    available the choice stays ``ollama`` and the call fails with a clear error.
    """
    if explicit:
        if explicit not in PROVIDER_CHOICES:
            raise ValueError(f"LLM_PROVIDER must be one of {PROVIDER_CHOICES}, got {explicit!r}")
        return explicit
    if ollama_reachable:
        return "ollama"
    if is_configured_api_key(openai_key):
        return "openai"
    return "ollama"


@lru_cache(maxsize=1)
def detect_default_provider() -> Provider:
    """Resolve the default provider once, probing Ollama only when needed."""
    if LLM_PROVIDER:
        provider = choose_default_provider(LLM_PROVIDER, False, OPENAI_API_KEY)
        logger.info("LLM provider set explicitly: %s", provider)
        return provider

    reachable = ollama_available()
    provider = choose_default_provider(None, reachable, OPENAI_API_KEY)
    if reachable:
        logger.info("Ollama detected at %s; using local model %s", OLLAMA_BASE_URL, OLLAMA_MODEL)
    elif provider == "openai":
        logger.info("OPENAI_API_KEY configured; using %s at %s", OPENAI_MODEL, OPENAI_BASE_URL)
    else:
        logger.warning("No LLM backend available: start Ollama at %s or set OPENAI_API_KEY", OLLAMA_BASE_URL)
    return provider


def resolve_provider(requested: str | None) -> Provider:
    """Validate a provider named in a request, or fall back to the detected default."""
    if requested is None:
        return detect_default_provider()
    if requested not in PROVIDER_CHOICES:
        raise ValueError(f"provider must be one of {PROVIDER_CHOICES}, got {requested!r}")
    return requested
