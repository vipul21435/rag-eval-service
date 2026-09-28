"""Typed settings: providers, models, retrieval hyperparameters and the HTTP API.

Settings are read from environment variables prefixed with ``RAG_``
(``RAG_CHUNK_SIZE``, ``RAG_OLLAMA_MODEL``, ...) and, below them, from a
``.env`` file in the working directory. Every setting has a local default,
so the service runs with no ``.env`` file, no API keys and no network access
beyond the first model download. Third-party credentials are also accepted
under their conventional unprefixed names (``OPENAI_API_KEY``,
``OPENAI_BASE_URL``, ``SERPAPI_KEY``); the ``RAG_`` name wins when both are
set.

Validation happens once, when the settings are loaded: an out-of-range
value or an unknown provider name fails at startup with a message naming
the variable, not later inside a request.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Literal, get_args

import requests
from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

logger = logging.getLogger(__name__)

# "ollama": a local Ollama server. "openai": any OpenAI-compatible Chat
# Completions endpoint (OpenAI, vLLM, LM Studio, hosted providers).
Provider = Literal["ollama", "openai"]
PROVIDER_CHOICES: tuple[Provider, ...] = get_args(Provider)

RerankMethod = Literal["cross_encoder", "llm", "none"]

# "sentence-transformers": a neural model from the Hugging Face Hub (the
# production default). "hash": deterministic feature hashing with no model,
# used by the tests and the demo; see ragsvc.embeddings.
EmbeddingProviderName = Literal["hash", "sentence-transformers"]
EMBEDDING_PROVIDER_CHOICES: tuple[EmbeddingProviderName, ...] = get_args(EmbeddingProviderName)

LogFormat = Literal["json", "text"]

# Placeholder values such as ``Your_OPENAI_API_KEY`` left over from an example
# file are treated as unset.
_PLACEHOLDER_PREFIX = "Your"


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return value


class Settings(BaseSettings):
    """Every configurable value of the service, validated on load."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- LLM providers ------------------------------------------------------
    llm_provider: Provider | None = Field(
        default=None,
        description="Force a provider instead of auto-detecting (running Ollama, then an OpenAI key).",
    )
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("RAG_OPENAI_API_KEY", "OPENAI_API_KEY")
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("RAG_OPENAI_BASE_URL", "OPENAI_BASE_URL"),
    )
    openai_model: str = "gpt-4o-mini"

    # --- Embeddings ---------------------------------------------------------
    embedding_provider: EmbeddingProviderName = Field(
        default="sentence-transformers",
        description="sentence-transformers (neural, downloaded on first use) or hash (deterministic).",
    )
    embed_model_name: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence-transformers model, from the Hugging Face Hub."
    )
    hash_embedding_dimension: int = Field(
        default=256, ge=8, le=65536, description="Vector size of the hash provider."
    )
    embedding_cache_enabled: bool = Field(default=True, description="Keep computed vectors in SQLite.")
    embedding_cache_path: Path = Field(
        default=Path(".cache/recallmcp/embeddings.sqlite3"),
        description="SQLite file of the embedding cache; created on first use.",
    )

    # --- Reranking (model downloaded from the Hugging Face Hub on first use)
    rerank_method: RerankMethod = "cross_encoder"
    rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Retrieval hyperparameters ------------------------------------------
    chunk_size: int = Field(default=400, ge=1, description="Characters per chunk.")
    chunk_overlap: int = Field(default=40, ge=0, description="Characters shared by adjacent chunks.")
    hybrid_alpha: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight of dense vs. BM25 scores.")
    retrieval_top_k: int = Field(default=10, ge=1, description="Candidates per retriever.")
    rerank_top_k: int = Field(default=5, ge=1, description="Candidates kept after reranking.")
    max_retrieval_iterations: int = Field(default=3, ge=1, description="Recursive retrieval rounds.")

    # --- Optional web search ------------------------------------------------
    serpapi_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("RAG_SERPAPI_KEY", "SERPAPI_KEY")
    )
    search_engine: str = "google"

    # --- HTTP API -----------------------------------------------------------
    # The API has no authentication unless api_token is set, so it listens on
    # the loopback interface by default.
    api_host: str = "127.0.0.1"
    api_port: int | None = Field(
        default=None, ge=1, le=65535, description="Unset: first free candidate port."
    )
    # Browser origins allowed to call the API. Empty means no CORS headers at
    # all: pages on other origins cannot read responses.
    cors_allow_origins: Annotated[tuple[str, ...], NoDecode] = ()
    max_upload_mb: int = Field(default=50, ge=1)
    # When set, every /api request must carry "Authorization: Bearer <token>".
    api_token: SecretStr | None = None

    # --- MCP server ---------------------------------------------------------
    # The MCP ingest_document tool only reads files under this directory.
    mcp_document_root: Path = Field(
        default=Path("."), description="Directory the MCP server may ingest documents from."
    )

    # --- Logging ------------------------------------------------------------
    log_level: str = Field(default="INFO", description="Root log level name (DEBUG, INFO, WARNING, ...).")
    log_format: LogFormat = Field(default="json", description="json: one object per line; text: readable.")

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        name = value.strip().upper()
        if name not in logging.getLevelNamesMapping():
            raise ValueError(f"unknown log level {value!r}")
        return name

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("ollama_base_url", "openai_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.strip().rstrip("/")

    @field_validator("api_token", "openai_api_key", "serpapi_key")
    @classmethod
    def _drop_blank_or_placeholder_secrets(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None or not is_configured_secret(value):
            return None
        return value

    @model_validator(mode="after")
    def _overlap_must_leave_room_for_new_text(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than chunk_size ({self.chunk_size})"
            )
        return self

    # --- Derived values -----------------------------------------------------

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def openai_configured(self) -> bool:
        return self.openai_api_key is not None

    @property
    def serpapi_configured(self) -> bool:
        return self.serpapi_key is not None


def is_configured_secret(secret: SecretStr | str | None) -> bool:
    """True when ``secret`` is a real value rather than empty or a ``Your...`` placeholder."""
    if secret is None:
        return False
    value = secret.get_secret_value() if isinstance(secret, SecretStr) else secret
    value = value.strip()
    return bool(value) and not value.startswith(_PLACEHOLDER_PREFIX)


# --- Process-wide access ----------------------------------------------------

_settings: Settings | None = None
# The default provider resolved for the current settings; probing Ollama is
# done once, not on every request.
_default_provider: Provider | None = None


def get_settings() -> Settings:
    """The process-wide settings, loaded from the environment on first use."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(settings: Settings | None) -> None:
    """Install ``settings`` process-wide; ``None`` reloads from the environment on next use.

    Used by tests and by programs embedding the service. Cached decisions
    derived from the previous settings, such as the default provider, are
    discarded.
    """
    global _settings, _default_provider
    _settings = settings
    _default_provider = None


# --- Provider selection -----------------------------------------------------


def ollama_available(base_url: str, timeout: float = 2.0) -> bool:
    """True when an Ollama server answers at ``base_url``."""
    try:
        return requests.get(f"{base_url}/api/tags", timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def choose_default_provider(
    explicit: Provider | None, ollama_reachable: bool, openai_configured: bool
) -> Provider:
    """Pick the provider to use when a request does not name one.

    An explicit ``RAG_LLM_PROVIDER`` wins. Otherwise local Ollama is
    preferred, then an OpenAI-compatible endpoint with a configured key. With
    nothing available the choice stays ``ollama`` and the call fails with a
    clear error.
    """
    if explicit is not None:
        return explicit
    if ollama_reachable:
        return "ollama"
    if openai_configured:
        return "openai"
    return "ollama"


def detect_default_provider() -> Provider:
    """Resolve the default provider once per settings, probing Ollama only when needed."""
    global _default_provider
    if _default_provider is None:
        _default_provider = _detect_default_provider(get_settings())
    return _default_provider


def _detect_default_provider(settings: Settings) -> Provider:
    if settings.llm_provider is not None:
        logger.info("LLM provider set explicitly: %s", settings.llm_provider)
        return settings.llm_provider

    reachable = ollama_available(settings.ollama_base_url)
    provider = choose_default_provider(None, reachable, settings.openai_configured)
    if reachable:
        logger.info("Ollama detected at %s; using model %s", settings.ollama_base_url, settings.ollama_model)
    elif provider == "openai":
        logger.info(
            "OpenAI API key configured; using %s at %s", settings.openai_model, settings.openai_base_url
        )
    else:
        logger.warning(
            "No LLM backend available: start Ollama at %s or set OPENAI_API_KEY", settings.ollama_base_url
        )
    return provider


def resolve_provider(requested: str | None) -> Provider:
    """Validate a provider named in a request, or fall back to the detected default."""
    if requested is None:
        return detect_default_provider()
    if requested not in PROVIDER_CHOICES:
        raise ValueError(f"provider must be one of {PROVIDER_CHOICES}, got {requested!r}")
    return requested
