"""Pluggable embedding providers.

- :class:`EmbeddingProvider` is the protocol every provider implements.
- :class:`HashEmbedder` is deterministic feature hashing: no model, no
  download, identical vectors everywhere. Tests and demos use it.
- :class:`SentenceTransformerEmbedder` wraps a sentence-transformers model
  loaded on first use; it is the production default.
- :class:`EmbeddingCache` and :class:`CachedEmbedder` keep vectors on disk
  in SQLite, keyed by provider, model and text hash.

:func:`build_embedder` assembles the provider named by the settings
(``RAG_EMBEDDING_PROVIDER``) behind the cache when it is enabled. The
process-wide instance used by the retrieval pipeline lives in
``ragsvc.core.embeddings``.
"""

from __future__ import annotations

from ragsvc.config import Settings
from ragsvc.embeddings.base import EmbeddingError, EmbeddingProvider, Matrix, Vector
from ragsvc.embeddings.cache import CachedEmbedder, CacheStats, EmbeddingCache, text_key
from ragsvc.embeddings.hashing import HashEmbedder
from ragsvc.embeddings.sentence_transformer import SentenceTransformerEmbedder

__all__ = [
    "CacheStats",
    "CachedEmbedder",
    "EmbeddingCache",
    "EmbeddingError",
    "EmbeddingProvider",
    "HashEmbedder",
    "Matrix",
    "SentenceTransformerEmbedder",
    "Vector",
    "build_embedder",
    "text_key",
]


def build_embedder(settings: Settings) -> EmbeddingProvider:
    """The provider ``settings`` describe, wrapped in the on-disk cache when enabled.

    Building is cheap: a sentence-transformers model is not loaded until
    the first text is embedded.
    """
    provider: EmbeddingProvider
    if settings.embedding_provider == "hash":
        provider = HashEmbedder(dimension=settings.hash_embedding_dimension)
    else:
        provider = SentenceTransformerEmbedder(model_name=settings.embed_model_name)
    if settings.embedding_cache_enabled:
        return CachedEmbedder(provider, EmbeddingCache(settings.embedding_cache_path))
    return provider
