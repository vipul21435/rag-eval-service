"""Embeddings: the process-wide embedding provider used by the retrieval pipeline.

The provider itself is built by ``ragsvc.embeddings.build_embedder`` from
the installed settings (``RAG_EMBEDDING_PROVIDER``): a sentence-transformers
model in production, deterministic feature hashing in tests and demos,
either one behind the on-disk cache. It is built on first use and rebuilt
when the settings object changes, so ``set_settings`` in a test or an
embedding program takes effect without an explicit reset.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ragsvc.config import Settings, get_settings
from ragsvc.embeddings import EmbeddingProvider, build_embedder

logger = logging.getLogger(__name__)

_lock = threading.Lock()
# The provider in use and the settings it was built from.
_current: tuple[Settings, EmbeddingProvider] | None = None


def get_embedder() -> EmbeddingProvider:
    """The provider for the installed settings, built on first use."""
    global _current
    settings = get_settings()
    current = _current
    if current is None or current[0] is not settings:
        with _lock:
            current = _current
            if current is None or current[0] is not settings:
                embedder = build_embedder(settings)
                logger.info("Embedding provider: %s (%s)", embedder.name, embedder.model_name)
                current = _current = (settings, embedder)
    return current[1]


def set_embedder(embedder: EmbeddingProvider | None) -> None:
    """Install ``embedder`` for the installed settings; ``None`` rebuilds from settings on next use."""
    global _current
    with _lock:
        _current = None if embedder is None else (get_settings(), embedder)


def encode_texts(texts: Sequence[str], show_progress: bool = False) -> NDArray[np.float32]:
    """Encode ``texts`` into a float32 array of shape (n_texts, dimension)."""
    del show_progress  # kept for callers of the previous signature
    return get_embedder().embed_documents(texts)


def encode_query(query: str) -> NDArray[np.float32]:
    """Encode a single query into a float32 array of shape (1, dimension)."""
    return get_embedder().embed_query(query)[np.newaxis, :]
