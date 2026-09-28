"""Embeddings: map text into a vector space with a sentence-transformers model.

Semantically similar texts end up close together, which is what the FAISS
index searches on. The default model (``all-MiniLM-L6-v2``, 384 dimensions,
about 80 MB) is English-oriented and fast on CPU; it is downloaded from the
Hugging Face Hub on first use. Override it with ``RAG_EMBED_MODEL_NAME``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ragsvc.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embed_model() -> SentenceTransformer:
    """Load the embedding model once and cache the instance."""
    from sentence_transformers import SentenceTransformer

    model_name = get_settings().embed_model_name
    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    logger.info("Embedding model loaded; dimension: %s", model.get_sentence_embedding_dimension())
    return model


def encode_texts(texts: Sequence[str], show_progress: bool = False) -> NDArray[np.float32]:
    """Encode ``texts`` into a float32 array of shape (n_texts, dimension)."""
    model = get_embed_model()
    embeddings = model.encode(list(texts), show_progress_bar=show_progress)
    return np.asarray(embeddings, dtype=np.float32)


def encode_query(query: str) -> NDArray[np.float32]:
    """Encode a single query into a float32 array of shape (1, dimension)."""
    model = get_embed_model()
    embedding = model.encode([query])
    return np.asarray(embedding, dtype=np.float32)
