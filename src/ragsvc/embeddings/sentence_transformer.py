"""The production embedder: a sentence-transformers model loaded on first use.

Importing this module is cheap; ``sentence_transformers`` (and torch) are
imported inside :meth:`SentenceTransformerEmbedder._model`, and the model
weights are fetched from the Hugging Face Hub the first time a vector is
requested. The default model (``all-MiniLM-L6-v2``, 384 dimensions, about
80 MB) is English-oriented and fast on CPU; set ``RAG_EMBED_MODEL_NAME`` to
use another one.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ragsvc.embeddings.base import EmbeddingError, Matrix, Vector

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class SentenceTransformerEmbedder:
    """Wrap a sentence-transformers model behind the :class:`EmbeddingProvider` protocol."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None) -> None:
        self._model_name = model_name
        self._device = device
        self._instance: SentenceTransformer | None = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def loaded(self) -> bool:
        """True once the model weights are in memory."""
        return self._instance is not None

    @property
    def dimension(self) -> int:
        """The model's output size; loads the model when it has not been loaded yet."""
        dimension = self._model().get_sentence_embedding_dimension()
        if not dimension:
            raise EmbeddingError(f"model {self._model_name!r} does not report an embedding dimension")
        return int(dimension)

    def _model(self) -> SentenceTransformer:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._load()
        return self._instance

    def _load(self) -> SentenceTransformer:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed; install it or set RAG_EMBEDDING_PROVIDER=hash"
            ) from exc
        logger.info("Loading embedding model %s", self._model_name)
        try:
            model = SentenceTransformer(self._model_name, device=self._device)
        except Exception as exc:  # download, disk and framework errors alike
            raise EmbeddingError(f"could not load embedding model {self._model_name!r}: {exc}") from exc
        logger.info(
            "Embedding model %s loaded; dimension %s",
            self._model_name,
            model.get_sentence_embedding_dimension(),
        )
        return model

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        embeddings = self._model().encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> Vector:
        vector: Vector = self.embed_documents([text])[0]
        return vector
