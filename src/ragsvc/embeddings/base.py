"""The embedding provider protocol and the errors providers raise.

An embedding provider maps text to fixed-size float32 vectors. The service
talks to providers only through :class:`EmbeddingProvider`, so the vector
store, the ingestion pipeline and the evaluation harness do not care whether
vectors come from a neural model or from feature hashing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float32]
Matrix = NDArray[np.float32]


class EmbeddingError(RuntimeError):
    """An embedding provider cannot be loaded or failed to produce vectors."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Anything that turns text into vectors of a fixed ``dimension``.

    ``name`` identifies the provider family (``hash``,
    ``sentence-transformers``) and ``model_name`` the concrete model; the
    two together key the on-disk cache, so vectors from different models are
    never mixed.
    """

    @property
    def name(self) -> str:
        """Provider family, stable across releases."""
        ...

    @property
    def model_name(self) -> str:
        """Concrete model identifier (a Hub name, or ``hash-<dimension>``)."""
        ...

    @property
    def dimension(self) -> int:
        """Length of every vector this provider returns."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        """Embed ``texts`` into a float32 array of shape ``(len(texts), dimension)``."""
        ...

    def embed_query(self, text: str) -> Vector:
        """Embed one query into a float32 array of shape ``(dimension,)``."""
        ...
