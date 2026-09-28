"""Vector store: FAISS index management.

FAISS index types trade accuracy for speed:

- ``IndexFlatL2``: exhaustive, exact search; fine for small collections.
- ``IndexIVFFlat``: clusters vectors first and searches a few clusters.
- ``IndexIVFPQ``: adds product quantization for very large collections.

``AutoFaissIndex`` picks one of them from the number of vectors.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from faiss import Index, IndexFlatL2, IndexIVF, IndexIVFFlat, IndexIVFPQ
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

Metadata = dict[str, Any]
SearchResult = tuple[list[str], list[str], list[Metadata]]


class AutoFaissIndex:
    """FAISS index wrapper that selects the index type from the dataset size."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.index: Index | None = None
        self.index_type: str | None = None
        self.nlist: int | None = None
        self.m: int | None = None
        self.nprobe: int | None = None
        self.small_dataset_threshold = 10_000
        self.medium_dataset_threshold = 100_000

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal) if self.index is not None else 0

    def select_index_type(self, num_vectors: int) -> str:
        """Create the index best suited to ``num_vectors`` and return its name."""
        if num_vectors <= self.small_dataset_threshold:
            self.index_type = "FlatL2"
            self.index = IndexFlatL2(self.dimension)
            self.nprobe = 1
        elif num_vectors <= self.medium_dataset_threshold:
            self.index_type = "IVFFlat"
            self.nlist = min(100, int(np.sqrt(num_vectors)))
            quantizer = IndexFlatL2(self.dimension)
            self.index = IndexIVFFlat(quantizer, self.dimension, self.nlist)
            self.nprobe = min(10, max(1, int(self.nlist * 0.1)))
        else:
            self.index_type = "IVFPQ"
            self.nlist = min(256, int(np.sqrt(num_vectors)))
            self.m = min(8, self.dimension // 4)
            quantizer = IndexFlatL2(self.dimension)
            self.index = IndexIVFPQ(quantizer, self.dimension, self.nlist, self.m, 8)
            self.nprobe = min(32, max(1, int(self.nlist * 0.05)))

        logger.info("Selected FAISS index type %s for %d vectors", self.index_type, num_vectors)
        return self.index_type

    def _require_index(self) -> Index:
        if self.index is None:
            raise RuntimeError("select_index_type() must be called before using the index")
        return self.index

    def train(self, vectors: NDArray[np.float32]) -> None:
        index = self._require_index()
        if isinstance(index, IndexIVF):
            index.train(vectors)

    def add(self, vectors: NDArray[np.float32]) -> None:
        index = self._require_index()
        if isinstance(index, IndexIVF) and not index.is_trained:
            self.train(vectors)
        index.add(vectors)

    def search(self, query_vectors: NDArray[np.float32], k: int = 5) -> tuple[NDArray[Any], NDArray[Any]]:
        """Return (distances, indices) for the nearest ``k`` vectors of each query."""
        index = self._require_index()
        if isinstance(index, IndexIVF) and self.nprobe is not None:
            index.nprobe = self.nprobe
        distances, indices = index.search(query_vectors, k)
        return distances, indices

    def get_index_info(self) -> dict[str, Any]:
        return {
            "index_type": self.index_type,
            "dimension": self.dimension,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "size": self.ntotal,
        }


class VectorStore:
    """FAISS index plus the chunk texts and metadata it points at."""

    def __init__(self) -> None:
        self.index: AutoFaissIndex | None = None
        self.contents_map: dict[str, str] = {}
        self.metadatas_map: dict[str, Metadata] = {}
        self.id_order: list[str] = []

    def build_index(
        self,
        chunks: list[str],
        chunk_ids: list[str],
        metadatas: list[Metadata],
        embeddings: NDArray[np.float32],
    ) -> None:
        """Build a fresh FAISS index over ``embeddings`` and record the chunk texts.

        Positions in ``chunks``, ``chunk_ids``, ``metadatas`` and ``embeddings``
        must correspond; ``chunk_ids`` are the stable identifiers returned by
        ``search``.
        """
        dimension = int(embeddings.shape[1])
        num_vectors = len(chunks)

        auto_index = AutoFaissIndex(dimension=dimension)
        auto_index.select_index_type(num_vectors)

        for chunk_id, chunk, meta in zip(chunk_ids, chunks, metadatas, strict=True):
            self.contents_map[chunk_id] = chunk
            self.metadatas_map[chunk_id] = meta
            self.id_order.append(chunk_id)

        auto_index.add(embeddings)
        self.index = auto_index
        logger.info("FAISS index built: %d chunks, type %s", auto_index.ntotal, auto_index.index_type)

    def search(self, query_embedding: NDArray[np.float32], k: int = 10) -> SearchResult:
        """Return the ``k`` nearest chunks as ``(texts, chunk_ids, metadatas)``."""
        if self.index is None or self.index.ntotal == 0:
            return [], [], []
        try:
            _, indices = self.index.search(query_embedding, k=k)
        except Exception as exc:  # noqa: BLE001 - FAISS raises plain RuntimeError
            logger.error("FAISS search failed: %s", exc)
            return [], [], []

        docs: list[str] = []
        doc_ids: list[str] = []
        metadatas: list[Metadata] = []
        for faiss_idx in indices[0]:
            if faiss_idx == -1 or faiss_idx >= len(self.id_order):
                continue
            chunk_id = self.id_order[faiss_idx]
            if chunk_id in self.contents_map:
                docs.append(self.contents_map[chunk_id])
                doc_ids.append(chunk_id)
                metadatas.append(self.metadatas_map.get(chunk_id, {}))
        return docs, doc_ids, metadatas

    @property
    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    def clear(self) -> None:
        self.index = None
        self.contents_map.clear()
        self.metadatas_map.clear()
        self.id_order.clear()
        logger.info("Vector store cleared")


# Module-level singleton shared by the retriever and the ingestion pipeline.
vector_store = VectorStore()
