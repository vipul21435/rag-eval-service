"""BM25 index: sparse, keyword-based retrieval.

BM25 complements dense retrieval: embeddings capture intent, BM25 rewards
exact term matches (identifiers, part numbers, rare words). The two are
combined by ``core.retriever.hybrid_merge``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_WORD = re.compile(r"\w+")
# CJK Unified Ideographs (basic block and extension A).
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_CJK_OR_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]|[^\u3400-\u4dbf\u4e00-\u9fff]+")


class BM25Hit(TypedDict):
    id: str
    score: float
    content: str


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; CJK ideographs become one token per character.

    Character unigrams are a dependency-free stand-in for word segmentation
    and work well enough for BM25 on CJK text.
    """
    tokens: list[str] = []
    for word in _WORD.findall(text.lower()):
        if _CJK.search(word):
            tokens.extend(_CJK_OR_RUN.findall(word))
        else:
            tokens.append(word)
    return tokens


@dataclass(frozen=True)
class _Snapshot:
    """One BM25 index and the corpus it scores, installed with one assignment."""

    index: BM25Okapi
    doc_ids: tuple[str, ...]
    corpus: tuple[str, ...]


class BM25IndexManager:
    """Build, search and clear a BM25 index over chunk texts."""

    def __init__(self) -> None:
        self._snapshot: _Snapshot | None = None

    @property
    def is_ready(self) -> bool:
        return self._snapshot is not None

    @property
    def bm25_index(self) -> BM25Okapi | None:
        snapshot = self._snapshot
        return snapshot.index if snapshot is not None else None

    def build_index(self, documents: list[str], doc_ids: list[str]) -> bool:
        """Replace the index with one over ``documents``; ``doc_ids`` are returned by ``search``."""
        if len(documents) != len(doc_ids):
            raise ValueError(f"{len(documents)} documents but {len(doc_ids)} ids")
        index = BM25Okapi([tokenize(doc) for doc in documents])
        # A single reference assignment: a concurrent search sees either the
        # previous corpus or the new one, never a mix.
        self._snapshot = _Snapshot(index=index, doc_ids=tuple(doc_ids), corpus=tuple(documents))
        logger.info("BM25 index built over %d documents", len(documents))
        return True

    def search(self, query: str, top_k: int = 5) -> list[BM25Hit]:
        """Return up to ``top_k`` documents with a positive BM25 score, best first."""
        snapshot = self._snapshot
        if snapshot is None:
            return []

        scores = snapshot.index.get_scores(tokenize(query))
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results: list[BM25Hit] = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    {
                        "id": snapshot.doc_ids[int(idx)],
                        "score": float(scores[idx]),
                        "content": snapshot.corpus[int(idx)],
                    }
                )
        return results

    def clear(self) -> None:
        self._snapshot = None


# Module-level singleton shared by the retriever and the ingestion pipeline.
bm25_manager = BM25IndexManager()
