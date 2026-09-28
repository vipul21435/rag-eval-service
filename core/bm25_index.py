"""BM25 index: sparse, keyword-based retrieval.

BM25 complements dense retrieval: embeddings capture intent, BM25 rewards
exact term matches (identifiers, part numbers, rare words). The two are
combined by ``core.retriever.hybrid_merge``.
"""

from __future__ import annotations

import logging
import re
from typing import TypedDict

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_WORD = re.compile(r"\w+")
# CJK Unified Ideographs (basic block and extension A).
_CJK = re.compile(r"[㐀-䶿一-鿿]")
_CJK_OR_RUN = re.compile(r"[㐀-䶿一-鿿]|[^㐀-䶿一-鿿]+")


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


class BM25IndexManager:
    """Build, search and clear a BM25 index over chunk texts."""

    def __init__(self) -> None:
        self.bm25_index: BM25Okapi | None = None
        self.doc_mapping: dict[int, str] = {}
        self.tokenized_corpus: list[list[str]] = []
        self.raw_corpus: list[str] = []

    def build_index(self, documents: list[str], doc_ids: list[str]) -> bool:
        """Index ``documents``; ``doc_ids`` are returned by ``search``."""
        self.raw_corpus = list(documents)
        self.doc_mapping = dict(enumerate(doc_ids))
        self.tokenized_corpus = [tokenize(doc) for doc in documents]
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        logger.info("BM25 index built over %d documents", len(documents))
        return True

    def search(self, query: str, top_k: int = 5) -> list[BM25Hit]:
        """Return up to ``top_k`` documents with a positive BM25 score, best first."""
        if self.bm25_index is None:
            return []

        scores = self.bm25_index.get_scores(tokenize(query))
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results: list[BM25Hit] = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    {"id": self.doc_mapping[int(idx)], "score": float(scores[idx]), "content": self.raw_corpus[idx]}
                )
        return results

    def clear(self) -> None:
        self.bm25_index = None
        self.doc_mapping = {}
        self.tokenized_corpus = []
        self.raw_corpus = []


# Module-level singleton shared by the retriever and the ingestion pipeline.
bm25_manager = BM25IndexManager()
