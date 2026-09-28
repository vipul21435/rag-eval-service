"""Reranker: second-stage scoring of retrieved candidates.

Recall (FAISS, BM25) is cheap and approximate; reranking applies a more
accurate but slower model to the few candidates that survived. A
cross-encoder reads the query and the document together, which is more
precise than comparing two independent embeddings.
"""

from __future__ import annotations

import logging
import re
import threading
from functools import lru_cache
from typing import TYPE_CHECKING, TypedDict

from ragsvc.config import get_settings
from ragsvc.core.vector_store import Metadata

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class ScoredDoc(TypedDict):
    score: float
    content: str
    metadata: Metadata


# (chunk_id, scored document) pairs, best first.
RankedDocs = list[tuple[str, ScoredDoc]]

_cross_encoder: CrossEncoder | None = None
_cross_encoder_lock = threading.Lock()


def get_cross_encoder() -> CrossEncoder | None:
    """Load the cross-encoder lazily (double-checked locking); None if loading fails."""
    global _cross_encoder
    if _cross_encoder is None:
        with _cross_encoder_lock:
            if _cross_encoder is None:
                try:
                    from sentence_transformers import CrossEncoder

                    model_name = get_settings().rerank_model_name
                    _cross_encoder = CrossEncoder(model_name)
                    logger.info("Cross-encoder loaded: %s", model_name)
                except Exception as exc:  # noqa: BLE001 - model download or load failure
                    logger.error("Failed to load cross-encoder: %s", exc)
                    _cross_encoder = None
    return _cross_encoder


def rerank_with_cross_encoder(
    query: str,
    docs: list[str],
    doc_ids: list[str],
    metadata_list: list[Metadata],
    top_k: int = 5,
) -> RankedDocs:
    """Score each (query, doc) pair with the cross-encoder and keep the best ``top_k``."""
    if not docs:
        return []

    encoder = get_cross_encoder()
    if encoder is None:
        logger.warning("Cross-encoder unavailable; skipping reranking")
        return _fallback_results(doc_ids, docs, metadata_list)

    try:
        scores = encoder.predict([(query, doc) for doc in docs])
    except Exception as exc:  # noqa: BLE001 - inference failure
        logger.error("Cross-encoder reranking failed: %s", exc)
        return _fallback_results(doc_ids, docs, metadata_list)

    results: RankedDocs = [
        (doc_id, {"content": doc, "metadata": meta, "score": float(score)})
        for doc_id, doc, meta, score in zip(doc_ids, docs, metadata_list, scores, strict=True)
    ]
    results.sort(key=lambda item: item[1]["score"], reverse=True)
    return results[:top_k]


@lru_cache(maxsize=32)
def get_llm_relevance_score(query: str, doc: str) -> float:
    """Ask the local Ollama model for a 0-10 relevance score (cached per pair)."""
    from ragsvc.utils.network import get_session

    prompt = f"""Rate how relevant the document excerpt is to the query.
Scale: 0 means completely unrelated, 10 means highly relevant.
Reply with a single integer between 0 and 10 and nothing else.

Query: {query}
Document excerpt: {doc}
Relevance score (0-10):"""

    settings = get_settings()
    try:
        response = get_session().post(
            f"{settings.ollama_base_url}/api/generate",
            json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
            timeout=180,
        )
        result = str(response.json().get("response", "")).strip()
    except Exception as exc:  # noqa: BLE001 - network or decode failure
        logger.error("LLM relevance scoring failed: %s", exc)
        return 5.0

    try:
        return max(0.0, min(10.0, float(result)))
    except ValueError:
        match = re.search(r"\b([0-9]|10)\b", result)
        return float(match.group(1)) if match else 5.0


def rerank_with_llm(
    query: str,
    docs: list[str],
    doc_ids: list[str],
    metadata_list: list[Metadata],
    top_k: int = 5,
) -> RankedDocs:
    """Score each document with the LLM and keep the best ``top_k``."""
    if not docs:
        return []
    results: RankedDocs = [
        (doc_id, {"content": doc, "metadata": meta, "score": get_llm_relevance_score(query, doc) / 10.0})
        for doc_id, doc, meta in zip(doc_ids, docs, metadata_list, strict=True)
    ]
    results.sort(key=lambda item: item[1]["score"], reverse=True)
    return results[:top_k]


def rerank_results(
    query: str,
    docs: list[str],
    doc_ids: list[str],
    metadata_list: list[Metadata],
    method: str | None = None,
    top_k: int = 5,
) -> RankedDocs:
    """Rerank with ``method``: ``cross_encoder`` or ``llm``; any other value keeps the input order."""
    if method is None:
        method = get_settings().rerank_method

    if method == "llm":
        return rerank_with_llm(query, docs, doc_ids, metadata_list, top_k)
    if method == "cross_encoder":
        return rerank_with_cross_encoder(query, docs, doc_ids, metadata_list, top_k)
    return _fallback_results(doc_ids, docs, metadata_list)


def _fallback_results(doc_ids: list[str], docs: list[str], metadata_list: list[Metadata]) -> RankedDocs:
    """Keep the input order, with scores decreasing by rank."""
    return [
        (doc_id, {"content": doc, "metadata": meta, "score": 1.0 - idx / len(docs)})
        for idx, (doc_id, doc, meta) in enumerate(zip(doc_ids, docs, metadata_list, strict=True))
    ]
