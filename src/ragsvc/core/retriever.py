"""Retriever: hybrid (dense + sparse) retrieval with optional recursive refinement.

Hybrid search scores each candidate as ``alpha * dense + (1 - alpha) * sparse``
so that semantic matches and exact keyword matches both surface. Recursive
retrieval runs several rounds, asking the LLM to rewrite the query when the
first round's context looks insufficient.
"""

from __future__ import annotations

import logging
from typing import Any

from ragsvc.config import Provider, get_settings
from ragsvc.core.bm25_index import BM25Hit, bm25_manager
from ragsvc.core.embeddings import encode_query
from ragsvc.core.reranker import RankedDocs, ScoredDoc, rerank_results
from ragsvc.core.vector_store import Metadata, vector_store
from ragsvc.features.web_search import check_serpapi_key, search_web

logger = logging.getLogger(__name__)

# Sentinel the query-rewriting LLM returns when the context is already sufficient.
NO_FURTHER_QUERY = "NO_FURTHER_QUERY"
MAX_REWRITTEN_QUERY_LENGTH = 100

# (contexts, doc_ids, metadata) with parallel positions.
RetrievalResult = tuple[list[str], list[str], list[Metadata]]


def _semantic_results_valid(semantic_results: dict[str, Any] | None) -> bool:
    if not semantic_results:
        return False
    keys = ("documents", "metadatas", "ids")
    if not all(isinstance(semantic_results.get(key), list) and semantic_results[key] for key in keys):
        return False
    if not isinstance(semantic_results["documents"][0], list):
        return False
    return (
        len(semantic_results["documents"][0])
        == len(semantic_results["metadatas"][0])
        == len(semantic_results["ids"][0])
    )


def hybrid_merge(
    semantic_results: dict[str, Any] | None,
    bm25_results: list[BM25Hit] | None,
    alpha: float | None = None,
) -> RankedDocs:
    """Merge dense and sparse results into one ranking.

    Dense results contribute a rank-based score in ``[0, 1]`` weighted by
    ``alpha``; BM25 scores are normalized by the best BM25 score and weighted
    by ``1 - alpha``. A document found by both retrievers gets the sum.

    ``semantic_results`` has the shape ``{"ids": [[...]], "documents": [[...]],
    "metadatas": [[...]]}``; ``bm25_results`` is what ``BM25IndexManager.search``
    returns. The result is sorted best first.
    """
    if alpha is None:
        alpha = get_settings().hybrid_alpha

    merged: dict[str, ScoredDoc] = {}

    if _semantic_results_valid(semantic_results):
        assert semantic_results is not None
        ids = semantic_results["ids"][0]
        docs = semantic_results["documents"][0]
        metas = semantic_results["metadatas"][0]
        num_results = len(docs)
        for rank, (doc_id, doc, meta) in enumerate(zip(ids, docs, metas, strict=True)):
            rank_score = 1.0 - (rank / max(1, num_results))
            merged[doc_id] = {"score": alpha * rank_score, "content": doc, "metadata": meta}
    else:
        logger.warning("Semantic results are empty or malformed")

    if not bm25_results:
        return sorted(merged.items(), key=lambda item: item[1]["score"], reverse=True)

    valid_scores = [r["score"] for r in bm25_results if isinstance(r, dict) and "score" in r]
    max_bm25 = max(valid_scores) if valid_scores else 1.0

    for result in bm25_results:
        if not (isinstance(result, dict) and {"id", "score", "content"} <= result.keys()):
            continue
        doc_id = result["id"]
        norm_score = result["score"] / max_bm25 if max_bm25 > 0 else 0.0

        if doc_id in merged:
            merged[doc_id]["score"] += (1 - alpha) * norm_score
        else:
            merged[doc_id] = {
                "score": (1 - alpha) * norm_score,
                "content": result["content"],
                "metadata": vector_store.metadatas_map.get(doc_id, {}),
            }

    return sorted(merged.items(), key=lambda item: item[1]["score"], reverse=True)


def hybrid_round(query: str, top_k: int, alpha: float) -> RankedDocs:
    """One dense + BM25 round for ``query``, merged and cut to the best ``top_k``."""
    query_embedding = encode_query(query)
    sem_docs, sem_ids, sem_metas = vector_store.search(query_embedding, k=top_k)
    semantic = {"ids": [sem_ids], "documents": [sem_docs], "metadatas": [sem_metas]}
    bm25_res = bm25_manager.search(query, top_k=top_k)
    return hybrid_merge(semantic, bm25_res, alpha=alpha)[:top_k]


def search_chunks(query: str, top_k: int | None = None) -> RankedDocs:
    """Retrieve the chunks that best match ``query``, best first, with scores.

    One hybrid round (``RAG_RETRIEVAL_TOP_K`` candidates from FAISS and
    BM25, merged with ``RAG_HYBRID_ALPHA``) followed by the configured
    reranker; ``top_k`` defaults to ``RAG_RERANK_TOP_K``. With
    ``RAG_RERANK_METHOD=none`` the scores are the hybrid scores. There is no
    query rewriting and no generation, so this never calls an LLM: it is
    the read path of the demo and of retrieval evaluation.
    """
    settings = get_settings()
    if top_k is None:
        top_k = settings.rerank_top_k
    hybrid = hybrid_round(query, settings.retrieval_top_k, settings.hybrid_alpha)
    if not hybrid or settings.rerank_method == "none":
        return hybrid[:top_k]
    ids = [doc_id for doc_id, _ in hybrid]
    docs = [data["content"] for _, data in hybrid]
    metas = [data["metadata"] for _, data in hybrid]
    return rerank_results(query, docs, ids, metas, top_k=top_k)[:top_k]


def _build_rewrite_prompt(initial_query: str, summary: str) -> str:
    return f"""You are a query optimization assistant. Decide whether a follow-up search is needed.

[Original question]
{initial_query}

[Retrieved context summary]
{summary}

Rules:
1. If the context is sufficient to answer, reply with exactly: {NO_FURTHER_QUERY}
2. Otherwise reply with a single, more precise search query and nothing else.
"""


def _web_search_round(
    query: str,
    seen_web_sources: set[str],
    all_contexts: list[str],
    all_doc_ids: list[str],
    all_metadata: list[Metadata],
) -> list[str]:
    """Run one web search, append unseen hits to the accumulators, return their texts."""
    web_texts: list[str] = []
    try:
        for res in search_web(query):
            title = res.get("title") or ""
            url = res.get("url") or ""
            snippet = res.get("snippet") or ""
            web_texts.append(f"Title: {title}\nSnippet: {snippet}")
            source_key = url or f"{title}\n{snippet}"
            if snippet and source_key not in seen_web_sources:
                seen_web_sources.add(source_key)
                all_contexts.append(snippet)
                all_doc_ids.append(f"web:{source_key}")
                all_metadata.append(
                    {"source": "web", "title": title, "url": url, "timestamp": res.get("timestamp")}
                )
    except Exception as exc:  # noqa: BLE001 - web search is best effort
        logger.error("Web search failed: %s", exc)
    return web_texts


def recursive_retrieval(
    initial_query: str,
    max_iterations: int | None = None,
    enable_web_search: bool = False,
    provider: Provider = "ollama",
) -> RetrievalResult:
    """Retrieve context for ``initial_query`` over up to ``max_iterations`` rounds.

    Each round runs dense + BM25 retrieval, merges and reranks the candidates,
    then asks the LLM (via ``provider``) whether a refined query is worth
    another round. Web results, when enabled, are added once per source.

    Returns ``(contexts, doc_ids, metadata)`` with parallel positions.
    """
    settings = get_settings()
    if max_iterations is None:
        max_iterations = settings.max_retrieval_iterations
    retrieval_top_k = settings.retrieval_top_k

    query = initial_query
    all_contexts: list[str] = []
    all_doc_ids: list[str] = []
    all_metadata: list[Metadata] = []
    seen_web_sources: set[str] = set()

    for iteration in range(max_iterations):
        logger.info("Retrieval round %d/%d, query: %s", iteration + 1, max_iterations, query)

        web_texts: list[str] = []
        if enable_web_search and check_serpapi_key():
            web_texts = _web_search_round(query, seen_web_sources, all_contexts, all_doc_ids, all_metadata)

        hybrid = hybrid_round(query, retrieval_top_k, settings.hybrid_alpha)
        ids_iter = [doc_id for doc_id, _ in hybrid]
        docs_iter = [data["content"] for _, data in hybrid]
        meta_iter = [data["metadata"] for _, data in hybrid]

        reranked: RankedDocs = []
        if docs_iter:
            try:
                reranked = rerank_results(query, docs_iter, ids_iter, meta_iter, top_k=settings.rerank_top_k)
            except Exception as exc:  # noqa: BLE001 - fall back to the hybrid order
                logger.error("Reranking failed: %s", exc)
                reranked = [
                    (doc_id, {"content": doc, "metadata": meta, "score": 1.0})
                    for doc_id, doc, meta in zip(ids_iter, docs_iter, meta_iter, strict=True)
                ]

        current_contexts = list(web_texts)
        for doc_id, data in reranked:
            if doc_id not in all_doc_ids:
                all_doc_ids.append(doc_id)
                all_contexts.append(data["content"])
                all_metadata.append(data["metadata"])
            current_contexts.append(data["content"])

        if iteration == max_iterations - 1 or not current_contexts:
            break

        summary = "\n".join(current_contexts[:3])
        try:
            from ragsvc.core.generator import call_llm_simple

            next_query = call_llm_simple(_build_rewrite_prompt(initial_query, summary), provider)
        except Exception as exc:  # noqa: BLE001 - keep what we have if the LLM is unavailable
            logger.error("Query rewriting failed: %s", exc)
            break
        if not next_query or NO_FURTHER_QUERY in next_query:
            # An empty rewrite (reasoning-only output, or nothing at all) is
            # treated as "sufficient": searching for "" would only add noise.
            logger.info("LLM reported the context is sufficient")
            break
        if len(next_query) > MAX_REWRITTEN_QUERY_LENGTH:
            logger.warning("Rewritten query too long; treating it as invalid")
            break
        query = next_query
        logger.info("Next query: %s", query)

    return all_contexts, all_doc_ids, all_metadata
