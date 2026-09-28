"""Ranking metrics with binary relevance: recall@k, MRR and nDCG@k.

Every function takes the ranked ids a retriever returned (best first) and
the set of ids the golden set marks relevant for the query. Relevance is
binary and ids are compared exactly, so the caller decides the granularity
(chunk ids or document ids) by what it puts in both sequences.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from statistics import fmean


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of ``relevant`` ids found in the first ``k`` of ``ranked``.

    An empty ``relevant`` set scores 0.0: a query with nothing to find
    cannot count as a hit.
    """
    _check_k(k)
    if not relevant:
        return 0.0
    found = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return found / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str]) -> float:
    """``1 / rank`` of the first relevant id in ``ranked``, 0.0 when there is none."""
    for position, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Normalized discounted cumulative gain with binary gains over the first ``k``.

    The ideal ordering puts every relevant id first, so the normalizer is
    the DCG of ``min(len(relevant), k)`` hits at the top ranks.
    """
    _check_k(k)
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, doc_id in enumerate(ranked[:k], start=1)
        if doc_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / ideal


@dataclass(frozen=True)
class QueryScores:
    """The metrics of one query at one cut-off ``k``."""

    query: str
    k: int
    recall: float
    mrr: float
    ndcg: float


def score_query(query: str, ranked: Sequence[str], relevant: Collection[str], k: int) -> QueryScores:
    """Score one query's ranking against its relevant ids at cut-off ``k``."""
    return QueryScores(
        query=query,
        k=k,
        recall=recall_at_k(ranked, relevant, k),
        mrr=reciprocal_rank(ranked, relevant),
        ndcg=ndcg_at_k(ranked, relevant, k),
    )


def mean_scores(scores: Iterable[QueryScores]) -> dict[str, float]:
    """Macro-average ``recall``, ``mrr`` and ``ndcg`` over the given queries.

    Raises ``ValueError`` on an empty iterable: an average over no queries
    would silently pass any threshold.
    """
    collected = list(scores)
    if not collected:
        raise ValueError("cannot average metrics over zero queries")
    return {
        "recall": fmean(score.recall for score in collected),
        "mrr": fmean(score.mrr for score in collected),
        "ndcg": fmean(score.ndcg for score in collected),
    }
