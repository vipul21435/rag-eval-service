"""Retrieval evaluation: ranking metrics over a golden set of queries.

``ragsvc.eval.metrics`` holds the pure, dependency-free metric functions
(recall@k, MRR, nDCG@k) so they can be unit-tested against hand-computed
values before anything touches an index.
"""

from ragsvc.eval.metrics import QueryScores, mean_scores, ndcg_at_k, recall_at_k, reciprocal_rank, score_query

__all__ = ["QueryScores", "mean_scores", "ndcg_at_k", "recall_at_k", "reciprocal_rank", "score_query"]
