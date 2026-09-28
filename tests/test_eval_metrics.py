"""The ranking metrics agree with hand-computed values."""

from __future__ import annotations

import math

import pytest

from ragsvc.eval import QueryScores, mean_scores, ndcg_at_k, recall_at_k, reciprocal_rank, score_query

RANKED = ["a", "b", "c", "d", "e"]


def test_recall_counts_relevant_ids_within_the_cut_off() -> None:
    assert recall_at_k(RANKED, {"a", "c", "z"}, k=3) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, {"a", "c"}, k=2) == pytest.approx(0.5)
    assert recall_at_k(RANKED, {"e"}, k=5) == 1.0
    assert recall_at_k([], {"a"}, k=5) == 0.0


def test_recall_is_zero_for_a_query_with_no_relevant_ids() -> None:
    assert recall_at_k(RANKED, set(), k=3) == 0.0


def test_reciprocal_rank_uses_the_first_hit() -> None:
    assert reciprocal_rank(RANKED, {"a"}) == 1.0
    assert reciprocal_rank(RANKED, {"c", "e"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, {"z"}) == 0.0
    assert reciprocal_rank([], {"a"}) == 0.0


def test_ndcg_matches_the_closed_form() -> None:
    # hits at ranks 1 and 3 out of two relevant ids: dcg = 1 + 1/log2(4), ideal = 1 + 1/log2(3)
    expected = (1.0 + 1.0 / math.log2(4)) / (1.0 + 1.0 / math.log2(3))
    assert ndcg_at_k(RANKED, {"a", "c"}, k=3) == pytest.approx(expected)
    assert ndcg_at_k(RANKED, {"a", "b"}, k=2) == 1.0
    assert ndcg_at_k(RANKED, {"z"}, k=5) == 0.0
    assert ndcg_at_k(RANKED, set(), k=5) == 0.0


def test_ndcg_ideal_is_capped_at_k_when_there_are_more_relevant_ids_than_slots() -> None:
    # Three relevant ids but k=2 and both slots hit: a perfect score, not 2/3 of one.
    assert ndcg_at_k(RANKED, {"a", "b", "c"}, k=2) == 1.0


@pytest.mark.parametrize("function", [recall_at_k, ndcg_at_k])
def test_a_cut_off_below_one_is_rejected(function) -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        function(RANKED, {"a"}, 0)


def test_score_query_bundles_the_three_metrics() -> None:
    scores = score_query("q", RANKED, {"b"}, k=3)
    assert isinstance(scores, QueryScores)
    assert (scores.query, scores.k, scores.recall, scores.mrr) == ("q", 3, 1.0, 0.5)
    assert scores.ndcg == pytest.approx(1 / math.log2(3))


def test_mean_scores_macro_averages_over_queries() -> None:
    first = score_query("q1", RANKED, {"a"}, k=1)
    second = score_query("q2", RANKED, {"z"}, k=1)
    assert mean_scores([first, second]) == {"recall": 0.5, "mrr": 0.5, "ndcg": 0.5}


def test_mean_scores_refuses_an_empty_run() -> None:
    with pytest.raises(ValueError, match="zero queries"):
        mean_scores([])
