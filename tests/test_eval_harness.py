"""The evaluation harness parses its inputs strictly and scores a small knowledge base."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragsvc.core.ingest import SourceFile, ingest_files
from ragsvc.eval.harness import (
    EvalReport,
    GoldenQuery,
    GoldenSetError,
    Thresholds,
    evaluate,
    load_golden,
    load_thresholds,
    rank_sources,
)

GOLDEN_LINES = [
    '{"id": "q1", "query": "apples and pears", "relevant": ["fruit.md"]}',
    "",
    '{"id": "q2", "query": "carrots and leeks", "relevant": ["veg.md", "fruit.md"]}',
]

THRESHOLDS_TOML = """
k = 2
[dense]
recall = 0.5
mrr = 0.5
ndcg = 0.5
[hybrid]
recall = 0.5
mrr = 0.5
ndcg = 0.5
"""


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_golden_parses_lines_and_skips_blanks(tmp_path: Path) -> None:
    golden = load_golden(write(tmp_path / "g.jsonl", "\n".join(GOLDEN_LINES)))
    assert golden == [
        GoldenQuery(id="q1", query="apples and pears", relevant=frozenset({"fruit.md"})),
        GoldenQuery(id="q2", query="carrots and leeks", relevant=frozenset({"veg.md", "fruit.md"})),
    ]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not json", "not valid JSON"),
        ('{"query": "x", "relevant": ["a"]}', "expected an object with id"),
        ('{"id": "q", "query": "x", "relevant": []}', "non-empty list"),
        ('{"id": "q", "query": "x", "relevant": [1]}', "non-empty list"),
        (
            '{"id": "q", "query": "x", "relevant": ["a"]}\n{"id": "q", "query": "y", "relevant": ["a"]}',
            "duplicate",
        ),
        ("\n\n", "empty"),
    ],
)
def test_load_golden_rejects_malformed_files(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(GoldenSetError, match=message):
        load_golden(write(tmp_path / "g.jsonl", text))


def test_load_thresholds_reads_k_and_both_modes(tmp_path: Path) -> None:
    thresholds = load_thresholds(write(tmp_path / "t.toml", THRESHOLDS_TOML))
    assert thresholds.k == 2
    assert thresholds.minimums["dense"] == {"recall": 0.5, "mrr": 0.5, "ndcg": 0.5}
    assert set(thresholds.minimums) == {"dense", "hybrid"}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("k = [", "not valid TOML"),
        ("k = 0\n", "positive integer"),
        ("k = 1\n[dense]\nrecall = 1\nmrr = 1\nndcg = 1\n", r"missing \[hybrid\]"),
        ("k = 1\n[dense]\nrecall = 2\nmrr = 1\nndcg = 1\n[hybrid]\n", r"\[dense\].recall"),
        ("k = 1\n[dense]\nrecall = 1\nmrr = 'x'\nndcg = 1\n[hybrid]\n", r"\[dense\].mrr"),
    ],
)
def test_load_thresholds_rejects_malformed_files(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(GoldenSetError, match=message):
        load_thresholds(write(tmp_path / "t.toml", text))


@pytest.fixture
def small_knowledge_base(tmp_path: Path) -> list[GoldenQuery]:
    write(
        tmp_path / "fruit.md",
        "Apples and pears are fruit. Pears ripen after picking. Apples keep for months.",
    )
    write(
        tmp_path / "veg.md",
        "Carrots and leeks are vegetables. Leeks like cold weather. Carrots need loose soil.",
    )
    report = ingest_files([SourceFile.from_path(tmp_path / name) for name in ("fruit.md", "veg.md")])
    assert report.succeeded
    return load_golden(write(tmp_path / "g.jsonl", "\n".join(GOLDEN_LINES)))


def test_rank_sources_lists_each_document_once_in_rank_order(small_knowledge_base: list[GoldenQuery]) -> None:
    for mode in ("dense", "hybrid"):
        ranked = rank_sources("carrots and leeks", mode, top_k=10)
        assert sorted(ranked) == ["fruit.md", "veg.md"]
        assert ranked[0] == "veg.md"


def test_evaluate_scores_both_modes_and_serialises(small_knowledge_base: list[GoldenQuery]) -> None:
    report = evaluate(small_knowledge_base, k=2, golden_set="g.jsonl")

    assert [mode.mode for mode in report.modes] == ["dense", "hybrid"]
    assert report.k == 2 and report.embedding_provider == "hash"
    assert report.means("hybrid") == {"recall": 1.0, "mrr": 1.0, "ndcg": 1.0}
    with pytest.raises(KeyError):
        report.means("sparse")  # type: ignore[arg-type]

    parsed = json.loads(report.to_json())
    assert parsed["golden_set"] == "g.jsonl"
    assert len(parsed["modes"][0]["queries"]) == 2
    markdown = report.to_markdown()
    assert "| Mode | Recall@2 | MRR | nDCG@2 |" in markdown
    assert "| hybrid | 1.000 | 1.000 | 1.000 |" in markdown
    assert "## Per query" in markdown


def test_failures_name_every_metric_below_its_threshold(small_knowledge_base: list[GoldenQuery]) -> None:
    report: EvalReport = evaluate(small_knowledge_base, k=2)
    passing = Thresholds(k=2, minimums={"dense": {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0}, "hybrid": {}})
    assert report.failures(passing) == []

    impossible = Thresholds(
        k=2, minimums={"hybrid": {"recall": 1.0, "mrr": 1.0, "ndcg": 1.0}, "dense": {"mrr": 1.01}}
    )
    failures = report.failures(impossible)
    assert failures == [f"dense mrr@2 = {report.means('dense')['mrr']:.3f} < 1.010"]
