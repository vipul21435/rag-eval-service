"""Score the knowledge base against a golden set of queries.

A golden set is a JSONL file, one query per line, naming the documents
(by source file name, as recorded in chunk metadata) that a good retriever
should rank first. The harness runs each query in two modes, dense-only
(FAISS alone) and hybrid (the service's dense + BM25 merge), collapses the
ranked chunks to their source documents in rank order, and scores the
document ranking with ``ragsvc.eval.metrics``. Thresholds live in a TOML
file so the pytest gate and ``make eval`` read the same numbers.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from ragsvc.config import get_settings
from ragsvc.core.embeddings import encode_query
from ragsvc.core.retriever import hybrid_round
from ragsvc.core.vector_store import vector_store
from ragsvc.eval.metrics import QueryScores, mean_scores, score_query

Mode = Literal["dense", "hybrid"]
MODES: tuple[Mode, ...] = ("dense", "hybrid")
METRICS: tuple[str, ...] = ("recall", "mrr", "ndcg")


class GoldenSetError(ValueError):
    """The golden set or the thresholds file is malformed."""


@dataclass(frozen=True)
class GoldenQuery:
    """One query and the source documents that count as relevant for it."""

    id: str
    query: str
    relevant: frozenset[str]


def load_golden(path: Path) -> list[GoldenQuery]:
    """Parse a JSONL golden set; every line needs ``id``, ``query`` and a non-empty ``relevant`` list."""
    queries: list[GoldenQuery] = []
    seen: set[str] = set()
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GoldenSetError(f"{path}:{number}: not valid JSON: {exc}") from exc
        if not isinstance(record, dict) or not {"id", "query", "relevant"} <= record.keys():
            raise GoldenSetError(f"{path}:{number}: expected an object with id, query and relevant")
        relevant = record["relevant"]
        if not isinstance(relevant, list) or not relevant or not all(isinstance(r, str) for r in relevant):
            raise GoldenSetError(f"{path}:{number}: relevant must be a non-empty list of source names")
        query_id = str(record["id"])
        if query_id in seen:
            raise GoldenSetError(f"{path}:{number}: duplicate query id {query_id!r}")
        seen.add(query_id)
        queries.append(GoldenQuery(id=query_id, query=str(record["query"]), relevant=frozenset(relevant)))
    if not queries:
        raise GoldenSetError(f"{path}: the golden set is empty")
    return queries


@dataclass(frozen=True)
class Thresholds:
    """Minimum macro-averaged scores per mode, at cut-off ``k``."""

    k: int
    minimums: dict[Mode, dict[str, float]]


def load_thresholds(path: Path) -> Thresholds:
    """Parse the TOML thresholds file: a top-level ``k`` and one table per mode."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise GoldenSetError(f"{path}: not valid TOML: {exc}") from exc
    k = data.get("k")
    if not isinstance(k, int) or k < 1:
        raise GoldenSetError(f"{path}: k must be a positive integer")
    minimums: dict[Mode, dict[str, float]] = {}
    for mode in MODES:
        table = data.get(mode)
        if not isinstance(table, dict):
            raise GoldenSetError(f"{path}: missing [{mode}] table")
        values: dict[str, float] = {}
        for metric in METRICS:
            value = table.get(metric)
            if not isinstance(value, int | float) or not 0.0 <= float(value) <= 1.0:
                raise GoldenSetError(f"{path}: [{mode}].{metric} must be a number between 0 and 1")
            values[metric] = float(value)
        minimums[mode] = values
    return Thresholds(k=k, minimums=minimums)


def rank_sources(query: str, mode: Mode, top_k: int) -> list[str]:
    """Source documents in the order the retriever first surfaces one of their chunks."""
    if mode == "dense":
        _, _, metas = vector_store.search(encode_query(query), k=top_k)
    else:
        metas = [data["metadata"] for _, data in hybrid_round(query, top_k, get_settings().hybrid_alpha)]
    sources: list[str] = []
    for meta in metas:
        source = str(meta.get("source", ""))
        if source and source not in sources:
            sources.append(source)
    return sources


@dataclass(frozen=True)
class ModeReport:
    """Per-query scores and their macro average for one retrieval mode."""

    mode: Mode
    k: int
    queries: list[QueryScores]
    means: dict[str, float]


@dataclass(frozen=True)
class EvalReport:
    """The outcome of one evaluation run over both modes."""

    golden_set: str
    embedding_provider: str
    hybrid_alpha: float
    retrieval_top_k: int
    k: int
    modes: list[ModeReport]

    def means(self, mode: Mode) -> dict[str, float]:
        for report in self.modes:
            if report.mode == mode:
                return report.means
        raise KeyError(mode)

    def failures(self, thresholds: Thresholds) -> list[str]:
        """Human-readable lines for every metric below its threshold; empty when the gate passes."""
        lines: list[str] = []
        for mode, minimums in thresholds.minimums.items():
            means = self.means(mode)
            for metric, minimum in minimums.items():
                if means[metric] < minimum:
                    lines.append(f"{mode} {metric}@{self.k} = {means[metric]:.3f} < {minimum:.3f}")
        return lines

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"

    def to_markdown(self) -> str:
        header = [
            f"# Retrieval evaluation: {self.golden_set}",
            "",
            f"Embedder: `{self.embedding_provider}`, hybrid alpha: {self.hybrid_alpha}, "
            f"candidates per retriever: {self.retrieval_top_k}, k: {self.k}.",
            "",
            f"| Mode | Recall@{self.k} | MRR | nDCG@{self.k} |",
            "| --- | --- | --- | --- |",
        ]
        rows = [
            f"| {r.mode} | {r.means['recall']:.3f} | {r.means['mrr']:.3f} | {r.means['ndcg']:.3f} |"
            for r in self.modes
        ]
        per_query = [
            "",
            "## Per query",
            "",
            "| Query | Mode | Recall | RR | nDCG |",
            "| --- | --- | --- | --- | --- |",
        ]
        for report in self.modes:
            per_query.extend(
                f"| {s.query} | {report.mode} | {s.recall:.2f} | {s.mrr:.2f} | {s.ndcg:.2f} |"
                for s in report.queries
            )
        return "\n".join(header + rows + per_query) + "\n"


def evaluate(golden: Sequence[GoldenQuery], k: int, golden_set: str = "golden") -> EvalReport:
    """Run every golden query in both modes against the current knowledge base."""
    settings = get_settings()
    top_k = max(settings.retrieval_top_k, k)
    modes: list[ModeReport] = []
    for mode in MODES:
        scores = [score_query(q.query, rank_sources(q.query, mode, top_k), q.relevant, k) for q in golden]
        modes.append(ModeReport(mode=mode, k=k, queries=scores, means=mean_scores(scores)))
    return EvalReport(
        golden_set=golden_set,
        embedding_provider=settings.embedding_provider,
        hybrid_alpha=settings.hybrid_alpha,
        retrieval_top_k=top_k,
        k=k,
        modes=modes,
    )
