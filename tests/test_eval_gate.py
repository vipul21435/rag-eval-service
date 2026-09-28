"""Retrieval quality gate: the bundled golden set must clear the committed thresholds.

The gate ingests ``examples/docs`` with the hash embedder at its default
dimension (what ``make eval`` uses) and fails when any macro-averaged
metric in either mode drops below ``examples/eval/thresholds.toml``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from ragsvc.config import Settings
from ragsvc.core.ingest import SourceFile, ingest_files
from ragsvc.eval.harness import evaluate, load_golden, load_thresholds

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "examples" / "docs"
GOLDEN_PATH = REPO_ROOT / "examples" / "eval" / "golden.v1.jsonl"
THRESHOLDS_PATH = REPO_ROOT / "examples" / "eval" / "thresholds.toml"
SCRIPT_PATH = REPO_ROOT / "examples" / "eval_retrieval.py"


def test_golden_set_clears_the_thresholds(settings: Callable[..., Settings]) -> None:
    settings(hash_embedding_dimension=Settings.model_fields["hash_embedding_dimension"].default)
    golden = load_golden(GOLDEN_PATH)
    thresholds = load_thresholds(THRESHOLDS_PATH)
    assert ingest_files([SourceFile.from_path(p) for p in sorted(DOCS_DIR.glob("*.md"))]).succeeded

    report = evaluate(golden, k=thresholds.k, golden_set=GOLDEN_PATH.name)

    assert report.failures(thresholds) == []
    assert report.means("hybrid")["mrr"] >= report.means("dense")["mrr"]


def test_golden_set_names_only_bundled_documents() -> None:
    bundled = {path.name for path in DOCS_DIR.glob("*.md")}
    for query in load_golden(GOLDEN_PATH):
        assert query.relevant <= bundled, query.id


def run_script(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {name: value for name, value in os.environ.items() if not name.startswith("RAG_")}
    env.update(
        {
            "RAG_EMBEDDING_PROVIDER": "sentence-transformers",
            "RAG_RETRIEVAL_TOP_K": "30",
            "HF_HUB_OFFLINE": "1",
        }
    )
    (tmp_path / ".env").write_text("RAG_HYBRID_ALPHA=0.1\n", encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--output-dir", str(tmp_path / "out"), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=120,
        check=False,
    )


def test_eval_script_writes_both_reports_and_passes(tmp_path: Path) -> None:
    completed = run_script(tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert "| Mode | Recall@3 | MRR | nDCG@3 |" in completed.stdout
    written = json.loads((tmp_path / "out" / "retrieval-eval.json").read_text(encoding="utf-8"))
    assert written["embedding_provider"] == "hash"
    assert written["hybrid_alpha"] == 0.7
    assert written["retrieval_top_k"] == 10
    assert written["modes"][1]["means"]["mrr"] == 1.0
    assert (
        (tmp_path / "out" / "retrieval-eval.md")
        .read_text()
        .startswith("# Retrieval evaluation: golden.v1.jsonl")
    )


def test_eval_script_fails_when_a_threshold_is_not_met(tmp_path: Path) -> None:
    strict = tmp_path / "strict.toml"
    strict.write_text(THRESHOLDS_PATH.read_text().replace("mrr = 0.85", "mrr = 1.0"), encoding="utf-8")

    completed = run_script(tmp_path, "--thresholds", str(strict))

    assert completed.returncode == 1
    assert "THRESHOLD FAILED: dense mrr@3" in completed.stderr
