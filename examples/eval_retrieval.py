"""Offline retrieval evaluation: ingest the sample documents, score the golden set, write reports.

Pins the hash embedder and no reranker like ``examples/demo.py`` so it runs
with no network, then evaluates ``examples/eval/golden.v1.jsonl`` in
dense-only and hybrid mode, writes ``retrieval-eval.json`` and
``retrieval-eval.md`` into ``--output-dir`` (default ``eval-reports/``) and
exits 1 when a mean drops below ``examples/eval/thresholds.toml``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ragsvc.config import Settings, set_settings
from ragsvc.core.ingest import SourceFile, ingest_files
from ragsvc.eval.harness import evaluate, load_golden, load_thresholds
from ragsvc.logging_setup import configure_logging

EXAMPLES_DIR = Path(__file__).resolve().parent
DOCS_DIR = EXAMPLES_DIR / "docs"
GOLDEN_PATH = EXAMPLES_DIR / "eval" / "golden.v1.jsonl"
THRESHOLDS_PATH = EXAMPLES_DIR / "eval" / "thresholds.toml"


def eval_settings() -> Settings:
    """Offline settings: hash embedder, no reranker, no LLM probe, no cache file."""
    return Settings(
        embedding_provider="hash",
        rerank_method="none",
        llm_provider="ollama",
        embedding_cache_enabled=False,
        log_level="WARNING",
        log_format="text",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--thresholds", type=Path, default=THRESHOLDS_PATH)
    parser.add_argument("--docs", type=Path, default=DOCS_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("eval-reports"))
    args = parser.parse_args(argv)

    settings = eval_settings()
    set_settings(settings)
    configure_logging(settings.log_level, settings.log_format)

    golden = load_golden(args.golden)
    thresholds = load_thresholds(args.thresholds)
    sources = [SourceFile.from_path(path) for path in sorted(args.docs.glob("*.md"))]
    if not sources:
        print(f"no sample documents under {args.docs}", file=sys.stderr)
        return 1

    started = time.perf_counter()
    ingest_report = ingest_files(sources)
    report = evaluate(golden, k=thresholds.k, golden_set=args.golden.name)
    elapsed = time.perf_counter() - started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "retrieval-eval.json").write_text(report.to_json(), encoding="utf-8")
    (args.output_dir / "retrieval-eval.md").write_text(report.to_markdown(), encoding="utf-8")

    print(report.to_markdown().split("## Per query")[0].rstrip())
    print(f"\n{len(golden)} queries over {ingest_report.total_chunks} chunks in {elapsed * 1000:.0f} ms;")
    print(f"reports written to {args.output_dir}/retrieval-eval.json and retrieval-eval.md")

    failures = report.failures(thresholds)
    for line in failures:
        print(f"THRESHOLD FAILED: {line}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
