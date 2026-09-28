"""The bundled demo stays offline however it is launched.

``examples/demo.py`` must not depend on the Makefile exporting
``RAG_EMBEDDING_PROVIDER``: run directly it still has to pin the hash
embedder and skip the reranker, or a fresh clone would download models
from the Hugging Face Hub.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = REPO_ROOT / "examples" / "demo.py"


def load_demo() -> ModuleType:
    spec = importlib.util.spec_from_file_location("recallmcp_demo", DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_settings_pin_the_offline_providers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "sentence-transformers")
    monkeypatch.setenv("RAG_RERANK_METHOD", "cross_encoder")
    monkeypatch.setenv("RAG_LLM_PROVIDER", "openai")
    cache_path = tmp_path / "embeddings.sqlite3"

    settings = load_demo().demo_settings(cache_path)

    assert settings.embedding_provider == "hash"
    assert settings.rerank_method == "none"
    assert settings.llm_provider == "ollama"
    assert settings.embedding_cache_enabled is True
    assert settings.embedding_cache_path == cache_path


def test_demo_settings_honour_the_other_variables(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RAG_HASH_EMBEDDING_DIMENSION", "32")

    settings = load_demo().demo_settings(tmp_path / "embeddings.sqlite3")

    assert settings.hash_embedding_dimension == 32


def test_demo_runs_offline_without_the_makefile(tmp_path: Path) -> None:
    env = {name: value for name, value in os.environ.items() if not name.startswith("RAG_")}
    env.update(
        {
            "RAG_EMBEDDING_PROVIDER": "sentence-transformers",
            "RAG_RERANK_METHOD": "cross_encoder",
            "HF_HUB_OFFLINE": "1",  # any attempt to fetch a model fails instead of reaching the network
            "TRANSFORMERS_OFFLINE": "1",
        }
    )

    completed = subprocess.run(
        [sys.executable, str(DEMO_PATH)],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,  # no repository .env, nothing written next to the sources
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert lines[0] == "RecallMCP demo: embedder=hash, reranker=none"
    assert any(line.startswith("Re-ingest (unchanged files): 22 chunks") for line in lines)
    assert any(line.startswith("Embedding cache at exit: entries=25 hits=79 misses=25") for line in lines)
