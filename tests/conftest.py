"""Shared fixtures.

Every test runs against a fresh, hermetic ``Settings`` instance: ``RAG_*``
variables and third-party credentials are removed from the environment and
no ``.env`` file is read, so results do not depend on the developer's
machine. Embeddings come from the deterministic hash provider with the
on-disk cache off, so no test downloads a model or writes outside its
``tmp_path``. Override individual values with the ``settings`` fixture.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator

import pytest

from ragsvc.config import Settings, set_settings
from ragsvc.core.embeddings import set_embedder

_UNPREFIXED_VARIABLES = frozenset({"OPENAI_API_KEY", "OPENAI_BASE_URL", "SERPAPI_KEY"})

# Test-suite defaults that differ from production; the ``settings`` factory
# applies them below any per-test overrides.
TEST_SETTINGS: dict[str, object] = {
    "embedding_provider": "hash",
    "hash_embedding_dimension": 64,
    "embedding_cache_enabled": False,
    "log_format": "text",
}


def test_settings(**overrides: object) -> Settings:
    """A hermetic ``Settings`` with the suite defaults under ``overrides``."""
    return Settings(_env_file=None, **{**TEST_SETTINGS, **overrides})  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def hermetic_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """Install the suite's default settings with no environment or ``.env`` influence."""
    for name in list(os.environ):
        if name.startswith("RAG_") or name in _UNPREFIXED_VARIABLES:
            monkeypatch.delenv(name)
    defaults = test_settings()
    set_settings(defaults)
    set_embedder(None)
    yield defaults
    set_embedder(None)
    set_settings(None)


@pytest.fixture
def settings() -> Callable[..., Settings]:
    """Factory installing settings with the given field overrides for the test."""

    def install(**overrides: object) -> Settings:
        installed = test_settings(**overrides)
        set_settings(installed)
        return installed

    return install
