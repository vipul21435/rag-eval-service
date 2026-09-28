"""Shared fixtures.

Every test runs against a fresh, hermetic ``Settings`` instance: ``RAG_*``
variables and third-party credentials are removed from the environment and
no ``.env`` file is read, so results do not depend on the developer's
machine. Override individual values with the ``settings`` fixture.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator

import pytest

from ragsvc.config import Settings, set_settings

_UNPREFIXED_VARIABLES = frozenset({"OPENAI_API_KEY", "OPENAI_BASE_URL", "SERPAPI_KEY"})


@pytest.fixture(autouse=True)
def hermetic_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """Install default settings with no environment or ``.env`` influence."""
    for name in list(os.environ):
        if name.startswith("RAG_") or name in _UNPREFIXED_VARIABLES:
            monkeypatch.delenv(name)
    defaults = Settings(_env_file=None)
    set_settings(defaults)
    yield defaults
    set_settings(None)


@pytest.fixture
def settings() -> Callable[..., Settings]:
    """Factory installing settings with the given field overrides for the test."""

    def install(**overrides: object) -> Settings:
        installed = Settings(_env_file=None, **overrides)  # type: ignore[arg-type]
        set_settings(installed)
        return installed

    return install
