"""Conflict detection: flag disagreeing facts across context sources.

When local documents and web results are combined, they may state different
numbers for the same thing. A detected conflict makes the answer prompt ask
the LLM to point out the discrepancy instead of silently picking one.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

# Four-digit years and percentages; the values most likely to disagree across sources.
_NUMERIC_FACT = re.compile(r"\b\d{4}\b|\b\d+(?:\.\d+)?%")


def detect_conflicts(sources: Iterable[dict[str, Any]]) -> bool:
    """True when two sources disagree on the same extracted fact.

    Each source is a mapping with the text under ``text`` (or ``excerpt``).
    """
    key_facts: dict[str, frozenset[str]] = {}
    for item in sources:
        text = item["text"] if "text" in item else item.get("excerpt", "")
        for fact, value in _extract_facts(text).items():
            if fact in key_facts and key_facts[fact] != value:
                return True
            key_facts[fact] = value
    return False


def _extract_facts(text: str) -> dict[str, frozenset[str]]:
    """Extract comparable facts; order of mention does not matter."""
    facts: dict[str, frozenset[str]] = {}
    numbers = _NUMERIC_FACT.findall(text)
    if numbers:
        facts["numbers"] = frozenset(numbers)
    return facts
