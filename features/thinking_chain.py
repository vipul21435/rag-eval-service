"""Thinking-chain formatting for reasoning models.

Models such as DeepSeek-R1 wrap their reasoning in ``<think>...</think>``.
This module turns each such block into a collapsible HTML ``<details>``
element and escapes every other angle bracket so model output cannot inject
markup.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_KEPT_TAG_PREFIXES = ("<details", "</details", "<summary", "</summary")


def process_thinking_content(text: Any) -> str:
    """Render ``<think>`` blocks as ``<details>`` and escape all other tags."""
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            processed = str(text)
        except Exception:  # noqa: BLE001 - defensive: __str__ of arbitrary objects
            return "Unprocessable content"
    else:
        processed = text

    try:
        while _THINK_OPEN in processed and _THINK_CLOSE in processed:
            start = processed.find(_THINK_OPEN)
            end = processed.find(_THINK_CLOSE)
            if end <= start:
                break
            reasoning = processed[start + len(_THINK_OPEN) : end]
            processed = (
                processed[:start]
                + "\n\n<details>\n<summary>Reasoning (click to expand)</summary>\n\n"
                + reasoning
                + "\n\n</details>\n\n"
                + processed[end + len(_THINK_CLOSE) :]
            )
        return _escape_except_details(processed)
    except Exception as exc:  # noqa: BLE001 - never let formatting take the answer down
        logger.error("Failed to format thinking content: %s", exc)
        try:
            return str(text).replace("<", "&lt;").replace(">", "&gt;")
        except Exception:  # noqa: BLE001
            return "Failed to format content"


def _escape_except_details(text: str) -> str:
    """HTML-escape angle brackets, keeping ``<details>`` and ``<summary>`` tags intact."""
    out: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith(_KEPT_TAG_PREFIXES, i):
            tag_end = text.find(">", i)
            if tag_end != -1:
                out.append(text[i : tag_end + 1])
                i = tag_end + 1
                continue
        char = text[i]
        if char == "<":
            out.append("&lt;")
        elif char == ">":
            out.append("&gt;")
        else:
            out.append(char)
        i += 1
    return "".join(out)
