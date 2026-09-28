"""Thinking-chain handling for reasoning models.

Models such as DeepSeek-R1 and Qwen3 wrap their reasoning in
``<think>...</think>``. The service is API-only, so the reasoning is not
rendered: it is separated from the answer and returned as its own field,
leaving any HTML rendering (and escaping) to the client.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass(frozen=True)
class ThinkingSplit:
    """A model response split into its visible answer and its reasoning."""

    answer: str
    reasoning: str | None = None


def split_thinking(text: Any) -> ThinkingSplit:
    """Separate ``<think>`` blocks from the rest of a model response.

    Every ``<think>...</think>`` span, wherever it appears, is removed from the
    answer and collected as reasoning. Two truncated shapes are handled as
    well: a ``<think>`` that is never closed (the model hit its token limit
    mid-reasoning) takes the rest of the text as reasoning, and a ``</think>``
    with no opening tag (chat templates that pre-fill ``<think>``) takes the
    text before it. Both parts are stripped of surrounding whitespace.
    """
    if text is None:
        return ThinkingSplit(answer="")
    processed = text if isinstance(text, str) else str(text)

    reasoning_parts: list[str] = []

    first_close = processed.find(_THINK_CLOSE)
    first_open = processed.find(_THINK_OPEN)
    if first_close != -1 and (first_open == -1 or first_close < first_open):
        reasoning_parts.append(processed[:first_close])
        processed = processed[first_close + len(_THINK_CLOSE) :]

    def collect(match: re.Match[str]) -> str:
        reasoning_parts.append(match.group(1))
        return ""

    processed = _THINK_BLOCK.sub(collect, processed)

    unclosed = processed.find(_THINK_OPEN)
    if unclosed != -1:
        reasoning_parts.append(processed[unclosed + len(_THINK_OPEN) :])
        processed = processed[:unclosed]

    reasoning = "\n\n".join(part.strip() for part in reasoning_parts if part.strip())
    return ThinkingSplit(answer=processed.strip(), reasoning=reasoning or None)
