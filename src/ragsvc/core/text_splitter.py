"""Text splitter: cut long text into retrieval-sized, overlapping chunks.

``chunk_size`` bounds each chunk in characters: too large and retrieval gets
coarse, too small and chunks lose context. ``chunk_overlap`` repeats the tail
of one chunk at the head of the next so facts are not cut in half.

The splitting is recursive: the text is cut on the first separator it
contains (paragraph, line, CJK punctuation, space, then single characters),
pieces that fit are merged back together up to ``chunk_size`` with the
requested overlap, and pieces that are still too long are cut again with
the remaining separators. Separators stay attached to the start of the
piece that follows them. This is the scheme of LangChain's recursive
character splitter, kept dependency-free here because importing that
package loads sentence-transformers and torch when they are installed,
which cost the demo and the tests tens of seconds on a cold start.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ragsvc.config import get_settings

# Tried in order: paragraph, line, CJK full stop / comma / semicolon / colon,
# space, then individual characters as a last resort.
SEPARATORS = ["\n\n", "\n", "\u3002", "\uff0c", "\uff1b", "\uff1a", " ", ""]


def split_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[str]:
    """Split ``text`` recursively by ``SEPARATORS`` until chunks fit ``chunk_size``.

    ``chunk_size`` and ``chunk_overlap`` default to the configured values.
    """
    settings = get_settings()
    size = settings.chunk_size if chunk_size is None else chunk_size
    overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    if size < 1:
        raise ValueError(f"chunk_size must be at least 1, got {size}")
    if overlap < 0 or overlap >= size:
        raise ValueError(f"chunk_overlap must be in [0, chunk_size), got {overlap}")
    return _split(text, SEPARATORS, size, overlap)


def _split_on(text: str, separator: str) -> list[str]:
    """Cut ``text`` on ``separator``, keeping each separator at the start of the next piece."""
    if separator == "":
        return [char for char in text if char]
    parts = re.split(f"({re.escape(separator)})", text)
    pieces = [parts[0]] + [parts[i] + parts[i + 1] for i in range(1, len(parts) - 1, 2)]
    if len(parts) % 2 == 0:
        pieces.append(parts[-1])
    return [piece for piece in pieces if piece]


def _split(text: str, separators: Sequence[str], size: int, overlap: int) -> list[str]:
    separator = separators[-1]
    remaining: Sequence[str] = []
    for index, candidate in enumerate(separators):
        if candidate == "" or candidate in text:
            separator = candidate
            remaining = separators[index + 1 :]
            break

    chunks: list[str] = []
    fitting: list[str] = []
    for piece in _split_on(text, separator):
        if len(piece) < size:
            fitting.append(piece)
            continue
        if fitting:
            chunks.extend(_merge(fitting, size, overlap))
            fitting = []
        if remaining:
            chunks.extend(_split(piece, remaining, size, overlap))
        else:
            chunks.append(piece)
    if fitting:
        chunks.extend(_merge(fitting, size, overlap))
    return chunks


def _merge(pieces: Sequence[str], size: int, overlap: int) -> list[str]:
    """Concatenate consecutive pieces up to ``size`` characters, repeating up to ``overlap`` of the tail."""
    chunks: list[str] = []
    window: list[str] = []
    total = 0
    for piece in pieces:
        if total + len(piece) > size and window:
            chunk = "".join(window).strip()
            if chunk:
                chunks.append(chunk)
            while total > overlap or (total + len(piece) > size and total > 0):
                total -= len(window[0])
                window = window[1:]
        window.append(piece)
        total += len(piece)
    chunk = "".join(window).strip()
    if chunk:
        chunks.append(chunk)
    return chunks
