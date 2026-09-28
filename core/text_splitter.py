"""Text splitter: cut long text into retrieval-sized, overlapping chunks.

``chunk_size`` bounds each chunk in characters: too large and retrieval gets
coarse, too small and chunks lose context. ``chunk_overlap`` repeats the tail
of one chunk at the head of the next so facts are not cut in half.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE

# Tried in order: paragraph, line, CJK full stop / comma / semicolon / colon,
# space, then individual characters as a last resort.
SEPARATORS = ["\n\n", "\n", "\u3002", "\uff0c", "\uff1b", "\uff1a", " ", ""]


def split_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[str]:
    """Split ``text`` recursively by ``SEPARATORS`` until chunks fit ``chunk_size``.

    ``chunk_size`` and ``chunk_overlap`` default to the configured values.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE if chunk_size is None else chunk_size,
        chunk_overlap=CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap,
        separators=SEPARATORS,
    )
    return splitter.split_text(text)
