from ragsvc.core.text_splitter import split_text


def test_split_text_respects_chunk_size_and_overlap():
    words = [f"word{i}" for i in range(60)]
    text = " ".join(words)

    chunks = split_text(text, chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(len(chunk) <= 50 for chunk in chunks)
    # Every word survives splitting (overlap only repeats, never drops, text).
    assert set(words) <= {word for chunk in chunks for word in chunk.split()}


def test_split_text_prefers_paragraph_boundaries():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

    chunks = split_text(text, chunk_size=20, chunk_overlap=0)

    assert chunks == ["First paragraph.", "Second paragraph.", "Third paragraph."]


def test_split_text_uses_configured_defaults():
    from ragsvc.config import CHUNK_SIZE

    chunks = split_text("short text")

    assert chunks == ["short text"]
    assert all(len(chunk) <= CHUNK_SIZE for chunk in chunks)
