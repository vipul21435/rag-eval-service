from pathlib import Path

from core.document_loader import extract_text


def test_extract_utf8_text_and_markdown(tmp_path: Path):
    text_file = tmp_path / "sample.txt"
    # Non-ASCII (CJK) content must round-trip through UTF-8.
    text_file.write_text(
        "RAG combines retrieval and generation.\n\u4e2d\u6587\u5185\u5bb9\u3002", encoding="utf-8"
    )

    markdown_file = tmp_path / "sample.md"
    markdown_file.write_text("# Heading\n\nHybrid retrieval", encoding="utf-8")

    assert "\u4e2d\u6587\u5185\u5bb9" in extract_text(str(text_file))
    assert "Hybrid retrieval" in extract_text(str(markdown_file))


def test_unsupported_extension_returns_empty_string(tmp_path: Path):
    unsupported = tmp_path / "sample.bin"
    unsupported.write_bytes(b"not a supported document")

    assert extract_text(str(unsupported)) == ""
