import sys
from pathlib import Path

import pytest

from ragsvc.core.document_loader import (
    SUPPORTED_EXTENSIONS,
    MissingParserError,
    UnsupportedFormatError,
    describe_supported_formats,
    extract_text,
)


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


def test_unsupported_extension_raises_with_the_supported_list(tmp_path: Path):
    unsupported = tmp_path / "sample.bin"
    unsupported.write_bytes(b"not a supported document")

    with pytest.raises(UnsupportedFormatError, match=r"unsupported file format '\.bin'") as excinfo:
        extract_text(str(unsupported))
    assert describe_supported_formats() in str(excinfo.value)
    assert set(describe_supported_formats().split(", ")) == set(SUPPORTED_EXTENSIONS)


def test_non_utf8_text_file_is_reported_clearly(tmp_path: Path):
    latin1 = tmp_path / "latin1.txt"
    latin1.write_bytes("caf\u00e9".encode("latin-1"))

    with pytest.raises(ValueError, match="not UTF-8"):
        extract_text(str(latin1))


@pytest.mark.parametrize(
    ("extension", "module", "package"),
    [(".docx", "docx", "python-docx"), (".pptx", "pptx", "python-pptx"), (".xlsx", "pandas", "pandas")],
)
def test_missing_optional_parser_names_the_documents_extra(
    tmp_path: Path, monkeypatch, extension, module, package
):
    # A None entry in sys.modules makes the import fail as if the package were absent.
    monkeypatch.setitem(sys.modules, module, None)
    document = tmp_path / f"sample{extension}"
    document.write_bytes(b"PK\x03\x04")

    with pytest.raises(MissingParserError, match=f"requires {package}; install the 'documents' extra"):
        extract_text(str(document))
