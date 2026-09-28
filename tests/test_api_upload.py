import os

import pytest
from fastapi.testclient import TestClient

from ragsvc import api
from ragsvc.core.ingest import FileResult, IngestReport


@pytest.fixture
def client():
    return TestClient(api.create_app())


def test_upload_returns_chunk_count_and_removes_temp_file(client, monkeypatch):
    seen: dict[str, object] = {}

    def fake_ingest(sources, progress=None):
        (source,) = sources
        seen["name"] = source.name
        seen["path"] = str(source.path)
        seen["content"] = source.path.read_bytes()
        return IngestReport(files=[FileResult(name=source.name, chunks=3)], total_chunks=3)

    monkeypatch.setattr(api, "ingest_files", fake_ingest)

    response = client.post(
        "/api/upload", files={"file": ("notes.md", b"# Notes\n\nHybrid retrieval", "text/markdown")}
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": "notes.md: indexed 3 chunk(s)",
        "file_info": {"filename": "notes.md", "chunks": 3},
    }
    assert seen["name"] == "notes.md"
    assert str(seen["path"]).endswith(".md")
    assert seen["content"] == b"# Notes\n\nHybrid retrieval"
    assert not os.path.exists(str(seen["path"]))


@pytest.mark.parametrize("filename", ["report.csv", "archive", "image.PNG"])
def test_upload_rejects_unsupported_formats_before_ingesting(client, monkeypatch, filename):
    def unexpected_ingest(sources, progress=None):
        raise AssertionError("an unsupported format must be rejected before ingestion starts")

    monkeypatch.setattr(api, "ingest_files", unexpected_ingest)

    response = client.post("/api/upload", files={"file": (filename, b"payload", "application/octet-stream")})

    assert response.status_code == 415
    detail = response.json()["detail"]
    assert detail.startswith("unsupported file format")
    assert ".pdf" in detail and ".md" in detail


def test_upload_reports_parse_failures_as_error_status(client, monkeypatch):
    monkeypatch.setattr(
        api,
        "ingest_files",
        lambda sources, progress=None: IngestReport(
            files=[FileResult(name="broken.pdf", chunks=0, error="document is empty")],
            total_chunks=0,
        ),
    )

    response = client.post("/api/upload", files={"file": ("broken.pdf", b"", "application/pdf")})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert body["message"] == "broken.pdf: document is empty"
    assert body["file_info"] == {"filename": "broken.pdf", "chunks": 0}
