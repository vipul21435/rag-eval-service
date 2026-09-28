import os
import tempfile
from typing import Any

import pytest
from fastapi.testclient import TestClient

import api_router
from core.ingest import FileResult, IngestReport

EVIL = "https://evil.example"
ALLOWED = "https://ui.example"


@pytest.fixture(autouse=True)
def fixed_provider(monkeypatch):
    monkeypatch.setattr(api_router, "detect_default_provider", lambda: "ollama")


def preflight(client: TestClient, origin: str, path: str = "/api/ask"):
    return client.options(
        path,
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )


# --- CORS -------------------------------------------------------------------


def test_default_app_sends_no_cors_headers_to_any_origin():
    client = TestClient(api_router.app)

    assert "access-control-allow-origin" not in preflight(client, EVIL).headers
    response = client.get("/api/status", headers={"Origin": EVIL, "Cookie": "session=abc"})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
    assert "access-control-allow-credentials" not in response.headers


def test_allowlisted_origin_is_admitted_without_credentials_and_others_are_not():
    client = TestClient(api_router.create_app(cors_origins=[ALLOWED]))

    allowed = preflight(client, ALLOWED)
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == ALLOWED
    assert "access-control-allow-credentials" not in allowed.headers

    denied = preflight(client, EVIL)
    assert "access-control-allow-origin" not in denied.headers
    actual = client.get("/api/status", headers={"Origin": EVIL})
    assert "access-control-allow-origin" not in actual.headers


def test_app_factory_defaults_to_the_configured_allowlist(monkeypatch):
    monkeypatch.setattr(api_router, "CORS_ALLOW_ORIGINS", (ALLOWED,))
    client = TestClient(api_router.create_app())

    assert preflight(client, ALLOWED).headers["access-control-allow-origin"] == ALLOWED


# --- Upload size ------------------------------------------------------------


def test_upload_over_the_size_limit_is_rejected_before_ingestion(monkeypatch, tmp_path):
    monkeypatch.setattr(api_router, "MAX_UPLOAD_BYTES", 16)
    monkeypatch.setattr(api_router, "_UPLOAD_CHUNK_BYTES", 4)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    def unexpected_ingest(sources, progress=None):
        raise AssertionError("an oversized upload must be rejected before ingestion starts")

    monkeypatch.setattr(api_router, "ingest_files", unexpected_ingest)
    client = TestClient(api_router.app)

    response = client.post("/api/upload", files={"file": ("big.md", b"x" * 17, "text/markdown")})

    assert response.status_code == 413
    assert "upload limit of 16 bytes" in response.json()["detail"]
    assert list(tmp_path.iterdir()) == [], "the partial temp file must be removed"


def test_upload_at_the_size_limit_is_accepted(monkeypatch):
    monkeypatch.setattr(api_router, "MAX_UPLOAD_BYTES", 16)
    monkeypatch.setattr(api_router, "_UPLOAD_CHUNK_BYTES", 4)
    seen: dict[str, object] = {}

    def fake_ingest(sources, progress=None):
        (source,) = sources
        seen["content"] = source.path.read_bytes()
        seen["path"] = str(source.path)
        return IngestReport(files=[FileResult(name=source.name, chunks=1)], total_chunks=1)

    monkeypatch.setattr(api_router, "ingest_files", fake_ingest)
    client = TestClient(api_router.app)

    response = client.post("/api/upload", files={"file": ("ok.md", b"x" * 16, "text/markdown")})

    assert response.status_code == 200
    assert seen["content"] == b"x" * 16
    assert not os.path.exists(str(seen["path"]))


# --- Bearer token -----------------------------------------------------------


@pytest.fixture
def token_client(monkeypatch):
    monkeypatch.setattr(api_router, "API_TOKEN", "s3cret")
    return TestClient(api_router.app)


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic s3cret"}, {"Authorization": "s3cret"}],
    ids=["missing", "wrong", "wrong-scheme", "no-scheme"],
)
def test_every_api_route_requires_the_token_when_configured(token_client, headers):
    requests: list[tuple[str, str, dict[str, Any]]] = [
        ("get", "/api/status", {}),
        ("post", "/api/ask", {"json": {"question": "hi"}}),
        ("post", "/api/upload", {"files": {"file": ("a.md", b"text", "text/markdown")}}),
    ]
    for method, path, kwargs in requests:
        response = getattr(token_client, method)(path, headers=headers, **kwargs)
        assert response.status_code == 401, (method, path)
        assert response.headers["www-authenticate"] == "Bearer"


def test_valid_bearer_token_is_accepted(token_client):
    response = token_client.get("/api/status", headers={"Authorization": "Bearer s3cret"})

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_no_token_configured_means_open_access(monkeypatch):
    monkeypatch.setattr(api_router, "API_TOKEN", None)

    assert TestClient(api_router.app).get("/api/status").status_code == 200
