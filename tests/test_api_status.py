from fastapi.testclient import TestClient

import api_router
from version import __version__


def test_status_reports_version_and_local_defaults_without_credentials(monkeypatch):
    monkeypatch.setattr(api_router, "detect_default_provider", lambda: "ollama")
    monkeypatch.setattr(api_router, "OPENAI_API_KEY", None)
    client = TestClient(api_router.app)

    status = client.get("/api/status").json()

    assert status["status"] == "healthy"
    assert status["version"] == __version__
    assert status["default_provider"] == "ollama"
    assert status["openai_configured"] is False
    assert status["serpapi_configured"] is False
    assert status["vector_store_ready"] is False
    assert status["total_chunks"] == 0
