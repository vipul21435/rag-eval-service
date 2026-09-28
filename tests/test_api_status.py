from fastapi.testclient import TestClient

from ragsvc import __version__, api


def test_status_reports_version_and_local_defaults_without_credentials(monkeypatch):
    monkeypatch.setattr(api, "detect_default_provider", lambda: "ollama")
    client = TestClient(api.create_app())

    status = client.get("/api/status").json()

    assert status["status"] == "healthy"
    assert status["version"] == __version__
    assert status["default_provider"] == "ollama"
    assert status["openai_configured"] is False
    assert status["serpapi_configured"] is False
    assert status["vector_store_ready"] is False
    assert status["total_chunks"] == 0
