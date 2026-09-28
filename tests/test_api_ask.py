import pytest
from fastapi.testclient import TestClient

import api_router
from core.generator import Answer, KnowledgeBaseEmptyError, ProviderError


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_router, "detect_default_provider", lambda: "ollama")
    return TestClient(api_router.app)


def test_ask_returns_answer_sources_and_metadata(client, monkeypatch):
    calls: list[tuple[str, bool, str]] = []

    def fake_answer(question, enable_web_search, provider):
        calls.append((question, enable_web_search, provider))
        return Answer(
            text="Hybrid retrieval merges dense and sparse scores.",
            sources=[{"type": "local", "source": "notes.md"}],
            conflict_detected=False,
            provider=provider,
        )

    monkeypatch.setattr(api_router, "answer_question", fake_answer)

    response = client.post("/api/ask", json={"question": "What is hybrid retrieval?", "provider": "openai"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Hybrid retrieval merges dense and sparse scores.",
        "sources": [{"type": "local", "source": "notes.md"}],
        "metadata": {"enable_web_search": False, "provider": "openai", "conflict_detected": False},
    }
    assert calls == [("What is hybrid retrieval?", False, "openai")]


def test_ask_uses_detected_default_provider(client, monkeypatch):
    seen: list[str] = []

    def fake_answer(question, enable_web_search, provider):
        seen.append(provider)
        return Answer(text="ok", provider=provider)

    monkeypatch.setattr(api_router, "answer_question", fake_answer)

    response = client.post("/api/ask", json={"question": "hello"})

    assert response.status_code == 200
    assert seen == ["ollama"]


def test_ask_rejects_blank_question_and_unknown_provider(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/ask", json={"question": "hi", "provider": "magick"}).status_code == 422


def test_ask_maps_empty_knowledge_base_to_409(client, monkeypatch):
    def empty(question, enable_web_search, provider):
        raise KnowledgeBaseEmptyError("The knowledge base is empty; upload documents first")

    monkeypatch.setattr(api_router, "answer_question", empty)

    response = client.post("/api/ask", json={"question": "anything"})

    assert response.status_code == 409
    assert "knowledge base is empty" in response.json()["detail"]


def test_ask_maps_provider_failure_to_502(client, monkeypatch):
    def failing(question, enable_web_search, provider):
        raise ProviderError("Ollama request to http://localhost:11434 failed: connection refused")

    monkeypatch.setattr(api_router, "answer_question", failing)

    response = client.post("/api/ask", json={"question": "anything"})

    assert response.status_code == 502
    assert "connection refused" in response.json()["detail"]
