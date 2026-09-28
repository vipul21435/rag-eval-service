import pytest
import requests

from ragsvc import config


def test_api_key_validation_rejects_placeholders():
    assert config.is_configured_api_key(None) is False
    assert config.is_configured_api_key("") is False
    assert config.is_configured_api_key("Your_OPENAI_API_KEY") is False
    assert config.is_configured_api_key("sk-real-value") is True


def test_default_provider_prefers_local_then_configured_key():
    assert config.choose_default_provider(None, True, "sk-openai") == "ollama"
    assert config.choose_default_provider(None, False, "sk-openai") == "openai"
    assert config.choose_default_provider(None, False, None) == "ollama"
    assert config.choose_default_provider(None, False, "Your_OPENAI_API_KEY") == "ollama"


def test_explicit_provider_wins_and_is_validated():
    assert config.choose_default_provider("openai", True, None) == "openai"
    assert config.choose_default_provider("ollama", False, "sk-openai") == "ollama"
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        config.choose_default_provider("siliconflow", False, None)


def test_resolve_provider_validates_request_and_falls_back_to_default(monkeypatch):
    monkeypatch.setattr(config, "detect_default_provider", lambda: "openai")

    assert config.resolve_provider(None) == "openai"
    assert config.resolve_provider("ollama") == "ollama"
    with pytest.raises(ValueError, match="provider must be one of"):
        config.resolve_provider("magick")


def test_detect_default_provider_does_not_probe_when_set_explicitly(monkeypatch):
    def unexpected_probe(*args, **kwargs):
        raise AssertionError("an explicit LLM_PROVIDER must not probe Ollama")

    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "ollama_available", unexpected_probe)
    config.detect_default_provider.cache_clear()
    try:
        assert config.detect_default_provider() == "openai"
    finally:
        config.detect_default_provider.cache_clear()


def test_ollama_available_is_false_when_nothing_listens(monkeypatch):
    def refused(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "get", refused)

    assert config.ollama_available("http://127.0.0.1:1") is False


def test_defaults_are_local():
    import os

    assert config.OLLAMA_BASE_URL.startswith("http://localhost")
    assert config.EMBED_MODEL_NAME == "all-MiniLM-L6-v2"
    assert config.OPENAI_BASE_URL == "https://api.openai.com/v1"
    assert os.environ.get("HF_ENDPOINT", "") != "https://hf-mirror.com"
