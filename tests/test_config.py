from pathlib import Path

import pytest
import requests
from pydantic import SecretStr, ValidationError

from ragsvc import config
from ragsvc.config import Settings, choose_default_provider, is_configured_secret


def test_defaults_are_local_and_need_no_credentials():
    settings = Settings(_env_file=None)

    assert settings.llm_provider is None
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.openai_api_key is None and not settings.openai_configured
    assert settings.openai_base_url == "https://api.openai.com/v1"
    assert settings.embed_model_name == "all-MiniLM-L6-v2"
    assert settings.rerank_method == "cross_encoder"
    assert (settings.chunk_size, settings.chunk_overlap, settings.hybrid_alpha) == (400, 40, 0.7)
    assert settings.api_host == "127.0.0.1"
    assert settings.api_port is None
    assert settings.cors_allow_origins == ()
    assert settings.max_upload_bytes == 50 * 1024 * 1024
    assert settings.api_token is None
    assert not settings.serpapi_configured


def test_rag_prefixed_environment_variables_are_read(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "120")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "12")
    monkeypatch.setenv("RAG_HYBRID_ALPHA", "0.25")
    monkeypatch.setenv("RAG_LLM_PROVIDER", "openai")
    monkeypatch.setenv("RAG_RERANK_METHOD", "none")
    monkeypatch.setenv("RAG_API_PORT", "8080")
    monkeypatch.setenv("RAG_API_TOKEN", "s3cret")
    monkeypatch.setenv("RAG_OLLAMA_BASE_URL", "http://ollama.internal:11434/")

    settings = Settings(_env_file=None)

    assert (settings.chunk_size, settings.chunk_overlap, settings.hybrid_alpha) == (120, 12, 0.25)
    assert settings.llm_provider == "openai"
    assert settings.rerank_method == "none"
    assert settings.api_port == 8080
    assert settings.api_token is not None and settings.api_token.get_secret_value() == "s3cret"
    assert settings.ollama_base_url == "http://ollama.internal:11434"


def test_cors_origins_are_parsed_from_a_comma_separated_list(monkeypatch):
    monkeypatch.setenv("RAG_CORS_ALLOW_ORIGINS", " http://localhost:3000, https://ui.example ,")

    assert Settings(_env_file=None).cors_allow_origins == ("http://localhost:3000", "https://ui.example")
    assert Settings(_env_file=None, cors_allow_origins=["https://a.example"]).cors_allow_origins == (
        "https://a.example",
    )


def test_conventional_credential_names_are_accepted_and_the_prefixed_name_wins(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-conventional")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.example/v1/")
    monkeypatch.setenv("SERPAPI_KEY", "serp-key")
    assert Settings(_env_file=None).openai_api_key == SecretStr("sk-conventional")
    assert Settings(_env_file=None).openai_base_url == "https://llm.example/v1"
    assert Settings(_env_file=None).serpapi_configured

    monkeypatch.setenv("RAG_OPENAI_API_KEY", "sk-prefixed")

    assert Settings(_env_file=None).openai_api_key == SecretStr("sk-prefixed")


def test_placeholder_and_blank_secrets_count_as_unset(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "Your_OPENAI_API_KEY")
    monkeypatch.setenv("RAG_API_TOKEN", "   ")

    settings = Settings(_env_file=None)

    assert settings.openai_api_key is None and not settings.openai_configured
    assert settings.api_token is None
    assert is_configured_secret(None) is False
    assert is_configured_secret("") is False
    assert is_configured_secret(SecretStr("Your_KEY")) is False
    assert is_configured_secret(SecretStr("sk-real-value")) is True


def test_dotenv_file_in_the_working_directory_is_read_below_the_environment(tmp_path: Path, monkeypatch):
    (tmp_path / ".env").write_text("RAG_CHUNK_SIZE=99\nRAG_OLLAMA_MODEL=phi3\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAG_OLLAMA_MODEL", "llama3.2")

    settings = Settings()

    assert settings.chunk_size == 99
    assert settings.ollama_model == "llama3.2"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"llm_provider": "siliconflow"}, "llm_provider"),
        ({"rerank_method": "bge"}, "rerank_method"),
        ({"hybrid_alpha": 1.5}, "hybrid_alpha"),
        ({"chunk_size": 0}, "chunk_size"),
        ({"chunk_size": 100, "chunk_overlap": 100}, "chunk_overlap"),
        ({"api_port": 70000}, "api_port"),
        ({"max_upload_mb": 0}, "max_upload_mb"),
    ],
)
def test_invalid_values_are_rejected_on_load_with_the_field_named(overrides, message):
    with pytest.raises(ValidationError, match=message):
        Settings(_env_file=None, **overrides)


def test_unknown_rag_variables_are_ignored(monkeypatch):
    monkeypatch.setenv("RAG_SILICONFLOW_API_KEY", "leftover")

    Settings(_env_file=None)


def test_settings_are_frozen():
    settings = Settings(_env_file=None)

    with pytest.raises(ValidationError):
        settings.chunk_size = 10  # type: ignore[misc]


def test_get_settings_loads_once_and_set_settings_replaces(settings):
    first = config.get_settings()
    assert config.get_settings() is first

    replaced = settings(chunk_size=64, chunk_overlap=8)

    assert config.get_settings() is replaced
    assert config.get_settings().chunk_size == 64


# --- Provider selection -----------------------------------------------------


def test_default_provider_prefers_local_then_configured_key():
    assert choose_default_provider(None, True, True) == "ollama"
    assert choose_default_provider(None, False, True) == "openai"
    assert choose_default_provider(None, False, False) == "ollama"


def test_explicit_provider_wins():
    assert choose_default_provider("openai", True, False) == "openai"
    assert choose_default_provider("ollama", False, True) == "ollama"


def test_resolve_provider_validates_request_and_falls_back_to_default(monkeypatch):
    monkeypatch.setattr(config, "detect_default_provider", lambda: "openai")

    assert config.resolve_provider(None) == "openai"
    assert config.resolve_provider("ollama") == "ollama"
    with pytest.raises(ValueError, match="provider must be one of"):
        config.resolve_provider("magick")


def test_detect_default_provider_does_not_probe_when_set_explicitly(settings, monkeypatch):
    def unexpected_probe(*args, **kwargs):
        raise AssertionError("an explicit RAG_LLM_PROVIDER must not probe Ollama")

    settings(llm_provider="openai")
    monkeypatch.setattr(config, "ollama_available", unexpected_probe)

    assert config.detect_default_provider() == "openai"


def test_detect_default_provider_probes_once_and_is_reset_by_new_settings(settings, monkeypatch):
    probes: list[str] = []

    def probe(base_url, timeout=2.0):
        probes.append(base_url)
        return False

    monkeypatch.setattr(config, "ollama_available", probe)
    settings(openai_api_key="sk-test")

    assert config.detect_default_provider() == "openai"
    assert config.detect_default_provider() == "openai"
    assert probes == ["http://localhost:11434"]

    settings(ollama_base_url="http://other:11434")

    assert config.detect_default_provider() == "ollama"
    assert probes == ["http://localhost:11434", "http://other:11434"]


def test_ollama_available_is_false_when_nothing_listens(monkeypatch):
    def refused(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "get", refused)

    assert config.ollama_available("http://127.0.0.1:1") is False
