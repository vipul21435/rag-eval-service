import pytest

from ragsvc.core.generator import ProviderError, _call_openai_compatible_api


def test_cloud_provider_fails_cleanly_without_api_key(monkeypatch):
    def unexpected_network_call(*args, **kwargs):
        raise AssertionError("a missing API key must not trigger an HTTP request")

    monkeypatch.setattr("ragsvc.core.generator.requests.post", unexpected_network_call)

    with pytest.raises(ProviderError, match="Test Provider API key is not configured"):
        _call_openai_compatible_api(
            provider_name="Test Provider",
            api_key=None,
            api_url="https://example.com/v1",
            model_name="example-model",
            prompt="hello",
        )


def test_cloud_provider_fails_cleanly_without_api_url(monkeypatch):
    def unexpected_network_call(*args, **kwargs):
        raise AssertionError("a missing API URL must not trigger an HTTP request")

    monkeypatch.setattr("ragsvc.core.generator.requests.post", unexpected_network_call)

    with pytest.raises(ProviderError, match="Test Provider API URL is not configured"):
        _call_openai_compatible_api(
            provider_name="Test Provider",
            api_key="sk-test",
            api_url="",
            model_name="example-model",
            prompt="hello",
        )
