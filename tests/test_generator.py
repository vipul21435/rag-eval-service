import numpy as np
import pytest

import core.generator as generator
from core.generator import (
    Answer,
    KnowledgeBaseEmptyError,
    ProviderError,
    _extract_openai_compatible_content,
    _normalize_chat_completions_url,
    answer_question,
    call_llm_simple,
)
from core.vector_store import vector_store


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://api.example.com/v1", "https://api.example.com/v1/chat/completions"),
        ("https://api.example.com/v1/", "https://api.example.com/v1/chat/completions"),
        ("https://api.example.com/v1/chat/completions", "https://api.example.com/v1/chat/completions"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_chat_completions_url(url, expected):
    assert _normalize_chat_completions_url(url) == expected


def test_extract_content_handles_text_parts_and_reasoning():
    result = {
        "choices": [
            {
                "message": {
                    "content": [{"type": "text", "text": "Hello "}, "world"],
                    "reasoning_content": "thinking",
                }
            }
        ]
    }

    assert _extract_openai_compatible_content(result) == "Hello world<think>thinking</think>"


def test_extract_content_rejects_empty_choices():
    with pytest.raises(ProviderError, match="no choices"):
        _extract_openai_compatible_content({"choices": []})


@pytest.mark.parametrize(
    "raw",
    [
        "  refined query <think>why</think>",
        "<think>why</think>\n\nrefined query",
        "<think>first</think>refined <think>second</think>query",
    ],
    ids=["trailing", "leading", "interleaved"],
)
def test_call_llm_simple_drops_reasoning_wherever_it_appears(monkeypatch, raw):
    monkeypatch.setattr(generator, "call_llm", lambda prompt, provider, **kw: raw)

    assert call_llm_simple("prompt", "ollama") == "refined query"


def test_call_llm_simple_keeps_sentinel_after_leading_reasoning(monkeypatch):
    # deepseek-r1 / qwen3 served by Ollama reason first, then answer.
    monkeypatch.setattr(
        generator,
        "call_llm",
        lambda prompt, provider, **kw: "<think>The summary already answers it.</think>\n\nNO_FURTHER_QUERY",
    )

    assert call_llm_simple("prompt", "ollama") == "NO_FURTHER_QUERY"


def test_answer_question_requires_documents_unless_web_search():
    vector_store.clear()

    with pytest.raises(KnowledgeBaseEmptyError):
        answer_question("anything", enable_web_search=False, provider="ollama")


def test_answer_question_builds_prompt_from_retrieved_context(monkeypatch):
    prompts: list[tuple[str, str]] = []

    vector_store.build_index(
        ["chunk"], ["doc_1_chunk_0"], [{"source": "report.pdf"}], np.zeros((1, 4), dtype=np.float32)
    )
    monkeypatch.setattr(
        generator,
        "recursive_retrieval",
        lambda initial_query, enable_web_search, provider: (
            ["Revenue grew 12% in 2024."],
            ["doc_1_chunk_0"],
            [{"source": "report.pdf", "doc_id": "doc_1"}],
        ),
    )

    def fake_llm(prompt, provider, temperature=0.7, max_tokens=1024):
        prompts.append((prompt, provider))
        return "Revenue grew 12%.<think>checked the report</think>"

    monkeypatch.setattr(generator, "call_llm", fake_llm)

    try:
        answer = answer_question("How much did revenue grow?", provider="openai")
    finally:
        vector_store.clear()

    assert isinstance(answer, Answer)
    assert answer.text == "Revenue grew 12%."
    assert answer.reasoning == "checked the report"
    assert answer.sources == [{"type": "local", "source": "report.pdf"}]
    assert answer.conflict_detected is False
    assert answer.provider == "openai"
    prompt, provider = prompts[0]
    assert provider == "openai"
    assert "[Local document: report.pdf]\nRevenue grew 12% in 2024." in prompt
    assert "User question: How much did revenue grow?" in prompt
