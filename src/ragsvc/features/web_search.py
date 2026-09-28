"""Web search: fetch live results through SerpAPI.

Web results never enter the FAISS or BM25 indexes; they are only appended to
the context handed to the LLM for the current question.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import requests

from ragsvc.config import get_settings

logger = logging.getLogger(__name__)


class WebResult(TypedDict, total=False):
    title: str | None
    url: str | None
    snippet: str | None
    timestamp: str | None
    source: str


def check_serpapi_key() -> bool:
    """True when a SerpAPI key is configured (``RAG_SERPAPI_KEY`` or ``SERPAPI_KEY``)."""
    return get_settings().serpapi_configured


def serpapi_search(query: str, num_results: int = 5) -> list[WebResult]:
    """Query SerpAPI; returns an empty list on any request failure."""
    settings = get_settings()
    if settings.serpapi_key is None:
        raise ValueError("SERPAPI_KEY is not set")
    params: dict[str, str | int] = {
        "engine": settings.search_engine,
        "q": query,
        "api_key": settings.serpapi_key.get_secret_value(),
        "num": num_results,
        "hl": "en",
        "gl": "us",
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=15)
        response.raise_for_status()
        return _parse_serpapi_results(response.json())
    except Exception as exc:  # noqa: BLE001 - web search is best effort
        logger.error("Web search failed: %s", exc)
        return []


def _parse_serpapi_results(data: dict[str, Any]) -> list[WebResult]:
    results: list[WebResult] = []
    for item in data.get("organic_results", []):
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
                "timestamp": item.get("date"),
            }
        )
    if "knowledge_graph" in data:
        kg = data["knowledge_graph"]
        results.insert(
            0,
            {
                "title": kg.get("title"),
                "url": kg.get("source", {}).get("link", ""),
                "snippet": kg.get("description"),
                "source": "knowledge_graph",
            },
        )
    return results


def search_web(query: str, num_results: int = 5) -> list[WebResult]:
    """Search the web for ``query``; results are context only and are not indexed."""
    results = serpapi_search(query, num_results)
    if results:
        logger.info("Web search returned %d result(s)", len(results))
    else:
        logger.info("Web search returned no results")
    return results
