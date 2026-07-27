"""Characterization tests pinning the observable contract of ``web_search``.

Every assertion is about behaviour that reaches the model's context — exact
error strings, output normalization — not implementation details.
``call_tool`` (conftest.py) is the only place that knows how a tool name is
resolved to a callable.
"""

from unittest.mock import Mock

import pytest

from research_agent.tools import search

from conftest import call_tool

pytestmark = pytest.mark.usefixtures("patch_tool_settings")


def test_web_search_rejects_empty_query() -> None:
    result = call_tool("web_search", query="   ")

    assert (
        result
        == "ERROR: Search query cannot be empty. Provide a specific question or phrase."
    )


def test_web_search_normalizes_and_removes_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_client = Mock()
    search_client.text.return_value = [
        {
            "title": "First result",
            "href": "https://example.com/first",
            "body": "First snippet",
        },
        {
            "title": "Duplicate result",
            "href": "https://example.com/first",
            "body": "Duplicate snippet",
        },
        {
            "title": "Second result",
            "href": "https://example.com/second",
            "body": "Second snippet",
        },
    ]
    ddgs_class = Mock(return_value=search_client)
    monkeypatch.setattr(search, "DDGS", ddgs_class)

    result = call_tool("web_search", query="  RAG retrieval  ")

    assert result == [
        {
            "title": "First result",
            "url": "https://example.com/first",
            "snippet": "First snippet",
        },
        {
            "title": "Second result",
            "url": "https://example.com/second",
            "snippet": "Second snippet",
        },
    ]
    ddgs_class.assert_called_once_with()
    search_client.text.assert_called_once_with("RAG retrieval", max_results=3)


def test_web_search_handles_missing_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_client = Mock()
    search_client.text.return_value = [
        {"href": "https://example.com/no-fields"},
        {"title": "Missing URL", "body": "This result must be skipped"},
    ]
    monkeypatch.setattr(search, "DDGS", Mock(return_value=search_client))

    result = call_tool("web_search", query="test")

    assert result == [
        {
            "title": "Untitled",
            "url": "https://example.com/no-fields",
            "snippet": "",
        }
    ]


def test_web_search_truncates_snippet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_client = Mock()
    search_client.text.return_value = [
        {
            "title": "Long result",
            "href": "https://example.com/long",
            "body": "x" * 150,
        }
    ]
    monkeypatch.setattr(search, "DDGS", Mock(return_value=search_client))

    result = call_tool("web_search", query="long text")

    assert isinstance(result, list)
    assert result[0]["snippet"] == "x" * 100


def test_web_search_returns_safe_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_client = Mock()
    search_client.text.side_effect = RuntimeError("private DNS and system details")
    monkeypatch.setattr(search, "DDGS", Mock(return_value=search_client))

    result = call_tool("web_search", query="test")

    assert result == "ERROR: Web search is temporarily unavailable."
