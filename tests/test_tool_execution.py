"""Tests for `_execute_tool_call`: parsing, validation and the search cache.

Loop-level concerns -- stop reasons, forcing, message history -- live in
`test_react_loop.py` instead; everything here is about one tool call's own
outcome.
"""

from unittest.mock import MagicMock, Mock

import pytest

from agent import ResearchAgent, RunState, react_step
from research_agent.settings import Settings
from research_agent.tools import fetch, search

from fakes import ScriptedChatClient, ScriptedTurn

pytestmark = pytest.mark.usefixtures("patch_tool_settings")


def _patch_search(
    monkeypatch: pytest.MonkeyPatch, *, snippet: str = "A snippet"
) -> Mock:
    search_client = Mock()
    search_client.text.return_value = [
        {
            "title": "Result",
            "href": "https://example.com/one",
            "body": snippet,
        }
    ]
    monkeypatch.setattr(search, "DDGS", Mock(return_value=search_client))
    return search_client


def _agent(
    configured_settings: Settings, script: list[ScriptedTurn]
) -> tuple[ResearchAgent, ScriptedChatClient]:
    client = ScriptedChatClient(script)
    return ResearchAgent(configured_settings, client=client), client


def _patch_read_url(monkeypatch: pytest.MonkeyPatch, *, text: str = "Hello") -> None:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.headers = {"content-type": "text/html; charset=utf-8"}
    response.iter_bytes.return_value = [b"<html><body><p>Hello</p></body></html>"]
    response.encoding = "utf-8"
    stream_context = MagicMock()
    stream_context.__enter__.return_value = response
    stream_context.__exit__.return_value = False
    monkeypatch.setattr(fetch.httpx, "stream", Mock(return_value=stream_context))
    monkeypatch.setattr(fetch.trafilatura, "extract", Mock(return_value=text))


def test_non_function_tool_call_is_rejected(configured_settings: Settings) -> None:
    agent, _ = _agent(
        configured_settings,
        [
            ScriptedTurn(
                raw_tool_calls=[
                    {
                        "id": "call_custom",
                        "type": "custom",
                        "custom": {"name": "shell", "input": "ls"},
                    }
                ]
            ),
            ScriptedTurn(content="Recovered."),
        ],
    )

    result = agent.run("Run a shell command.")

    assert result.steps[0].ok is False
    assert result.steps[0].result == "ERROR: Only function tool calls are supported."
    assert result.final_answer == "Recovered."


def test_malformed_json_arguments_returns_error(configured_settings: Settings) -> None:
    agent, _ = _agent(
        configured_settings,
        [
            ScriptedTurn(tool_calls=[("web_search", "not valid json")]),
            ScriptedTurn(content="Recovered."),
        ],
    )

    result = agent.run("Search for something.")

    assert result.steps[0].ok is False
    assert result.steps[0].result == "ERROR: Tool arguments are not valid JSON."
    assert result.final_answer == "Recovered."


def test_non_object_arguments_returns_error(configured_settings: Settings) -> None:
    agent, _ = _agent(
        configured_settings,
        [
            ScriptedTurn(tool_calls=[("web_search", "[1, 2, 3]")]),
            ScriptedTurn(content="Recovered."),
        ],
    )

    result = agent.run("Search for something.")

    assert result.steps[0].ok is False
    assert result.steps[0].result == "ERROR: Tool arguments must be a JSON object."
    assert result.final_answer == "Recovered."


def test_unknown_tool_returns_error_and_continues(
    configured_settings: Settings,
) -> None:
    agent, _ = _agent(
        configured_settings,
        [
            ScriptedTurn(tool_calls=[("delete_everything", {})]),
            ScriptedTurn(content="Recovered."),
        ],
    )

    result = agent.run("Do something unsupported.")

    assert result.steps[0].ok is False
    assert result.steps[0].result == "ERROR: Unknown tool 'delete_everything'."
    assert result.final_answer == "Recovered."


def test_missing_required_argument_returns_error(
    configured_settings: Settings,
) -> None:
    agent, _ = _agent(
        configured_settings,
        [
            ScriptedTurn(tool_calls=[("web_search", {})]),
            ScriptedTurn(content="Recovered."),
        ],
    )

    result = agent.run("Search for something.")

    assert result.steps[0].ok is False
    assert result.steps[0].result == "ERROR: Invalid arguments for 'web_search'."
    assert result.final_answer == "Recovered."


def test_repeated_web_search_query_is_cached_within_a_run(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    search_client = _patch_search(monkeypatch)
    agent, _ = _agent(
        configured_settings,
        [
            ScriptedTurn(tool_calls=[("web_search", {"query": "RAG"})]),
            ScriptedTurn(tool_calls=[("web_search", {"query": "RAG"})]),
            ScriptedTurn(content="Done."),
        ],
    )

    result = agent.run("What is RAG?")

    search_client.text.assert_called_once()
    tool_messages = [message for message in agent.messages if message["role"] == "tool"]
    assert tool_messages[0]["content"] == tool_messages[1]["content"]
    assert [step.ok for step in result.steps] == [True, True]


def test_search_cache_does_not_survive_across_runs(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    search_client = _patch_search(monkeypatch)
    agent, _ = _agent(
        configured_settings,
        [
            ScriptedTurn(tool_calls=[("web_search", {"query": "RAG"})]),
            ScriptedTurn(content="First answer."),
            ScriptedTurn(tool_calls=[("web_search", {"query": "RAG"})]),
            ScriptedTurn(content="Second answer."),
        ],
    )

    agent.run("What is RAG?")
    agent.run("What is RAG, again?")

    assert search_client.text.call_count == 2


def test_failed_web_search_is_not_cached(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    search_client = Mock()
    search_client.text.side_effect = RuntimeError("backend down")
    monkeypatch.setattr(search, "DDGS", Mock(return_value=search_client))
    agent, _ = _agent(
        configured_settings,
        [
            ScriptedTurn(tool_calls=[("web_search", {"query": "RAG"})]),
            ScriptedTurn(tool_calls=[("web_search", {"query": "RAG"})]),
            ScriptedTurn(content="Done."),
        ],
    )

    result = agent.run("What is RAG?")

    assert search_client.text.call_count == 2
    assert [step.ok for step in result.steps] == [False, False]


def test_successful_read_url_increments_state(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    _patch_read_url(monkeypatch)

    client = ScriptedChatClient(
        [ScriptedTurn(tool_calls=[("read_url", {"url": "https://example.com"})])]
    )
    state = RunState()

    result = react_step(
        [{"role": "user", "content": "Read this page."}],
        client,
        configured_settings,
        state,
    )

    assert result.steps[0].ok is True
    assert state.successful_reads == 1
