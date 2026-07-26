"""Integration tests for the ReAct loop, driven by a scripted fake client.

No test reaches the network: the model is scripted (``fakes.py``) and the
search backend is patched, while the tools themselves run for real.
"""

import json
import re
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

import tools
from agent import ResearchAgent, RunState, react_step
from config import BUDGET_NUDGE_MESSAGE, Settings

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
    monkeypatch.setattr(tools, "DDGS", Mock(return_value=search_client))
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
    monkeypatch.setattr(tools.httpx, "stream", Mock(return_value=stream_context))
    monkeypatch.setattr(tools.trafilatura, "extract", Mock(return_value=text))


def test_single_tool_exchange_extends_history(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    _patch_search(monkeypatch)
    agent, _ = _agent(
        configured_settings,
        [
            ScriptedTurn(tool_calls=[("web_search", {"query": "RAG"})]),
            ScriptedTurn(content="RAG trades index freshness for context size."),
        ],
    )

    result = agent.run("What is RAG?")

    assert [message["role"] for message in agent.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result.final_answer == "RAG trades index freshness for context size."
    assert result.stop_reason == "goal_satisfied"
    assert result.iterations_used == 2
    assert [step.name for step in result.steps] == ["web_search"]


def test_parallel_tool_calls_all_executed(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    _patch_search(monkeypatch)
    agent, client = _agent(
        configured_settings,
        [
            ScriptedTurn(
                tool_calls=[
                    ("web_search", {"query": "angle a"}),
                    ("web_search", {"query": "angle b"}),
                    ("web_search", {"query": "angle c"}),
                ]
            ),
            ScriptedTurn(content="Done."),
        ],
    )

    result = agent.run("Research three angles at once.")

    assert result.iterations_used == 2
    assert [step.name for step in result.steps] == ["web_search"] * 3

    tool_messages = [message for message in agent.messages if message["role"] == "tool"]
    assert len(tool_messages) == 3
    assert len({message["tool_call_id"] for message in tool_messages}) == 3
    assert client.requests[0]["parallel_tool_calls"] is True


def test_system_prompt_is_first_message(configured_settings: Settings) -> None:
    agent, _ = _agent(configured_settings, [ScriptedTurn(content="Answer.")])

    agent.run("A question.")

    assert agent.messages[0]["role"] == "system"
    assert "research agent" in agent.messages[0]["content"].lower()


def test_message_history_has_no_orphan_tool_call_id(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    _patch_search(monkeypatch)
    agent, _ = _agent(
        configured_settings,
        [
            ScriptedTurn(tool_calls=[("web_search", {"query": "RAG"})]),
            ScriptedTurn(content="Done."),
        ],
    )

    agent.run("What is RAG?")

    announced = {
        tool_call["id"]
        for message in agent.messages
        if message["role"] == "assistant"
        for tool_call in message.get("tool_calls", [])
    }
    answered = [
        message["tool_call_id"]
        for message in agent.messages
        if message["role"] == "tool"
    ]

    assert announced == set(answered)
    assert len(answered) == len(set(answered))


def test_message_history_is_json_serializable(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    _patch_search(monkeypatch)
    agent, _ = _agent(
        configured_settings,
        [
            ScriptedTurn(tool_calls=[("web_search", {"query": "RAG"})]),
            ScriptedTurn(content="Done."),
        ],
    )

    agent.run("What is RAG?")

    assert json.loads(json.dumps(agent.messages)) == agent.messages


def test_search_results_are_serialized_as_readable_json(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    _patch_search(monkeypatch, snippet="Пошук по-українськи")
    agent, _ = _agent(
        configured_settings,
        [
            ScriptedTurn(tool_calls=[("web_search", {"query": "RAG"})]),
            ScriptedTurn(content="Done."),
        ],
    )

    agent.run("What is RAG?")

    tool_message = agent.messages[3]
    assert isinstance(tool_message["content"], str)
    assert "Пошук по-українськи" in tool_message["content"]


def test_react_step_does_not_mutate_input_messages(
    configured_settings: Settings,
) -> None:
    client = ScriptedChatClient([ScriptedTurn(content="Answer.")])
    messages: list[dict[str, Any]] = [{"role": "user", "content": "A question."}]

    result = react_step(messages, client, configured_settings, RunState())

    assert messages == [{"role": "user", "content": "A question."}]
    assert len(result.messages) == 2


def test_failed_turn_leaves_history_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    _patch_search(monkeypatch)
    agent, _ = _agent(
        configured_settings,
        [ScriptedTurn(tool_calls=[("web_search", {"query": "RAG"})])],
    )
    history_before = list(agent.messages)

    with pytest.raises(AssertionError):
        agent.run("What is RAG?")

    assert agent.messages == history_before


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


def test_consecutive_tool_errors_stop_loop(configured_settings: Settings) -> None:
    agent, client = _agent(
        configured_settings,
        [
            ScriptedTurn(tool_calls=[("no_such_tool", {})]),
            ScriptedTurn(tool_calls=[("no_such_tool", {})]),
            ScriptedTurn(tool_calls=[("no_such_tool", {})]),
        ],
    )

    result = agent.run("Do something impossible.")

    assert len(client.requests) == 3
    assert result.stop_reason == "tool_failures"
    assert result.budget_exhausted is False
    assert all(not step.ok for step in result.steps)


def test_refusal_stops_the_turn(configured_settings: Settings) -> None:
    agent, _ = _agent(
        configured_settings,
        [ScriptedTurn(refusal="I cannot help with that.")],
    )

    result = agent.run("Something disallowed.")

    assert result.stop_reason == "refusal"
    assert result.final_answer == "I cannot help with that."


def test_truncated_response_stops_cleanly(configured_settings: Settings) -> None:
    agent, _ = _agent(
        configured_settings,
        [ScriptedTurn(content="Half an ans", finish_reason="length")],
    )

    result = agent.run("A very long question.")

    assert result.stop_reason == "truncated"
    assert result.final_answer == "Half an ans"


def test_iteration_limit_stops_loop(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    _patch_search(monkeypatch)
    # One scripted turn per allowed iteration, every one still asking for a
    # tool: if the loop ever asked for one call too many, the fake client's
    # "script exhausted" assertion would fail this test for us.
    script = [
        ScriptedTurn(tool_calls=[("web_search", {"query": f"q{i}"})])
        for i in range(configured_settings.max_iterations)
    ]
    agent, client = _agent(configured_settings, script)

    result = agent.run("Keep searching forever.")

    assert len(client.requests) == configured_settings.max_iterations
    assert result.iterations_used == configured_settings.max_iterations
    assert result.stop_reason == "iteration_limit"
    assert result.budget_exhausted is True


def test_saved_report_path_is_reported(configured_settings: Settings) -> None:
    agent, _ = _agent(
        configured_settings,
        [
            ScriptedTurn(
                tool_calls=[("write_report", {"filename": "rag", "content": "# RAG\n"})]
            ),
            ScriptedTurn(content="Saved."),
        ],
    )

    result = agent.run("Research RAG.")

    assert result.report_source == "tool"
    assert result.saved_report_path is not None
    assert re.fullmatch(r".*[/\\]rag_\d{8}-\d{6}\.md", result.saved_report_path)


def test_last_iteration_forces_write_report(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    _patch_read_url(monkeypatch)
    configured_settings.max_iterations = 2
    client = ScriptedChatClient(
        [
            ScriptedTurn(tool_calls=[("read_url", {"url": "https://example.com"})]),
            ScriptedTurn(content="Best effort report."),
        ]
    )
    agent = ResearchAgent(configured_settings, client=client)

    agent.run("Research something huge.")

    assert client.requests[1]["tool_choice"] == {
        "type": "function",
        "function": {"name": "write_report"},
    }


def test_budget_nudge_not_persisted_in_history(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    _patch_read_url(monkeypatch)
    configured_settings.max_iterations = 2
    client = ScriptedChatClient(
        [
            ScriptedTurn(tool_calls=[("read_url", {"url": "https://example.com"})]),
            ScriptedTurn(content="Best effort report."),
        ]
    )
    agent = ResearchAgent(configured_settings, client=client)

    agent.run("Research something huge.")

    assert client.requests[1]["messages"][-1] == BUDGET_NUDGE_MESSAGE
    assert all(
        message.get("content") != BUDGET_NUDGE_MESSAGE["content"]
        for message in agent.messages
    )


def test_forced_write_report_skipped_without_reads(
    configured_settings: Settings,
) -> None:
    configured_settings.max_iterations = 1
    client = ScriptedChatClient([ScriptedTurn(content="I could not research this.")])
    agent = ResearchAgent(configured_settings, client=client)

    agent.run("Research something huge.")

    assert client.requests[0]["tool_choice"] == "auto"
    assert client.requests[0]["messages"][-1] != BUDGET_NUDGE_MESSAGE


def test_forced_write_report_skipped_when_already_saved(
    configured_settings: Settings,
) -> None:
    configured_settings.max_iterations = 2
    client = ScriptedChatClient(
        [
            ScriptedTurn(
                tool_calls=[("write_report", {"filename": "r", "content": "# R\n"})]
            ),
            ScriptedTurn(content="Done."),
        ]
    )
    agent = ResearchAgent(configured_settings, client=client)

    result = agent.run("Research something.")

    assert result.saved_report_path is not None
    assert client.requests[1]["tool_choice"] == "auto"


def test_system_prompt_states_the_iteration_budget(
    configured_settings: Settings,
) -> None:
    configured_settings.max_iterations = 5
    client = ScriptedChatClient([ScriptedTurn(content="Answer.")])

    agent = ResearchAgent(configured_settings, client=client)

    assert "5 tool-call turns" in agent.messages[0]["content"]


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
    monkeypatch.setattr(tools, "DDGS", Mock(return_value=search_client))
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
