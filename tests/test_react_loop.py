"""Integration tests for the ReAct loop, driven by a scripted fake client.

No test reaches the network: the model is scripted (``fakes.py``) and the
search backend is patched, while the tools themselves run for real.
"""

import json
from typing import Any
from unittest.mock import Mock

import pytest

import tools
from agent import ResearchAgent, RunState, react_step
from config import Settings

from fakes import ScriptedChatClient, ScriptedTurn

pytestmark = pytest.mark.usefixtures("patch_tool_settings")


def _patch_search(
    monkeypatch: pytest.MonkeyPatch, *, snippet: str = "A snippet"
) -> None:
    search_client = Mock()
    search_client.text.return_value = [
        {
            "title": "Result",
            "href": "https://example.com/one",
            "body": snippet,
        }
    ]
    monkeypatch.setattr(tools, "DDGS", Mock(return_value=search_client))


def _agent(
    configured_settings: Settings, script: list[ScriptedTurn]
) -> tuple[ResearchAgent, ScriptedChatClient]:
    client = ScriptedChatClient(script)
    return ResearchAgent(configured_settings, client=client), client


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
    assert result.saved_report_path.endswith("rag.md")
