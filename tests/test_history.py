"""History compaction between turns.

Compaction shrinks the ``content`` of stale tool results and nothing else:
dropping a message would leave an assistant ``tool_call`` unanswered and the
next request would fail with HTTP 400 (invariant A.2, plan H.6).
"""

import json
from typing import Any
from unittest.mock import Mock

import pytest

from agent import Messages, ResearchAgent, compact_history
from research_agent.settings import Settings
from research_agent.tools import search
from research_agent.tools.contract import REPORT_SAVED_PREFIX

from fakes import ScriptedChatClient, ScriptedTurn

SEARCH_RESULTS = json.dumps(
    [
        {
            "title": f"Result {index}",
            "url": f"https://example.com/{index}",
            "snippet": "A long snippet that costs tokens on every later turn.",
        }
        for index in range(5)
    ]
)

PAGE_TEXT = "Full article text. " * 200

SAVED_REPORT = f"{REPORT_SAVED_PREFIX}output/20260726-101010_rag.md"


def _exchange(
    index: int, name: str, arguments: dict[str, Any], result: str
) -> Messages:
    call_id = f"call_{index}"
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def _sample_history() -> Messages:
    return [
        {"role": "system", "content": "You are a research agent."},
        {"role": "user", "content": "Compare RAG approaches."},
        *_exchange(1, "web_search", {"query": "naive RAG"}, SEARCH_RESULTS),
        *_exchange(2, "read_url", {"url": "https://example.com/1"}, PAGE_TEXT),
        *_exchange(
            3,
            "write_report",
            {"filename": "rag", "content": "# RAG"},
            SAVED_REPORT,
        ),
        {"role": "assistant", "content": "Saved."},
    ]


def _tool_contents(messages: Messages) -> list[str]:
    return [message["content"] for message in messages if message["role"] == "tool"]


def test_compact_preserves_message_structure() -> None:
    messages = _sample_history()

    compacted = compact_history(messages, keep_recent=1)

    assert [message["role"] for message in compacted] == [
        message["role"] for message in messages
    ]
    requested = [
        call["id"] for message in compacted for call in message.get("tool_calls") or []
    ]
    answered = [
        message["tool_call_id"] for message in compacted if message["role"] == "tool"
    ]
    assert answered == requested


def test_compact_keeps_recent_tool_results_intact() -> None:
    messages = _sample_history()

    compacted = compact_history(messages, keep_recent=1)

    assert _tool_contents(compacted)[-1] == SAVED_REPORT


def test_compact_summarises_older_tool_results() -> None:
    messages = _sample_history()

    search_result, page_result, report_result = _tool_contents(
        compact_history(messages, keep_recent=0)
    )

    assert search_result == '[web_search: 5 results for "naive RAG", payload dropped]'
    assert page_result == (
        f"[read_url: https://example.com/1 ({len(PAGE_TEXT)} chars), payload dropped]"
    )
    assert report_result == "[write_report: saved to output/20260726-101010_rag.md]"


def test_compact_keeps_error_results_verbatim() -> None:
    error = "ERROR: The page is unavailable. Pick another URL from your search results."
    messages = [
        {"role": "user", "content": "Read this page."},
        *_exchange(1, "read_url", {"url": "https://example.com/gone"}, error),
        *_exchange(2, "read_url", {"url": "https://example.com/1"}, PAGE_TEXT),
    ]

    compacted = compact_history(messages, keep_recent=0)

    assert _tool_contents(compacted)[0] == error


def test_compact_survives_malformed_tool_arguments() -> None:
    messages: Messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_url", "arguments": "not json at all"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": PAGE_TEXT},
    ]

    summary = _tool_contents(compact_history(messages, keep_recent=0))[0]

    assert summary.startswith("[read_url:")
    assert "Full article text" not in summary


def test_compact_counts_nothing_when_search_output_is_not_a_list() -> None:
    messages = _exchange(1, "web_search", {"query": "naive RAG"}, "5")

    compacted = compact_history(messages, keep_recent=0)

    assert _tool_contents(compacted)[0] == (
        '[web_search: 0 results for "naive RAG", payload dropped]'
    )


def test_compact_summarises_an_unknown_tool() -> None:
    messages = _exchange(1, "unknown_tool", {"anything": 1}, "2026")

    compacted = compact_history(messages, keep_recent=0)

    assert _tool_contents(compacted)[0] == "[unknown_tool: 4 chars, payload dropped]"


def test_compact_never_touches_user_or_system() -> None:
    messages = _sample_history()

    compacted = compact_history(messages, keep_recent=0)

    assert [message for message in compacted if message["role"] != "tool"] == [
        message for message in messages if message["role"] != "tool"
    ]


def test_compacted_history_is_still_json_serializable() -> None:
    compacted = compact_history(_sample_history(), keep_recent=0)

    json.dumps(compacted)


def test_agent_keeps_earlier_queries_visible_after_compaction(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    search_client = Mock()
    search_client.text.return_value = [
        {
            "title": "Result",
            "href": "https://example.com/one",
            "body": "A snippet that should not survive compaction.",
        }
    ]
    monkeypatch.setattr(search, "DDGS", Mock(return_value=search_client))
    settings = configured_settings.model_copy(update={"compact_keep_recent": 0})
    client = ScriptedChatClient(
        [
            ScriptedTurn(tool_calls=[("web_search", {"query": "naive RAG"})]),
            ScriptedTurn(content="First answer."),
            ScriptedTurn(content="Second answer."),
        ]
    )
    agent = ResearchAgent(settings, client=client)

    agent.run("What is naive RAG?")
    agent.run("And parent-child retrieval?")

    sent = json.dumps(client.requests[-1]["messages"], ensure_ascii=False)
    assert "naive RAG" in sent
    assert "should not survive compaction" not in sent
