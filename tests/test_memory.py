"""Session memory: the message list, and the business state beside it.

``self._messages`` carries the dialogue; ``SessionState`` carries what the
turns produced -- read URLs and saved reports -- so that business state no
longer dies with each run's ``RunState`` (12-factor #5, plan H.3).
"""

from unittest.mock import MagicMock, Mock

import pytest

from agent import ResearchAgent
from research_agent.settings import Settings
from research_agent.tools import fetch

from fakes import ScriptedChatClient, ScriptedTurn

pytestmark = pytest.mark.usefixtures("patch_tool_settings")


def _patch_read_url(monkeypatch: pytest.MonkeyPatch) -> None:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.headers = {"content-type": "text/html; charset=utf-8"}
    response.iter_bytes.return_value = [b"<html><body><p>Hello</p></body></html>"]
    response.encoding = "utf-8"
    stream_context = MagicMock()
    stream_context.__enter__.return_value = response
    stream_context.__exit__.return_value = False
    monkeypatch.setattr(fetch.httpx, "stream", Mock(return_value=stream_context))
    monkeypatch.setattr(fetch.trafilatura, "extract", Mock(return_value="Hello"))


def _research_turns(url: str, filename: str) -> list[ScriptedTurn]:
    """Script one question answered by a read, a saved report and an answer."""
    return [
        ScriptedTurn(tool_calls=[("read_url", {"url": url})]),
        ScriptedTurn(
            tool_calls=[("write_report", {"filename": filename, "content": "# Report"})]
        ),
        ScriptedTurn(content=f"Saved as {filename}."),
    ]


def test_second_question_sees_the_first_exchange(
    configured_settings: Settings,
) -> None:
    client = ScriptedChatClient(
        [
            ScriptedTurn(content="First answer."),
            ScriptedTurn(content="Second answer."),
        ]
    )
    agent = ResearchAgent(configured_settings, client=client)

    agent.run("First question.")
    agent.run("Second question.")

    sent = client.requests[1]["messages"]

    assert [message["role"] for message in sent] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert sent[1]["content"] == "First question."
    assert sent[2]["content"] == "First answer."


def test_separate_agents_do_not_share_history(
    configured_settings: Settings,
) -> None:
    first_client = ScriptedChatClient([ScriptedTurn(content="First answer.")])
    second_client = ScriptedChatClient([ScriptedTurn(content="Second answer.")])

    ResearchAgent(configured_settings, client=first_client).run("First question.")
    second_agent = ResearchAgent(configured_settings, client=second_client)
    second_agent.run("Second question.")

    sent = second_client.requests[0]["messages"]

    assert [message["role"] for message in sent] == ["system", "user"]
    assert sent[1]["content"] == "Second question."


def test_session_records_one_run_per_turn(configured_settings: Settings) -> None:
    client = ScriptedChatClient(
        [
            ScriptedTurn(content="First answer."),
            ScriptedTurn(content="Second answer."),
        ]
    )
    agent = ResearchAgent(configured_settings, client=client)

    agent.run("First question.")
    agent.run("Second question.")

    assert len(agent.session.runs) == 2


def test_session_collects_read_urls_across_turns(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    _patch_read_url(monkeypatch)
    client = ScriptedChatClient(
        [
            *_research_turns("https://example.com/one", "first"),
            *_research_turns("https://example.com/two", "second"),
        ]
    )
    agent = ResearchAgent(configured_settings, client=client)

    agent.run("First question.")
    agent.run("Second question.")

    assert agent.session.all_read_urls == {
        "https://example.com/one",
        "https://example.com/two",
    }


def test_session_lists_saved_reports_oldest_first(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    _patch_read_url(monkeypatch)
    client = ScriptedChatClient(
        [
            *_research_turns("https://example.com/one", "first"),
            *_research_turns("https://example.com/two", "second"),
        ]
    )
    agent = ResearchAgent(configured_settings, client=client)

    agent.run("First question.")
    agent.run("Second question.")

    saved = agent.session.all_saved_reports
    assert len(saved) == 2
    assert saved[0].endswith("_first.md")
    assert saved[1].endswith("_second.md")


def test_answer_only_turn_records_no_sources(configured_settings: Settings) -> None:
    client = ScriptedChatClient([ScriptedTurn(content="No research needed.")])
    agent = ResearchAgent(configured_settings, client=client)

    agent.run("Where did you save it?")

    assert agent.session.all_saved_reports == []
    assert agent.session.all_read_urls == set()
