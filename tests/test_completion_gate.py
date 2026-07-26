"""Tests for the completion gate: what a finished turn reports as its report.

The gate runs after the loop, so every case here scripts what the model does
and then asserts both on `AgentResult` and on the files that actually exist
in the output directory.
"""

import re
from pathlib import Path
from unittest.mock import Mock

import pytest

import tools
from agent import ResearchAgent
from config import Settings

from fakes import ScriptedChatClient, ScriptedTurn

pytestmark = pytest.mark.usefixtures("patch_tool_settings")


def _agent(configured_settings: Settings, script: list[ScriptedTurn]) -> ResearchAgent:
    return ResearchAgent(configured_settings, client=ScriptedChatClient(script))


def _patch_search(monkeypatch: pytest.MonkeyPatch) -> None:
    search_client = Mock()
    search_client.text.return_value = [
        {"title": "Result", "href": "https://example.com/one", "body": "A snippet"}
    ]
    monkeypatch.setattr(tools, "DDGS", Mock(return_value=search_client))


def _saved_reports(settings: Settings) -> list[Path]:
    """Every report file that exists in the configured output directory."""
    output_dir = Path(settings.output_dir)
    return sorted(output_dir.glob("*.md")) if output_dir.exists() else []


def test_report_saved_by_agent(configured_settings: Settings) -> None:
    agent = _agent(
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
    assert Path(result.saved_report_path).read_text(encoding="utf-8") == "# RAG\n"
    saved_names = [path.name for path in _saved_reports(configured_settings)]
    assert len(saved_names) == 1
    assert re.fullmatch(r"rag_\d{8}-\d{6}\.md", saved_names[0])


def test_no_report_for_non_research_turn(configured_settings: Settings) -> None:
    agent = _agent(
        configured_settings,
        [ScriptedTurn(content="The report is in the output directory.")],
    )

    result = agent.run("Where exactly did you save the report?")

    assert result.report_source == "none"
    assert result.saved_report_path is None
    assert _saved_reports(configured_settings) == []


def test_no_report_when_search_only(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    _patch_search(monkeypatch)
    agent = _agent(
        configured_settings,
        [
            ScriptedTurn(tool_calls=[("web_search", {"query": "RAG"})]),
            ScriptedTurn(content="Here is what the snippets say."),
        ],
    )

    result = agent.run("What is RAG?")

    assert result.report_source == "none"
    assert result.saved_report_path is None
    assert _saved_reports(configured_settings) == []


def test_fallback_reuses_markdown_from_failed_write(
    configured_settings: Settings,
) -> None:
    client = ScriptedChatClient(
        [
            ScriptedTurn(
                tool_calls=[("write_report", {"filename": "???", "content": "# RAG\n"})]
            ),
            ScriptedTurn(content="Done."),
        ]
    )
    agent = ResearchAgent(configured_settings, client=client)

    result = agent.run("Tell me about RAG.")

    assert result.report_source == "fallback"
    assert result.saved_report_path is not None
    assert Path(result.saved_report_path).read_text(encoding="utf-8") == "# RAG\n"
    assert len(client.requests) == 2
    