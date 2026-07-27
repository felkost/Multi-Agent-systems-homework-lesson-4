"""Tests for the completion gate: what a finished turn reports as its report.

The gate runs after the loop, so every case here scripts what the model does
and then asserts both on `AgentResult` and on the files that actually exist
in the output directory.

Pure `render_report` behaviour (numbering, anchors, folding) lives in
`test_report_rendering.py` instead -- those cases need no `ResearchAgent`
at all.
"""

import re
from pathlib import Path
from unittest.mock import MagicMock, Mock

import httpx
import pytest
from openai import BadRequestError

from agent import ResearchAgent
from research_agent.prompts.requests import (
    FALLBACK_REPORT_REQUEST,
    FALLBACK_STRUCTURED_REPORT_REQUEST,
)
from research_agent.report import (
    ReportSection,
    ResearchReport,
    SourceRef,
    render_report,
)
from research_agent.settings import Settings
from research_agent.tools import fetch, report_writer, search

from fakes import ScriptedChatClient, ScriptedTurn

pytestmark = pytest.mark.usefixtures("patch_tool_settings")


def _agent(configured_settings: Settings, script: list[ScriptedTurn]) -> ResearchAgent:
    return ResearchAgent(configured_settings, client=ScriptedChatClient(script))


def _patch_search(monkeypatch: pytest.MonkeyPatch) -> None:
    search_client = Mock()
    search_client.text.return_value = [
        {"title": "Result", "href": "https://example.com/one", "body": "A snippet"}
    ]
    monkeypatch.setattr(search, "DDGS", Mock(return_value=search_client))


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
    assert re.fullmatch(r"\d{8}-\d{6}_rag\.md", saved_names[0])


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


def test_fallback_asks_model_when_no_markdown(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    _patch_read_url(monkeypatch)
    client = ScriptedChatClient(
        [
            ScriptedTurn(tool_calls=[("read_url", {"url": "https://example.com"})]),
            ScriptedTurn(content="Here is what I found, in prose."),
            ScriptedTurn(content="# Report\n\nSynthesized from what was read.\n"),
        ]
    )
    agent = ResearchAgent(configured_settings, client=client)

    result = agent.run("What is RAG?")

    assert result.report_source == "fallback"
    assert result.saved_report_path is not None
    assert Path(result.saved_report_path).read_text(encoding="utf-8") == (
        "# Report\n\nSynthesized from what was read.\n"
    )
    assert len(client.requests) == 3
    assert "tools" not in client.requests[2]


def test_fallback_uses_structured_output(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    _patch_read_url(monkeypatch)
    report = ResearchReport(
        title="RAG",
        summary="A synthesized summary.",
        sections=[ReportSection(heading="Findings", body="RAG helps.")],
        limitations="Only one source was read.",
        sources=[SourceRef(url="https://example.com", title="Example")],
    )
    client = ScriptedChatClient(
        [
            ScriptedTurn(tool_calls=[("read_url", {"url": "https://example.com"})]),
            ScriptedTurn(content="Here is what I found, in prose."),
        ],
        parsed_report=report,
    )
    agent = ResearchAgent(configured_settings, client=client)

    result = agent.run("What is RAG?")

    assert result.report_source == "fallback"
    assert result.saved_report_path is not None
    saved_text = Path(result.saved_report_path).read_text(encoding="utf-8")
    assert saved_text == render_report(report)
    assert len(client.parse_requests) == 1
    assert client.parse_requests[0]["response_format"] is ResearchReport
    assert len(client.requests) == 2


def test_fallback_falls_back_to_plain_text_when_parse_fails(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    _patch_read_url(monkeypatch)
    bad_response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        json={"error": {"message": "model does not support response_format"}},
    )
    client = ScriptedChatClient(
        [
            ScriptedTurn(tool_calls=[("read_url", {"url": "https://example.com"})]),
            ScriptedTurn(content="Here is what I found, in prose."),
            ScriptedTurn(content="# Report\n\nPlain markdown fallback.\n"),
        ],
        parse_error=BadRequestError(
            "model does not support response_format",
            response=bad_response,
            body=None,
        ),
    )
    agent = ResearchAgent(configured_settings, client=client)

    result = agent.run("What is RAG?")

    assert result.report_source == "fallback"
    assert result.saved_report_path is not None
    assert Path(result.saved_report_path).read_text(encoding="utf-8") == (
        "# Report\n\nPlain markdown fallback.\n"
    )
    assert len(client.parse_requests) == 1
    assert len(client.requests) == 3


def test_fallback_gives_up_when_model_returns_nothing(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    _patch_read_url(monkeypatch)
    bad_response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        json={"error": {"message": "model does not support response_format"}},
    )
    client = ScriptedChatClient(
        [
            ScriptedTurn(tool_calls=[("read_url", {"url": "https://example.com"})]),
            ScriptedTurn(content="Here is what I found, in prose."),
            ScriptedTurn(content=""),
        ],
        parse_error=BadRequestError(
            "model does not support response_format",
            response=bad_response,
            body=None,
        ),
    )
    agent = ResearchAgent(configured_settings, client=client)

    result = agent.run("What is RAG?")

    assert result.report_source == "none"
    assert result.saved_report_path is None


def test_fallback_reports_none_when_final_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    _patch_read_url(monkeypatch)
    client = ScriptedChatClient(
        [
            ScriptedTurn(tool_calls=[("read_url", {"url": "https://example.com"})]),
            ScriptedTurn(content="Here is what I found, in prose."),
            ScriptedTurn(content="# Report\n\nSynthesized text.\n"),
        ]
    )
    agent = ResearchAgent(configured_settings, client=client)
    monkeypatch.setattr(
        report_writer.Path, "mkdir", Mock(side_effect=OSError("disk full"))
    )

    result = agent.run("What is RAG?")

    assert result.report_source == "none"
    assert result.saved_report_path is None


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


def test_model_report_citing_an_unread_source_is_flagged(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    """R7b: the tool path keeps free-form Markdown, so the code cannot fix
    a phantom source there -- but it must not pretend not to see one."""
    _patch_read_url(monkeypatch)
    markdown = (
        "# RAG\n\n## Findings\nClaim. [1](#source-1)\n\n## Sources\n"
        '<a id="source-1"></a>1. [Read](https://example.com)\n'
        '<a id="source-2"></a>2. [Never opened](https://phantom.example)\n'
    )
    client = ScriptedChatClient(
        [
            ScriptedTurn(tool_calls=[("read_url", {"url": "https://example.com"})]),
            ScriptedTurn(
                tool_calls=[("write_report", {"filename": "rag", "content": markdown})]
            ),
            ScriptedTurn(content="Saved."),
        ]
    )
    agent = ResearchAgent(configured_settings, client=client)

    result = agent.run("What is RAG?")

    assert result.report_source == "tool"
    assert result.cites_unread_sources is True


def test_model_report_citing_only_read_sources_is_not_flagged(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    _patch_read_url(monkeypatch)
    markdown = (
        "# RAG\n\n## Findings\nClaim. [1](#source-1)\n\n## Sources\n"
        '<a id="source-1"></a>1. [Read](https://example.com)\n'
    )
    client = ScriptedChatClient(
        [
            ScriptedTurn(tool_calls=[("read_url", {"url": "https://example.com"})]),
            ScriptedTurn(
                tool_calls=[("write_report", {"filename": "rag", "content": markdown})]
            ),
            ScriptedTurn(content="Saved."),
        ]
    )
    agent = ResearchAgent(configured_settings, client=client)

    result = agent.run("What is RAG?")

    assert result.report_source == "tool"
    assert result.cites_unread_sources is False


def test_fallback_report_never_cites_an_unread_source(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    """The invariant R7a buys: the gate's own report cannot carry a source
    the agent failed to open, even when the model lists one."""
    _patch_read_url(monkeypatch)
    report = ResearchReport(
        title="RAG",
        summary="A synthesized summary.",
        sections=[
            ReportSection(
                heading="Findings",
                body="RAG helps.",
                source_urls=["https://example.com", "https://phantom.example"],
            )
        ],
        limitations="Only one source was read.",
        sources=[
            SourceRef(url="https://example.com", title="Read"),
            SourceRef(url="https://phantom.example", title="Never opened"),
        ],
    )
    client = ScriptedChatClient(
        [
            ScriptedTurn(tool_calls=[("read_url", {"url": "https://example.com"})]),
            ScriptedTurn(content="Here is what I found, in prose."),
        ],
        parsed_report=report,
    )
    agent = ResearchAgent(configured_settings, client=client)

    result = agent.run("What is RAG?")

    assert result.report_source == "fallback"
    assert result.cites_unread_sources is False
    assert result.saved_report_path is not None
    saved = Path(result.saved_report_path).read_text(encoding="utf-8")
    assert "phantom.example" not in saved


def test_structured_fallback_uses_its_own_request(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    _patch_read_url(monkeypatch)
    report = ResearchReport(
        title="RAG",
        summary="S",
        sections=[
            ReportSection(
                heading="Findings", body="B", source_urls=["https://example.com"]
            )
        ],
        limitations="L",
        sources=[SourceRef(url="https://example.com", title="Example")],
    )
    client = ScriptedChatClient(
        [
            ScriptedTurn(tool_calls=[("read_url", {"url": "https://example.com"})]),
            ScriptedTurn(content="Here is what I found, in prose."),
        ],
        parsed_report=report,
    )
    agent = ResearchAgent(configured_settings, client=client)

    agent.run("What is RAG?")

    sent = client.parse_requests[0]["messages"][-1]["content"]
    assert sent == FALLBACK_STRUCTURED_REPORT_REQUEST


def test_plain_text_fallback_uses_the_plain_request(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    _patch_read_url(monkeypatch)
    bad_response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        json={"error": {"message": "model does not support response_format"}},
    )
    client = ScriptedChatClient(
        [
            ScriptedTurn(tool_calls=[("read_url", {"url": "https://example.com"})]),
            ScriptedTurn(content="Here is what I found, in prose."),
            ScriptedTurn(content="# Report\n\nPlain markdown fallback.\n"),
        ],
        parse_error=BadRequestError(
            "model does not support response_format",
            response=bad_response,
            body=None,
        ),
    )
    agent = ResearchAgent(configured_settings, client=client)

    agent.run("What is RAG?")

    assert client.requests[2]["messages"][-1]["content"] == FALLBACK_REPORT_REQUEST
