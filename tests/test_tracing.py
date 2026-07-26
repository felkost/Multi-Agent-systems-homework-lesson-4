"""Tracing configuration: what reaches LangSmith and what never does.

Every test here runs offline. No test in this suite may need a LangSmith key
or a network call, so any test that could reach the SDK's transport replaces
it first.
"""

from __future__ import annotations

from typing import Any

import pytest
from langsmith.client import Client
from langsmith.run_helpers import get_current_run_tree, tracing_context
from langsmith.utils import get_env_var, tracing_is_enabled

import agent as agent_module
import tools
from agent import ResearchAgent, ToolStep, build_client, configure_tracing
from config import PROMPT_VERSION, Settings
from fakes import ScriptedChatClient, ScriptedTurn


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)
    return Settings.model_validate({})


def test_tracing_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)

    assert configure_tracing(settings) is False


def test_configure_tracing_exports_what_the_sdk_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(
        monkeypatch,
        LANGSMITH_TRACING="true",
        LANGSMITH_API_KEY="lsv2_pt_test",
        LANGSMITH_PROJECT="research-agent-test",
        LANGSMITH_ENDPOINT="https://eu.api.smith.langchain.com",
        LANGSMITH_WORKSPACE_ID="11111111-2222-3333-4444-555555555555",
    )
    # Cleared so the assertions below cannot pass on a value the shell that
    # started pytest had already exported.
    for name in (
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_WORKSPACE_ID",
    ):
        monkeypatch.delenv(name, raising=False)

    assert configure_tracing(settings) is True
    assert get_env_var("TRACING") == "true"
    assert get_env_var("API_KEY") == "lsv2_pt_test"
    assert get_env_var("PROJECT") == "research-agent-test"
    assert get_env_var("ENDPOINT") == "https://eu.api.smith.langchain.com"
    assert get_env_var("WORKSPACE_ID") == "11111111-2222-3333-4444-555555555555"


def test_tracing_without_a_key_stays_off(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, LANGSMITH_TRACING="true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    assert configure_tracing(settings) is False
    assert get_env_var("TRACING") == "false"


def test_configure_tracing_follows_settings_not_the_live_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The decision comes from `Settings`, read once, not from `os.environ`.

    Notes
    -----
    `Settings` itself does take the environment ahead of ``.env``, which is
    how a single run is switched from the shell; what this pins is that
    nothing can move the decision *after* the configuration was resolved.
    """
    settings = _settings(monkeypatch, LANGSMITH_TRACING="false")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    assert configure_tracing(settings) is False
    assert get_env_var("TRACING") == "false"


def test_a_leftover_langchain_flag_cannot_keep_tracing_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`LANGCHAIN_TRACING_V2` outranks `LANGSMITH_TRACING` inside the SDK.

    Notes
    -----
    A machine that ever ran a LangChain project still exports it, and
    writing only `LANGSMITH_TRACING` left it winning -- the agent went on
    tracing after this function had returned ``False``.
    """
    settings = _settings(monkeypatch)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    assert configure_tracing(settings) is False
    assert tracing_is_enabled() is False


def test_a_leftover_langchain_flag_cannot_keep_tracing_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(
        monkeypatch,
        LANGSMITH_TRACING="true",
        LANGSMITH_API_KEY="lsv2_pt_test",
    )
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    assert configure_tracing(settings) is True
    assert tracing_is_enabled() is True


def test_build_client_wraps_only_when_tracing_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapped: list[object] = []

    def spy(client: Any) -> Any:
        wrapped.append(client)
        return client

    monkeypatch.setattr(agent_module, "wrap_openai", spy)

    build_client(_settings(monkeypatch))
    assert wrapped == []

    build_client(
        _settings(
            monkeypatch,
            LANGSMITH_TRACING="true",
            LANGSMITH_API_KEY="lsv2_pt_test",
        )
    )
    assert len(wrapped) == 1


def test_tracing_disabled_makes_no_langsmith_calls(
    monkeypatch: pytest.MonkeyPatch,
    patch_tool_settings: None,
    configured_settings: Settings,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a run reached LangSmith with tracing disabled")

    monkeypatch.setattr(Client, "request_with_retries", fail)
    monkeypatch.setattr(Client, "multipart_ingest", fail)
    monkeypatch.setattr(
        tools, "DDGS", lambda: _FakeDDGS([{"title": "T", "href": "u", "body": "b"}])
    )
    client = ScriptedChatClient(
        [
            ScriptedTurn(tool_calls=[("web_search", {"query": "rag"})]),
            ScriptedTurn(content="Done."),
        ]
    )
    agent = ResearchAgent(configured_settings, client=client)

    result = agent.run("What is RAG?")

    assert result.final_answer == "Done."


def test_tool_call_becomes_a_tool_span_with_its_arguments(
    monkeypatch: pytest.MonkeyPatch,
    patch_tool_settings: None,
    configured_settings: Settings,
) -> None:
    _silence_langsmith_transport(monkeypatch)
    spans: list[Any] = []

    def search(query: str, max_results: int) -> list[dict[str, str]]:
        spans.append(get_current_run_tree())
        return [{"title": "T", "href": "https://example.com", "body": "b"}]

    monkeypatch.setattr(tools, "DDGS", lambda: _FakeDDGS(search=search))

    with tracing_context(enabled="local"):
        tools.web_search("rag chunking")

    span = spans[0]

    assert (span.name, span.run_type) == ("web_search", "tool")
    assert span.inputs == {"query": "rag chunking"}


def test_tool_spans_nest_inside_the_research_run(
    monkeypatch: pytest.MonkeyPatch,
    patch_tool_settings: None,
    configured_settings: Settings,
) -> None:
    """A tool's span must be a child of the run, not a sibling of it.

    Notes
    -----
    Nesting is what makes a trace readable, and it rests on the contextvar
    LangSmith sets: run the tools in another thread and every span becomes a
    root. Plan B.3 names the flat trace as this stage's typical failure, so
    it is worth one test rather than one look at the UI.
    """
    _silence_langsmith_transport(monkeypatch)
    tool_spans: list[Any] = []
    root_spans: list[Any] = []

    def search(query: str, max_results: int) -> list[dict[str, str]]:
        tool_spans.append(get_current_run_tree())
        return [{"title": "T", "href": "https://example.com", "body": "b"}]

    monkeypatch.setattr(tools, "DDGS", lambda: _FakeDDGS(search=search))
    client = ScriptedChatClient(
        [
            ScriptedTurn(tool_calls=[("web_search", {"query": "rag"})]),
            ScriptedTurn(content="Done."),
        ]
    )
    agent = ResearchAgent(configured_settings, client=client)

    # on_step fires after the tool's span has closed, so the tree current at
    # that moment is the run itself -- the parent the assertions below expect.
    def capture_root(step: ToolStep) -> None:
        root_spans.append(get_current_run_tree())

    with tracing_context(enabled="local"):
        agent.run("What is RAG?", on_step=capture_root)

    tool_span, root_span = tool_spans[0], root_spans[0]

    assert (tool_span.name, tool_span.run_type) == ("web_search", "tool")
    assert (root_span.name, root_span.run_type) == ("research_run", "chain")
    assert tool_span.parent_run_id == root_span.id
    assert root_span.metadata["prompt_version"] == PROMPT_VERSION
    assert root_span.metadata["model"] == configured_settings.model_name
    assert root_span.metadata["max_iterations"] == configured_settings.max_iterations


class _FakeResponse:
    """The two methods LangSmith calls on an HTTP response it gets back."""

    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {}


def _silence_langsmith_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cut every HTTP path out of the SDK before a test enables tracing.

    Notes
    -----
    Tracing builds a real `Client` even with ``enabled="local"``, and that
    client fetches ``/info`` and posts runs. Patched here so the suite keeps
    its promise of running with no network and no API key. The dummy key only
    keeps the SDK from warning about a hosted endpoint with no credentials --
    nothing can leave the process once the two methods below are replaced.
    """
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_pt_test")
    monkeypatch.setattr(
        Client, "request_with_retries", lambda *a, **kw: _FakeResponse()
    )
    monkeypatch.setattr(Client, "multipart_ingest", lambda *a, **kw: None)


class _FakeDDGS:
    """Stand-in for `ddgs.DDGS` whose `text()` never leaves the process."""

    def __init__(
        self,
        results: list[dict[str, str]] | None = None,
        search: Any = None,
    ) -> None:
        self._results = results or []
        self._search = search

    def text(self, query: str, max_results: int) -> list[dict[str, str]]:
        if self._search is not None:
            return list(self._search(query, max_results))
        return self._results
