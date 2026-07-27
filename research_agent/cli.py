"""The interactive REPL: console formatting and the ``main()`` loop.

Reads questions from the terminal until the user types ``exit`` or sends
EOF, printing each tool call and its result as the agent makes them. One
``ResearchAgent`` serves the whole session, so its message list is the
session's memory.
"""

import io
import json
import sys
from typing import Any, Callable, cast
from urllib.parse import urlsplit, urlunsplit

from openai import APIError, OpenAIError
from pydantic import ValidationError

from agent import ResearchAgent
from research_agent.settings import Settings, load_settings
from research_agent.state import AgentResult, SessionState, ToolStep
from research_agent.tracing import configure_tracing

_MAX_ARGUMENT_LENGTH = 80
_MAX_RESULT_PREVIEW = 60

_TOOL_CALL_ICON = "🔧"
_RESULT_ICON = "📎"
_ASCII_TOOL_CALL_ICON = "[tool]"
_ASCII_RESULT_ICON = "[result]"


def _truncate(text: str, limit: int) -> str:
    """Cut `text` to `limit` characters, marking a real cut with `...`."""
    return text if len(text) <= limit else f"{text[:limit]}..."


def _strip_query_string(url: str) -> str:
    """Drop the query string and fragment from a URL before it is logged.

    Notes
    -----
    A query string can carry an API key or session token; the console log is
    not the place for it (plan Etap 5, "safe log rules").
    """
    scheme, netloc, path, _, _ = urlsplit(url)
    return urlunsplit((scheme, netloc, path, "", ""))


def _format_argument(key: str, value: Any) -> str:
    """Render one keyword argument the way a Python call would show it."""
    if not isinstance(value, str):
        return f"{key}={value!r}"
    text = _strip_query_string(value) if key == "url" else value
    return f'{key}="{_truncate(text, _MAX_ARGUMENT_LENGTH)}"'


def format_tool_call(step: ToolStep, icon: str = _TOOL_CALL_ICON) -> str:
    """Render the tool-call log line (`🔧 Tool call: ...` by default)."""
    arguments = ", ".join(
        _format_argument(key, value) for key, value in step.arguments.items()
    )
    return f"{icon} Tool call: {step.name}({arguments})"


def format_tool_result(step: ToolStep, icon: str = _RESULT_ICON) -> str:
    """Render the result log line (`📎 Result: ...` by default).

    Notes
    -----
    Never the full payload: a saved report's Markdown or a page's extracted
    text can run to thousands of characters, and the model already has them
    in `self.messages` -- the console only needs enough to follow along.
    """
    if not step.ok:
        # ERROR: messages are already one short, safe sentence (plan A.4).
        return f"{icon} Result: {step.result}"
    if step.name == "web_search":
        try:
            count = len(json.loads(step.result))
        except (json.JSONDecodeError, TypeError):
            count = 0
        return f"{icon} Result: Found {count} results..."
    if step.name == "write_report":
        return f"{icon} Result: {step.result}"
    preview = _truncate(step.result, _MAX_RESULT_PREVIEW)
    return f"📎 Result: [{len(step.result)} chars] {preview}"


def _configure_console() -> tuple[str, str]:
    """Reconfigure stdout to UTF-8; fall back to ASCII log markers if it can't.

    Returns
    -------
    tuple of (str, str)
        Icons for the tool-call and result log lines: the emoji markers when
        the console accepts UTF-8, plain ASCII markers otherwise.

    Notes
    -----
    A Windows console still on a legacy code page (cp1251/cp866) raises
    `UnicodeEncodeError` the moment `print` sees an emoji -- not hypothetical,
    it broke an unrelated PDF-extraction script during this project's own
    planning (`insights.md`).
    """
    try:
        # sys.stdout is typed as the narrower `TextIO`, which has no
        # `reconfigure`; the cast tells mypy what every real stream (and
        # pytest's own capture object) actually is, without changing runtime
        # behaviour -- the `except` below still fires if that trust is wrong.
        cast(io.TextIOWrapper, sys.stdout).reconfigure(
            encoding="utf-8", errors="replace"
        )
        return _TOOL_CALL_ICON, _RESULT_ICON
    except (AttributeError, ValueError):
        return _ASCII_TOOL_CALL_ICON, _ASCII_RESULT_ICON


def format_session_stats(session: SessionState) -> str:
    """Render the `:stats` summary of what the session has done so far."""
    lines = [
        f"Turns: {len(session.runs)}",
        f"Sources read: {len(session.all_read_urls)}",
        f"Reports saved: {len(session.all_saved_reports)}",
    ]
    lines.extend(f"  {path}" for path in session.all_saved_reports)
    return "\n".join(lines)


def format_tracing_notice(settings: Settings, active: bool) -> str | None:
    """Say where this session's traces go, or why they go nowhere.

    Parameters
    ----------
    settings : Settings
        Configuration for this session.
    active : bool
        What `configure_tracing` decided.

    Returns
    -------
    str or None
        One line for the console, or ``None`` when tracing is off and nobody
        asked for it -- the default needs no announcement.

    Notes
    -----
    The second branch is the reason this function exists: tracing switched on
    without a key is a request that silently does nothing, which is the same
    class of failure this stage removed from `configure_tracing` itself.
    """
    if active:
        return f"LangSmith: tracing to project {settings.langsmith_project}."
    if settings.langsmith_tracing:
        return (
            "LangSmith: tracing is enabled but LANGSMITH_API_KEY is missing "
            "-- this session is not traced."
        )
    return None


def _handle_repl_command(user_input: str, agent: ResearchAgent) -> str | None:
    """Handle a local REPL command that never reaches the agent.

    Returns
    -------
    str or None
        ``"stop"`` when the REPL should end, ``"handled"`` when the input
        was a recognised command already acted on, or ``None`` when
        `user_input` is a real question for the agent.
    """
    if user_input.lower() in ("exit", "quit"):
        print("Goodbye!")
        return "stop"
    if user_input.lower() == ":stats":
        print(format_session_stats(agent.session))
        return "handled"
    return None


def _run_turn(
    agent: ResearchAgent,
    user_input: str,
    on_step: Callable[[ToolStep], None],
) -> AgentResult | None:
    """Run one question through the agent, surviving a failed request.

    Returns
    -------
    AgentResult or None
        The turn's result, or ``None`` when the request failed and the
        session should continue -- a `KeyboardInterrupt` is deliberately
        not caught here, since that means "stop the session", not "retry".

    Notes
    -----
    A failed request kills the turn, not the session: the user may want to
    retry the same question or ask a different one. `APIError` comes first
    because it is the more specific of the two.
    """
    try:
        return agent.run(user_input, on_step=on_step)
    except APIError:
        print(
            "\nAgent error: OpenAI API request failed. "
            "Check the API key and connection."
        )
        return None
    except OpenAIError:
        print("\nAgent error: the OpenAI client could not complete the request.")
        return None


def _print_turn_result(result: AgentResult) -> None:
    """Print the agent's final answer and any unread-source warning."""
    if result.final_answer is not None:
        print(f"\nAgent: {result.final_answer}")

    # The agent cannot rewrite a report the model wrote itself (hl-4
    # wants free-form Markdown there), so the honest move is to say so
    # rather than let an unread source pass as evidence.
    if result.cites_unread_sources:
        print(
            "\nWarning: the report lists a source that could not be "
            "opened -- treat those claims as unverified."
        )


def _start_session() -> tuple[ResearchAgent, str, str] | None:
    """Print the banner, load settings, and build the session's agent.

    Returns
    -------
    tuple of (ResearchAgent, str, str) or None
        The agent and its console icons, or ``None`` when configuration
        failed and the caller should exit before the loop starts.
    """
    tool_icon, result_icon = _configure_console()
    print("Research Agent (type 'exit' to quit, ':stats' for session stats)")
    print("-" * 40)

    try:
        settings = load_settings()
    except ValidationError:
        print("Configuration error: check OPENAI_API_KEY and values in .env.")
        return None

    # The agent configures tracing again when it builds its own client; the
    # call is idempotent, and doing it here is what lets the REPL report the
    # decision instead of leaving the user to guess.
    notice = format_tracing_notice(settings, configure_tracing(settings))
    if notice is not None:
        print(notice)

    try:
        agent = ResearchAgent(settings)
    except ValueError as error:
        # Only get_system_prompt raises this here, and its message already
        # lists the versions that do exist.
        print(f"Configuration error: {error}")
        return None

    return agent, tool_icon, result_icon


def main() -> None:
    """Run the interactive research REPL.

    Reads questions from the terminal until the user types ``exit`` or sends
    EOF, printing each tool call and its result as the agent makes them. One
    `ResearchAgent` serves the whole session, so its message list is the
    session's memory.
    """
    started = _start_session()
    if started is None:
        return
    agent, tool_icon, result_icon = started

    def log_step(step: ToolStep) -> None:
        print(f"\n{format_tool_call(step, tool_icon)}")
        print(format_tool_result(step, result_icon))

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        command_outcome = _handle_repl_command(user_input, agent)
        if command_outcome == "stop":
            break
        if command_outcome == "handled":
            continue

        try:
            result = _run_turn(agent, user_input, log_step)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        if result is None:
            continue
        _print_turn_result(result)
