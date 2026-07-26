import io
import json
import sys
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

from openai import APIError, OpenAIError
from pydantic import ValidationError

from agent import ResearchAgent, SessionState, ToolStep
from config import load_settings

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


def main() -> None:
    """Run the interactive research REPL.

    Reads questions from the terminal until the user types ``exit`` or sends
    EOF, printing each tool call and its result as the agent makes them. One
    `ResearchAgent` serves the whole session, so its message list is the
    session's memory.
    """
    tool_icon, result_icon = _configure_console()

    def log_step(step: ToolStep) -> None:
        print(f"\n{format_tool_call(step, tool_icon)}")
        print(format_tool_result(step, result_icon))

    print("Research Agent (type 'exit' to quit, ':stats' for session stats)")
    print("-" * 40)

    try:
        settings = load_settings()
    except ValidationError:
        print("Configuration error: check OPENAI_API_KEY and values in .env.")
        return

    agent = ResearchAgent(settings)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if user_input.lower() == ":stats":
            print(format_session_stats(agent.session))
            continue

        try:
            result = agent.run(user_input, on_step=log_step)
        # A failed request kills the turn, not the session: the user may want
        # to retry the same question or ask a different one. APIError comes
        # first because it is the more specific of the two.
        except APIError:
            print(
                "\nAgent error: OpenAI API request failed. "
                "Check the API key and connection."
            )
            continue
        except OpenAIError:
            print("\nAgent error: the OpenAI client could not complete the request.")
            continue
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if result.final_answer is not None:
            print(f"\nAgent: {result.final_answer}")


if __name__ == "__main__":
    main()
