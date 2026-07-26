"""REPL orchestration: commands, error handling, console encoding.

``main()`` itself sits outside the coverage bar (plan G.5, interactive REPL),
but its branches are still worth pinning directly: a broken request must not
end the session, and local commands like `:stats` must never reach the model.
``ResearchAgent.run`` is replaced with a spy in every test that reaches the
REPL loop body, so nothing here calls the network.
"""

from unittest.mock import Mock

import httpx
import pytest
from openai import APIError, OpenAIError

import agent as agent_module
import main
from agent import AgentResult, SessionState
from config import Settings


def _run_repl(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, inputs: list[str]
) -> None:
    monkeypatch.setattr(main, "load_settings", lambda: settings)
    monkeypatch.setattr("builtins.input", Mock(side_effect=inputs))
    main.main()


def test_repl_ignores_empty_input_without_calling_run(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    run_spy = Mock()
    monkeypatch.setattr(agent_module.ResearchAgent, "run", run_spy)

    _run_repl(monkeypatch, configured_settings, ["", "   ", "exit"])

    run_spy.assert_not_called()


@pytest.mark.parametrize("command", ["exit", "quit", "EXIT", "Quit"])
def test_repl_exit_command_stops_without_calling_run(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
    command: str,
) -> None:
    run_spy = Mock()
    monkeypatch.setattr(agent_module.ResearchAgent, "run", run_spy)

    _run_repl(monkeypatch, configured_settings, [command])

    run_spy.assert_not_called()


def test_repl_eof_on_input_stops_the_loop(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(main, "load_settings", lambda: configured_settings)
    monkeypatch.setattr("builtins.input", Mock(side_effect=EOFError()))

    main.main()

    assert "Goodbye!" in capsys.readouterr().out


def test_repl_keyboard_interrupt_on_input_stops_the_loop(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(main, "load_settings", lambda: configured_settings)
    monkeypatch.setattr("builtins.input", Mock(side_effect=KeyboardInterrupt()))

    main.main()

    assert "Goodbye!" in capsys.readouterr().out


def test_repl_stats_command_does_not_call_run(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_spy = Mock()
    monkeypatch.setattr(agent_module.ResearchAgent, "run", run_spy)

    _run_repl(monkeypatch, configured_settings, [":stats", "exit"])

    run_spy.assert_not_called()
    assert "Sources read: 0" in capsys.readouterr().out


def test_repl_runs_the_agent_for_a_real_question(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = AgentResult(
        final_answer="RAG trades freshness for context size.",
        steps=[],
        iterations_used=1,
        budget_exhausted=False,
        stop_reason="goal_satisfied",
        saved_report_path=None,
        report_source="none",
    )
    run_spy = Mock(return_value=result)
    monkeypatch.setattr(agent_module.ResearchAgent, "run", run_spy)

    _run_repl(monkeypatch, configured_settings, ["What is RAG?", "exit"])

    run_spy.assert_called_once()
    assert run_spy.call_args.args[0] == "What is RAG?"
    assert "on_step" in run_spy.call_args.kwargs
    assert "Agent: RAG trades freshness for context size." in capsys.readouterr().out


def test_repl_survives_an_api_error(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    run_spy = Mock(side_effect=APIError("boom", request=request, body=None))
    monkeypatch.setattr(agent_module.ResearchAgent, "run", run_spy)

    _run_repl(monkeypatch, configured_settings, ["What is RAG?", "exit"])

    assert "OpenAI API request failed" in capsys.readouterr().out


def test_repl_survives_a_generic_openai_error(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_spy = Mock(side_effect=OpenAIError("boom"))
    monkeypatch.setattr(agent_module.ResearchAgent, "run", run_spy)

    _run_repl(monkeypatch, configured_settings, ["What is RAG?", "exit"])

    assert "could not complete the request" in capsys.readouterr().out


def test_repl_survives_a_keyboard_interrupt_during_run(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_spy = Mock(side_effect=KeyboardInterrupt())
    monkeypatch.setattr(agent_module.ResearchAgent, "run", run_spy)

    _run_repl(monkeypatch, configured_settings, ["What is RAG?"])

    assert "Goodbye!" in capsys.readouterr().out


def test_repl_settings_error_exits_before_the_loop(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    main.main()

    assert "Configuration error" in capsys.readouterr().out


def test_repl_unknown_prompt_version_exits_before_the_loop(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = configured_settings.model_copy(update={"prompt_version": "v9"})
    monkeypatch.setattr(main, "load_settings", lambda: settings)
    input_spy = Mock()
    monkeypatch.setattr("builtins.input", input_spy)

    main.main()

    input_spy.assert_not_called()
    assert "Unknown prompt version 'v9'" in capsys.readouterr().out


def test_configure_console_uses_emoji_icons_by_default() -> None:
    tool_icon, result_icon = main._configure_console()

    assert (tool_icon, result_icon) == ("🔧", "📎")


def test_configure_console_falls_back_to_ascii_icons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken_stdout = Mock()
    broken_stdout.reconfigure.side_effect = AttributeError("no reconfigure")
    monkeypatch.setattr(main.sys, "stdout", broken_stdout)

    tool_icon, result_icon = main._configure_console()

    assert (tool_icon, result_icon) == ("[tool]", "[result]")


def test_format_session_stats_reports_an_empty_session() -> None:
    text = main.format_session_stats(SessionState())

    assert "Turns: 0" in text
    assert "Sources read: 0" in text
    assert "Reports saved: 0" in text


def test_format_session_stats_lists_saved_report_paths() -> None:
    session = SessionState()
    session.runs.append(agent_module.RunState(saved_report_path="output/one.md"))
    session.runs.append(agent_module.RunState(saved_report_path="output/two.md"))

    text = main.format_session_stats(session)

    assert "Reports saved: 2" in text
    assert "output/one.md" in text
    assert "output/two.md" in text


def test_tracing_notice_names_the_project_when_tracing_is_on(
    configured_settings: Settings,
) -> None:
    notice = main.format_tracing_notice(configured_settings, True)

    assert notice is not None
    assert configured_settings.langsmith_project in notice


def test_tracing_notice_warns_when_the_key_is_missing(
    configured_settings: Settings,
) -> None:
    settings = configured_settings.model_copy(update={"langsmith_tracing": True})

    notice = main.format_tracing_notice(settings, False)

    assert notice is not None
    assert "LANGSMITH_API_KEY" in notice


def test_tracing_notice_stays_quiet_when_nobody_asked_for_tracing(
    configured_settings: Settings,
) -> None:
    assert main.format_tracing_notice(configured_settings, False) is None


def test_repl_reports_that_a_requested_trace_will_not_happen(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = configured_settings.model_copy(update={"langsmith_tracing": True})
    monkeypatch.setattr(agent_module.ResearchAgent, "run", Mock())

    _run_repl(monkeypatch, settings, ["exit"])

    assert "not traced" in capsys.readouterr().out
