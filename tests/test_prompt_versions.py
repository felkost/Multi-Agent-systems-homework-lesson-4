"""Prompt versions: the registry, the placeholders, the loud failure.

Stage 7 replaces one hardcoded prompt with a registry keyed by version,
because stage 8 has to run the same questions on v1 and v2 and attribute
each result to the text that produced it.
"""

from datetime import date

import pytest

from agent import ResearchAgent
from config import SYSTEM_PROMPTS, Settings, build_system_prompt, get_system_prompt

from fakes import ScriptedChatClient, ScriptedTurn


@pytest.mark.parametrize("version", sorted(SYSTEM_PROMPTS))
def test_every_registered_version_is_non_empty(version: str) -> None:
    assert SYSTEM_PROMPTS[version].strip()


def test_get_system_prompt_returns_the_registered_text() -> None:
    assert get_system_prompt("v1") is SYSTEM_PROMPTS["v1"]


def test_unknown_prompt_version_names_the_available_ones() -> None:
    with pytest.raises(ValueError) as error:
        get_system_prompt("v9")

    message = str(error.value)

    assert "v9" in message
    assert "v1" in message


def test_build_fills_in_the_iteration_budget() -> None:
    prompt = build_system_prompt("v1", max_iterations=5)

    assert "5 tool-call turns" in prompt
    assert "{max_iterations}" not in prompt


def test_build_fills_in_the_date(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        SYSTEM_PROMPTS, "test", "Today is {today}, budget {max_iterations}."
    )

    prompt = build_system_prompt("test", max_iterations=3, today=date(2026, 7, 26))

    assert prompt == "Today is 2026-07-26, budget 3."


def test_build_defaults_to_the_current_date(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(SYSTEM_PROMPTS, "test", "Today is {today}.")

    prompt = build_system_prompt("test", max_iterations=1)

    assert prompt == f"Today is {date.today().isoformat()}."


def test_agent_starts_the_session_with_the_selected_prompt(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    monkeypatch.setitem(
        SYSTEM_PROMPTS, "test", "Prompt under test, {max_iterations} turns."
    )
    settings = configured_settings.model_copy(update={"prompt_version": "test"})
    client = ScriptedChatClient([ScriptedTurn(content="Done.")])

    agent = ResearchAgent(settings, client=client)

    assert agent.messages[0] == {
        "role": "system",
        "content": f"Prompt under test, {settings.max_iterations} turns.",
    }
