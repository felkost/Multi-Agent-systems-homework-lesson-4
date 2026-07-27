"""Prompt versions: the registry, the placeholders, the loud failure.

Stage 7 replaces one hardcoded prompt with a registry keyed by version,
because stage 8 has to run the same questions on v1 and v2 and attribute
each result to the text that produced it.
"""

from datetime import date

import pytest

from agent import ResearchAgent
from research_agent.prompts import (
    SYSTEM_PROMPTS,
    build_system_prompt,
    get_system_prompt,
)
from research_agent.prompts.v2 import SYSTEM_PROMPT_V2_MIN
from research_agent.settings import Settings
from research_agent.tools import TOOL_REGISTRY

from fakes import ScriptedChatClient, ScriptedTurn


@pytest.mark.parametrize("version", sorted(SYSTEM_PROMPTS))
def test_every_registered_version_is_non_empty(version: str) -> None:
    assert SYSTEM_PROMPTS[version].strip()


def test_v2_min_contains_its_three_sections() -> None:
    prompt = SYSTEM_PROMPTS["v2-min"]

    assert "# Role" in prompt
    assert "## Core rules" in prompt
    assert "# Output contract" in prompt


def test_v2_min_has_no_later_rung_sections() -> None:
    prompt = SYSTEM_PROMPTS["v2-min"]

    for marker in (
        "# Example",
        "## Boundaries",
        "# Tool policy",
        "# Research protocol",
        "# Before you answer",
    ):
        assert marker not in prompt


def test_v2_min_output_contract_uses_the_citation_format() -> None:
    prompt = SYSTEM_PROMPTS["v2-min"]

    assert "[1](#source-1)" in prompt
    assert '<a id="source-1">' in prompt
    assert "## Sources" in prompt


def test_v2_min_fills_in_date_and_budget() -> None:
    prompt = build_system_prompt("v2-min", max_iterations=6, today=date(2026, 7, 26))

    assert "Today is 2026-07-26." in prompt
    assert "6 reasoning steps" in prompt
    assert "{today}" not in prompt
    assert "{max_iterations}" not in prompt


def test_v2_few_extends_v2_min_with_an_example() -> None:
    prompt = SYSTEM_PROMPTS["v2-few"]

    assert prompt.startswith(SYSTEM_PROMPT_V2_MIN)
    assert "# Example" in prompt


def test_v2_few_has_no_full_rung_sections() -> None:
    prompt = SYSTEM_PROMPTS["v2-few"]

    for marker in (
        "## Boundaries",
        "# Tool policy",
        "# Research protocol",
        "# Before you answer",
    ):
        assert marker not in prompt


def test_v2_few_example_shows_one_search_per_subquestion() -> None:
    prompt = SYSTEM_PROMPTS["v2-few"]

    assert prompt.count("web_search(") == 2
    assert prompt.count("read_url(") == 2
    assert prompt.count("write_report(") == 1


def test_v2_contains_every_full_rung_section() -> None:
    prompt = SYSTEM_PROMPTS["v2"]

    for marker in (
        "# Role",
        "## Core rules",
        "## Boundaries",
        "# Tool policy",
        "# Research protocol",
        "# Output contract",
        "# Example",
        "# Stop criterion",
        "# Before you answer",
    ):
        assert marker in prompt


def test_v2_documents_every_registered_tool() -> None:
    """Plan E.4: a tool added to `TOOL_REGISTRY` but never named in the
    prompt is a defect the schema oracle cannot catch, since the model
    still receives its schema either way."""
    prompt = SYSTEM_PROMPTS["v2"]

    for name in TOOL_REGISTRY:
        assert name in prompt


def test_v2_boundaries_rule_out_answering_from_memory() -> None:
    prompt = SYSTEM_PROMPTS["v2"]

    assert "writing or debugging code" in prompt
    assert "answering from your own memory when no source supports the claim" in prompt


def test_v2_before_you_answer_requires_inline_citations() -> None:
    """The citation format is specified in `# Output contract`, which the
    model demonstrably read and ignored: stage 7's ladder measured in-text
    references in only 2 of 13 runs where the model wrote the report
    itself. This repeats the requirement in the checklist closest to the
    moment of writing (plan E.5's minimal-change rule: a clause on an
    existing item, not a new section)."""
    checklist = SYSTEM_PROMPTS["v2"].split("# Before you answer")[1]

    assert "[n](#source-n)" in checklist


def test_v2_before_you_answer_forbids_restating_the_report() -> None:
    """Stage 6's third measured defect: the final chat message repeated the
    whole saved report (8.4K tokens) instead of just its path."""
    prompt = SYSTEM_PROMPTS["v2"]

    assert "your final message repeats that exact path" in prompt


def test_settings_default_prompt_version_is_v2(
    configured_settings: Settings,
) -> None:
    assert configured_settings.prompt_version == "v2"


def test_agent_defaults_to_the_full_v2_prompt(
    configured_settings: Settings,
) -> None:
    client = ScriptedChatClient([ScriptedTurn(content="Done.")])

    agent = ResearchAgent(configured_settings, client=client)

    assert agent.messages[0]["content"].startswith("# Role")
    assert "# Before you answer" in agent.messages[0]["content"]


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


def test_v2_example_avoids_the_agents_own_subject_matter() -> None:
    """The worked example must not be about what the agent gets asked.

    Measured on 35 real runs: with a RAG-topic example, 30 of 30 opened
    with the example's literal first query, `"naive RAG pipeline
    explained"`; with the same example rewritten about databases, 0 of 5
    did. A topic-matched example stops being a format demonstration and
    becomes an answer key. Plan L.4 rules out the obvious alternative
    (more examples), so the lever is the topic of the one example.
    """
    example = SYSTEM_PROMPTS["v2"].split("# Example")[1]

    assert "RAG" not in example
    assert "retrieval" not in example.lower()
