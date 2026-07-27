"""The evaluation dataset: coverage, split hygiene, and the payload shape.

A dataset defect is expensive in a way a code defect is not -- it is only
visible after the experiment has been paid for and run, and it invalidates
every number the experiment produced. These tests are cheap by comparison.
"""

import re

import pytest

from research_agent.prompts import SYSTEM_PROMPTS
from evals.dataset import (
    COMMON_KNOWN_BAD,
    DEV_DATASET,
    DEV_EXAMPLES,
    HOLDOUT_DATASET,
    HOLDOUT_EXAMPLES,
    INJECTION_CANARY,
    INJECTION_PAGE_URL,
    JUDGED_CASES,
    EvalExample,
    get_split,
)

ALL_EXAMPLES: tuple[EvalExample, ...] = DEV_EXAMPLES + HOLDOUT_EXAMPLES

# Subjects of the worked example in `# Example`. Stage 7 measured what
# happens when they overlap with what the agent is asked: 30 of 30 runs
# opened with the example's literal first query. Update this list if the
# example's topic ever changes.
PROMPT_EXAMPLE_SUBJECTS = ("postgresql", "mongodb")

REQUIRED_DEV_CASES = frozenset(
    {
        "research",
        "factual",
        "conflicting",
        "unreachable_source",
        "injection",
        "followup",
        "non_research",
        "tight_budget",
        "duplicate_sources",
        "early_answer",
        "out_of_coverage",
        "gibberish",
    }
)

NO_REPORT_CASES = frozenset({"non_research", "gibberish"})

RESEARCH_EXAMPLES = tuple(e for e in ALL_EXAMPLES if e.case == "research")
NO_REPORT_EXAMPLES = tuple(e for e in ALL_EXAMPLES if e.case in NO_REPORT_CASES)


def test_dev_split_has_fifteen_examples() -> None:
    assert len(DEV_EXAMPLES) == 15


def test_holdout_split_has_six_examples() -> None:
    assert len(HOLDOUT_EXAMPLES) == 6


def test_example_ids_are_unique() -> None:
    ids = [example.id for example in ALL_EXAMPLES]

    assert len(ids) == len(set(ids))


def test_dev_covers_every_planned_case() -> None:
    assert {example.case for example in DEV_EXAMPLES} == REQUIRED_DEV_CASES


def test_holdout_reuses_dev_cases_without_reusing_dev_questions() -> None:
    """Held-out has to measure the same behaviours on unseen material: same
    case labels, no shared question (plan J.2)."""
    dev_questions = {example.question for example in DEV_EXAMPLES}

    for example in HOLDOUT_EXAMPLES:
        assert example.case in REQUIRED_DEV_CASES
        assert example.question not in dev_questions


@pytest.mark.parametrize("example", ALL_EXAMPLES, ids=lambda e: e.id)
def test_no_question_shares_the_prompt_examples_topic(example: EvalExample) -> None:
    """A question about the worked example's own subject stops measuring the
    prompt and starts measuring how well the model copies the example."""
    question = example.question.lower()
    prompt_queries = re.findall(r'web_search\("([^"]+)"\)', SYSTEM_PROMPTS["v2"])

    for subject in PROMPT_EXAMPLE_SUBJECTS:
        assert subject not in question
    for query in prompt_queries:
        assert query.lower() not in question


@pytest.mark.parametrize("example", ALL_EXAMPLES, ids=lambda e: e.id)
def test_every_example_carries_the_common_known_bad_strings(
    example: EvalExample,
) -> None:
    for phrase in COMMON_KNOWN_BAD:
        assert phrase in example.known_bad


@pytest.mark.parametrize("example", ALL_EXAMPLES, ids=lambda e: e.id)
def test_the_judge_is_used_exactly_where_prose_is_the_defect(
    example: EvalExample,
) -> None:
    assert bool(example.assertions) == (example.case in JUDGED_CASES)


@pytest.mark.parametrize("example", ALL_EXAMPLES, ids=lambda e: e.id)
def test_the_tool_call_band_is_coherent(example: EvalExample) -> None:
    assert example.min_tool_calls >= 0
    if example.max_tool_calls is not None:
        assert example.max_tool_calls >= example.min_tool_calls


@pytest.mark.parametrize("example", RESEARCH_EXAMPLES, ids=lambda e: e.id)
def test_a_research_question_needs_several_tool_calls_and_a_report(
    example: EvalExample,
) -> None:
    assert example.min_tool_calls >= 3
    assert example.expect_report


@pytest.mark.parametrize("example", NO_REPORT_EXAMPLES, ids=lambda e: e.id)
def test_a_non_research_question_expects_no_report_and_no_tools(
    example: EvalExample,
) -> None:
    """The completion gate guarantees a report for research. This is the
    other half of the contract: it must not invent one for a question that
    never asked for research."""
    assert not example.expect_report
    assert example.max_tool_calls == 0


@pytest.mark.parametrize("example", ALL_EXAMPLES, ids=lambda e: e.id)
def test_only_followup_examples_carry_a_second_question(
    example: EvalExample,
) -> None:
    assert (example.followup is not None) == (example.case == "followup")


def test_only_the_tight_budget_example_overrides_the_iteration_budget() -> None:
    overridden = [e for e in ALL_EXAMPLES if e.max_iterations is not None]

    assert [e.case for e in overridden] == ["tight_budget"]
    assert overridden[0].max_iterations == 3
    assert overridden[0].expect_report


def test_the_injection_example_points_at_the_trap_and_names_the_canary() -> None:
    example = next(e for e in DEV_EXAMPLES if e.case == "injection")

    assert INJECTION_PAGE_URL in example.question
    assert INJECTION_CANARY in example.known_bad


def test_langsmith_payload_carries_inputs_reference_and_metadata() -> None:
    example = EvalExample(
        id="dev-00-shape",
        case="research",
        question="Q?",
        known_bad=("bad",),
        expected_signals=("signal",),
        assertions=("The report exists.",),
        followup="And then?",
        max_iterations=4,
        min_tool_calls=2,
        max_tool_calls=7,
    )

    payload = example.as_langsmith_example()

    assert payload == {
        "inputs": {
            "question": "Q?",
            "followup": "And then?",
            "max_iterations": 4,
        },
        "outputs": {
            "case": "research",
            "min_tool_calls": 2,
            "max_tool_calls": 7,
            "expect_report": True,
            "expected_signals": ["signal"],
            "known_bad": ["bad"],
            "assertions": ["The report exists."],
        },
        "metadata": {"id": "dev-00-shape", "case": "research"},
    }


@pytest.mark.parametrize("example", ALL_EXAMPLES, ids=lambda e: e.id)
def test_every_example_renders_json_ready_values(example: EvalExample) -> None:
    payload = example.as_langsmith_example()

    for key in ("expected_signals", "known_bad", "assertions"):
        assert isinstance(payload["outputs"][key], list)


def test_get_split_returns_the_dataset_name_and_its_examples() -> None:
    assert get_split("dev").name == DEV_DATASET
    assert get_split("dev").examples == DEV_EXAMPLES
    assert get_split("holdout").name == HOLDOUT_DATASET
    assert get_split("holdout").examples == HOLDOUT_EXAMPLES


def test_unknown_split_names_the_available_ones() -> None:
    with pytest.raises(ValueError) as error:
        get_split("prod")

    message = str(error.value)

    assert "prod" in message
    assert "dev" in message
    assert "holdout" in message
