"""`build_target` and `ensure_dataset`, run offline against scripted tools.

No test reaches the network or LangSmith: the model is scripted (`fakes.py`),
`DDGS`/`httpx` are patched the way `tests/test_react_loop.py` already does,
and `ensure_dataset` is exercised against a stand-in `Client`.
"""

import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
from langsmith.utils import ContextThreadPoolExecutor
from pydantic import SecretStr

import evals.run_eval as run_eval
from agent import AgentResult, ToolStep
from research_agent.settings import Settings
from research_agent.tools import fetch, search
from evals.dataset import DEV_DATASET, get_split
from evals.run_eval import (
    _final_result,
    _read_if_exists,
    _record_experiment,
    _search_result_urls,
    build_target,
    ensure_dataset,
)

from fakes import ScriptedChatClient, ScriptedTurn

pytestmark = pytest.mark.usefixtures("configured_settings")


def _patch_search(
    monkeypatch: pytest.MonkeyPatch, url: str = "https://a.example"
) -> None:
    search_client = Mock()
    search_client.text.return_value = [
        {"title": "Result", "href": url, "body": "A snippet"}
    ]
    monkeypatch.setattr(search, "DDGS", Mock(return_value=search_client))


def _patch_read_url(monkeypatch: pytest.MonkeyPatch, text: str = "Hello") -> None:
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


def _agent_result(
    saved_report_path: str | None,
    report_source: str = "none",
) -> AgentResult:
    return AgentResult(
        final_answer="Done.",
        steps=[],
        iterations_used=1,
        budget_exhausted=False,
        stop_reason="goal_satisfied",
        saved_report_path=saved_report_path,
        report_source=report_source,  # type: ignore[arg-type]
    )


def test_search_result_urls_collects_only_successful_web_search_steps() -> None:
    steps = [
        ToolStep(
            name="web_search",
            arguments={"query": "chunking"},
            result=(
                '[{"title": "A", "url": "https://a.example", "snippet": "..."}, '
                '{"title": "B", "url": "https://b.example", "snippet": "..."}]'
            ),
            ok=True,
        ),
        ToolStep(
            name="web_search",
            arguments={"query": "broken"},
            result="ERROR: Web search is temporarily unavailable.",
            ok=False,
        ),
        ToolStep(
            name="read_url",
            arguments={"url": "https://a.example"},
            result="Page text.",
            ok=True,
        ),
    ]

    assert _search_result_urls(steps) == ["https://a.example", "https://b.example"]


def test_search_result_urls_ignores_malformed_json() -> None:
    steps = [
        ToolStep(
            name="web_search",
            arguments={"query": "q"},
            result="not json",
            ok=True,
        )
    ]

    assert _search_result_urls(steps) == []


def test_search_result_urls_ignores_json_that_is_not_a_list() -> None:
    steps = [
        ToolStep(
            name="web_search",
            arguments={"query": "q"},
            result='{"unexpected": "shape"}',
            ok=True,
        )
    ]

    assert _search_result_urls(steps) == []


def test_read_if_exists_returns_none_for_no_path(tmp_path: Path) -> None:
    assert _read_if_exists(None) is None
    assert _read_if_exists("") is None


def test_read_if_exists_returns_none_for_a_missing_file(tmp_path: Path) -> None:
    missing = str(tmp_path / "never-written.md")

    assert _read_if_exists(missing) is None


def test_read_if_exists_returns_the_files_text(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    path.write_text("# Report", encoding="utf-8")

    assert _read_if_exists(str(path)) == "# Report"


def test_final_result_picks_the_last_turn_that_saved_a_report() -> None:
    first = _agent_result("first.md", "tool")
    second = _agent_result(None, "none")

    assert _final_result([first, second]) is first


def test_final_result_falls_back_to_the_last_turn_when_none_saved() -> None:
    first = _agent_result(None, "none")
    second = _agent_result(None, "none")

    assert _final_result([first, second]) is second


def test_build_target_runs_a_full_research_trajectory(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
    tmp_path: Path,
) -> None:
    _patch_search(monkeypatch, url="https://a.example/chunking")
    _patch_read_url(monkeypatch, text="Chunking splits text into pieces.")
    script = [
        ScriptedTurn(
            tool_calls=[("web_search", {"query": "chunking"})],
        ),
        ScriptedTurn(
            tool_calls=[("read_url", {"url": "https://a.example/chunking"})],
        ),
        ScriptedTurn(
            tool_calls=[
                (
                    "write_report",
                    {
                        "filename": "chunking.md",
                        "content": "# Chunking\n\nBody [1](#source-1).",
                    },
                )
            ],
        ),
        ScriptedTurn(content="Saved to output/chunking.md."),
    ]
    output_root = tmp_path / "eval-runs"

    target = build_target(
        "v2", output_root, client_factory=lambda: ScriptedChatClient(script)
    )
    result = target({"question": "Explain chunking.", "followup": None})

    assert result["report_source"] == "tool"
    assert result["saved_report_path"] is not None
    assert Path(result["saved_report_path"]).is_file()
    assert Path(result["saved_report_path"]).parent.parent == output_root
    assert result["report_markdown"] == "# Chunking\n\nBody [1](#source-1)."
    assert [c["name"] for c in result["tool_calls"]] == [
        "web_search",
        "read_url",
        "write_report",
    ]
    assert all(c["ok"] for c in result["tool_calls"])
    assert result["read_urls"] == ["https://a.example/chunking"]
    assert result["search_result_urls"] == ["https://a.example/chunking"]
    assert result["final_answer"] == "Saved to output/chunking.md."
    assert result["stop_reason"] == "goal_satisfied"
    assert result["wall_time_seconds"] >= 0.0


def test_build_target_isolates_each_examples_report_directory(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
    tmp_path: Path,
) -> None:
    def script() -> ScriptedChatClient:
        return ScriptedChatClient(
            [
                ScriptedTurn(
                    tool_calls=[
                        (
                            "write_report",
                            {"filename": "r.md", "content": "# R"},
                        )
                    ]
                ),
                ScriptedTurn(content="Done."),
            ]
        )

    output_root = tmp_path / "eval-runs"
    target = build_target("v2", output_root, client_factory=script)

    first = target({"question": "Q1", "followup": None})
    second = target({"question": "Q2", "followup": None})

    assert first["saved_report_path"] != second["saved_report_path"]
    assert (
        Path(first["saved_report_path"]).parent
        != Path(second["saved_report_path"]).parent
    )


def test_build_target_leaves_the_process_environment_alone(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
    tmp_path: Path,
) -> None:
    """The redirect is contextual, not global. An `os.environ` assignment
    would be visible here -- and to every other example running at the same
    time, which is the defect this replaced."""
    before = os.environ.get("OUTPUT_DIR")
    target = build_target(
        "v2",
        tmp_path / "eval-runs",
        client_factory=lambda: ScriptedChatClient([ScriptedTurn(content="Done.")]),
    )

    target({"question": "Where did you save the file?", "followup": None})

    assert os.environ.get("OUTPUT_DIR") == before


def test_concurrent_examples_keep_their_reports_apart(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
    tmp_path: Path,
) -> None:
    """Run two examples the way `evaluate(max_concurrency=2)` does.

    `ContextThreadPoolExecutor` submits each example through
    `copy_context().run(...)`; the barrier makes both examples hold their own
    redirect at the same moment, which is exactly when a process-global
    `OUTPUT_DIR` would hand one example's report to the other.
    """
    barrier = threading.Barrier(2, timeout=5)

    def script() -> ScriptedChatClient:
        return ScriptedChatClient(
            [
                ScriptedTurn(
                    tool_calls=[
                        ("write_report", {"filename": "r.md", "content": "# R"})
                    ]
                ),
                ScriptedTurn(content="Done."),
            ]
        )

    output_root = tmp_path / "eval-runs"
    target = build_target("v2", output_root, client_factory=script)

    def one_example(question: str) -> dict[str, Any]:
        barrier.wait()
        return target({"question": question, "followup": None})

    with ContextThreadPoolExecutor(2) as executor:
        first, second = list(executor.map(one_example, ["Q1", "Q2"]))

    first_directory = Path(first["saved_report_path"]).parent
    second_directory = Path(second["saved_report_path"]).parent

    # samefile, not path equality: under concurrent mkdir Windows sometimes
    # resolves one of the paths into its extended `\\?\C:\...` spelling, and
    # a string comparison would then fail on two names for the same place.
    assert not first_directory.samefile(second_directory)
    assert first_directory.parent.samefile(output_root)
    assert second_directory.parent.samefile(output_root)
    assert [p.name for p in first_directory.iterdir()] == [
        Path(first["saved_report_path"]).name
    ]
    assert [p.name for p in second_directory.iterdir()] == [
        Path(second["saved_report_path"]).name
    ]


def test_build_target_runs_the_followup_on_the_same_agent(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
    tmp_path: Path,
) -> None:
    """A follow-up that never writes again must still be graded against the
    report the first turn saved."""
    script = [
        ScriptedTurn(
            tool_calls=[("write_report", {"filename": "r.md", "content": "# R"})]
        ),
        ScriptedTurn(content="Saved to r.md."),
        ScriptedTurn(content="It also does X, building on that report."),
    ]

    target = build_target(
        "v2", tmp_path / "eval-runs", client_factory=lambda: ScriptedChatClient(script)
    )
    result = target({"question": "Explain X.", "followup": "Now compare it with Y."})

    assert result["report_source"] == "tool"
    assert result["saved_report_path"] is not None
    assert result["final_answer"] == "It also does X, building on that report."


def test_build_target_overrides_max_iterations_from_the_example(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
    tmp_path: Path,
) -> None:
    """The tight-budget example needs `max_iterations=3` without affecting
    every other example's own budget."""
    script = [ScriptedTurn(content="No time to research, here is a guess.")]
    target = build_target(
        "v2", tmp_path / "eval-runs", client_factory=lambda: ScriptedChatClient(script)
    )

    result = target({"question": "Q", "followup": None, "max_iterations": 3})

    assert result["iterations_used"] == 1


def test_build_langsmith_client_uses_the_configured_region(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    """A bare `Client()` defaults to the US endpoint while reading nothing
    from `.env`, so an EU key produced `401 Invalid token` -- a region
    mismatch reported as a bad key."""
    client_class = Mock()
    monkeypatch.setattr(run_eval, "Client", client_class)
    settings = configured_settings.model_copy(
        update={
            "langsmith_api_key": SecretStr("lsv2_pt_test"),
            "langsmith_endpoint": "https://eu.api.smith.langchain.com",
            "langsmith_workspace_id": "workspace-1",
        }
    )

    run_eval.build_langsmith_client(settings)

    client_class.assert_called_once_with(
        api_key="lsv2_pt_test",
        api_url="https://eu.api.smith.langchain.com",
        workspace_id="workspace-1",
    )


def test_build_langsmith_client_ignores_the_tracing_flag(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
) -> None:
    """Uploading a dataset is not tracing. Gating it on `langsmith_tracing`
    would turn "tracing is off" into an authentication error."""
    client_class = Mock()
    monkeypatch.setattr(run_eval, "Client", client_class)
    settings = configured_settings.model_copy(
        update={
            "langsmith_api_key": SecretStr("lsv2_pt_test"),
            "langsmith_tracing": False,
        }
    )

    run_eval.build_langsmith_client(settings)

    client_class.assert_called_once()


def test_build_langsmith_client_says_which_key_is_missing(
    configured_settings: Settings,
) -> None:
    settings = configured_settings.model_copy(update={"langsmith_api_key": None})

    with pytest.raises(ValueError) as error:
        run_eval.build_langsmith_client(settings)

    assert "LANGSMITH_API_KEY" in str(error.value)


def test_ensure_dataset_skips_upload_when_the_dataset_already_exists() -> None:
    client = Mock()
    client.has_dataset.return_value = True
    split = get_split("dev")

    name = ensure_dataset(client, split)

    assert name == DEV_DATASET
    client.create_dataset.assert_not_called()
    client.create_examples.assert_not_called()


def test_ensure_dataset_uploads_every_example_once() -> None:
    client = Mock()
    client.has_dataset.return_value = False
    split = get_split("dev")

    name = ensure_dataset(client, split)

    assert name == DEV_DATASET
    client.create_dataset.assert_called_once_with(
        dataset_name=DEV_DATASET,
        description=client.create_dataset.call_args.kwargs["description"],
    )
    uploaded = client.create_examples.call_args.kwargs["examples"]
    assert len(uploaded) == len(split.examples)
    assert uploaded[0] == split.examples[0].as_langsmith_example()


def test_build_arg_parser_uses_the_documented_defaults() -> None:
    args = run_eval._build_arg_parser().parse_args(
        ["--prompt-version", "v2", "--dataset", "dev"]
    )

    assert args.num_repetitions == 3
    assert args.max_concurrency == 2
    assert args.output_root == run_eval.DEFAULT_OUTPUT_ROOT


def test_build_arg_parser_rejects_an_unregistered_prompt_version() -> None:
    with pytest.raises(SystemExit):
        run_eval._build_arg_parser().parse_args(
            ["--prompt-version", "v9", "--dataset", "dev"]
        )


def test_build_arg_parser_rejects_an_unknown_dataset_split() -> None:
    with pytest.raises(SystemExit):
        run_eval._build_arg_parser().parse_args(
            ["--prompt-version", "v2", "--dataset", "prod"]
        )


def test_main_uploads_the_dataset_then_evaluates_it(
    monkeypatch: pytest.MonkeyPatch,
    configured_settings: Settings,
    tmp_path: Path,
) -> None:
    """Only `Client` and `evaluate` are replaced -- `ensure_dataset` and
    `build_target` run for real against the fake client, which is what
    actually exercises the CLI's own wiring rather than just asserting it."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_pt_test")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com")
    client_instance = Mock()
    client_instance.has_dataset.return_value = True
    client_class = Mock(return_value=client_instance)
    monkeypatch.setattr(run_eval, "Client", client_class)
    evaluate_mock = Mock(
        return_value=SimpleNamespace(
            experiment_name="prompt-v2-abc123",
            url="https://smith.langchain.com/o/x/datasets/y/compare?ex=z",
        )
    )
    monkeypatch.setattr(run_eval, "evaluate", evaluate_mock)
    output_root = tmp_path / "eval-runs"
    log_path = tmp_path / "eval_runs.log"
    monkeypatch.setattr(run_eval, "DEFAULT_EXPERIMENT_LOG", log_path)

    run_eval.main(
        [
            "--prompt-version",
            "v2",
            "--dataset",
            "dev",
            "--output-root",
            str(output_root),
        ]
    )

    # The CLI must reach the region `.env` names, not the SDK's US default.
    assert (
        client_class.call_args.kwargs["api_url"] == "https://eu.api.smith.langchain.com"
    )
    client_instance.has_dataset.assert_called_once_with(dataset_name=DEV_DATASET)
    client_instance.create_dataset.assert_not_called()
    evaluate_mock.assert_called_once()
    call = evaluate_mock.call_args
    assert call.args[0].__name__ == "target"
    assert call.kwargs["data"] == DEV_DATASET
    assert call.kwargs["experiment_prefix"] == "prompt-v2"
    assert call.kwargs["max_concurrency"] == 2
    assert call.kwargs["num_repetitions"] == 3
    assert call.kwargs["metadata"]["prompt_version"] == "v2"
    assert call.kwargs["metadata"]["dataset_split"] == "dev"
    assert call.kwargs["client"] is client_instance
    assert output_root.is_dir()
    logged = log_path.read_text(encoding="utf-8")
    assert "prompt-v2-abc123" in logged
    assert "dataset=dev" in logged


def test_record_experiment_names_the_experiment_and_where_to_find_it(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "eval_runs.log"
    results = SimpleNamespace(
        experiment_name="prompt-v1-abc123",
        url="https://smith.langchain.com/o/x/datasets/y/compare?ex=z",
    )

    line = _record_experiment(
        results,
        prompt_version="v1",
        dataset="dev",
        num_repetitions=3,
        max_concurrency=2,
        log_path=log_path,
    )

    assert log_path.read_text(encoding="utf-8") == line + "\n"
    assert "experiment=prompt-v1-abc123" in line
    assert "url=https://smith.langchain.com/o/x/datasets/y/compare?ex=z" in line
    assert "prompt_version=v1" in line
    assert "dataset=dev" in line
    assert "num_repetitions=3" in line
    assert "max_concurrency=2" in line


def test_record_experiment_appends_without_overwriting(tmp_path: Path) -> None:
    log_path = tmp_path / "eval_runs.log"
    log_path.write_text("earlier run\n", encoding="utf-8")
    results = SimpleNamespace(
        experiment_name="prompt-v2-xyz", url="https://example.com"
    )

    _record_experiment(
        results,
        prompt_version="v2",
        dataset="holdout",
        num_repetitions=1,
        max_concurrency=2,
        log_path=log_path,
    )

    content = log_path.read_text(encoding="utf-8")
    assert content.startswith("earlier run\n")
    assert "prompt-v2-xyz" in content


def test_record_experiment_creates_the_log_directory(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "eval_runs.log"
    results = SimpleNamespace(
        experiment_name="prompt-v1-abc", url="https://example.com"
    )

    _record_experiment(
        results,
        prompt_version="v1",
        dataset="dev",
        num_repetitions=3,
        max_concurrency=2,
        log_path=log_path,
    )

    assert log_path.is_file()
