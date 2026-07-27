"""Turn one dataset example into an agent run, and run a whole dataset.

``build_target`` is what LangSmith's ``evaluate()`` calls once per example: a
fresh ``ResearchAgent`` every time, since memory is ``self._messages`` and a
shared agent would leak state between examples (dataset example 9's own
follow-up would then "work" only because of whatever the previous example
left behind, not because memory actually functions). The dict it returns is
exactly the ``RunOutputs`` shape ``evals.evaluators.EVALUATORS`` grades.

``ensure_dataset`` uploads ``evals.dataset`` into LangSmith once, idempotent
by name, so a repeated run never creates a duplicate.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import uuid4

from langsmith import Client
from langsmith.evaluation import evaluate

from agent import AgentResult, LLMClient, ResearchAgent, ToolStep
from research_agent.prompts import SYSTEM_PROMPTS
from research_agent.settings import Settings, load_settings, output_directory
from evals.dataset import Split, get_split
from evals.evaluators import EVALUATORS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# `experiments/` is gitignored wholesale (plan K.3 convention): a saved
# report only has to exist on disk long enough for `report_saved` and
# `report_markdown` to read it back, it is not the deliverable.
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "output" / "eval-runs"
# LangSmith is the system of record for results (handoff.md's stage-8
# decisions); this is only an index of which local command produced which
# LangSmith experiment name, since `evaluate()` mints an unpredictable
# suffix and a browser tab full of them is not searchable by command line.
DEFAULT_EXPERIMENT_LOG = PROJECT_ROOT / "experiments" / "output" / "eval_runs.log"


class _NamedExperiment(Protocol):
    """The two attributes `_record_experiment` needs from `evaluate()`'s
    return value -- narrower than `ExperimentResults` so a test can pass a
    plain stand-in instead of running a real experiment."""

    @property
    def experiment_name(self) -> str: ...

    @property
    def url(self) -> str | None: ...


def _search_result_urls(steps: list[ToolStep]) -> list[str]:
    """URLs every successful ``web_search`` call in the run returned."""
    urls: list[str] = []
    for step in steps:
        if step.name != "web_search" or not step.ok:
            continue
        try:
            results = json.loads(step.result)
        except json.JSONDecodeError:
            continue
        if not isinstance(results, list):
            continue
        urls.extend(
            str(item["url"])
            for item in results
            if isinstance(item, dict) and item.get("url")
        )
    return urls


def _read_if_exists(path: str | None) -> str | None:
    """The saved report's own text, or ``None`` when there is no file."""
    if not path:
        return None
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def _final_result(results: list[AgentResult]) -> AgentResult:
    """The turn the grading evaluators should read.

    The last turn that actually saved a report, or the last turn overall
    when none did -- a follow-up commonly reuses the first turn's report
    without writing a new one, and that reuse is not a failure to grade.
    """
    for result in reversed(results):
        if result.saved_report_path is not None:
            return result
    return results[-1]


def build_target(
    prompt_version: str,
    output_root: Path,
    client_factory: Callable[[], LLMClient] | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the ``target(inputs) -> outputs`` function ``evaluate()`` calls.

    Parameters
    ----------
    prompt_version : str
        Key into `research_agent.prompts.SYSTEM_PROMPTS`. Rides in the run's own trace
        metadata (`agent.ResearchAgent.run`), which is how an experiment is
        attributed to the text that produced it.
    output_root : Path
        Parent directory for this experiment's saved reports. Each example
        gets its own subdirectory, named from a fresh UUID: `write_report`
        stamps filenames to the second, so two repetitions of one question
        finishing in the same second would otherwise overwrite each other.
    client_factory : callable, optional
        Builds the chat-completions client for one example. ``None`` (every
        real run) leaves it to `ResearchAgent` itself, which builds the real
        OpenAI client. A test passes a factory that returns a fresh
        `fakes.ScriptedChatClient`, since a real client would either reach
        the network or fail on the placeholder key tests use.

    Returns
    -------
    callable
        ``target(inputs)``, shaped for ``evals.evaluators.EVALUATORS``.
    """

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        with output_directory(str(output_root / uuid4().hex)):
            settings = load_settings().model_copy(
                update={"prompt_version": prompt_version}
            )
            if inputs.get("max_iterations"):
                settings = settings.model_copy(
                    update={"max_iterations": inputs["max_iterations"]}
                )

            client = client_factory() if client_factory is not None else None
            agent = ResearchAgent(settings, client=client)

            started = time.perf_counter()
            results = [agent.run(inputs["question"])]
            if inputs.get("followup"):
                results.append(agent.run(inputs["followup"]))
            elapsed = time.perf_counter() - started

        graded = _final_result(results)
        all_steps = [step for result in results for step in result.steps]
        return {
            "final_answer": results[-1].final_answer or "",
            "saved_report_path": graded.saved_report_path,
            "report_markdown": _read_if_exists(graded.saved_report_path),
            "report_source": graded.report_source,
            "tool_calls": [
                {"name": s.name, "arguments": s.arguments, "ok": s.ok}
                for s in all_steps
            ],
            "read_urls": sorted(agent.session.all_read_urls),
            "search_result_urls": _search_result_urls(all_steps),
            "iterations_used": results[-1].iterations_used,
            "stop_reason": results[-1].stop_reason,
            "wall_time_seconds": elapsed,
        }

    return target


def build_langsmith_client(settings: Settings) -> Client:
    """Build a LangSmith client from `Settings`, not from the environment.

    Parameters
    ----------
    settings : Settings
        Configuration for this process.

    Returns
    -------
    Client
        Client pointed at the configured region, authenticated with the
        configured key.

    Raises
    ------
    ValueError
        If no LangSmith key is configured. Failing here beats a 401 raised
        several frames deep inside the SDK.

    Notes
    -----
    A bare ``Client()`` reads its key and endpoint from `os.environ` alone,
    and this project's values arrive through ``.env``, which
    pydantic-settings parses without exporting (`agent.configure_tracing`
    exists for the same reason). The endpoint is the part that fails loudly:
    with an EU key and the SDK's default US endpoint, every call returns
    ``401 Invalid token`` -- a real region mismatch reported as if the key
    itself were wrong.

    Deliberately independent of `Settings.langsmith_tracing`: uploading a
    dataset and running an experiment are not tracing, and gating them on
    that flag would turn "tracing is off" into an authentication error.
    """
    api_key = settings.langsmith_api_key
    if api_key is None:
        raise ValueError(
            "LANGSMITH_API_KEY is not set. The evaluation suite reads and "
            "writes datasets in LangSmith, so a key is required."
        )
    return Client(
        api_key=api_key.get_secret_value(),
        api_url=settings.langsmith_endpoint,
        workspace_id=settings.langsmith_workspace_id,
    )


def ensure_dataset(client: Client, split: Split) -> str:
    """Upload ``split`` into LangSmith unless a dataset by that name exists.

    Parameters
    ----------
    client : langsmith.Client
    split : evals.dataset.Split
        Name and examples to upload.

    Returns
    -------
    str
        ``split.name``, ready to pass as ``evaluate(data=...)``.

    Notes
    -----
    Idempotent by name, not by content: editing a question in
    `evals/dataset.py` after the dataset already exists in LangSmith needs
    the dataset deleted by hand before the next run picks up the change.
    `Client` has no built-in diff-and-patch for examples, and silently
    overwriting a dataset mid-comparison would be worse than requiring a
    manual reset.
    """
    if client.has_dataset(dataset_name=split.name):
        return split.name

    client.create_dataset(
        dataset_name=split.name,
        description=(
            "Research agent behaviour suite: multi-step research, memory, "
            "error recovery, injection resistance and the report-write "
            "guarantee (plan Appendix B.5)."
        ),
    )
    client.create_examples(
        dataset_name=split.name,
        examples=[example.as_langsmith_example() for example in split.examples],
    )
    return split.name


def _record_experiment(
    results: _NamedExperiment,
    *,
    prompt_version: str,
    dataset: str,
    num_repetitions: int,
    max_concurrency: int,
    log_path: Path = DEFAULT_EXPERIMENT_LOG,
) -> str:
    """Append one line naming the LangSmith experiment this run produced.

    Parameters
    ----------
    results : _NamedExperiment
        Return value of `evaluate()` (or a stand-in with the same two
        attributes, for tests).
    prompt_version, dataset, num_repetitions, max_concurrency
        The command's own arguments, recorded alongside the experiment name
        so a later reader does not have to guess them from the name alone.
    log_path : Path, optional
        Defaults to `DEFAULT_EXPERIMENT_LOG`. Overridable so tests never
        touch the real file.

    Returns
    -------
    str
        The line written, so the caller can also print it.

    Notes
    -----
    Append-only, one line per invocation -- never truncated or rewritten, the
    same convention as `docs/prompt-log.md` and `docs/learning-log.md`.
    """
    line = (
        f"{datetime.now().isoformat(timespec='seconds')} "
        f"prompt_version={prompt_version} dataset={dataset} "
        f"num_repetitions={num_repetitions} max_concurrency={max_concurrency} "
        f"experiment={results.experiment_name} url={results.url}"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(line + "\n")
    return line


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the hl-4 research agent through a LangSmith dataset."
    )
    parser.add_argument(
        "--prompt-version",
        required=True,
        choices=sorted(SYSTEM_PROMPTS),
        help="Key into research_agent.prompts.SYSTEM_PROMPTS.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("dev", "holdout"),
        help="dev while tuning the prompt; holdout once, at the end.",
    )
    parser.add_argument(
        "--num-repetitions",
        type=int,
        default=3,
        help="Repeats per example (plan B.8: the model is variable, one run "
        "proves nothing).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=2,
        help="Examples to run at once (plan B.8). Deliberately low: ddgs "
        "rate-limits parallel searches, and a difference explained by "
        "DuckDuckGo throttling is not a difference between prompts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Where per-example saved reports go.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Upload the chosen split if needed, then run the agent across it."""
    args = _build_arg_parser().parse_args(argv)
    settings = load_settings()
    client = build_langsmith_client(settings)
    split = get_split(args.dataset)
    dataset_name = ensure_dataset(client, split)

    args.output_root.mkdir(parents=True, exist_ok=True)
    target = build_target(args.prompt_version, args.output_root)

    results = evaluate(
        target,
        data=dataset_name,
        evaluators=list(EVALUATORS),
        experiment_prefix=f"prompt-{args.prompt_version}",
        metadata={
            "prompt_version": args.prompt_version,
            "model": settings.model_name,
            "dataset_split": args.dataset,
        },
        max_concurrency=args.max_concurrency,
        num_repetitions=args.num_repetitions,
        client=client,
        description=(
            f"hl-4 stage 8: prompt {args.prompt_version} on the "
            f"{args.dataset} split."
        ),
    )
    print(
        _record_experiment(
            results,
            prompt_version=args.prompt_version,
            dataset=args.dataset,
            num_repetitions=args.num_repetitions,
            max_concurrency=args.max_concurrency,
            # Read afresh at call time, not `_record_experiment`'s own default
            # parameter (bound once at import) -- a test that monkeypatches
            # this module attribute has to actually redirect the write.
            log_path=DEFAULT_EXPERIMENT_LOG,
        )
    )


if __name__ == "__main__":
    main()
