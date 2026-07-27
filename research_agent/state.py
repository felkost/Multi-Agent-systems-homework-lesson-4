"""The plain dataclasses one turn and one session pass around.

No logic lives here on purpose: everything in this module is data, so it
can sit at the bottom of the import graph and be imported by every other
module in this package without risk of a cycle.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

Messages = list[dict[str, Any]]

StopReason = Literal[
    "goal_satisfied",
    "iteration_limit",
    "token_budget",
    "stagnation",
    "tool_failures",
    "refusal",
    "truncated",
]

ReportSource = Literal["tool", "fallback", "none"]


@dataclass(slots=True)
class ToolStep:
    """One executed tool call: what the CLI logs and what evaluators read."""

    name: str
    arguments: dict[str, Any]
    result: str
    ok: bool


@dataclass(slots=True)
class RunState:
    """What one turn accumulated while its tools ran."""

    tool_calls_made: int = 0
    read_urls: list[str] = field(default_factory=list)
    consecutive_tool_errors: int = 0
    # Keyed on the raw query string. Lives only as long as this RunState (one
    # run()); a repeat of the same phrase in a later question is not a cache
    # hit, since the web has had a chance to change by then.
    search_cache: dict[str, str] = field(default_factory=dict)
    last_report_markdown: str | None = None
    last_report_filename: str | None = None
    saved_report_path: str | None = None
    # The Markdown that actually reached disk, whichever path wrote it.
    # Kept so a finished turn can be checked against `read_urls` without
    # reading the file back.
    saved_report_markdown: str | None = None

    @property
    def successful_reads(self) -> int:
        """How many `read_url` calls returned a page during this turn."""
        return len(self.read_urls)


@dataclass(slots=True)
class SessionState:
    """What the session accumulated, one `RunState` per finished turn.

    Notes
    -----
    Without it a turn's business state dies with its `RunState` while the
    messages live on, so "what have you already read?" has two answers that
    drift apart (12-factor #5, plan H.3).
    """

    runs: list[RunState] = field(default_factory=list)

    @property
    def all_read_urls(self) -> set[str]:
        """Every URL read successfully so far, deduplicated."""
        return {url for run in self.runs for url in run.read_urls}

    @property
    def all_saved_reports(self) -> list[str]:
        """Paths of the reports saved so far, oldest first."""
        return [run.saved_report_path for run in self.runs if run.saved_report_path]


@dataclass(slots=True)
class StepResult:
    """Outcome of one model call and the tool calls it requested."""

    messages: Messages
    steps: list[ToolStep]
    final_answer: str | None
    stop_reason: StopReason | None


@dataclass(slots=True)
class AgentResult:
    """What one user question produced."""

    final_answer: str | None
    steps: list[ToolStep]
    iterations_used: int
    budget_exhausted: bool
    stop_reason: StopReason
    saved_report_path: str | None
    report_source: ReportSource
    # True when the saved report lists a URL no `read_url` returned. The
    # gate's own path cannot produce this (render_report filters), so it
    # only ever flags a report the model wrote itself -- where hl-4
    # requires free-form Markdown, leaving the code able to report the
    # problem but not to fix it.
    cites_unread_sources: bool = False
