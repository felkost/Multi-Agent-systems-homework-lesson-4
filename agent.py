"""The research agent: session memory around the ReAct loop, plus the
completion gate that guarantees a report gets saved.
"""

from typing import Callable

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from openai import BadRequestError
from pydantic import ValidationError

from research_agent.history import compact_history
from research_agent.llm import LLMClient, build_client
from research_agent.loop import react_step
from research_agent.prompts import build_system_prompt
from research_agent.prompts.requests import (
    FALLBACK_REPORT_REQUEST,
    FALLBACK_STRUCTURED_REPORT_REQUEST,
)
from research_agent.report import (
    ResearchReport,
    _build_report_filename,
    _cites_unread_sources,
    render_report,
)
from research_agent.settings import Settings
from research_agent.state import (
    AgentResult,
    Messages,
    ReportSource,
    RunState,
    SessionState,
    StepResult,
    StopReason,
    ToolStep,
)
from research_agent.tools.contract import REPORT_SAVED_PREFIX
from research_agent.tools.report_writer import write_report


class ResearchAgent:
    """Research agent that runs its own ReAct loop.

    Parameters
    ----------
    settings : Settings
        Model, temperature and budget for the session.
    client : LLMClient, optional
        Chat-completions client. Tests pass a scripted one, which is what lets
        the suite run without an API key.

    Notes
    -----
    The session's memory is `messages` and nothing else: one list of dicts
    living on the instance for as long as the process runs.
    """

    def __init__(self, settings: Settings, client: LLMClient | None = None) -> None:
        self._settings = settings
        if client is None:
            client = build_client(settings)
        # Annotated explicitly: without it, mypy narrows the attribute to the
        # concrete OpenAI type through this conditional assignment, and any
        # later call through self._client is then checked against the SDK's
        # own strict overloads instead of this Protocol.
        self._client: LLMClient = client
        system_prompt = build_system_prompt(
            settings.prompt_version, settings.max_iterations
        )
        self._messages: Messages = [{"role": "system", "content": system_prompt}]
        self._session = SessionState()

    @property
    def messages(self) -> Messages:
        """Conversation history of the whole session, oldest message first."""
        return self._messages

    @property
    def session(self) -> SessionState:
        """Sources read and reports saved, accumulated over every turn."""
        return self._session

    @traceable(run_type="chain", name="research_run")
    def run(
        self,
        user_input: str,
        on_step: Callable[[ToolStep], None] | None = None,
    ) -> AgentResult:
        """Answer one question, calling tools until the model stops asking.

        Parameters
        ----------
        user_input : str
            The user's question.
        on_step : callable, optional
            Called once per executed tool call, as soon as it finishes, so a
            caller such as the CLI can print progress during the turn instead
            of only after ``run`` returns.

        Returns
        -------
        AgentResult
            Final answer, executed steps and why the turn ended.

        Notes
        -----
        The metadata is attached from inside the body because the values live
        on the instance: a decorator argument is evaluated once at import
        time, when no `Settings` exists yet. `get_current_run_tree` returns
        ``None`` whenever tracing is off, which is what keeps this block free
        for untraced runs.
        """
        self._attach_run_metadata()

        state = RunState()
        pending = self._prepare_turn(user_input)
        step, steps, iteration = self._run_react_loop(pending, state, on_step)

        # Assigned once, after the turn survived. An exception mid-turn would
        # otherwise leave a tool_call_id without its tool result, and every
        # later request would fail with HTTP 400.
        self._messages = step.messages
        result = self._finish(step, steps, state, iteration, user_input)
        # The completion gate can still save a report after the loop ended, so
        # the turn's own record is only complete once _finish has run.
        state.saved_report_path = result.saved_report_path
        self._session.runs.append(state)
        return result

    def _attach_run_metadata(self) -> None:
        """Attach this run's model, budget and prompt version to its trace.

        Notes
        -----
        `get_current_run_tree` returns ``None`` whenever tracing is off,
        which is what keeps this a no-op for untraced runs.
        """
        run_tree = get_current_run_tree()
        if run_tree is not None:
            run_tree.add_metadata(
                {
                    "model": self._settings.model_name,
                    "max_iterations": self._settings.max_iterations,
                    "prompt_version": self._settings.prompt_version,
                }
            )

    def _run_react_loop(
        self,
        pending: Messages,
        state: RunState,
        on_step: Callable[[ToolStep], None] | None,
    ) -> tuple[StepResult, list[ToolStep], int]:
        """Call `react_step` repeatedly until it reports a stop reason.

        Returns
        -------
        tuple of (StepResult, list of ToolStep, int)
            The final step, every step executed across all iterations, and
            how many iterations were used.
        """
        max_iterations = self._settings.max_iterations
        steps: list[ToolStep] = []

        step = react_step(
            pending,
            self._client,
            self._settings,
            state,
            force_write_report=max_iterations == 1,
            on_step=on_step,
        )
        steps.extend(step.steps)
        iteration = 1

        while step.stop_reason is None and iteration < max_iterations:
            iteration += 1
            step = react_step(
                step.messages,
                self._client,
                self._settings,
                state,
                force_write_report=iteration == max_iterations,
                on_step=on_step,
            )
            steps.extend(step.steps)

        return step, steps, iteration

    def _prepare_turn(self, user_input: str) -> Messages:
        """Compact stale history and append the new question.

        Notes
        -----
        Compacted here and not inside the loop: within one question the
        pages just read are the evidence the model is reasoning over, and
        only a finished turn's payloads have earned their summary (plan
        H.6).
        """
        history = compact_history(self._messages, self._settings.compact_keep_recent)
        return [*history, {"role": "user", "content": user_input}]

    def _finish(
        self,
        step: StepResult,
        steps: list[ToolStep],
        state: RunState,
        iterations_used: int,
        question: str,
    ) -> AgentResult:
        """Assemble the result of a finished turn."""
        stop_reason: StopReason = step.stop_reason or "iteration_limit"
        saved_report_path, report_source = self._ensure_report_saved(state, question)
        return AgentResult(
            final_answer=step.final_answer,
            steps=steps,
            iterations_used=iterations_used,
            budget_exhausted=step.stop_reason is None,
            stop_reason=stop_reason,
            saved_report_path=saved_report_path,
            report_source=report_source,
            cites_unread_sources=_cites_unread_sources(
                state.saved_report_markdown, state.read_urls
            ),
        )

    def _ensure_report_saved(
        self, state: RunState, question: str
    ) -> tuple[str | None, ReportSource]:
        """Decide what report this turn produced, once the loop has ended.

        Parameters
        ----------
        state : RunState
            What the turn's tool calls accumulated.
        question : str
            The user's question, used to name a fallback report file.

        Returns
        -------
        tuple of (str or None, ReportSource)
            Path of the saved report, and how that file came to exist.

        Notes
        -----
        The guarantee lives here, in Python after the loop, rather than in
        the system prompt: an instruction the model is asked to follow is
        not something the code can rely on.

        A turn with no successful read and no attempted report is treated as
        a non-research turn (e.g. "where did you save it?"): a report with
        no evidence behind it has no value, and writing one on every reply
        would litter `output/` with noise.

        When the model attempted `write_report` but the call failed, this
        retries with the same markdown under a filename built from the
        question, deliberately not the model's own filename: reusing it
        would repeat the exact failure whenever the name itself was the
        problem, as happened during this stage's own K.3 probe, where a
        forced write reused a name that collided with an existing report.

        When the model never attempted `write_report` at all, one extra
        model call asks for the report directly -- see
        `_request_report_markdown`.
        """
        if state.saved_report_path is not None:
            return state.saved_report_path, "tool"
        if state.successful_reads == 0 and state.last_report_markdown is None:
            return None, "none"

        markdown = state.last_report_markdown
        if markdown is None:
            markdown = self._request_report_markdown(set(state.read_urls))
        if markdown is None or not markdown.strip():
            return None, "none"

        result = write_report(
            filename=_build_report_filename(question), content=markdown
        )
        if result.startswith(REPORT_SAVED_PREFIX):
            state.saved_report_markdown = markdown
            return result.removeprefix(REPORT_SAVED_PREFIX).strip(), "fallback"
        return None, "none"

    def _request_report_markdown(self, read_urls: set[str]) -> str | None:
        """Ask the model for a final report with no tools available.

        Returns
        -------
        str or None
            The model's markdown, or ``None`` if it returned nothing.

        Notes
        -----
        Called only when the loop read at least one source but the model
        never attempted `write_report`. The request is transient: neither
        this prompt nor the response is appended to `self._messages`, so the
        next question does not open with "your budget is exhausted." Passing
        no `tools` is itself the guarantee that the model cannot go search
        instead of writing -- there is nothing left it could call.

        `parse()` is tried first: a structured `ResearchReport` means the
        renderer builds source numbering and anchors deterministically,
        instead of trusting the model to count correctly. A model or request
        that does not support strict structured output falls back to a plain
        `create()` call with free-form markdown.

        The two paths ask for different things on purpose. Only the
        structured one is followed by `render_report`, so only it can tell
        the model that citation markers are added for it; saying the same
        on the plain path would leave that report with no citations at all.
        """
        try:
            completion = self._client.chat.completions.parse(
                model=self._settings.model_name,
                messages=[
                    *self._messages,
                    {
                        "role": "user",
                        "content": FALLBACK_STRUCTURED_REPORT_REQUEST,
                    },
                ],
                response_format=ResearchReport,
            )
            report = completion.choices[0].message.parsed
            if report is not None:
                return render_report(report, read_urls=read_urls)
        except (BadRequestError, ValidationError):
            pass
        response = self._client.chat.completions.create(
            model=self._settings.model_name,
            messages=[
                *self._messages,
                {"role": "user", "content": FALLBACK_REPORT_REQUEST},
            ],
            temperature=self._settings.temperature,
        )
        return response.choices[0].message.content
