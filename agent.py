"""The ReAct loop: model call, tool calls, tool results, model call again.

The loop is a plain function over a list of messages. ``ResearchAgent`` only
keeps that list between turns, which is all the memory this agent has: no
checkpointer, no graph, no framework state.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCallUnion,
)

from config import (
    BUDGET_NUDGE_MESSAGE,
    ERROR_PREFIX,
    REPORT_SAVED_PREFIX,
    SYSTEM_PROMPT,
    Settings,
)
from tools import TOOL_REGISTRY, TOOL_SCHEMAS, write_report

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


class CompletionsProtocol(Protocol):
    """The single SDK call this agent makes.

    Notes
    -----
    The parameters are named but typed ``Any`` on purpose. Spelling out the
    SDK's own parameter types would make the real client fail this protocol
    check: its ``create`` is overloaded and expects TypedDict unions rather
    than the plain dicts this project builds.
    """

    def create(
        self,
        *,
        model: Any,
        messages: Any,
        tools: Any,
        tool_choice: Any,
        temperature: Any,
        parallel_tool_calls: Any,
    ) -> ChatCompletion: ...


class ChatProtocol(Protocol):
    @property
    def completions(self) -> CompletionsProtocol: ...


class LLMClient(Protocol):
    @property
    def chat(self) -> ChatProtocol: ...


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
    successful_reads: int = 0
    consecutive_tool_errors: int = 0
    # Keyed on the raw query string. Lives only as long as this RunState (one
    # run()); a repeat of the same phrase in a later question is not a cache
    # hit, since the web has had a chance to change by then.
    search_cache: dict[str, str] = field(default_factory=dict)
    last_report_markdown: str | None = None
    last_report_filename: str | None = None
    saved_report_path: str | None = None


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


def _assistant_message_to_dict(message: ChatCompletionMessage) -> dict[str, Any]:
    """Convert a model message into the plain dict the history stores.

    Parameters
    ----------
    message : ChatCompletionMessage
        Message returned by the model.

    Returns
    -------
    dict
        ``role`` and ``content``, plus ``tool_calls`` when the model asked for
        any.

    Notes
    -----
    Built field by field rather than with ``model_dump()``: the SDK object
    also carries ``annotations``, ``audio`` and the deprecated
    ``function_call``, which the API does not need back and which would make
    the history brittle to SDK changes. The same pass narrows the tool-call
    union to its function variant.
    """
    payload: dict[str, Any] = {"role": "assistant", "content": message.content}
    function_calls = [
        tool_call
        for tool_call in (message.tool_calls or [])
        if tool_call.type == "function"
    ]
    if function_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in function_calls
        ]
    return payload


def _record_failure(
    state: RunState, name: str, arguments: dict[str, Any], message: str
) -> ToolStep:
    """Build an ERROR step and count it toward consecutive tool failures."""
    state.consecutive_tool_errors += 1
    return ToolStep(
        name=name,
        arguments=arguments,
        result=f"{ERROR_PREFIX}{message}",
        ok=False,
    )


def _execute_tool_call(
    tool_call: ChatCompletionMessageToolCallUnion,
    state: RunState,
) -> ToolStep:
    """Run one tool call and describe the outcome as a step.

    Parameters
    ----------
    tool_call : ChatCompletionMessageToolCallUnion
        One entry of the model's ``tool_calls``.
    state : RunState
        Updated in place with what the call produced.

    Returns
    -------
    ToolStep
        Executed step, whose ``result`` is the string handed back to the
        model.

    Notes
    -----
    Every failure here is returned as a step, never raised: a broken tool
    call is data the model can act on and recover from, not a reason to
    crash the loop.

    A repeated `web_search` query is served from `state.search_cache` instead
    of hitting the backend again; only a successful search is ever cached, so
    a transient failure still gets a genuine retry.
    """
    state.tool_calls_made += 1

    if tool_call.type != "function":
        return _record_failure(
            state, tool_call.type, {}, "Only function tool calls are supported."
        )

    name = tool_call.function.name
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return _record_failure(state, name, {}, "Tool arguments are not valid JSON.")
    if not isinstance(arguments, dict):
        return _record_failure(state, name, {}, "Tool arguments must be a JSON object.")

    function = TOOL_REGISTRY.get(name)
    if function is None:
        return _record_failure(state, name, arguments, f"Unknown tool '{name}'.")

    if name == "web_search":
        query = arguments.get("query")
        if isinstance(query, str) and query in state.search_cache:
            state.consecutive_tool_errors = 0
            return ToolStep(
                name=name,
                arguments=arguments,
                result=state.search_cache[query],
                ok=True,
            )

    if name == "write_report":
        # Recorded before the call: when saving fails, the completion gate
        # still has the markdown the model produced.
        state.last_report_markdown = arguments.get("content")
        state.last_report_filename = arguments.get("filename")

    try:
        result = function(**arguments)
    except TypeError:
        return _record_failure(
            state, name, arguments, f"Invalid arguments for '{name}'."
        )

    # A tool result reaches the model as text, so anything that is not already
    # a string is serialized here. ensure_ascii=False keeps Cyrillic readable
    # and roughly three times cheaper in tokens than its escaped form.
    content = (
        result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    )
    ok = not content.startswith(ERROR_PREFIX)
    state.consecutive_tool_errors = 0 if ok else state.consecutive_tool_errors + 1

    if name == "web_search" and ok and isinstance(arguments.get("query"), str):
        state.search_cache[arguments["query"]] = content
    if name == "read_url" and ok:
        state.successful_reads += 1
    if name == "write_report" and content.startswith(REPORT_SAVED_PREFIX):
        state.saved_report_path = content.removeprefix(REPORT_SAVED_PREFIX).strip()

    return ToolStep(name=name, arguments=arguments, result=content, ok=ok)


def react_step(
    messages: Messages,
    client: LLMClient,
    settings: Settings,
    state: RunState,
    *,
    force_write_report: bool = False,
) -> StepResult:
    """Call the model once and run whatever tools it asked for.

    Parameters
    ----------
    messages : list of dict
        Conversation so far. Never mutated: the extended history comes back in
        the result.
    client : LLMClient
        Chat-completions client.
    settings : Settings
        Model name and temperature for the request.
    state : RunState
        Updated in place by the tool calls of this step.
    force_write_report : bool, optional
        Requested by the caller on the last iteration. Takes effect only when
        no report has been saved yet and at least one source was read: forcing
        it after a successful save would loop the model into writing forever,
        and forcing it with no evidence makes the model invent its sources.

    Returns
    -------
    StepResult
        Extended history, executed steps, and either a stop reason or ``None``
        when the turn should continue.

    Notes
    -----
    The assistant message is appended before any tool runs. The API rejects a
    request in which a ``tool_call_id`` has no assistant message announcing
    it, so this ordering is part of the protocol, not bookkeeping.
    """
    force = (
        force_write_report
        and state.saved_report_path is None
        # A forced write with nothing read produces a complete-looking report
        # whose sources no tool ever returned -- observed directly in the K.3
        # max_iterations=1 experiment, not hypothesised.
        and state.successful_reads > 0
    )
    # The nudge is a request-only addition: request_messages carries it to
    # the API, but `updated` (the persisted history) is built from the
    # original `messages` a few lines down, so the next question never sees
    # it.
    request_messages = [*messages, BUDGET_NUDGE_MESSAGE] if force else messages
    tool_choice: str | dict[str, Any] = (
        {"type": "function", "function": {"name": "write_report"}} if force else "auto"
    )

    response = client.chat.completions.create(
        model=settings.model_name,
        messages=request_messages,
        tools=TOOL_SCHEMAS,
        tool_choice=tool_choice,
        temperature=settings.temperature,
        parallel_tool_calls=True,
    )

    choice = response.choices[0]
    message = choice.message
    updated: Messages = [*messages, _assistant_message_to_dict(message)]

    if message.refusal:
        return StepResult(updated, [], message.refusal, "refusal")
    if choice.finish_reason == "length":
        # The response was cut mid-token, so any tool arguments in it are
        # almost certainly broken JSON. Stopping beats feeding the model back
        # its own truncated request.
        return StepResult(updated, [], message.content, "truncated")
    if not message.tool_calls:
        return StepResult(updated, [], message.content, "goal_satisfied")

    steps: list[ToolStep] = []
    for tool_call in message.tool_calls:
        step = _execute_tool_call(tool_call, state)
        steps.append(step)
        updated.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": step.result,
            }
        )
    if state.consecutive_tool_errors >= settings.max_consecutive_tool_errors:
        # [12-factor #9]: an unreachable backend would otherwise burn the
        # whole iteration budget on calls that were never going to succeed.
        return StepResult(updated, steps, None, "tool_failures")
    return StepResult(updated, steps, None, None)


def _build_report_filename(question: str) -> str:
    """Build a fallback report filename from the user's question.

    Parameters
    ----------
    question : str
        The user's question for this turn.

    Returns
    -------
    str
        A slug derived from the question, without a directory or extension.
        ``write_report`` sanitizes and timestamps it further, so this name
        only has to be descriptive, not unique or already safe.

    Examples
    --------
    >>> _build_report_filename("What is RAG?")
    'research_what_is_rag'
    """
    slug = re.sub(r"[^\w]+", "_", question.lower(), flags=re.UNICODE).strip("_")
    return f"research_{slug[:60] or 'report'}"


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
            client = OpenAI(api_key=settings.api_key.get_secret_value())
        self._client = client
        system_prompt = SYSTEM_PROMPT.format(max_iterations=settings.max_iterations)
        self._messages: Messages = [{"role": "system", "content": system_prompt}]

    @property
    def messages(self) -> Messages:
        """Conversation history of the whole session, oldest message first."""
        return self._messages

    def run(self, user_input: str) -> AgentResult:
        """Answer one question, calling tools until the model stops asking.

        Parameters
        ----------
        user_input : str
            The user's question.

        Returns
        -------
        AgentResult
            Final answer, executed steps and why the turn ended.
        """
        state = RunState()
        steps: list[ToolStep] = []
        pending: Messages = [*self._messages, {"role": "user", "content": user_input}]
        max_iterations = self._settings.max_iterations

        step = react_step(
            pending,
            self._client,
            self._settings,
            state,
            force_write_report=max_iterations == 1,
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
            )
            steps.extend(step.steps)

        # Assigned once, after the turn survived. An exception mid-turn would
        # otherwise leave a tool_call_id without its tool result, and every
        # later request would fail with HTTP 400.
        self._messages = step.messages
        return self._finish(step, steps, state, iteration, user_input)

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
        """
        if state.saved_report_path is not None:
            return state.saved_report_path, "tool"
        if state.successful_reads == 0 and state.last_report_markdown is None:
            return None, "none"

        markdown = state.last_report_markdown
        if markdown is None or not markdown.strip():
            return None, "none"

        result = write_report(
            filename=_build_report_filename(question), content=markdown
        )
        if result.startswith(REPORT_SAVED_PREFIX):
            return result.removeprefix(REPORT_SAVED_PREFIX).strip(), "fallback"
        return None, "none"
