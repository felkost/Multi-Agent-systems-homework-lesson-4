"""The ReAct loop: model call, tool calls, tool results, model call again.

The loop is a plain function over a list of messages. ``ResearchAgent`` only
keeps that list between turns, which is all the memory this agent has: no
checkpointer, no graph, no framework state.
"""

import json
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCallUnion,
)

from config import ERROR_PREFIX, REPORT_SAVED_PREFIX, SYSTEM_PROMPT, Settings
from tools import TOOL_REGISTRY, TOOL_SCHEMAS

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
    """
    if tool_call.type != "function":
        return ToolStep(
            name=tool_call.type,
            arguments={},
            result=f"{ERROR_PREFIX}Only function tool calls are supported.",
            ok=False,
        )

    name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    function = TOOL_REGISTRY[name]

    state.tool_calls_made += 1
    if name == "write_report":
        # Recorded before the call: when saving fails, the completion gate
        # still has the markdown the model produced.
        state.last_report_markdown = arguments.get("content")
        state.last_report_filename = arguments.get("filename")

    result = function(**arguments)
    # A tool result reaches the model as text, so anything that is not already
    # a string is serialized here. ensure_ascii=False keeps Cyrillic readable
    # and roughly three times cheaper in tokens than its escaped form.
    content = (
        result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    )
    ok = not content.startswith(ERROR_PREFIX)

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
    response = client.chat.completions.create(
        model=settings.model_name,
        messages=messages,
        tools=TOOL_SCHEMAS,
        tool_choice="auto",
        temperature=settings.temperature,
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
    return StepResult(updated, steps, None, None)


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
        self._messages: Messages = [{"role": "system", "content": SYSTEM_PROMPT}]

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

        step = react_step(pending, self._client, self._settings, state)
        steps.extend(step.steps)
        iterations_used = 1
        if step.stop_reason is None:
            step = react_step(step.messages, self._client, self._settings, state)
            steps.extend(step.steps)
            iterations_used = 2

        # Assigned once, after the turn survived. An exception mid-turn would
        # otherwise leave a tool_call_id without its tool result, and every
        # later request would fail with HTTP 400.
        self._messages = step.messages
        return self._finish(step, steps, state, iterations_used)

    def _finish(
        self,
        step: StepResult,
        steps: list[ToolStep],
        state: RunState,
        iterations_used: int,
    ) -> AgentResult:
        """Assemble the result of a finished turn."""
        stop_reason: StopReason = step.stop_reason or "iteration_limit"
        return AgentResult(
            final_answer=step.final_answer,
            steps=steps,
            iterations_used=iterations_used,
            budget_exhausted=step.stop_reason is None,
            stop_reason=stop_reason,
            saved_report_path=state.saved_report_path,
            report_source="tool" if state.saved_report_path is not None else "none",
        )
