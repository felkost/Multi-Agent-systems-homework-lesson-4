"""The ReAct step: one model call, then whatever tools it asked for.

A plain function over a list of messages -- ``ResearchAgent`` (agent.py)
owns the loop that calls this repeatedly and the session state around it;
nothing here keeps state of its own between calls.
"""

import json
from typing import Any, Callable

from openai.types.chat import ChatCompletionMessageToolCallUnion

from research_agent.history import _assistant_message_to_dict
from research_agent.llm import LLMClient
from research_agent.prompts.requests import BUDGET_NUDGE_MESSAGE
from research_agent.settings import Settings
from research_agent.state import Messages, RunState, StepResult, ToolStep
from research_agent.tools import TOOL_REGISTRY, TOOL_SCHEMAS
from research_agent.tools.contract import ERROR_PREFIX, REPORT_SAVED_PREFIX


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
        state.read_urls.append(str(arguments["url"]))
    if name == "write_report" and content.startswith(REPORT_SAVED_PREFIX):
        state.saved_report_path = content.removeprefix(REPORT_SAVED_PREFIX).strip()
        state.saved_report_markdown = state.last_report_markdown

    return ToolStep(name=name, arguments=arguments, result=content, ok=ok)


def react_step(
    messages: Messages,
    client: LLMClient,
    settings: Settings,
    state: RunState,
    *,
    force_write_report: bool = False,
    on_step: Callable[[ToolStep], None] | None = None,
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
    on_step : callable, optional
        Called once per executed tool call, immediately after it finishes --
        the hook the CLI uses to print a step while the turn is still running,
        instead of waiting for the whole turn to end.

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
        if on_step is not None:
            on_step(step)
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
