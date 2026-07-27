"""Turning a model message into history, and shrinking stale history back
down.
"""

import json
from typing import Any

from openai.types.chat import ChatCompletionMessage

from research_agent.state import Messages
from research_agent.tools.contract import ERROR_PREFIX, REPORT_SAVED_PREFIX


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


def _summarize_tool_result(name: str, arguments: dict[str, Any], content: str) -> str:
    """Describe a tool result in one line, without its payload."""
    if content.startswith(ERROR_PREFIX):
        # Already one short line, and the model still needs to know that this
        # source failed: shrinking it would drop the reason, not the bulk.
        return content
    if name == "web_search":
        try:
            count = len(json.loads(content))
        except (json.JSONDecodeError, TypeError):
            count = 0
        query = arguments.get("query", "")
        return f'[web_search: {count} results for "{query}", payload dropped]'
    if name == "read_url":
        url = arguments.get("url", "")
        return f"[read_url: {url} ({len(content)} chars), payload dropped]"
    if name == "write_report":
        return f"[write_report: saved to {content.removeprefix(REPORT_SAVED_PREFIX)}]"
    return f"[{name}: {len(content)} chars, payload dropped]"


def compact_history(messages: Messages, keep_recent: int) -> Messages:
    """Replace the payloads of older tool results with one-line summaries.

    Parameters
    ----------
    messages : list of dict
        Conversation so far. Not mutated.
    keep_recent : int
        How many of the most recent tool results keep their full payload.

    Returns
    -------
    list of dict
        New list of the same shape: same messages in the same order, with only
        the ``content`` of stale tool results replaced.

    Notes
    -----
    Deleting a stale message instead would leave an assistant ``tool_call``
    without its answer, and the next request would fail with HTTP 400
    (invariant A.2). What survives compaction is the fact of the call, its
    arguments and its outcome, so the agent still knows what it has already
    searched and read; only the raw page text is dropped.

    Examples
    --------
    >>> messages = [
    ...     {
    ...         "role": "assistant",
    ...         "content": None,
    ...         "tool_calls": [
    ...             {
    ...                 "id": "call_1",
    ...                 "type": "function",
    ...                 "function": {
    ...                     "name": "read_url",
    ...                     "arguments": '{"url": "https://example.com"}',
    ...                 },
    ...             }
    ...         ],
    ...     },
    ...     {"role": "tool", "tool_call_id": "call_1", "content": "Long page text"},
    ... ]
    >>> compact_history(messages, keep_recent=0)[1]["content"]
    '[read_url: https://example.com (14 chars), payload dropped]'
    """
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            function = tool_call["function"]
            try:
                arguments = json.loads(function["arguments"])
            except json.JSONDecodeError:
                arguments = {}
            calls[tool_call["id"]] = (
                function["name"],
                arguments if isinstance(arguments, dict) else {},
            )

    results = [
        position
        for position, message in enumerate(messages)
        if message.get("role") == "tool"
    ]
    # Sliced by count rather than [:-keep_recent], which silently keeps
    # everything when keep_recent is 0.
    stale = set(results[: len(results) - keep_recent])

    compacted: Messages = []
    for position, message in enumerate(messages):
        if position not in stale:
            compacted.append(message)
            continue
        name, arguments = calls.get(message.get("tool_call_id", ""), ("tool", {}))
        compacted.append(
            {
                **message,
                "content": _summarize_tool_result(name, arguments, message["content"]),
            }
        )
    return compacted
