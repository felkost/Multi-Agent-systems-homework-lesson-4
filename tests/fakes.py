"""Test doubles for the ReAct loop.

The fake client returns real ``ChatCompletion`` objects built by
``model_validate``: a ``Mock`` would accept any attribute and hide a
malformed message, while validation fails on the spot when the shape drifts
from the SDK's.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Self

from openai.types.chat import ChatCompletion


@dataclass
class ScriptedTurn:
    """One planned model response.

    Attributes
    ----------
    tool_calls : list of (str, Any)
        Tool name and arguments. A ``dict`` is serialized to JSON the way the
        API does; a ``str`` is passed through verbatim, which is how a test
        scripts malformed arguments.
    content : str, optional
        Assistant text of the response.
    refusal : str, optional
        Refusal text, when the turn should model a refused request.
    finish_reason : str, optional
        Overrides the reason derived from the presence of tool calls.
    raw_tool_calls : list of dict, optional
        Fully-formed tool-call payloads, for shapes ``tool_calls`` cannot
        express, such as a custom (non-function) tool call.
    """

    tool_calls: list[tuple[str, Any]] = field(default_factory=list)
    content: str | None = None
    refusal: str | None = None
    finish_reason: str | None = None
    raw_tool_calls: list[dict[str, Any]] | None = None


def _tool_call_payloads(turn: ScriptedTurn, turn_number: int) -> list[dict[str, Any]]:
    if turn.raw_tool_calls is not None:
        return turn.raw_tool_calls
    return [
        {
            "id": f"call_{turn_number}_{position}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": (
                    arguments
                    if isinstance(arguments, str)
                    else json.dumps(arguments, ensure_ascii=False)
                ),
            },
        }
        for position, (name, arguments) in enumerate(turn.tool_calls)
    ]


def build_completion(turn: ScriptedTurn, turn_number: int) -> ChatCompletion:
    """Build the ``ChatCompletion`` a scripted turn stands for.

    Parameters
    ----------
    turn : ScriptedTurn
        The planned response.
    turn_number : int
        Position in the script, used to make tool-call ids unique.

    Returns
    -------
    ChatCompletion
        Validated response object, identical in shape to a real one.
    """
    tool_calls = _tool_call_payloads(turn, turn_number)
    message: dict[str, Any] = {
        "role": "assistant",
        "content": turn.content,
        "refusal": turn.refusal,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    return ChatCompletion.model_validate(
        {
            "id": f"chatcmpl-test-{turn_number}",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": turn.finish_reason
                    or ("tool_calls" if tool_calls else "stop"),
                }
            ],
        }
    )


class ScriptedChatClient:
    """Stand-in for ``openai.OpenAI`` driven by a fixed script.

    Parameters
    ----------
    script : list of ScriptedTurn
        Responses to return, in order.

    Attributes
    ----------
    requests : list of dict
        Every request payload the loop sent, so tests can assert on the
        message history and on request parameters such as ``tool_choice``.
    """

    def __init__(self, script: list[ScriptedTurn]) -> None:
        self._script = script
        self._index = 0
        self.requests: list[dict[str, Any]] = []

    @property
    def chat(self) -> Self:
        return self

    @property
    def completions(self) -> Self:
        return self

    def create(self, **kwargs: Any) -> ChatCompletion:
        """Return the next scripted response and record the request."""
        self.requests.append(kwargs)
        if self._index >= len(self._script):
            raise AssertionError(
                "script exhausted: the loop asked for one model call too many"
            )
        turn = self._script[self._index]
        self._index += 1
        return build_completion(turn, self._index)
