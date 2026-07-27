"""The one SDK call this agent makes, named as a Protocol so a scripted
fake client can stand in for it in tests without inheriting from anything.
"""

from typing import Any, Protocol

from langsmith.wrappers import wrap_openai
from openai import OpenAI
from openai.types.chat import ChatCompletion

from research_agent.settings import Settings
from research_agent.tracing import configure_tracing


class CompletionsProtocol(Protocol):
    """The single SDK call this agent makes.

    Notes
    -----
    The parameters are named but typed ``Any`` on purpose. Spelling out the
    SDK's own parameter types would make the real client fail this protocol
    check: its ``create`` is overloaded and expects TypedDict unions rather
    than the plain dicts this project builds. ``tools``, ``tool_choice`` and
    ``parallel_tool_calls`` default to ``None``: the completion gate's
    fallback call omits all three on purpose, since a request with no tools
    is the guarantee that the model cannot go search instead of writing.
    """

    def create(
        self,
        *,
        model: Any,
        messages: Any,
        temperature: Any,
        tools: Any = None,
        tool_choice: Any = None,
        parallel_tool_calls: Any = None,
    ) -> ChatCompletion: ...

    def parse(
        self,
        *,
        model: Any,
        messages: Any,
        response_format: Any,
    ) -> Any: ...


class ChatProtocol(Protocol):
    @property
    def completions(self) -> CompletionsProtocol: ...


class LLMClient(Protocol):
    @property
    def chat(self) -> ChatProtocol: ...


def build_client(settings: Settings) -> LLMClient:
    """Build the chat client, traced when tracing is configured.

    Parameters
    ----------
    settings : Settings
        Model credentials and tracing configuration.

    Returns
    -------
    LLMClient
        A plain `OpenAI` client, or the same client wrapped so that every
        completion becomes an LLM span with its own token counts.

    Notes
    -----
    `wrap_openai` patches the client's `create` and `parse` methods, so the
    ReAct loop keeps calling exactly what it called before -- tracing costs
    this project no change to the loop itself.
    """
    client = OpenAI(api_key=settings.api_key.get_secret_value())
    if not configure_tracing(settings):
        return client
    return wrap_openai(client)
