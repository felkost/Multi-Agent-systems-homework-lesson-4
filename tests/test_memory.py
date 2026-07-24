from collections.abc import Sequence
from typing import Any, Self

import pytest
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig

from agent import create_research_agent
from config import Settings


class ContextEchoChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "context-echo-test-model"

    def bind_tools(
        self,
        tools: Sequence[Any],
        **kwargs: Any,
    ) -> Self:
        del tools, kwargs
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs

        human_messages = [
            message.content
            for message in messages
            if isinstance(message, HumanMessage) and isinstance(message.content, str)
        ]
        response = " | ".join(human_messages)

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=response),
                )
            ]
        )


@pytest.fixture
def settings(
    monkeypatch: pytest.MonkeyPatch,
) -> Settings:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return Settings(_env_file=None)  # type: ignore[call-arg]


def _thread_config(thread_id: str) -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


def _last_ai_text(result: dict[str, Any]) -> str:
    message = result["messages"][-1]

    assert isinstance(message, AIMessage)
    assert isinstance(message.content, str)

    return message.content


def test_same_thread_id_preserves_context(
    settings: Settings,
) -> None:
    model = ContextEchoChatModel()
    agent = create_research_agent(
        settings,
        model=model,
    )
    config = _thread_config("thread-one")

    agent.invoke(
        {"messages": [("user", "first message")]},
        config=config,
    )
    result = agent.invoke(
        {"messages": [("user", "second message")]},
        config=config,
    )

    assert _last_ai_text(result) == ("first message | second message")


def test_different_thread_ids_are_isolated(
    settings: Settings,
) -> None:
    model = ContextEchoChatModel()
    agent = create_research_agent(
        settings,
        model=model,
    )

    agent.invoke(
        {"messages": [("user", "thread A message")]},
        config=_thread_config("thread-a"),
    )
    result = agent.invoke(
        {"messages": [("user", "thread B message")]},
        config=_thread_config("thread-b"),
    )

    assert _last_ai_text(result) == "thread B message"
