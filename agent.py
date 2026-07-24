from langchain.agents import create_agent  # створює single-agent ReAct loop
from langchain.agents.middleware import (
    ToolCallLimitMiddleware,
)  # обмежує tool calls одного запуску
from langchain_core.language_models import (
    BaseChatModel,
)  # спільний тип реальної та fake-моделі
from langchain_openai import ChatOpenAI  # модель за замовчуванням
from langgraph.checkpoint.memory import InMemorySaver  # пам’ять розмови
from langgraph.graph.state import CompiledStateGraph  # тип результату фабрики

from config import SYSTEM_PROMPT, Settings  # стратегія дослідження, готова конфігурація
from tools import read_url, web_search, write_report  # три tools


# create_agent повертає скомпільований граф,
# який викликає модель і tools у циклі до завершення
# Bідповідальності розділенi:
# config.py читає й перевіряє конфігурацію;
# agent.py створює агента;
# main.py керує запуском.
# Є два режими: create_research_agent(settings) cтворює справжній ChatOpenAI
# другий - використовує передану модель без створення ChatOpenAI
# create_research_agent(settings, model=fake_model)
def create_research_agent(
    settings: Settings, model: BaseChatModel | None = None
) -> CompiledStateGraph:
    """Create the configured research agent graph."""
    research_model: BaseChatModel
    if model is None:
        research_model = ChatOpenAI(
            model=settings.model_name,
            api_key=settings.api_key,
            temperature=settings.temperature,
        )
    else:
        research_model = model
    tools = [
        web_search,
        read_url,
        write_report,
    ]
    checkpointer = InMemorySaver()
    middleware = [
        ToolCallLimitMiddleware(
            run_limit=settings.max_tool_calls,
            exit_behavior="continue",
        )
    ]
    return create_agent(
        model=research_model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        middleware=middleware,
    )
