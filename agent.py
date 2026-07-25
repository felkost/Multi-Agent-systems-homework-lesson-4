from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from config import SYSTEM_PROMPT, Settings
from tools import TOOL_REGISTRY


def create_research_agent(
    settings: Settings, model: BaseChatModel | None = None
) -> CompiledStateGraph:
    """Create the configured research agent graph.

    Parameters
    ----------
    settings : Settings
        Model name, temperature and tool-call budget for the run.
    model : BaseChatModel, optional
        Chat model to use instead of building a `ChatOpenAI` from `settings`.
        Tests pass a scripted model here, which is what lets the suite run
        without an API key.

    Returns
    -------
    CompiledStateGraph
        Graph that alternates model calls and tool calls until the model
        answers without requesting another tool.
    """
    research_model: BaseChatModel
    if model is None:
        research_model = ChatOpenAI(
            model=settings.model_name,
            api_key=settings.api_key,
            temperature=settings.temperature,
        )
    else:
        research_model = model
    tools = list(TOOL_REGISTRY.values())
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
