from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from openai import OpenAIError
from pydantic import ValidationError

from agent import create_research_agent
from config import load_settings


def main() -> None:
    """Run the interactive research REPL.

    Reads questions from the terminal until the user types ``exit`` or sends
    EOF, printing the name of each tool the agent calls and, once the agent
    stops calling tools, the final answer.
    """
    print("Research Agent (type 'exit' to quit)")
    print("-" * 40)

    try:
        settings = load_settings()
        agent = create_research_agent(settings)
    except ValidationError:
        print("Configuration error: check OPENAI_API_KEY and values in .env.")
        return

    thread_id = str(uuid4())
    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": settings.recursion_limit,
    }

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        final_answer: str | None = None
        try:
            # Keep the last answer instead of printing every AIMessage as it
            # arrives: a streamed run emits several, and printing each one
            # would show intermediate text as if it were the answer.
            for chunk in agent.stream(
                {"messages": [("user", user_input)]},
                config=config,
                stream_mode="updates",
            ):
                tools_update = chunk.get("tools")
                if isinstance(tools_update, dict):
                    for message in tools_update.get("messages", []):
                        if isinstance(message, ToolMessage):
                            tool_name = message.name or "unknown_tool"
                            print(f"\nTool: {tool_name}")

                model_update = chunk.get("model")
                if isinstance(model_update, dict):
                    for message in model_update.get("messages", []):
                        if (
                            isinstance(message, AIMessage)
                            and not message.tool_calls
                            and isinstance(message.content, str)
                            and message.content.strip()
                        ):
                            final_answer = message.content

        except OpenAIError:
            # One failed API request must not end the session: the user may
            # want to retry the same question or ask a different one.
            print(
                "\nAgent error: OpenAI API request failed. "
                "Check the API key and connection."
            )
            continue
        except GraphRecursionError:
            print(
                "\nAgent error: graph recursion limit reached. "
                "Increase recursion_limit in .env or config.py, "
                "and ensure the agent has a proper stop condition."
            )
            continue
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if final_answer is not None:
            print(f"\nAgent: {final_answer}")


if __name__ == "__main__":
    main()
