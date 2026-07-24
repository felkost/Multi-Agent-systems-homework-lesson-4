from uuid import uuid4  # створює унікальний thread_id

# повідомлення моделі, результат виконання tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig  # тип конфігурації LangGraph
from langgraph.errors import GraphRecursionError
from openai import OpenAIError  # базовий виняток OpenAI API
from pydantic import ValidationError  # помилки Settings

from agent import create_research_agent  # фабрика
from config import load_settings  # фабрика конфігурації


def main() -> None:
    print("Research Agent (type 'exit' to quit)")
    print("-" * 40)

    try:
        settings = load_settings()
        agent = create_research_agent(settings)
    except ValidationError:
        print("Configuration error: check OPENAI_API_KEY " "and values in .env.")
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
            # Чому фінальна відповідь не дублюється
            # Під час streaming можуть надходити різні AIMessage.
            # Якщо друкувати кожне одразу,
            # проміжний текст можна показати кілька разів
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
            # Мета: не завершувати CLI через помилку одного API-запиту та
            # дозволити користувачу перервати виконання.
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
