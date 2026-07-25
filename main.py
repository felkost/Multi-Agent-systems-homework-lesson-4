from openai import APIError, OpenAIError
from pydantic import ValidationError

from agent import ResearchAgent
from config import load_settings


def main() -> None:
    """Run the interactive research REPL.

    Reads questions from the terminal until the user types ``exit`` or sends
    EOF, printing the name of each tool the agent calls and then the final
    answer. One `ResearchAgent` serves the whole session, so its message list
    is the session's memory.
    """
    print("Research Agent (type 'exit' to quit)")
    print("-" * 40)

    try:
        settings = load_settings()
    except ValidationError:
        print("Configuration error: check OPENAI_API_KEY and values in .env.")
        return

    agent = ResearchAgent(settings)

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

        try:
            result = agent.run(user_input)
        # A failed request kills the turn, not the session: the user may want
        # to retry the same question or ask a different one. APIError comes
        # first because it is the more specific of the two.
        except APIError:
            print(
                "\nAgent error: OpenAI API request failed. "
                "Check the API key and connection."
            )
            continue
        except OpenAIError:
            print("\nAgent error: the OpenAI client could not complete the request.")
            continue
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        for step in result.steps:
            print(f"\nTool: {step.name}")

        if result.final_answer is not None:
            print(f"\nAgent: {result.final_answer}")


if __name__ == "__main__":
    main()
