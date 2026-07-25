"""Session memory: one list of dicts on the agent, no checkpointer."""

from agent import ResearchAgent
from config import Settings

from fakes import ScriptedChatClient, ScriptedTurn


def test_second_question_sees_the_first_exchange(
    configured_settings: Settings,
) -> None:
    client = ScriptedChatClient(
        [
            ScriptedTurn(content="First answer."),
            ScriptedTurn(content="Second answer."),
        ]
    )
    agent = ResearchAgent(configured_settings, client=client)

    agent.run("First question.")
    agent.run("Second question.")

    sent = client.requests[1]["messages"]

    assert [message["role"] for message in sent] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert sent[1]["content"] == "First question."
    assert sent[2]["content"] == "First answer."


def test_separate_agents_do_not_share_history(
    configured_settings: Settings,
) -> None:
    first_client = ScriptedChatClient([ScriptedTurn(content="First answer.")])
    second_client = ScriptedChatClient([ScriptedTurn(content="Second answer.")])

    ResearchAgent(configured_settings, client=first_client).run("First question.")
    second_agent = ResearchAgent(configured_settings, client=second_client)
    second_agent.run("Second question.")

    sent = second_client.requests[0]["messages"]

    assert [message["role"] for message in sent] == ["system", "user"]
    assert sent[1]["content"] == "Second question."
