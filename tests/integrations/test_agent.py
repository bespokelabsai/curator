import hashlib
import os

os.environ["CURATOR_DISABLE_RICH_DISPLAY"] = "1"
import pytest

from bespokelabs.curator.agent.agent import Agent, MultiTurnAgents


class Client(Agent):
    def prompt(self, text: str):
        text = text["prompt"]
        return [
            {
                "role": "user",
                "content": text,
            },
        ]


class Advisor(Client): ...


@pytest.mark.parametrize("temp_working_dir", ([{"integration": "openai"}]), indirect=True)
def test_basic_conversation(temp_working_dir):
    temp_working_dir, backend, vcr_config = temp_working_dir

    # Create client and advisor agents
    client = Client(
        name="client",
        model_name="gpt-4o-mini",
        backend=backend,
        system_prompt="You are a client that asks questions to the advisor.",
    )
    advisor = Advisor(
        name="advisor",
        model_name="gpt-4o-mini",
        backend=backend,
        system_prompt="You are a helpful advisor that can answer questions and help with tasks.",
    )

    # Create a seed message to start the conversation
    seed_message = "I need help with my investment strategy. What should I do?"

    # Create the multi-turn conversation simulator
    simulator = MultiTurnAgents(
        seeder=client,
        partner=advisor,
        max_length=4,  # 2 turns each
        seed_message=seed_message,
    )

    with vcr_config.use_cassette("basic_agent_conversation.yaml"):
        result = simulator(working_dir=temp_working_dir)

        assert len(result) == 4
        df = result.to_pandas()
        convo = "".join([recipe[0] for recipe in df.values.tolist()])
        hash_convo = hashlib.sha256(convo.encode("utf-8")).hexdigest()
        assert hash_convo == "072b0624b1c4f74ad2d2da9952ca14580edbc150a569f59e4d5c5b451568dd27"
        assert "response" in df.columns
        assert "name" in df.columns
