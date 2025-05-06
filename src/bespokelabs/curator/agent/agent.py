import os
import typing as t

from xxhash import xxh64

from bespokelabs import curator
from bespokelabs.curator.agent.processor import MultiTurnAgenticProcessor
from bespokelabs.curator.llm.llm import _CURATOR_DEFAULT_CACHE_DIR
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop


class Agent(curator.LLM):
    """A specialized LLM agent that can participate in multi-turn conversations.

    This class extends the base LLM class to provide agent-specific functionality
    and identity through a name attribute.

    Attributes:
        name (str): The unique identifier for this agent.
    """

    def __init__(self, name: str, *args, **kwargs):
        """Initialize an Agent instance.

        Args:
            name (str): The unique identifier for this agent.
            *args: Additional positional arguments passed to the parent LLM class.
            **kwargs: Additional keyword arguments passed to the parent LLM class.
        """
        self.name = name
        super().__init__(*args, **kwargs)

    def __str__(self):
        """Return a string representation of the agent.

        Returns:
            str: A string in the format "Agent(name={name})".
        """
        return f"Agent(name={self.name})"

    def __repr__(self):
        """Return the official string representation of the agent.

        Returns:
            str: The same string representation as __str__.
        """
        return self.__str__()


class MultiTurnAgents:
    """A class that manages a conversation between two agents.

    This class orchestrates a multi-turn conversation between a seeder agent
    and a partner agent, with a specified maximum conversation length and
    initial seed message.

    Attributes:
        seeder (Agent): The agent that initiates the conversation.
        partner (Agent): The agent that responds to the seeder.
        max_length (int): The maximum number of turns in the conversation.
        seed_message (str): The initial message to start the conversation.
    """

    def __init__(self, seeder: Agent, partner: Agent, max_length: int, seed_message: str):
        """Initialize a MultiTurnAgents instance.

        Args:
            seeder (Agent): The agent that initiates the conversation.
            partner (Agent): The agent that responds to the seeder.
            max_length (int): The maximum number of turns in the conversation.
            seed_message (str): The initial message to start the conversation.
                                Note: This message is send to partner agent.

        Raises:
            AssertionError: If seeder and partner have the same name.
        """
        self.seeder = seeder
        self.partner = partner
        self.max_length = max_length
        self.seed_message = seed_message

        assert self.seeder.name != self.partner.name, "Seeder and partner must have different names"
        self._processor = MultiTurnAgenticProcessor(self.seeder, self.partner, self.max_length, self.seed_message)

    def __call__(self, working_dir: t.Optional[str] = None):
        """Execute the multi-turn conversation between the agents.

        This method runs the conversation simulation and saves the results
        in the specified working directory. If no working directory is provided,
        it uses the CURATOR_CACHE_DIR environment variable or a default location.

        Args:
            working_dir (Optional[str]): The directory where conversation results
                will be stored. If None, uses environment variable or default.

        Returns:
            Any: The result of the conversation simulation.
        """
        if working_dir is None:
            working_dir = os.environ.get(
                "CURATOR_CACHE_DIR",
                os.path.expanduser(_CURATOR_DEFAULT_CACHE_DIR),
            )
        disable_cache = os.getenv("CURATOR_DISABLE_CACHE", "").lower() in ["true", "1"]
        fingerprint = self.seeder._hash_fingerprint(disable_cache=disable_cache)
        fingerprint += xxh64(self.seed_message).hexdigest()
        fingerprint += self.partner._hash_fingerprint(disable_cache=disable_cache)
        working_dir = os.path.join(working_dir, fingerprint)
        os.makedirs(working_dir, exist_ok=True)
        print(f"Running multi turn simulation, find results in {working_dir}")
        return run_in_event_loop(self._processor.run(working_dir=working_dir))
