import json
import os
import typing as t

import aiofiles
import aiohttp
from datasets import Dataset
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

from bespokelabs.curator.agent.agent_response import AgentResponse
from bespokelabs.curator.request_processor.online.base_online_request_processor import APIRequest
from bespokelabs.curator.status_tracker.agent_status_tracker import AgentStatusTracker, AgentTurn
from bespokelabs.curator.types.generic_response import GenericResponse

if t.TYPE_CHECKING:
    from bespokelabs.curator.agent.agent import Agent


class MultiTurnAgenticProcessor:
    """A processor that manages multi-turn conversations between two agents.

    This class handles the orchestration of conversations between a seeder agent
    and a partner agent, managing the conversation history, caching, and dataset
    creation. It supports both new conversations and resuming from cached states.

    Attributes:
        seeder (Agent): The agent that initiates the conversation.
        partner (Agent): The agent that responds to the seeder.
        max_length (int): The maximum number of turns in the conversation.
        seed_message (str): The initial message to start the conversation.
        conversation_history (list): List of message dictionaries containing role and content.
        status_tracker (AgentStatusTracker): Tracks the status of the conversation.
    """

    def __init__(self, seeder: "Agent", partner: "Agent", max_length: int, seed_message: str):
        """Initialize a MultiTurnAgenticProcessor instance.

        Args:
            seeder (Agent): The agent that initiates the conversation.
            partner (Agent): The agent that responds to the seeder.
            max_length (int): The maximum number of turns in the conversation.
            seed_message (str): The initial message to start the conversation.
        """
        self.seeder = seeder
        self.partner = partner
        self.max_length = max_length
        self.seed_message = seed_message
        self.conversation_history = []
        self.status_tracker = AgentStatusTracker(seeder_name=self.seeder.name, partner_name=self.partner.name, max_turns=self.max_length)

    def load_cache(self, working_dir: str) -> int:
        """Load cached conversation history from a JSONL file.

        Args:
            working_dir (str): Directory containing the cached conversation file.

        Returns:
            int: The number of messages loaded from the cache.
        """
        cache_file = os.path.join(working_dir, "responses_0.jsonl")
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                for line in f:
                    response = AgentResponse.model_validate_json(line)
                    self.conversation_history.append({"role": response.name, "content": response.response_message})
                    # Update status tracker for cached responses
                    if response.name == self.seeder.name:
                        self.status_tracker.update_turn(AgentTurn.SEEDER)
                    else:
                        self.status_tracker.update_turn(AgentTurn.PARTNER)
        return len(self.conversation_history)

    async def run(self, working_dir: str) -> Dataset:
        """Execute the multi-turn conversation between agents.

        This method manages the conversation flow between the seeder and partner agents,
        handling caching, response generation, and dataset creation.

        Args:
            working_dir (str): Directory where conversation results will be stored.

        Returns:
            Dataset: A HuggingFace Dataset containing the conversation history.
        """
        request_file = os.path.join(working_dir, "responses_0.jsonl")
        if os.path.exists(request_file):
            start_step = self.load_cache(working_dir)
            if start_step == self.max_length:
                self.status_tracker.stop_tracker()
                return Dataset.from_file(self.create_dataset_file(working_dir))
        else:
            self.conversation_history.append({"role": self.seeder.name, "content": self.seed_message})
            start_step = 0

        async with aiohttp.ClientSession() as session:
            async with aiofiles.open(request_file, "a") as f:
                for step in range(start_step, self.max_length):
                    try:
                        if step % 2 == 0:
                            partener_request = self._transform_conversation_history(self.partner)
                            partener_request = APIRequest(
                                task_id=step,
                                generic_request=partener_request,
                                api_specific_request=self.partner._request_processor.create_api_specific_request_online(partener_request),
                                attempts_left=1,
                                prompt_formatter=self.partner.prompt_formatter,
                            )
                            partener_response = await self.partner._request_processor.call_single_request(partener_request, session, status_tracker=None)
                            await self.append_response(self.partner.name, f, partener_response)
                            self.conversation_history.append({"role": self.partner.name, "content": partener_response.response_message})
                            self.status_tracker.update_turn(AgentTurn.PARTNER, token_usage=partener_response.token_usage, cost=partener_response.response_cost)
                        else:
                            seeder_request = self._transform_conversation_history(self.seeder)
                            seeder_request = APIRequest(
                                task_id=step,
                                generic_request=seeder_request,
                                api_specific_request=self.seeder._request_processor.create_api_specific_request_online(seeder_request),
                                attempts_left=1,
                                prompt_formatter=self.seeder.prompt_formatter,
                            )
                            seeder_response = await self.seeder._request_processor.call_single_request(seeder_request, session, status_tracker=None)
                            await self.append_response(self.seeder.name, f, seeder_response)
                            self.conversation_history.append({"role": self.seeder.name, "content": seeder_response.response_message})
                            self.status_tracker.update_turn(AgentTurn.SEEDER, token_usage=seeder_response.token_usage, cost=seeder_response.response_cost)
                    except Exception as e:
                        # Update status tracker with error
                        if step % 2 == 0:
                            self.status_tracker.update_turn(AgentTurn.PARTNER, response_success=False)
                        else:
                            self.status_tracker.update_turn(AgentTurn.SEEDER, response_success=False)
                        raise e

        self.status_tracker.stop_tracker()
        return Dataset.from_file(self.create_dataset_file(working_dir))

    async def append_response(self, name: str, f, response: GenericResponse) -> None:
        """Append a response to the conversation history file.

        Args:
            name (str): The name of the agent generating the response.
            f: The file object to write to.
            response (GenericResponse): The response to append.
        """
        response = response.model_dump()
        response["name"] = name
        await f.write(json.dumps(response, default=str) + "\n")

    def _transform_conversation_history(self, target_agent: "Agent"):
        """Transform the conversation history into a format suitable for the target agent.

        This method converts the conversation history into a format that the target agent
        can process, including system prompts and proper role assignments.

        Args:
            target_agent (Agent): The agent for whom the conversation history is being transformed.

        Returns:
            APIRequest: A request object containing the transformed conversation history.
        """
        transformed_conversation_history = []
        if len(self.conversation_history) == 1:
            transformed_conversation_history.append({"role": "user", "content": self.conversation_history[0]["content"]})
        else:
            for message in self.conversation_history:
                if message["role"] == target_agent.name:
                    transformed_conversation_history.append({"role": "assistant", "content": message["content"]})
                else:
                    transformed_conversation_history.append({"role": "user", "content": message["content"]})

        request = target_agent.prompt_formatter.create_generic_request({"prompt": transformed_conversation_history[-1]["content"]}, 0)
        system_prompt = [msg for msg in request.messages if msg["role"] == "system"][0]
        request.messages = [msg for msg in request.messages if msg["role"] != "system"]

        del transformed_conversation_history[-1]
        transformed_conversation_history += request.messages
        request.messages = transformed_conversation_history
        request.messages.insert(0, system_prompt)
        return request

    def create_dataset_file(self, working_dir: str) -> str:
        """Create a dataset file from the conversation history.

        This method converts the JSONL conversation history into an Arrow dataset file
        for efficient storage and processing.

        Args:
            working_dir (str): Directory where the dataset file will be created.

        Returns:
            str: Path to the created dataset file.
        """
        response_file = os.path.join(working_dir, "responses_0.jsonl")
        dataset_file = os.path.join(working_dir, "dataset.arrow")

        with ArrowWriter(path=dataset_file) as writer:
            with open(response_file, "r") as f_in:
                for line in f_in:
                    response = AgentResponse.model_validate_json(line)
                    row = response.model_dump()
                    response = {"response": row["response_message"], "name": row["name"]}
                    # Write the row to the arrow file
                    writer.write(response)

            # Finalize the writer
            writer.finalize()

        return dataset_file
