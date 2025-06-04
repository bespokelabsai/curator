import os
import typing as t
from datetime import datetime

from xxhash import xxh64

from bespokelabs import curator
from bespokelabs.curator.agent.agent_response import MultiTurnResponse
from bespokelabs.curator.agent.processor import MultiTurnAgenticProcessor
from bespokelabs.curator.client import Client
from bespokelabs.curator.constants import PUBLIC_CURATOR_VIEWER_HOME_URL
from bespokelabs.curator.db import MetadataDB
from bespokelabs.curator.llm.llm import _CURATOR_DEFAULT_CACHE_DIR, _get_function_source
from bespokelabs.curator.log import logger
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop
from bespokelabs.curator.utils import push_to_viewer


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

    def is_completed(self, response: str) -> bool:
        """Check if the agent's response is signals conversation completion.

        Args:
            response (str): The response string to check.

        Returns:
            bool: True if the conversation is completed, False otherwise.
        """
        return False

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

    def _setup_metadata(self, fingerprint: str) -> dict:
        """Set up metadata for the curator.

        Args:
            fingerprint (str): The unique fingerprint for this run.

        Returns:
            dict: Metadata dictionary.
        """
        # Get the source code of both agents' prompt functions
        seeder_prompt_func = _get_function_source(self.seeder.prompt_formatter.prompt_func)
        partner_prompt_func = _get_function_source(self.partner.prompt_formatter.prompt_func)
        seeder_parse_func = _get_function_source(self.seeder.prompt_formatter.parse_func)
        partner_parse_func = _get_function_source(self.partner.prompt_formatter.parse_func)

        return {
            "run_hash": fingerprint,
            "dataset_hash": fingerprint,
            "prompt_func": f"Seeder ({self.seeder.name}):\n{seeder_prompt_func}\n\nPartner ({self.partner.name}):\n{partner_prompt_func}",
            "parse_func": f"Seeder ({self.seeder.name}):\n{seeder_parse_func}\n\nPartner ({self.partner.name}):\n{partner_parse_func}",
            "model_name": f"{self.seeder.prompt_formatter.model_name} + {self.partner.prompt_formatter.model_name}",
            "response_format": "N/A",
            "batch_mode": False,
            "status": "COMPLETED",
            "timestamp": datetime.now().isoformat(),
        }

    def _handle_viewer_push(self, viewer_client: Client, dataset, metadata_dict: dict, working_dir: str) -> str | None:
        """Handle pushing the dataset to the curator viewer.

        Args:
            viewer_client (Client): The curator viewer client.
            dataset: The dataset to push.
            metadata_dict (dict): Metadata for the viewer.
            working_dir (str): The working directory for this run.

        Returns:
            str | None: The viewer URL if successful, None otherwise.
        """
        # Create or get existing session
        metadata_db_path = os.path.join(working_dir, "metadata.db")
        metadata_db = MetadataDB(metadata_db_path)
        existing_session_id = metadata_db.get_existing_session_id(metadata_dict["run_hash"])
        existing_viewer_sync = metadata_db.check_existing_hosted_sync(metadata_dict["run_hash"])

        if not existing_viewer_sync and existing_session_id:
            session_id = viewer_client.create_session(metadata_dict)
        else:
            session_id = viewer_client.create_session(metadata_dict, session_id=existing_session_id)

        metadata_dict["session_id"] = session_id
        metadata_dict["is_hosted_viewer_synced"] = False
        metadata_db.store_metadata(metadata_dict)

        # Push dataset to viewer
        if existing_session_id is None:
            # No existing session, push the dataset
            push_to_viewer(dataset, session_id=session_id)
        else:
            # Warn about existing session
            msg = (
                f"Found existing session with run hash {metadata_dict['run_hash']}. "
                "The dataset has already been pushed to the Curator Viewer. "
                "If you want to push a new version, please use a different run hash."
            )
            logger.warning(msg)

        metadata_db.update_sync_viewer_flag(metadata_dict["run_hash"], True)
        return viewer_client.curator_viewer_url

    def __call__(self, working_dir: t.Optional[str] = None) -> MultiTurnResponse:
        """Execute the multi-turn conversation between the agents.

        This method runs the conversation simulation and saves the results
        in the specified working directory. If no working directory is provided,
        it uses the CURATOR_CACHE_DIR environment variable or a default location.

        Args:
            working_dir (Optional[str]): The directory where conversation results
                will be stored. If None, uses environment variable or default.

        Returns:
            MultiTurnResponse: A response object containing the conversation results and statistics.
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
        logger.info(f"Running multi turn simulation, find results in {working_dir}")

        viewer_client = Client()
        self._processor.viewer_client = viewer_client

        # Create metadata database
        metadata_db_path = os.path.join(working_dir, "metadata.db")
        metadata_db = MetadataDB(metadata_db_path)

        # Run the conversation and get the dataset
        dataset = run_in_event_loop(self._processor.run(run_hash=fingerprint, working_dir=working_dir, metadata_db=metadata_db))

        # Create and populate the response object
        response = MultiTurnResponse(
            dataset=dataset,
            cache_dir=working_dir,
            seeder_model=self.seeder.prompt_formatter.model_name,
            partner_model=self.partner.prompt_formatter.model_name,
            metadata={
                "max_length": self.max_length,
                "seeder_name": self.seeder.name,
                "partner_name": self.partner.name,
                "seed_message": self.seed_message,
                "timestamp": datetime.now().isoformat(),
                "run_hash": fingerprint,
            },
        )

        # Update statistics from the processor's status tracker
        response.update_tracker_stats(self._processor.status_tracker)

        # Setup metadata for viewer
        metadata_dict = self._setup_metadata(fingerprint)
        existing_session_id = metadata_db.get_existing_session_id(metadata_dict["run_hash"])
        existing_viewer_sync = metadata_db.check_existing_hosted_sync(metadata_dict["run_hash"])

        if not existing_viewer_sync and existing_session_id:
            session_id = viewer_client.create_session(metadata_dict)
        else:
            session_id = viewer_client.create_session(metadata_dict, session_id=existing_session_id)

        metadata_dict["session_id"] = session_id
        metadata_dict["is_hosted_viewer_synced"] = False
        metadata_db.store_metadata(metadata_dict)

        # Handle viewer push if enabled
        if viewer_client.hosted:
            viewer_url = self._handle_viewer_push(viewer_client, dataset, metadata_dict, working_dir)
            response.viewer_url = viewer_url
            if viewer_url:
                logger.info(f"Curator Viewer: {viewer_url}")
            metadata_db.update_sync_viewer_flag(metadata_dict["run_hash"], True)
        else:
            logger.info(f"Curator Viewer: Disabled (Set CURATOR_VIEWER=1 to view at {PUBLIC_CURATOR_VIEWER_HOME_URL})")

        return response
