import datetime
import json
import typing as t
from abc import ABC
from dataclasses import dataclass, field

from bespokelabs.curator.llm.prompt_formatter import PromptFormatter
from bespokelabs.curator.log import logger
from bespokelabs.curator.request_processor.base_request_processor import BaseRequestProcessor
from bespokelabs.curator.request_processor.config import OfflineRequestProcessorConfig
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse

if t.TYPE_CHECKING:
    from bespokelabs.curator.status_tracker.offline_status_tracker import OfflineStatusTracker


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata.

    Attributes:
        task_id: Unique identifier for this request
        generic_request: The generic request object containing prompt and parameters
        api_specific_request: Request formatted for the specific API being used
        result: List to store results from the API
        prompt_formatter: Formatter used to create prompts
        created_at: Timestamp when request was created
    """

    task_id: int
    generic_request: GenericRequest
    api_specific_request: dict
    result: list = field(default_factory=list)
    prompt_formatter: PromptFormatter = field(default=None)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)


class BaseOfflineRequestProcessor(BaseRequestProcessor, ABC):
    """Base class for offline request processors.

    Provides core functionality for processing requests through offline models, including:
    - Model loading and configuration
    - Request processing and response generation
    - File handling and resumption of interrupted processing

    Args:
        config (OfflineRequestProcessorConfig): Configuration for the request processor containing
            model parameters and processing settings
    """

    def __init__(self, config: OfflineRequestProcessorConfig):
        """Initialize the offline request processor.

        Args:
            config: Configuration object containing model and processing parameters
        """
        super().__init__(config)
        self.model: str = config.model
        self.max_model_length: int = config.max_model_length
        self.max_tokens: int = config.max_tokens
        self.enforce_eager: bool = config.enforce_eager
        self.tensor_parallel_size: int = config.tensor_parallel_size
        self.gpu_memory_utilization: float = config.gpu_memory_utilization
        self.min_tokens: int = config.min_tokens
        self.batch_size: int = config.batch_size
        self.generation_params = config.generation_params

    @property
    def backend(self):
        """Backend property."""
        return "base"

    def validate_config(self):
        """Validate offline request processor configuration.

        Ensures that configuration parameters are set correctly.

        Raises:
            ValueError: If configuration parameters are invalid
        """
        pass

    def load_offline_model(self):
        """Load the offline model into memory.

        Should be implemented by subclasses to handle specific model loading logic.
        """
        pass

    def destroy(self) -> None:
        """Clean up model resources.

        Should be implemented by subclasses to handle model cleanup.
        """
        pass

    def process_requests(self, requests: list[APIRequest], status_tracker: "OfflineStatusTracker") -> list[GenericResponse]:
        """Process a batch of requests through the model.

        Args:
            requests: List of API requests to process
            status_tracker: Tracker to monitor processing status

        Returns:
            List of responses from the model
        """
        pass

    def requests_to_responses(
        self,
        generic_request_files: list[str],
    ) -> None:
        """Process multiple request files and generate corresponding response files.

        Args:
            generic_request_files: List of paths to request files to process
        """
        for request_file in generic_request_files:
            response_file = request_file.replace("requests_", "responses_")
            self.process_requests_from_file(
                generic_request_filepath=request_file,
                save_filepath=response_file,
            )

    def process_requests_from_file(
        self,
        generic_request_filepath: str,
        save_filepath: str,
    ) -> None:
        """Process requests from a file and save responses.

        Handles resuming interrupted processing and retrying failed requests.
        Creates response files with results from model inference.

        Args:
            generic_request_filepath: Path to file containing requests
            save_filepath: Path to save response file
            resume: Whether to resume processing from previous state
            resume_no_retry: Whether to skip retrying failed requests when resuming

        Side Effects:
            - Creates/updates response file with model outputs
            - Logs progress and completion status
            - May prompt user for confirmation on file overwrite
        """
        from bespokelabs.curator.status_tracker.offline_status_tracker import OfflineStatusTracker

        status_tracker = OfflineStatusTracker()

        # Track completed requests for resume functionality
        completed_request_ids, completed_parsed_responses = self.validate_existing_response_file(save_filepath)
        status_tracker.num_parsed_responses = completed_parsed_responses

        if not hasattr(self, "model_class"):
            self.load_offline_model()  # Load the offline model if it hasn't been loaded yet
        # Count total requests
        total_requests = sum(1 for _ in open(generic_request_filepath))

        logger.info(f"Processing {total_requests} requests from {generic_request_filepath}")

        requests = []
        with open(generic_request_filepath, "r") as f:
            for line in f:
                request = GenericRequest.model_validate_json(line)
                if request.original_row_idx not in completed_request_ids:
                    requests.append(
                        APIRequest(
                            task_id=request.original_row_idx,
                            generic_request=request,
                            api_specific_request=self.create_api_specific_request(request),
                            prompt_formatter=self.prompt_formatter,
                        )
                    )
        responses = self.process_requests(
            requests=requests,
            status_tracker=status_tracker,
        )

        # Save responses
        with open(save_filepath, "a") as f:
            for response in responses:
                processed_response = self._process_response(response)
                response.parsed_response_message = processed_response
                json_string = json.dumps(response.model_dump(mode="json"), default=str)
                f.write(json_string + "\n")

                # Stream responses to viewer client
                idx = status_tracker.num_parsed_responses
                status_tracker.num_parsed_responses = idx + len(responses)
                run_in_event_loop(self.viewer_client.stream_response(json_string, idx))

        # Log final status
        logger.info(f"Processing complete. Results saved to {save_filepath}")
        logger.info(f"Status tracker: {status_tracker}")
