import datetime
import json
import logging
import os
import litellm

from bespokelabs.curator.dataset import Dataset
from bespokelabs.curator.prompter.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor.base_request_processor import (
    parse_response_message,
    BaseRequestProcessor,
)
from bespokelabs.curator.types.generic_batch_object import GenericBatchObject
from bespokelabs.curator.types.generic_batch_object import GenericBatchRequestCounts
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop
from bespokelabs.curator.types.token_usage import TokenUsage
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from bespokelabs.curator.batch_manager.base_batch_manager import BaseBatchManager
from bespokelabs.curator.batch_manager.anthropic_batch_manager import AnthropicBatchManager
from bespokelabs.curator.batch_manager.openai_batch_manager import OpenAIBatchManager

logger = logging.getLogger(__name__)


class BaseBatchRequestProcessor(BaseRequestProcessor):
    def __init__(
        self,
        batch_size: int,
        model: str,
        delete_successful_batch_files: bool,
        delete_failed_batch_files: bool,
        temperature: float | None = None,
        top_p: float | None = None,
        batch_check_interval: int = 60,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
    ):
        super().__init__(batch_size)
        self.model = model
        self.check_interval: int = batch_check_interval
        self.temperature: float | None = temperature
        self.top_p: float | None = top_p
        self.presence_penalty: float | None = presence_penalty
        self.frequency_penalty: float | None = frequency_penalty
        self.delete_successful_batch_files: bool = delete_successful_batch_files
        self.delete_failed_batch_files: bool = delete_failed_batch_files

    def get_rate_limits(self) -> dict:
        """Can't find any information on batch rate limits"""
        return None

    def create_api_specific_request(self, generic_request: GenericRequest) -> dict:
        """
        Creates an API-specific request body from a generic request body.

        This function transforms a GenericRequest into the format expected by OpenAI's batch API.
        It handles both standard requests and those with JSON schema response formats.

        Args:
            generic_request (GenericRequest): The generic request object containing model, messages,
                and optional response format.

        Returns:
            dict: API specific request body formatted for OpenAI's batch API, including:
                - custom_id: String identifier from the original row index
                - method: Always "POST"
                - url: OpenAI chat completions endpoint
                - body: Request parameters including model, messages, and optional formatting
        """
        if generic_request.response_format:
            # TODO(Ryan) how can we support this the way litellm does?
            raise NotImplementedError("response_format is not yet supported for Anthropic")

        params = {
            "model": generic_request.model,
        }
        if self.generic_request.messages[0]["role"] == "system":
            params["system"] = self.generic_request.messages[0]["content"]
            params["messages"] = self.generic_request.messages[1:]
        else:
            params["messages"] = self.generic_request.messages

        if self.temperature is not None:
            params["temperature"] = self.temperature

        if self.top_p is not None:
            params["top_p"] = self.top_p

        if self.presence_penalty is not None:
            raise NotImplementedError("presence_penalty is not yet supported for Anthropic")

        if self.frequency_penalty is not None:
            raise NotImplementedError("frequency_penalty is not yet supported for Anthropic")

        request = {
            "custom_id": str(generic_request.original_row_idx),
            "params": params,
        }

        return request

    def requests_from_generic_request_file(self, request_file: str) -> list[dict]:
        """
        Reads and converts generic requests from a file into API-specific request format.

        Args:
            request_file (str): Path to the file containing generic requests in JSONL format.

        Returns:
            list[dict]: List of API-specific request bodies ready for batch submission.
        """
        api_specific_requests = []

        with open(request_file, "r") as file:
            for line in file:
                request = GenericRequest.model_validate_json(line.strip())
                api_specific_request = self.create_api_specific_request(request)
                api_specific_requests.append(json.dumps(api_specific_request))

        return api_specific_requests

    def generic_response_file_from_responses(
        self, responses: str, batch: BetaMessageBatch, response_file: str
    ) -> str | None:
        """Processes API-specific responses and creates a generic response file.

        Takes raw API responses from a batch request and converts them into GenericResponse objects,
        writing them to a response file. Handles both successful and failed responses, including
        token usage tracking and cost calculation.

        Args:
            responses (str): Raw response text from the API containing JSONL formatted responses.
            batch (Batch): The OpenAI batch object containing metadata about the request batch.
            response_file (str): Path where the generic response file should be written.

        Returns:
            str | None: Path to the created response file, or None if creation failed.

        Note:
            The response file will contain one GenericResponse per line in JSONL format.
            Failed requests will have response_message=None and include error details.
            Costs are calculated using litellm with 50% discount applied for batch requests.
        """
        request_file = batch.metadata["request_file_name"]
        generic_request_map = {}
        batch_created_at = datetime.datetime.fromtimestamp(batch.created_at)
        with open(request_file, "r") as f:
            for line in f:
                generic_request = GenericRequest.model_validate_json(line)
                generic_request_map[generic_request.original_row_idx] = generic_request

        with open(response_file, "w") as f:
            for raw_response in responses.text.splitlines():
                raw_response = json.loads(raw_response)
                request_idx = int(raw_response["custom_id"])
                generic_request = generic_request_map[request_idx]

                if raw_response["response"]["status_code"] != 200:
                    logger.warning(
                        f"Request {generic_request} failed with status code {raw_response['response']['status_code']}"
                    )
                    generic_response = GenericResponse(
                        response_message=None,
                        response_errors=[
                            f"Request {generic_request} failed with status code {raw_response['response']['status_code']}"
                        ],
                        raw_response=raw_response,
                        raw_request=None,
                        generic_request=generic_request,
                        created_at=batch_created_at,
                        finished_at=datetime.datetime.now(),
                        token_usage=None,
                        response_cost=None,
                    )
                else:
                    response_body = raw_response["response"]["body"]
                    choices = response_body["choices"]
                    usage = response_body.get("usage", {})

                    token_usage = TokenUsage(
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                    )

                    # Calculate cost using litellm (50% off for batch)
                    cost = (
                        litellm.completion_cost(
                            model=generic_request.model,
                            prompt=str(generic_request.messages),
                            completion=choices[0]["message"]["content"],
                        )
                        * 0.5
                    )

                    response_message = choices[0]["message"]["content"]
                    response_message, response_errors = parse_response_message(
                        response_message, self.prompt_formatter.response_format
                    )

                    generic_response = GenericResponse(
                        response_message=response_message,
                        response_errors=response_errors,
                        raw_response=raw_response,
                        raw_request=None,
                        generic_request=generic_request,
                        created_at=batch_created_at,
                        finished_at=datetime.datetime.now(),
                        token_usage=token_usage,
                        response_cost=cost,
                    )
                json.dump(generic_response.model_dump(), f, default=str)
                f.write("\n")

    async def run_batch_operations(self, batch_manager, request_files):
        # For running in a single event loop (so sempahore does not change)
        await batch_manager.submit_batches_from_request_files(
            request_files, self.requests_from_generic_request_file
        )
        await batch_manager.poll_and_process_batches(self.generic_response_file_from_responses)

    def get_batch_manager(
        self, working_dir: str, prompt_formatter: PromptFormatter
    ) -> BaseBatchManager:
        MODEL_TO_BATCH_MANAGER = {
            "claude-3-5-sonnet-20240620": AnthropicBatchManager,
            "claude-3-5-sonnet-20241022": AnthropicBatchManager,
            "gpt-4o-mini": OpenAIBatchManager,
            "gpt-4o-2024-08-06": OpenAIBatchManager,
        }

        BatchManagerClass = MODEL_TO_BATCH_MANAGER.get(self.model)
        if not BatchManagerClass:
            raise ValueError(f"Model {self.model} is not supported for batch processing")

        return BatchManagerClass(
            working_dir,
            self.check_interval,
            prompt_formatter,
            delete_successful_batch_files=self.delete_successful_batch_files,
            delete_failed_batch_files=self.delete_failed_batch_files,
        )

    def run(
        self,
        dataset: Dataset | None,
        working_dir: str,
        parse_func_hash: str,
        prompt_formatter: PromptFormatter,
    ) -> Dataset:
        """
        Processes a dataset using OpenAI's batch API.

        This function orchestrates the complete batch processing workflow:
        1. Attempts to load cached results if available
        2. Creates request files from the dataset
        3. Submits and processes batches
        4. Creates output dataset files

        Args:
            dataset (Dataset | None): Input dataset to process.
            working_dir (str): Directory for storing intermediate files and results.
            parse_func_hash (str): Hash of the parsing function for cache identification.
            prompt_formatter (PromptFormatter): Formatter for processing prompts and responses.

        Returns:
            Dataset: Processed dataset

        Raises:
            RuntimeError: If batch processing fails or no successful responses are received.
        """
        batch_manager = self.get_batch_manager(working_dir, prompt_formatter)

        if self.batch_size > batch_manager.max_requests_per_batch:
            raise ValueError(
                f"batch_size {self.batch_size} is greater than the maximum of "
                f"{batch_manager.max_requests_per_batch:,} requests per batch that {batch_manager.__class__.__name__} supports. "
                f"Please set your batch_size to be less than or equal to {batch_manager.max_requests_per_batch:,}."
            )

        # load from already completed dataset
        output_dataset = self.attempt_loading_cached_dataset(working_dir, parse_func_hash)
        if output_dataset is not None:
            return output_dataset

        request_files = set(self.create_request_files(dataset, working_dir, prompt_formatter))
        self.prompt_formatter = prompt_formatter

        run_in_event_loop(self.run_batch_operations(batch_manager, request_files))

        return self.create_dataset_files(working_dir, parse_func_hash, prompt_formatter)

    def cancel_batches(self, working_dir: str) -> Dataset:
        """
        Cancels all submitted batches and exits the program.

        Args:
            working_dir (str): The directory where submitted batch object file is stored.
        """
        batch_manager = self.get_batch_manager(working_dir)
        run_in_event_loop(batch_manager.cancel_batches())
        logger.warning("Exiting program after batch cancellation.")
        os._exit(1)
