import asyncio
import json
import logging
import os
from dataclasses import dataclass

import aiofiles
from openai import AsyncOpenAI
from openai.types import Batch
from tqdm import tqdm

from bespokelabs.curator.dataset import Dataset
from bespokelabs.curator.prompter.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor.base_request_processor import (
    BaseRequestProcessor,
    GenericRequest,
    GenericResponse,
    parse_response_message,
)
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop

logger = logging.getLogger(__name__)

MAX_REQUESTS_PER_BATCH = 50_000
MAX_BYTES_PER_BATCH = 200 * 1024 * 1024


class OpenAIBatchRequestProcessor(BaseRequestProcessor):
    def __init__(
        self,
        batch_size: int,
        model: str,
        temperature: float | None = None,
        top_p: float | None = None,
        check_interval: int = 10,
        api_key: str = os.getenv("OPENAI_API_KEY"),
        url: str = "https://api.openai.com/v1/chat/completions",
    ):
        if batch_size > MAX_REQUESTS_PER_BATCH:
            raise ValueError(
                f"batch_size {batch_size} is greater than the maximum of "
                f"{MAX_REQUESTS_PER_BATCH:,} requests per batch that OpenAI supports. "
                f"Please set your batch_size to be less than or equal to {MAX_REQUESTS_PER_BATCH:,}."
            )
        super().__init__(batch_size)
        self.url: str = url
        self.api_key: str = api_key
        self.check_interval: int = check_interval
        self.temperature: float | None = temperature
        self.top_p: float | None = top_p

    def get_rate_limits(self) -> dict:
        """
        Function to get rate limits for a given annotator. Not available via response headers, so
        the following is based on tier 5 limits on Nov 6th, 2024.

        These rate limits vary per model
        and are determined by your organization's usage tier. View the following:
        https://platform.openai.com/docs/guides/rate-limits/usage-tiers
        https://platform.openai.com/settings/organization/limits

        Args:
            model (str): The model for which to get the rate limits.
            request_url (str): The request URL for which to get the rate limits.

        Returns:
            tuple[int, int]: A tuple containing the maximum number of requests and tokens per minute.
        """
        model_tpd = {
            "gpt-3.5-turbo": 5_000_000_000,
            "gpt-3.5-turbo-0125": 5_000_000_000,
            "gpt-3.5-turbo-1106": 5_000_000_000,
            "gpt-3.5-turbo-16k": 5_000_000_000,
            "gpt-3.5-turbo-instruct": 200_000,
            "gpt-3.5-turbo-instruct-0914": 200_000,
            "gpt-4": 150_000_000,
            "gpt-4-0613": 150_000_000,
            "gpt-4-turbo": 300_000_000,
            "gpt-4o": 10_000_000_000,
            "gpt-4o-mini": 15_000_000_000,
        }

        if self.model not in model_tpd:
            tpd = 1_000_000_000
        else:
            tpd = model_tpd[self.model]

        logger.info(
            f"Automatically set max_tokens_per_day to {tpd}, model: {self.model} "
        )

        rate_limits = {"max_tokens_per_day": tpd}

        return rate_limits

    def create_api_specific_request(
        self, generic_request: GenericRequest
    ) -> dict:
        """
        Creates a API-specific request body from a generic request body.

        Using the api_parallel_processor, we can store whatever we want in the metadata. We will store both the row and the index.
        This is so we can later construct the new dataset row.

        Returns:
            dict: API specific request body
        """
        # NOTE(Ryan): We can have a shared place that creates the body (since it is the same for both online and batch).
        if generic_request.response_format:
            body = {
                "model": generic_request.model,
                "messages": generic_request.messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        # TODO(ryan): not sure if this should be something else.
                        # TODO(ryan): also not sure if we should use strict: True
                        "name": "output_schema",
                        "schema": generic_request.response_format,
                    },
                },
            }
        else:
            body = {
                "model": generic_request.model,
                "messages": generic_request.messages,
            }

        if self.temperature is not None:
            body["temperature"] = self.temperature

        if self.top_p is not None:
            body["top_p"] = self.top_p

        request = {
            "custom_id": str(generic_request.original_row_idx),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

        return request

    async def asubmit_batch(self, batch_file: str) -> dict:
        async_client = AsyncOpenAI()
        # Create a list to store API-specific requests
        api_specific_requests = []

        line_count = 0
        async with aiofiles.open(batch_file, "r") as file:
            file_content = await file.read()
            for line in file_content.splitlines():
                request = GenericRequest.model_validate_json(line)
                api_specific_request = self.create_api_specific_request(request)
                api_specific_requests.append(json.dumps(api_specific_request))
                line_count += 1

        if line_count > MAX_REQUESTS_PER_BATCH:
            raise ValueError(
                f"Batch file {batch_file} contains {line_count:,} requests, "
                f"which is more than the maximum of {MAX_REQUESTS_PER_BATCH:,} requests per batch that OpenAI supports. "
                f"Preventing batch submission."
            )

        # Join requests with newlines and encode to bytes for upload
        file_content = "\n".join(api_specific_requests).encode()
        file_content_size = len(file_content)
        logger.debug(
            f"Batch file content size: {file_content_size / (1024*1024):.2f} MB ({file_content_size:,} bytes)"
        )
        if file_content_size > MAX_BYTES_PER_BATCH:
            raise ValueError(
                f"Batch file content size {file_content_size:,} bytes "
                f"is greater than the maximum of {MAX_BYTES_PER_BATCH:,} bytes per batch that OpenAI supports. "
                f"Please reduce your batch size or request content size (via prompt_func and response_format)."
            )

        # this let's you upload a file that is larger than 200MB and won't error, so we catch it above
        batch_file_upload = await async_client.files.create(
            file=file_content, purpose="batch"
        )

        logger.info(f"File uploaded: {batch_file_upload}")

        batch_object = await async_client.batches.create(
            input_file_id=batch_file_upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "request_file_name": batch_file
            },  # for downloading the batch to similarly named responses file
        )
        logger.info(
            f"Batch request submitted, received batch object: {batch_object}"
        )
        # Explicitly close the client. Otherwise we get something like
        # future: <Task finished name='Task-46' coro=<AsyncClient.aclose() done ... >>
        await async_client.close()

        return batch_object

    def run(
        self,
        dataset: Dataset | None,
        working_dir: str,
        parse_func_hash: str,
        prompt_formatter: PromptFormatter,
        load_from_cache: bool,
    ) -> Dataset:
        """
        Uses the API to completing the specific map by calling the LLM.

        Args:
            dataset (Dataset): Dataset that is being mapped over
            working_dir (str): Working directory to save files (requests.jsonl, responses.jsonl, dataset.arrow)
            parse_func_hash (str): Hash of the parse_func to be used as the dataset file name
            prompt_formatter (PromptFormatter): Prompt formatter to be used to format the prompt
            load_from_cache (bool): Whether to load the dataset from the cache (if false, then will overwrite cache)

        Returns:
            Dataset: Completed dataset
        """
        requests_files = self.create_request_files(
            dataset, working_dir, prompt_formatter, load_from_cache
        )
        batch_objects_file = f"{working_dir}/batch_objects.jsonl"

        # TODO(Ryan): we should have an easy way to cancel all batches in batch_objects.jsonl if the user realized they made a mistake
        if os.path.exists(batch_objects_file) and load_from_cache:
            logger.warning(
                f"Batch objects file already exists, skipping batch submission and resuming: {batch_objects_file}"
            )
        else:
            # upload requests files and submit batches
            # asyncio gather preserves order
            async def submit_all_batches():
                tasks = [
                    self.asubmit_batch(requests_files[i])
                    for i in range(len(requests_files))
                ]
                return await asyncio.gather(*tasks)

            batch_objects = run_in_event_loop(submit_all_batches())

            with open(batch_objects_file, "w") as f:
                # NOTE(Ryan): we can also store the request_file_name in this object here, instead of in the metadata during batch submission. Can find a nice abstraction across other batch APIs (e.g. claude)
                for obj in batch_objects:
                    f.write(json.dumps(obj.model_dump(), default=str) + "\n")
            logger.info(f"Batch objects written to {batch_objects_file}")

        # TODO(Ryan): Actually do accounting for tokens, so rate limits enforced locally.
        # NOTE(Ryan): Although this isn't really practical since the limits are for an entire day and an entire organization. Maybe skip this and just recognize what a rate limit error for batching looks like (need to try this on a low tier account).
        # rate_limits = self.get_rate_limits()
        # tpd = rate_limits["max_tokens_per_day"]
        # token_encoding_name = get_token_encoding_name(self.model)

        # TODO(Ryan): based on the files that are downloaded, update completed_ids. If any are errors, try to resubmit (depending on error type).
        # TODO(Ryan): This creates responses_0.jsonl, responses_1.jsonl, etc. errors named same way? or errors_0.jsonl, errors_1.jsonl?
        # TODO(Ryan): retries, resubmits on lagging batches - need to study this a little closer
        # TODO(Ryan): likely can add some logic for smarter check_interval based on batch size and if the batch has started or not, fine to do a dumb ping for now
        # NOTE(Ryan): If we allow for multiple heterogeneous requests per dataset row, we will need to update this.
        n_submitted_requests = 1 if dataset is None else len(dataset)

        async def watch_batches():
            batch_watcher = BatchWatcher(
                working_dir,
                check_interval=self.check_interval,
                n_submitted_requests=n_submitted_requests,
                prompt_formatter=prompt_formatter,
            )
            await batch_watcher.watch()
            # Explicitly close the client. Otherwise we get something like
            # future: <Task finished name='Task-46' coro=<AsyncClient.aclose() done ... >>
            await batch_watcher.close_client()

        run_in_event_loop(watch_batches())

        dataset = self.create_dataset_files(
            working_dir, parse_func_hash, prompt_formatter, load_from_cache
        )

        return dataset


@dataclass
class BatchStatusTracker:
    # BATCHES
    # returned batches = completed + failed + cancelled + expired
    n_submitted_batches: int = 0
    n_returned_batches: int = 0
    n_completed_batches: int = 0
    n_failed_batches: int = 0
    n_cancelled_batches: int = 0
    n_expired_batches: int = 0

    # REQUESTS
    n_submitted_requests: int = 0
    n_completed_returned_requests: int = 0
    n_failed_returned_requests: int = 0
    # requests in pending batches that have not been returned yet
    n_completed_in_progress_requests: int = 0
    n_failed_in_progress_requests: int = 0


class BatchWatcher:
    def __init__(
        self,
        working_dir: str,
        check_interval: int,
        prompt_formatter: PromptFormatter,
        n_submitted_requests: int,
    ) -> None:
        """Initialize BatchWatcher with batch objects file and check interval.

        Args:
            working_dir (str): Directory containing the batch objects JSON file.
            check_interval (int): Time interval (in seconds) to check batch status.
            prompt_formatter (PromptFormatter): Prompt formatter to be used to format the prompt
            n_submitted_requests (int): Number of requests submitted to the batches (used for progress bar)
        """
        self.client = AsyncOpenAI()
        with open(f"{working_dir}/batch_objects.jsonl", "r") as f:
            self.batch_objects = [json.loads(line) for line in f]
        self.batch_ids = [obj["id"] for obj in self.batch_objects]
        self.batch_id_to_request_file_name = {
            obj["id"]: obj["metadata"]["request_file_name"]
            for obj in self.batch_objects
        }
        self.check_interval = check_interval
        self.working_dir = working_dir
        self.tracker = BatchStatusTracker()
        self.tracker.n_submitted_batches = len(self.batch_ids)
        self.tracker.n_submitted_requests = n_submitted_requests
        self.remaining_batch_ids = set(self.batch_ids)
        self.prompt_formatter = prompt_formatter

    async def close_client(self):
        await self.client.close()

    async def check_batch_status(self, batch_id: str) -> Batch | None:
        """Check the status of a batch by its ID.

        Args:
            batch_id (str): The ID of the batch to check.

        Returns:
            Batch: The batch object. None if the batch has not returned yet.
        """
        batch = await self.client.batches.retrieve(batch_id)
        assert batch.id == batch_id

        n_completed_requests = batch.request_counts.completed
        n_failed_requests = batch.request_counts.failed
        n_total_requests = batch.request_counts.total

        logger.debug(
            f"Batch {batch.id} status: {batch.status} requests: "
            f"{n_completed_requests}/{n_failed_requests}/{n_total_requests} "
            "completed/failed/total"
        )

        batch_returned = False
        if batch.status == "completed":
            self.tracker.n_completed_batches += 1
            batch_returned = True
        elif batch.status == "failed":
            self.tracker.n_failed_batches += 1
            batch_returned = True
        elif batch.status == "expired":
            self.tracker.n_expired_batches += 1
            batch_returned = True
        elif batch.status == "cancelled":
            self.tracker.n_cancelled_batches += 1
            batch_returned = True
        else:
            if batch.status not in [
                "validating",
                "finalizing",
                "cancelling",
                "in_progress",
            ]:
                logger.warning(f"Unknown batch status: {batch.status}")

        if batch_returned:
            logger.info(
                f"Batch {batch.id} returned with status: {batch.status}"
            )
            self.tracker.n_returned_batches += 1
            self.tracker.n_completed_returned_requests += n_completed_requests
            self.tracker.n_failed_returned_requests += n_failed_requests
            self.remaining_batch_ids.remove(batch.id)
            return batch
        else:
            self.tracker.n_completed_in_progress_requests += (
                n_completed_requests
            )
            self.tracker.n_failed_in_progress_requests += n_failed_requests
            return None

    async def watch(self) -> None:
        """Monitor the status of batches until all are completed (includes successfully, failed, expired or cancelled)."""
        # progress bar for completed requests
        pbar = tqdm(
            total=self.tracker.n_submitted_requests,
            desc="Completed OpenAI requests in batches",
            unit="request",
        )
        all_response_files = []

        # loop until all batches have been returned
        while self.remaining_batch_ids:

            # need to reset in_progress numbers each time
            self.tracker.n_completed_in_progress_requests = 0
            self.tracker.n_failed_in_progress_requests = 0

            # check batch status also updates the tracker
            status_tasks = [
                self.check_batch_status(batch_id)
                for batch_id in self.remaining_batch_ids
            ]
            batches_to_download = await asyncio.gather(*status_tasks)
            batches_to_download = filter(None, batches_to_download)

            # update progress bar
            pbar.n = 0
            pbar.n += self.tracker.n_completed_returned_requests
            pbar.n += self.tracker.n_failed_returned_requests
            pbar.n += self.tracker.n_completed_in_progress_requests
            pbar.n += self.tracker.n_failed_in_progress_requests
            pbar.refresh()

            download_tasks = [
                self.download_batch_to_generic_responses_file(batch)
                for batch in batches_to_download
            ]
            # Failed downloads return None and print any errors that occurred
            all_response_files.extend(await asyncio.gather(*download_tasks))

            if (
                self.tracker.n_returned_batches
                < self.tracker.n_submitted_batches
            ):
                logger.debug(
                    f"Batches returned: {self.tracker.n_returned_batches}/{self.tracker.n_submitted_batches} "
                    f"Requests completed: {pbar.n}/{self.tracker.n_submitted_requests}"
                )
                logger.debug(f"Sleeping for {self.check_interval} seconds...")
                await asyncio.sleep(self.check_interval)

        pbar.close()
        response_files = filter(None, all_response_files)
        if self.tracker.n_completed_batches == 0 or not response_files:
            raise RuntimeError(
                "None of the submitted batches completed successfully. "
                "Please check the logs above and https://platform.openai.com/batches for errors."
            )

    async def download_batch_to_generic_responses_file(
        self, batch: Batch
    ) -> str | None:
        """Download the result of a completed batch to file.

        Args:
            batch: The batch object to download results from.

        Returns:
            str: Path to the downloaded result file.
        """
        if batch.status == "completed" and batch.output_file_id:
            file_content = await self.client.files.content(batch.output_file_id)
        elif batch.status == "failed" and batch.error_file_id:
            file_content = await self.client.files.content(batch.error_file_id)
            logger.warning(
                f"Batch {batch.id} failed\n. Errors will be parsed below."
            )
        elif batch.status == "failed" and not batch.error_file_id:
            errors = "\n".join([str(error) for error in batch.errors.data])
            logger.error(
                f"Batch {batch.id} failed and likely failed validation. "
                f"Batch errors: {errors}. "
                f"Check https://platform.openai.com/batches/{batch.id} for more details."
            )
            return None
        elif batch.status == "cancelled" or batch.status == "expired":
            logger.warning(f"Batch {batch.id} was cancelled or expired")
            return None

        # Naming is consistent with the request file (e.g. requests_0.jsonl -> responses_0.jsonl)
        request_file = self.batch_id_to_request_file_name[batch.id]
        request_file_idx = request_file.split("/")[-1].split("_", 1)[1]
        response_file = f"{self.working_dir}/responses_{request_file_idx}"

        generic_request_map = {}
        with open(request_file, "r") as f:
            for line in f:
                generic_request = GenericRequest.model_validate_json(line)
                generic_request_map[generic_request.original_row_idx] = (
                    generic_request
                )

        with open(response_file, "w") as f:
            for raw_response in file_content.text.splitlines():
                raw_response = json.loads(raw_response)
                generic_request = generic_request_map[
                    int(raw_response["custom_id"])
                ]

                generic_response = GenericResponse(
                    raw_response=raw_response,
                    raw_request=None,
                    generic_request=generic_request,
                )

                # TODO(Ryan): Add more specific error handling
                status_code = raw_response["response"]["status_code"]
                if status_code != 200:
                    logger.warning(
                        f"Request {generic_request} failed with status code {status_code}"
                    )
                    generic_response.response_errors = [
                        f"Request {generic_request} failed with status code {status_code}"
                    ]
                else:
                    choices = raw_response["response"]["body"]["choices"]
                    response_message = choices[0]["message"][
                        "content"
                    ]  # Assuming N = 1
                    response_message, response_errors = parse_response_message(
                        response_message, self.prompt_formatter.response_format
                    )
                    generic_response.response_message = response_message
                    generic_response.response_errors = response_errors
                f.write(
                    json.dumps(generic_response.model_dump(), default=str)
                    + "\n"
                )
        return response_file
