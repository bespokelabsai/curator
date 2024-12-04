import asyncio
import datetime
import json
import logging
import os
import resource
from dataclasses import dataclass

import aiofiles
import glob
import litellm
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
from bespokelabs.curator.request_processor.generic_response import TokenUsage

logger = logging.getLogger(__name__)

MAX_REQUESTS_PER_BATCH = 50_000
MAX_BYTES_PER_BATCH = 200 * 1024 * 1024

# NOTE(Ryan): This allows us to stay under the rate limit when submitting ~1,000 batches at a time
# When submitting >1,000 batches the batch submission and batch download operations get rate limited
MAX_CONCURRENT_BATCH_OPERATIONS = 100
MAX_RETRIES_PER_OPERATION = 5


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
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
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
        self.presence_penalty: float | None = presence_penalty
        self.frequency_penalty: float | None = frequency_penalty
        self._file_lock = asyncio.Lock()

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

        logger.info(f"Automatically set max_tokens_per_day to {tpd}, model: {self.model} ")

        rate_limits = {"max_tokens_per_day": tpd}

        return rate_limits

    def create_api_specific_request(self, generic_request: GenericRequest) -> dict:
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

        if self.presence_penalty is not None:
            body["presence_penalty"] = self.presence_penalty

        if self.frequency_penalty is not None:
            body["frequency_penalty"] = self.frequency_penalty

        request = {
            "custom_id": str(generic_request.original_row_idx),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

        return request

    async def asubmit_batch(
        self, batch_file: str, semaphore: asyncio.Semaphore | None = None
    ) -> dict:
        async with semaphore or asyncio.Semaphore():  # Use provided semaphore or dummy one
            async_client = AsyncOpenAI(max_retries=MAX_RETRIES_PER_OPERATION)
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

            try:
                # this let's you upload a file that is larger than 200MB and won't error, so we catch it above
                batch_file_upload = await async_client.files.create(
                    file=file_content, purpose="batch"
                )
            except Exception as e:
                logger.error(f"Error uploading batch file: {e}")
                raise e

            # When submitting a file, sometimes the file is not ready immediately for status checking
            await asyncio.sleep(1)

            try:
                batch_file_upload = await async_client.files.wait_for_processing(
                    batch_file_upload.id
                )
            except Exception as e:
                logger.error(f"Error waiting for batch file to be processed: {e}")
                raise e

            logger.info(f"File uploaded with id {batch_file_upload.id}")
            try:
                batch_object = await async_client.batches.create(
                    input_file_id=batch_file_upload.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                        "request_file_name": batch_file
                    },  # for downloading the batch to similarly named responses file
                )
                logger.info(f"Batch submitted with id {batch_object.id}")
            except Exception as e:
                logger.error(f"Error submitting batch: {e}")
                raise e

            # NOTE(Ryan): we can also store the request_file_name in this object here, instead of in the metadata during batch submission.
            # If we create "generic batch object" maybe we can find a nice abstraction across other batch APIs (e.g. claude)
            async with self._file_lock:
                async with aiofiles.open(
                    f"{os.path.dirname(batch_file)}/batch_objects.jsonl", "a"
                ) as f:
                    await f.write(json.dumps(batch_object.model_dump(), default=str) + "\n")
                    await f.flush()

            # Explicitly close the client
            await async_client.close()

            return batch_object

    def run(
        self,
        dataset: Dataset | None,
        working_dir: str,
        parse_func_hash: str,
        prompt_formatter: PromptFormatter,
    ) -> Dataset:
        """
        Uses the API to completing the specific map by calling the LLM.

        Args:
            dataset (Dataset): Dataset that is being mapped over
            working_dir (str): Working directory to save files (requests.jsonl, responses.jsonl, dataset.arrow)
            parse_func_hash (str): Hash of the parse_func to be used as the dataset file name
            prompt_formatter (PromptFormatter): Prompt formatter to be used to format the prompt

        Returns:
            Dataset: Completed dataset
        """
        # Increase the number of open file descriptors to avoid "Too many open files" errors
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired_limit = min(1_000_000, hard)
        logger.info(
            f"Adjusting file descriptor limit from {soft} to {desired_limit} (hard limit: {hard})"
        )
        resource.setrlimit(resource.RLIMIT_NOFILE, (desired_limit, hard))

        requests_files = set(self.create_request_files(dataset, working_dir, prompt_formatter))
        batch_objects_file = f"{working_dir}/batch_objects.jsonl"

        if os.path.exists(batch_objects_file):
            # Read existing batch objects from file
            already_submitted = 0
            with open(batch_objects_file, "r") as f:
                for line in f:
                    batch_object = json.loads(line)
                    request_file_name = batch_object["metadata"]["request_file_name"]
                    logger.info(f"Batch job already submitted for request file {request_file_name}")
                    requests_files.remove(request_file_name)
                    already_submitted += 1

            if len(requests_files) == 0:
                logger.info(
                    f"All {already_submitted:,} batches completed, skipping batch submission"
                )
            else:
                logger.info(
                    f"{already_submitted:,} out of {len(requests_files)+already_submitted:,} batch jobs already submitted. Submitting the remaining {len(requests_files):,} batch jobs."
                )

        # Limit concurrent batch submissions to 100 so avoid overwhelming the API
        async def submit_all_batches():
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCH_OPERATIONS)
            tasks = [
                self.asubmit_batch(requests_file, semaphore) for requests_file in requests_files
            ]
            return await asyncio.gather(*tasks)

        run_in_event_loop(submit_all_batches())
        logger.info(f"All batch objects submitted and written to {batch_objects_file}")

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
            await batch_watcher.close_client()

        run_in_event_loop(watch_batches())

        dataset = self.create_dataset_files(working_dir, parse_func_hash, prompt_formatter)

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
            obj["id"]: obj["metadata"]["request_file_name"] for obj in self.batch_objects
        }
        self.check_interval = check_interval
        self.working_dir = working_dir
        self.tracker = BatchStatusTracker()
        self.tracker.n_submitted_batches = len(self.batch_ids)
        self.tracker.n_submitted_requests = n_submitted_requests
        self.pbar = None
        self.remaining_batch_ids = set(self.batch_ids)
        self.prompt_formatter = prompt_formatter
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCH_OPERATIONS)

    async def close_client(self):
        await self.client.close()

    async def check_batch_status(self, batch_id: str) -> Batch | None:
        """Check the status of a batch by its ID.

        Args:
            batch_id (str): The ID of the batch to check.

        Returns:
            Batch: The batch object. None if the batch has not returned yet.
        """
        async with self.semaphore:
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
                logger.info(f"Batch {batch.id} returned with status: {batch.status}")
                self.tracker.n_returned_batches += 1
                self.tracker.n_completed_returned_requests += n_completed_requests
                self.tracker.n_failed_returned_requests += n_failed_requests
                self.remaining_batch_ids.remove(batch.id)
                return batch
            else:
                self.tracker.n_completed_in_progress_requests += n_completed_requests
                self.tracker.n_failed_in_progress_requests += n_failed_requests
                return None

    async def resume(self, batch_id: str, all_response_files: set[str]):
        request_file = self.batch_id_to_request_file_name[batch_id]
        request_file_idx = request_file.split("/")[-1].split("_", 1)[1]
        response_file = f"{self.working_dir}/responses_{request_file_idx}"
        if response_file in all_response_files:
            logger.info(
                f"File {response_file} found for batch {batch_id}, skipping status check and download."
            )
            self.remaining_batch_ids.remove(batch_id)
            # Update progress bar based on completed responses found in files
            async with aiofiles.open(response_file, "r") as f:
                n_completed = 0
                async for _ in f:
                    n_completed += 1
            self.tracker.n_completed_returned_requests += n_completed
            self.tracker.n_returned_batches += 1
            self.tracker.n_completed_batches += 1  # asumming completed instead of failed

    async def watch(self) -> None:
        """Monitor the status of batches until all are completed (includes successfully, failed, expired or cancelled)."""
        # progress bar for completed requests
        self.pbar = tqdm(
            total=self.tracker.n_submitted_requests,
            desc="Completed OpenAI requests in batches",
            unit="request",
        )

        # resume from already downloaded batches
        all_response_files = set(glob.glob(f"{self.working_dir}/responses_*.jsonl"))
        tasks = [self.resume(batch_id, all_response_files) for batch_id in self.remaining_batch_ids]
        await asyncio.gather(*tasks)

        if self.tracker.n_returned_batches > 0:
            logger.info(
                f"Found {self.tracker.n_returned_batches} out of {self.tracker.n_submitted_batches} completed batches, resuming polling for the remaining {len(self.remaining_batch_ids)} batches."
            )
        all_response_files = list(all_response_files)

        # loop until all batches have been returned
        while self.remaining_batch_ids:

            # need to reset in_progress numbers each time
            self.tracker.n_completed_in_progress_requests = 0
            self.tracker.n_failed_in_progress_requests = 0

            # check batch status also updates the tracker
            status_tasks = [
                self.check_batch_status(batch_id) for batch_id in self.remaining_batch_ids
            ]
            batches_to_download = await asyncio.gather(*status_tasks)
            batches_to_download = filter(None, batches_to_download)

            # update progress bar
            self.pbar.n = 0
            self.pbar.n += self.tracker.n_completed_returned_requests
            self.pbar.n += self.tracker.n_failed_returned_requests
            self.pbar.n += self.tracker.n_completed_in_progress_requests
            self.pbar.n += self.tracker.n_failed_in_progress_requests
            self.pbar.refresh()

            download_tasks = [
                self.download_batch_to_generic_responses_file(batch)
                for batch in batches_to_download
            ]
            # Failed downloads return None and print any errors that occurred
            all_response_files.extend(await asyncio.gather(*download_tasks))

            if self.tracker.n_returned_batches < self.tracker.n_submitted_batches:
                logger.debug(
                    f"Batches returned: {self.tracker.n_returned_batches}/{self.tracker.n_submitted_batches} "
                    f"Requests completed: {self.pbar.n}/{self.tracker.n_submitted_requests}"
                )
                logger.debug(f"Sleeping for {self.check_interval} seconds...")
                await asyncio.sleep(self.check_interval)

        self.pbar.close()
        response_files = filter(None, all_response_files)
        if self.tracker.n_completed_batches == 0 or not response_files:
            raise RuntimeError(
                "None of the submitted batches completed successfully. "
                "Please check the logs above and https://platform.openai.com/batches for errors."
            )

    async def download_batch_to_generic_responses_file(self, batch: Batch) -> str | None:
        """Download the result of a completed batch to file.

        Args:
            batch: The batch object to download results from.

        Returns:
            str: Path to the downloaded result file.
        """
        async with self.semaphore:
            if batch.status == "completed" and batch.output_file_id:
                file_content = await self.client.files.content(batch.output_file_id)
                logger.info(f"Batch {batch.id} completed and downloaded")
            elif batch.status == "failed" and batch.error_file_id:
                file_content = await self.client.files.content(batch.error_file_id)
                logger.warning(f"Batch {batch.id} failed\n. Errors will be parsed below.")
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
            request_creation_times = {}  # Track creation times for requests
            with open(request_file, "r") as f:
                for line in f:
                    generic_request = GenericRequest.model_validate_json(line)
                    generic_request_map[generic_request.original_row_idx] = generic_request
                    request_creation_times[generic_request.original_row_idx] = (
                        datetime.datetime.now()
                    )

            with open(response_file, "w") as f:
                for raw_response in file_content.text.splitlines():
                    raw_response = json.loads(raw_response)
                    request_idx = int(raw_response["custom_id"])
                    generic_request = generic_request_map[request_idx]

                    # TODO(Ryan): Add more specific error handling
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
                            created_at=request_creation_times[request_idx],
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

                        # Calculate cost using litellm
                        cost = litellm.completion_cost(
                            model=generic_request.model,
                            prompt=str(
                                generic_request.messages
                            ),  # Convert messages to string for cost calculation
                            completion=choices[0]["message"]["content"],
                        )
                        # Batch requests are 50% off
                        cost = cost * 0.5

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
                            created_at=request_creation_times[request_idx],
                            finished_at=datetime.datetime.now(),
                            token_usage=token_usage,
                            response_cost=cost,
                        )
                    f.write(json.dumps(generic_response.model_dump(), default=str) + "\n")
                logger.info(f"Batch {batch.id} written to {response_file}")
            return response_file
