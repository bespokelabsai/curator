import asyncio
import os
import json
import glob
from typing import Optional, Callable

from tqdm import tqdm

from abc import abstractmethod

from bespokelabs.curator.status_tracker.batch_status_tracker import BatchStatusTracker
from bespokelabs.curator.types.generic_batch_object import GenericBatchObject
from bespokelabs.curator.request_processor.base_request_processor import (
    request_file_to_response_file,
)
from bespokelabs.curator.request_processor.base_request_processor import (
    response_file_to_request_file,
)
import logging

logger = logging.getLogger(__name__)


class BaseBatchManager:
    def __init__(
        self,
        working_dir: str,
        max_concurrent_batch_operations: int,
        check_interval: int = 60,
        delete_successful_batch_files: bool = False,
        delete_failed_batch_files: bool = False,
    ) -> None:
        """Initialize BatchManager to handle OpenAI batch processing operations.

        Args:
            working_dir (str): Directory for storing batch-related files including requests, responses,
                and tracking files.
            check_interval (int): Time interval (in seconds) between batch status checks.
            delete_successful_batch_files (bool): Whether to delete input/output files from OpenAI
                after successful batch completion.
            delete_failed_batch_files (bool): Whether to delete input/error files from OpenAI
                after batch failure.
        """
        self.check_interval = check_interval
        self.working_dir = working_dir
        self.tracker = BatchStatusTracker()
        self.semaphore = asyncio.Semaphore(max_concurrent_batch_operations)
        self.delete_successful_batch_files = delete_successful_batch_files
        self.delete_failed_batch_files = delete_failed_batch_files
        self._batch_objects_file_lock = asyncio.Lock()
        self.batch_objects_file = f"{working_dir}/batch_objects_{self.client.api_key[-4:]}.jsonl"
        self.batch_submit_pbar: tqdm | None = None
        self.request_pbar: tqdm | None = None

    @abstractmethod
    async def submit_batch(
        self, requests: list[dict], metadata: Optional[dict] = None
    ) -> GenericBatchObject:
        """Needs to use self.semaphore"""
        pass

    @abstractmethod
    async def retrieve_batch(self, batch_id: str) -> GenericBatchObject:
        """Needs to use self.semaphore"""
        pass

    @abstractmethod
    async def cancel_batch(self, batch_id: str) -> GenericBatchObject:
        """Needs to use self.semaphore"""
        pass

    async def cancel_batches(self):
        if not os.path.exists(self.batch_objects_file):
            logger.warning("No batches to be cancelled, but cancel_batches=True.")
        else:
            logger.info(f"Batch objects file exists, cancelling all batches.")
            batch_ids = []
            with open(self.batch_objects_file, "r") as f:
                for line in f:
                    batch_obj = GenericBatchObject.model_validate_json(line.strip())
                    batch_ids.append(batch_obj.id)
            tasks = [self.cancel_batch(batch_id) for batch_id in batch_ids]
            await asyncio.gather(*tasks)

    async def submit_batch_from_request_file(
        self,
        request_file: str,
        requests_from_request_file_func: Callable = requests_from_api_specific_request_file,
    ):
        """
        Submits a batch from a request file.

        Args:
            request_file (str): Path to the file containing requests
            requests_from_request_file_func (Callable): Function to parse requests from file

        Side Effects:
            - Updates batch submission progress bar
            - Updates tracker with submitted batch status
        """
        metadata = {"request_file_name": request_file}
        requests = requests_from_request_file_func(request_file)
        batch_object = await self.submit_batch(requests, metadata)
        self.tracker.mark_as_submitted(request_file, batch_object, len(requests))
        with open(self.submitted_batch_objects_file, "a") as f:
            json.dump(batch_object.model_dump(), f, default=str)
            f.write("\n")
            f.flush()
        self.batch_submit_pbar.update(1)

    async def track_already_submitted_batches(self):
        """
        Tracks previously submitted batches from the submitted batch objects file.
        We need to check all submitted batch objects files because we might be looking at a cancelled batch
        or a batch from another key but same project.

        Side Effects:
            - Updates tracker with previously submitted batch statuses
        """
        all_submitted_batches_files = set(
            glob.glob(f"{self.working_dir}/batch_objects_submitted_*.jsonl")
        )

        existing_submitted_batches = {}
        for submitted_batch_objects_file in all_submitted_batches_files:
            logger.info(
                f"Processing submitted batch objects file: {submitted_batch_objects_file} Your API key is ***{self.client.api_key[-4:]}."
            )
            with open(submitted_batch_objects_file, "r") as f:
                for line in f:
                    batch_object = Batch.model_validate(json.loads(line))
                    request_file_name = batch_object.metadata["request_file_name"]
                    logger.debug(
                        f"Already submitted batch {batch_object.id} for request file {request_file_name}. "
                        f"Getting batch object to update tracker."
                    )
                    try:
                        batch_object = await self.retrieve_batch(batch_object.id)
                    except NotFoundError:
                        logger.warning(
                            f"Already submitted batch object {batch_object.id} not found. This might be fine since we might be "
                            "looking at a batch object submitted by another project. Will ignore this batch object..."
                        )
                        continue

                    # We skip the batch if it has a status that means it can no longer be used.
                    if batch_object.status in ["expired", "cancelling", "cancelled"]:
                        logger.info(
                            f"Batch {batch_object.id} has status {batch_object.status}, which means it can "
                            "no longer be used. Will ignore this batch object..."
                        )
                        continue

                    # Edge case where the batch is still validating, and we need to know the total number of requests
                    if batch_object.status == "validating":
                        n_requests = len(open(request_file_name, "r").readlines())
                        batch_object.request_counts.total = n_requests
                    else:
                        n_requests = batch_object.request_counts.total

                    # For each request file, we only want to keep the latest batch object.
                    if (
                        request_file_name not in existing_submitted_batches
                        or existing_submitted_batches[request_file_name].created_at
                        < batch_object.created_at
                    ):
                        existing_submitted_batches[request_file_name] = batch_object

        for request_file_name, batch_object in existing_submitted_batches.items():

            output_file_id = batch_object.output_file_id
            if output_file_id is not None:
                try:
                    await self.client.files.retrieve(output_file_id)
                except NotFoundError:
                    logger.warning(
                        f"Output file {output_file_id} exists in batch object but cannot be found "
                        "in OpenAI storage. The file may have been deleted. Will resubmit this batch..."
                    )
                    continue

            if request_file_name in self.tracker.unsubmitted_request_files:
                self.tracker.mark_as_submitted(request_file_name, batch_object, n_requests)
            else:
                response_file = request_file_to_response_file(request_file_name, self.working_dir)
                if not os.path.exists(response_file):
                    raise ValueError(
                        f"While processing {batch_object.id}, we found that its corresponding request_file_name {request_file_name} is "
                        f"not in tracker.unsubmitted_request_files, but its corresponding response_file {response_file} does not exist. "
                        f"This is an invalid state. \n"
                        f"batch_object: {batch_object} \n"
                        f"request_file_name: {request_file_name} \n"
                        f"tracker.unsubmitted_request_files: {self.tracker.unsubmitted_request_files} \n"
                        f"tracker.submitted_batches: {self.tracker.submitted_batches} \n"
                        f"tracker.downloaded_batches: {self.tracker.downloaded_batches} \n"
                    )

        if self.tracker.n_submitted_batches > 0:
            logger.info(
                f"{self.tracker.n_submitted_batches:,} out of {self.tracker.n_total_batches - self.tracker.n_downloaded_batches:,} remaining batches are already submitted."
            )

    async def track_already_downloaded_batches(self):
        """
        Tracks previously downloaded batches from the downloaded batch objects files.

        Side Effects:
            - Updates tracker with previously downloaded batch statuses
        """
        downloaded_batch_object_files = set(
            glob.glob(f"{self.working_dir}/batch_objects_downloaded_*.jsonl")
        )
        for downloaded_batch_object_file in downloaded_batch_object_files:
            logger.info(
                f"Processing downloaded batch objects file: {downloaded_batch_object_file} Your API key is ***{self.client.api_key[-4:]}."
            )
            with open(downloaded_batch_object_file, "r") as f:
                for line in f:
                    batch_object = Batch.model_validate(json.loads(line))
                    request_file = batch_object.metadata["request_file_name"]
                    response_file = request_file_to_response_file(request_file, self.working_dir)
                    assert (
                        request_file in self.tracker.unsubmitted_request_files
                    ), f"request_file {request_file} not in unsubmitted_request_files: {self.tracker.unsubmitted_request_files}"
                    if not os.path.exists(response_file):
                        logger.warning(
                            f"Downloaded batch object {batch_object.id} has a response_file {response_file} that does not exist. "
                            "Will resubmit this batch..."
                        )
                        continue

                    self.tracker.mark_as_submitted(
                        request_file, batch_object, batch_object.request_counts.total
                    )
                    self.tracker.mark_as_finished(batch_object)
                    self.tracker.mark_as_downloaded(batch_object)

        if self.tracker.n_downloaded_batches > 0:
            logger.info(
                f"{self.tracker.n_downloaded_batches:,} out of {self.tracker.n_total_batches:,} batches already downloaded."
            )

    async def submit_batches_from_request_files(
        self,
        request_files: set[str],
        requests_from_request_file_func: Callable = requests_from_api_specific_request_file,
    ):
        """
        Manages the submission of multiple request files as batches.

        Args:
            request_files (set[str]): Set of paths to request files to process
            requests_from_request_file_func (Callable): Function to parse requests from files

        Side Effects:
            - Updates tracker with batch statuses
            - Creates and updates batch submission progress bar
        """
        self.tracker.unsubmitted_request_files = request_files
        await self.track_already_downloaded_batches()
        await self.track_already_submitted_batches()
        # exit early
        if self.tracker.n_unsubmitted_request_files == 0:
            return

        # submit remaining batches
        self.batch_submit_pbar = tqdm(
            total=self.tracker.n_total_batches,
            desc="Submitting batches",
            unit="batch",
            initial=self.tracker.n_submitted_finished_or_downloaded_batches,
        )
        tasks = [
            self.submit_batch_from_request_file(f, requests_from_request_file_func)
            for f in self.tracker.unsubmitted_request_files
        ]
        await asyncio.gather(*tasks)
        self.batch_submit_pbar.close()
        assert self.tracker.unsubmitted_request_files == set()
        logger.debug(
            f"All batch objects submitted and written to {self.submitted_batch_objects_file}"
        )

    async def check_batch_status(self, batch_id: str) -> Batch | None:
        """
        Checks the current status of a batch job.

        Args:
            batch_id (str): The ID of the batch to check

        Returns:
            Batch | None: The batch object if completed (including failures), None if in progress

        Side Effects:
            - Updates tracker with current batch status
            - Updates request completion counts
        """
        async with self.semaphore:
            batch = await self.client.batches.retrieve(batch_id)
            self.tracker.update_submitted(batch)

            n_completed_requests = batch.request_counts.completed
            n_failed_requests = batch.request_counts.failed
            n_total_requests = batch.request_counts.total

            logger.debug(
                f"Batch {batch.id} status: {batch.status} requests: "
                f"{n_completed_requests}/{n_failed_requests}/{n_total_requests} "
                "completed/failed/total"
            )

            finished_statuses = ["completed", "failed", "expired", "cancelled"]
            batch_returned = batch.status in finished_statuses
            if not self._validate_batch_status(batch.status):
                logger.warning(f"Unknown batch status: {batch.status}")

            if batch_returned:
                logger.debug(f"Batch {batch.id} returned with status: {batch.status}")
                self.tracker.mark_as_finished(batch)
                return batch

    async def poll_and_process_batches(
        self,
        response_file_from_responses_func: Callable = api_specific_response_file_from_responses,
    ) -> None:
        """Monitors and processes batches until all are completed.

        Continuously polls the status of submitted batches and downloads their results
        when complete. Handles successful completions, failures, expirations, and
        cancellations. Progress is tracked via a progress bar showing completed requests.

        Returns:
            None

        Raises:
            RuntimeError: If none of the submitted batches complete successfully.

        Side Effects:
            - Updates the batch tracker state
            - Creates response files for completed batches
            - Creates and updates requests progress bar
        """
        # progress bar for finished requests
        self.request_pbar = tqdm(
            total=self.tracker.n_total_requests,
            desc="Finished requests in batches",
            unit="request",
            initial=self.tracker.n_finished_or_downloaded_requests,
        )

        # loop until all batches have been returned
        all_response_files = []
        while len(self.tracker.submitted_batches) > 0:
            # check batch status also updates the tracker
            status_tasks = [
                self.check_batch_status(batch_id) for batch_id in self.tracker.submitted_batches
            ]
            batches_to_download = await asyncio.gather(*status_tasks)
            batches_to_download = filter(None, batches_to_download)

            # update progress bari
            self.request_pbar.n = self.tracker.n_finished_or_downloaded_requests
            self.request_pbar.refresh()

            download_tasks = [
                self.download_batch_to_response_file(batch, response_file_from_responses_func)
                for batch in batches_to_download
            ]
            # Failed downloads return None and print any errors that occurred
            all_response_files.extend(await asyncio.gather(*download_tasks))
            if self.tracker.n_finished_or_downloaded_requests < self.tracker.n_total_requests:
                logger.debug(
                    f"Batches returned: {self.tracker.n_finished_or_downloaded_batches:,}/{self.tracker.n_total_batches:,} "
                    f"Requests completed: {self.tracker.n_finished_or_downloaded_requests:,}/{self.tracker.n_total_requests:,}"
                )
                logger.debug(f"Sleeping for {self.check_interval} seconds...")
                await asyncio.sleep(self.check_interval)

        self.request_pbar.close()
        response_files = filter(None, all_response_files)
        if len(self.tracker.downloaded_batches) == 0 or not response_files:
            raise RuntimeError(
                "None of the submitted batches completed successfully. "
                "Please check the logs above and https://platform.openai.com/batches for errors."
            )

    async def delete_file(self, file_id: str, semaphore: asyncio.Semaphore):
        """
        Deletes a file from OpenAI's storage.

        Args:
            file_id (str): The ID of the file to delete
            semaphore (asyncio.Semaphore): Semaphore to limit concurrent operations
        """
        async with semaphore:
            try:
                delete_response = await self.client.files.delete(file_id)
                if delete_response.deleted:
                    logger.debug(f"Deleted file {file_id}")
                else:
                    logger.warning(f"Failed to delete file {file_id}")
            except NotFoundError:
                # This is fine, the file may have been deleted already. Deletion should be best-effort.
                logger.warning(f"Trying to delete file {file_id} but it was not found.")

    async def download_batch(self, batch: Batch) -> str | None:
        file_content = None
        async with self.semaphore:
            # Completed batches have an output file
            if batch.status == "completed" and batch.output_file_id:
                file_content = await self.client.files.content(batch.output_file_id)
                logger.debug(f"Batch {batch.id} completed and downloaded")

            # Failed batches with an error file
            elif batch.status == "failed" and batch.error_file_id:
                file_content = await self.client.files.content(batch.error_file_id)
                logger.warning(f"Batch {batch.id} failed\n. Errors will be parsed below.")
                if self.delete_failed_batch_files:
                    await self.delete_file(batch.input_file_id, self.semaphore)
                    await self.delete_file(batch.error_file_id, self.semaphore)

            # Failed batches without an error file
            elif batch.status == "failed" and not batch.error_file_id:
                errors = "\n".join([str(error) for error in batch.errors.data])
                logger.error(
                    f"Batch {batch.id} failed and likely failed validation. "
                    f"Batch errors: {errors}. "
                    f"Check https://platform.openai.com/batches/{batch.id} for more details."
                )
                if self.delete_failed_batch_files:
                    await self.delete_file(batch.input_file_id, self.semaphore)

            # Cancelled or expired batches
            elif batch.status == "cancelled" or batch.status == "expired":
                logger.warning(f"Batch {batch.id} was cancelled or expired")
                if self.delete_failed_batch_files:
                    await self.delete_file(batch.input_file_id, self.semaphore)

        return file_content

    async def download_batch_to_response_file(
        self,
        batch: Batch,
        response_file_from_responses_func: Callable = api_specific_response_file_from_responses,
    ) -> str | None:
        """
        Downloads and processes the results of a completed batch.

        Handles successful completions, failures, and error cases. Converts API-specific
        responses to generic responses and calculates costs.

        Args:
            batch (Batch): The completed batch object to process

        Returns:
            str | None: Path to the response file if successful, None if batch failed

        Side Effects:
            - Creates response file with processed results
            - Updates batch tracking state
            - Appends batch object to downloaded batch objects file
            - Optionally deletes batch files from OpenAI
        """
        file_content = await self.download_batch(batch)

        if file_content is None:
            return None

        request_file = batch.metadata["request_file_name"]
        response_file = request_file_to_response_file(request_file, self.working_dir)
        response_file_from_responses_func(file_content, batch, response_file)

        logger.debug(f"Batch {batch.id} written to {response_file}")

        # Simplified file writing
        with open(self.downloaded_batch_objects_file, "a") as f:
            json.dump(batch.model_dump(), f, default=str)
            f.write("\n")
            f.flush()

        logger.debug(f"Batch {batch.id} written to {self.downloaded_batch_objects_file}")

        if self.delete_successful_batch_files:
            await self.delete_file(batch.input_file_id, self.semaphore)
            await self.delete_file(batch.output_file_id, self.semaphore)

        self.tracker.mark_as_downloaded(batch)

        return response_file

    def requests_from_api_specific_request_file(self, request_file: str) -> list[dict]:
        with open(request_file, "r") as file:
            return file.read().splitlines()

    def api_specific_response_file_from_responses(
        responses: str, batch: GenericBatchObject, response_file: str
    ) -> str | None:
        open(response_file, "w").write(responses.text)
