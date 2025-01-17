"""
Module for handling batch requests to Google's Gemini API via Vertex AI.

This module provides a request processor for submitting batch requests to Gemini models
using Google Cloud's Vertex AI batch prediction API. It supports both Cloud Storage
and BigQuery for input/output configurations.
"""

import asyncio
import datetime
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional, cast, Sequence

import aiofiles
from google.cloud import aiplatform
from datasets import Dataset
from tqdm import tqdm

from bespokelabs.curator.prompter.prompt_formatter import PromptFormatter
from bespokelabs.curator.request_processor.base_request_processor import (
    BaseRequestProcessor,
    GenericRequest,
    GenericResponse,
    parse_response_message,
)
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop

logger = logging.getLogger(__name__)


@dataclass
class BatchStatusTracker:
    """Tracks the status of batch prediction jobs."""
    
    # BATCHES
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
    n_completed_in_progress_requests: int = 0
    n_failed_in_progress_requests: int = 0

# Default values and constants
DEFAULT_LOCATION = "us-central1"
DEFAULT_CHECK_INTERVAL = 10  # seconds
MAX_CONCURRENT_JOBS = 4  # Vertex AI default quota





class GeminiBatchRequestProcessor(BaseRequestProcessor):
    """
    Request processor for handling batch requests to Gemini models via Vertex AI.

    This processor implements batch processing using Google Cloud's Vertex AI
    batch prediction API. It supports both Cloud Storage and BigQuery for
    input/output configurations.

    Args:
        batch_size (int): Number of requests to include in each batch.
        model (str): Gemini model identifier (e.g., "gemini-1.5-pro-001").
        project_id (str): Google Cloud project ID.
        location (str, optional): Google Cloud region. Defaults to "us-central1".
        gcs_input_uri_prefix (str, optional): Cloud Storage URI prefix for input files.
        gcs_output_uri_prefix (str, optional): Cloud Storage URI prefix for output files.
        check_interval (int, optional): Interval in seconds to check job status. Defaults to 10.
        api_key (str, optional): Gemini API key. Defaults to GEMINI_API_KEY environment variable.
    """

    def __init__(
        self,
        batch_size: int,
        model: str,
        project_id: str,
        location: str = DEFAULT_LOCATION,
        gcs_input_uri_prefix: Optional[str] = None,
        gcs_output_uri_prefix: Optional[str] = None,
        check_interval: int = DEFAULT_CHECK_INTERVAL,
        api_key: Optional[str] = None,
    ):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Please set the GEMINI_API_KEY environment variable or provide it in the constructor."
            )
        if not project_id:
            raise ValueError("Google Cloud project ID is required.")
        
        super().__init__(batch_size)
        self.model: str = model
        self.project_id: str = project_id
        self.location: str = location
        self.gcs_input_uri_prefix: Optional[str] = gcs_input_uri_prefix
        self.gcs_output_uri_prefix: Optional[str] = gcs_output_uri_prefix
        self.check_interval: int = check_interval
        self.api_key: str = api_key

        # Validate model name format
        valid_models = [
            "gemini-1.5-pro-001",
            "gemini-1.5-pro-002",
            "gemini-1.5-flash-001",
            "gemini-1.5-flash-002",
            "gemini-1.0-pro-001",
            "gemini-1.0-pro-002",
        ]
        if model not in valid_models:
            raise ValueError(
                f"Invalid model name: {model}. Must be one of: {', '.join(valid_models)}"
            )





    def get_rate_limits(self) -> dict:
        """Returns the rate limits for the Gemini API via Vertex AI.

        Returns:
            dict: A dictionary containing the rate limit information,
                  specifically the maximum number of concurrent batch jobs.
        """
        return {"max_concurrent_jobs": MAX_CONCURRENT_JOBS}

    def create_api_specific_request(self, generic_request: GenericRequest) -> dict:
        """Creates a Vertex AI batch prediction API-specific request body from a GenericRequest.

        Args:
            generic_request (GenericRequest): The generic request to convert.

        Returns:
            dict: API specific request body for Vertex AI batch prediction.
        """
        # Build structured JSON for Vertex AI batch job
        display_name = f"curator-gemini-job-{generic_request.original_row_idx}"
        
        # Convert model name to Vertex AI format
        vertex_model = f"publishers/google/models/{self.model}"
        
        # Prepare the request body
        body = {
            "displayName": display_name,
            "model": vertex_model,
            "inputConfig": {
                "instancesFormat": "jsonl",
                # We'll set the actual GCS URI when submitting the batch
                "gcsSource": {
                    "uris": [f"{self.gcs_input_uri_prefix}/requests_{generic_request.original_row_idx}.jsonl"]
                } if self.gcs_input_uri_prefix else None
            },
            "outputConfig": {
                "predictionsFormat": "jsonl",
                "gcsDestination": {
                    "outputUriPrefix": self.gcs_output_uri_prefix
                } if self.gcs_output_uri_prefix else None
            }
        }

        # Add any model-specific parameters from the generic request
        if generic_request.response_format:
            body["parameters"] = {
                "response_format": generic_request.response_format
            }
        
        # Include the messages in the format expected by Gemini
        instances = [{
            "prompt": {
                "messages": generic_request.messages
            }
        }]
        
        return {
            "instances": instances,
            "parameters": body
        }

    async def asubmit_batch(self, batch_file: str) -> dict:
        """Submit a batch file to Vertex AI for processing.

        Args:
            batch_file (str): Path to the JSONL file containing the batch requests.

        Returns:
            dict: The batch job information from Vertex AI.
        """
        # Read and prepare the batch file content
        async with aiofiles.open(batch_file, "r") as file:
            content = await file.read()
            requests = [json.loads(line) for line in content.split('\n') if line.strip()]

        # Initialize Vertex AI client
        aiplatform.init(
            project=self.project_id,
            location=self.location,
        )

        # Create a batch prediction job
        job_display_name = f"curator-gemini-batch-{os.path.basename(batch_file)}"
        
        try:
            batch_prediction_job = aiplatform.BatchPredictionJob.create(
                job_display_name=job_display_name,
                model_name=f"publishers/google/models/{self.model}",
                instances_format="jsonl",
                predictions_format="jsonl",
                gcs_source=[f"{self.gcs_input_uri_prefix}/{os.path.basename(batch_file)}"],
                gcs_destination_prefix=f"{self.gcs_output_uri_prefix}/responses_{os.path.basename(batch_file)}",
                sync=False  # Run asynchronously
            )
            
            # Return job information as a dictionary
            return {
                "id": batch_prediction_job.resource_name,
                "display_name": job_display_name,
                "state": batch_prediction_job.state.name,
                "create_time": batch_prediction_job.create_time.isoformat(),
                "metadata": {
                    "request_file_name": batch_file,
                    "model": self.model,
                    "input_uri": f"{self.gcs_input_uri_prefix}/{os.path.basename(batch_file)}",
                    "output_uri": f"{self.gcs_output_uri_prefix}/responses_{os.path.basename(batch_file)}"
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to submit batch job for {batch_file}: {str(e)}")
            raise

    class BatchWatcher:
        """Watches and manages Vertex AI batch prediction jobs."""

        def __init__(
            self,
            working_dir: str,
            check_interval: int,
            prompt_formatter: PromptFormatter,
            n_submitted_requests: int,
            project_id: str,
            location: str,
        ) -> None:
            self.working_dir = working_dir
            self.check_interval = check_interval
            self.prompt_formatter = prompt_formatter
            self.tracker = BatchStatusTracker()
            self.tracker.n_submitted_requests = n_submitted_requests
            
            # Initialize Vertex AI
            aiplatform.init(project=project_id, location=location)
            
            # Load batch objects
            with open(f"{working_dir}/batch_objects.jsonl", "r") as f:
                self.batch_objects = [json.loads(line) for line in f]
            self.batch_ids = [obj["id"] for obj in self.batch_objects]
            self.tracker.n_submitted_batches = len(self.batch_ids)
            self.remaining_batch_ids = set(self.batch_ids)

        async def check_batch_status(self, batch_id: str) -> Optional[dict]:
            """Check the status of a batch prediction job.

            Args:
                batch_id (str): The resource name of the batch prediction job.

            Returns:
                Optional[dict]: The batch object if completed, None if still running.
            """
            try:
                job = aiplatform.BatchPredictionJob.get(batch_id)
                state = job.state.name

                logger.debug(f"Batch {batch_id} status: {state}")

                batch_returned = False
                if state == "JOB_STATE_SUCCEEDED":
                    self.tracker.n_completed_batches += 1
                    batch_returned = True
                elif state == "JOB_STATE_FAILED":
                    self.tracker.n_failed_batches += 1
                    batch_returned = True
                elif state == "JOB_STATE_CANCELLED":
                    self.tracker.n_cancelled_batches += 1
                    batch_returned = True
                elif state == "JOB_STATE_EXPIRED":
                    self.tracker.n_expired_batches += 1
                    batch_returned = True

                if batch_returned:
                    logger.info(f"Batch {batch_id} returned with status: {state}")
                    self.tracker.n_returned_batches += 1
                    self.remaining_batch_ids.remove(batch_id)
                    
                    # Get completion counts from job
                    completed_count = job.completed_count
                    error_count = job.error_count
                    
                    self.tracker.n_completed_returned_requests += completed_count
                    self.tracker.n_failed_returned_requests += error_count
                    
                    return next(obj for obj in self.batch_objects if obj["id"] == batch_id)
                else:
                    # Update in-progress counts
                    self.tracker.n_completed_in_progress_requests = job.completed_count
                    self.tracker.n_failed_in_progress_requests = job.error_count
                    return None

            except Exception as e:
                logger.error(f"Error checking batch status for {batch_id}: {str(e)}")
                return None

        async def download_batch_results(self, batch_obj: dict) -> Optional[str]:
            """Download results from a completed batch job.

            Args:
                batch_obj (dict): The batch object containing job information.

            Returns:
                Optional[str]: Path to the downloaded responses file, or None if download failed.
            """
            try:
                job = aiplatform.BatchPredictionJob.get(batch_obj["id"])
                
                # Get the output URI from job metadata
                output_uri = job.output_info.gcs_output_directory
                
                if not output_uri:
                    logger.error(f"No output URI found for batch {batch_obj['id']}")
                    return None
                
                # Download results from GCS
                from google.cloud import storage
                
                client = storage.Client()
                bucket_name = output_uri.split("/")[2]
                prefix = "/".join(output_uri.split("/")[3:])
                
                bucket = client.bucket(bucket_name)
                blobs = bucket.list_blobs(prefix=prefix)
                
                # Create responses file
                responses_file = os.path.join(
                    self.working_dir,
                    f"responses_{os.path.basename(batch_obj['metadata']['request_file_name'])}",
                )
                
                # Download and process each prediction file
                async with aiofiles.open(responses_file, "w") as f:
                    for blob in blobs:
                        if not blob.name.endswith(".jsonl"):
                            continue
                            
                        content = blob.download_as_text()
                        for line in content.splitlines():
                            response_data = json.loads(line)
                            
                            # Convert to GenericResponse format
                            generic_response = {
                                "generic_request": response_data["instance"],
                                "response_message": response_data["prediction"],
                                "response_errors": None if "error" not in response_data else [response_data["error"]],
                            }
                            
                            await f.write(json.dumps(generic_response) + "\n")
                
                logger.info(f"Downloaded batch results to {responses_file}")
                return responses_file
                
            except Exception as e:
                logger.error(f"Failed to download results for batch {batch_obj['id']}: {str(e)}")
                return None

        async def watch(self) -> None:
            """Monitor batch jobs until all complete."""
            pbar = tqdm(
                total=self.tracker.n_submitted_requests,
                desc="Completed Gemini requests in batches",
                unit="request",
            )

            all_response_files = []

            while self.remaining_batch_ids:
                # Reset in-progress counts
                self.tracker.n_completed_in_progress_requests = 0
                self.tracker.n_failed_in_progress_requests = 0

                # Check all remaining batch statuses
                status_tasks = [
                    self.check_batch_status(batch_id)
                    for batch_id in self.remaining_batch_ids
                ]
                completed_batches = await asyncio.gather(*status_tasks)
                completed_batches = list(filter(None, completed_batches))

                # Download results from completed batches
                download_tasks = [
                    self.download_batch_results(batch)
                    for batch in completed_batches
                ]
                response_files = await asyncio.gather(*download_tasks)
                all_response_files.extend(response_files)

                # Update progress bar
                pbar.n = (
                    self.tracker.n_completed_returned_requests
                    + self.tracker.n_failed_returned_requests
                    + self.tracker.n_completed_in_progress_requests
                    + self.tracker.n_failed_in_progress_requests
                )
                pbar.refresh()

                if self.remaining_batch_ids:
                    logger.debug(
                        f"Batches returned: {self.tracker.n_returned_batches}/{self.tracker.n_submitted_batches} "
                        f"Requests completed: {pbar.n}/{self.tracker.n_submitted_requests}"
                    )
                    logger.debug(f"Sleeping for {self.check_interval} seconds...")
                    await asyncio.sleep(self.check_interval)

            pbar.close()

            # Check if any batches completed successfully
            response_files = list(filter(None, all_response_files))
            if self.tracker.n_completed_batches == 0 or not response_files:
                raise RuntimeError(
                    "None of the submitted batches completed successfully. "
                    "Please check the logs above for errors."
                )

    def run(
        self,
        dataset: Dataset,
        working_dir: str,
        parse_func_hash: str,
        prompt_formatter: PromptFormatter,
    ) -> Dataset:
        """Uses the Vertex AI API to process batch predictions using Gemini models.

        Args:
            dataset (Dataset): Dataset that is being mapped over
            working_dir (str): Working directory to save files (requests.jsonl, responses.jsonl)
            parse_func_hash (str): Hash of the parse_func to be used as the dataset file name
            prompt_formatter (PromptFormatter): Prompt formatter to be used to format the prompt

        Returns:
            Dataset: Completed dataset with model responses
        """
        if not self.gcs_input_uri_prefix or not self.gcs_output_uri_prefix:
            raise ValueError(
                "Both gcs_input_uri_prefix and gcs_output_uri_prefix must be provided for batch processing"
            )

        # Create request files using parent class method
        requests_files = self.create_request_files(dataset, working_dir, prompt_formatter)
        batch_objects_file = f"{working_dir}/batch_objects.jsonl"

        if os.path.exists(batch_objects_file):
            logger.warning(
                f"Batch objects file already exists, skipping batch submission and resuming: {batch_objects_file}"
            )
        else:
            # Submit all batches
            async def submit_all_batches():
                tasks = [self.asubmit_batch(requests_files[i]) for i in range(len(requests_files))]
                return await asyncio.gather(*tasks)

            batch_objects = run_in_event_loop(submit_all_batches())

            # Save batch job information
            with open(batch_objects_file, "w") as f:
                for obj in batch_objects:
                    f.write(json.dumps(obj, default=str) + "\n")
            logger.info(f"Batch objects written to {batch_objects_file}")

        # Watch batch jobs until completion
        n_submitted_requests = 1 if dataset is None else len(dataset)
        batch_watcher = self.BatchWatcher(
            working_dir=working_dir,
            check_interval=self.check_interval,
            prompt_formatter=prompt_formatter,
            n_submitted_requests=n_submitted_requests,
            project_id=self.project_id,
            location=self.location,
        )

        async def watch_batches():
            await batch_watcher.watch()

        run_in_event_loop(watch_batches())
        
        # Create the final dataset
        output_dataset = self.create_dataset_files(working_dir, parse_func_hash, prompt_formatter)
        if output_dataset is None:
            raise RuntimeError("Failed to create output dataset from responses")
        
        return cast(Dataset, output_dataset)
