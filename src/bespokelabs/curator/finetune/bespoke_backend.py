import logging
import os
from typing import Union

import requests

from bespokelabs.curator.finetune.base_backend import BaseFinetuneBackend
from bespokelabs.curator.finetune.types import FinetuneBackendParams

logger = logging.getLogger(__name__)


class BespokeFinetuneBackend(BaseFinetuneBackend):
    """A client for interacting with the Bespoke Labs Finetuning API.

    This class provides methods to create, list, and manage finetuning jobs
    on the Bespoke Labs platform.
    """

    def __init__(self, backend_params: Union[FinetuneBackendParams, dict]):
        """Initializes the BespokeFinetuneBackend.

        Args:
            backend_params: A dictionary or FinetuneBackendParams object containing
                            configuration for the backend, such as base_url and env.
        """
        super().__init__(backend_params)
        # Determine the environment suffix for the API URL.
        self.env = "-dev" if backend_params.get("env", "prod") == "dev" else ""
        # Set the base URL for the finetuning API.
        if not backend_params.get("base_url"):
            self.base_url = f"https://api{self.env}.bespokelabs.com/v0/finetune"
        else:
            self.base_url = backend_params["base_url"] + "/v0/finetune"
        # Retrieve the API key from environment variables.
        self.api_key = os.environ.get("BESPOKE_API_KEY")

    def create_job(self, *args, **kwargs):
        """Creates a new finetuning job.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments. Expected keys include:
                      'model_name': The name of the model to finetune.
                      'dataset_id': The ID of the dataset for training.
                      'seed': The random seed for reproducibility.
                      'suffix': A suffix to append to the finetuned model name.
                      'method': The finetuning method and its hyperparameters.
                      'job_name': A name for the finetuning job.
                      'num_gpus': The number of GPUs to use for finetuning.

        Returns:
            dict: The JSON response from the API, typically containing job details.
        """
        url = f"{self.base_url}/jobs/create"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        # Warn if Hugging Face token is not set, as it might be needed for private resources.
        if os.environ.get("HF_TOKEN") is None:
            logger.warning("HF_TOKEN is not set. Please set it to use private gated models or datasets.")

        # Prepare the data payload for the API request.
        data = {
            "model_name": kwargs["model_name"],
            "dataset_id": kwargs["dataset_id"],
            "seed": kwargs["seed"],
            "suffix": kwargs["suffix"],
            "method": kwargs["method"],
            "job_name": kwargs["job_name"],
            "num_gpus": kwargs["num_gpus"],
            "secrets": {
                "HF_TOKEN": os.environ.get("HF_TOKEN", None),  # Include HF_TOKEN if available.
            },
        }

        response = requests.post(url, headers=headers, json=data)
        return response.json()

    def list_jobs(self, *args, **kwargs):
        """Lists all finetuning jobs.

        Args:
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments (currently unused).

        Returns:
            dict: The JSON response from the API, typically a list of jobs.
        """
        url = f"{self.base_url}/jobs"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        response = requests.get(url, headers=headers)
        return response.json()

    def list_job_events(self, *args, **kwargs):
        """Lists events for a specific finetuning job.

        Args:
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments. Expected key:
                      'job_id': The ID of the job to retrieve events for.

        Returns:
            dict: The JSON response from the API, typically a list of job events.
        """
        url = f"{self.base_url}/jobs/{kwargs['job_id']}/events"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        response = requests.get(url, headers=headers)
        return response.json()

    def get_job_details(self, *args, **kwargs):
        """Retrieves details for a specific finetuning job.

        Args:
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments. Expected keys:
                      'job_id': The ID of the job to retrieve details for.
                      'type' (optional): The type of logs to retrieve (e.g., 'EVENTS').
                                         Defaults to 'EVENTS'.

        Returns:
            dict: The JSON response from the API, containing job details.
        """
        # Construct URL with optional log_type parameter.
        url = f"{self.base_url}/jobs/{kwargs['job_id']}?log_type={kwargs.get('type', 'EVENTS')}"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        response = requests.get(url, headers=headers)
        return response.json()

    def cancel_job(self, *args, **kwargs):
        """Cancels a specific finetuning job.

        Args:
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments. Expected key:
                      'job_id': The ID of the job to cancel.

        Returns:
            dict: The JSON response from the API, typically confirming cancellation.
        """
        url = f"{self.base_url}/jobs/{kwargs['job_id']}/cancel"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        response = requests.post(url, headers=headers)
        return response.json()
