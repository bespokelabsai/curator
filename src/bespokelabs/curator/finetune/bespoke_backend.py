from bespokelabs.curator.finetune.base_backend import BaseFinetuneBackend
import requests
import json
import os
import logging

logger = logging.getLogger(__name__)

class BespokeFinetuneBackend(BaseFinetuneBackend):

    def __init__(self, backend_params: dict):
        super().__init__(backend_params)
        self.env = "dev" if backend_params.get("env", "prod") == "dev" else "prod"
        if not backend_params.get("base_url"):
            self.base_url = f"https://api.bespokelabs.com/{self.env}/v0/finetune"
        else:
            self.base_url = backend_params["base_url"] + "/v0/finetune" 
        self.api_key = os.environ.get("BESPOKE_API_KEY")


    def create_job(self, *args, **kwargs):
        url = f"{self.base_url}/jobs/create"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        if os.environ.get("HF_TOKEN") is None:
            logger.warning("HF_TOKEN is not set. Please set it to use private gated models or datasets.")

        data = {
            "model_name": kwargs["model_name"],
            "dataset_id": kwargs["dataset_id"],
            "seed": kwargs["seed"],
            "suffix": kwargs["suffix"],
            "method": kwargs["method"],
            "job_name": kwargs["job_name"],
            "secrets": {
                'HF_TOKEN': os.environ.get("HF_TOKEN", None),
            }
        }

        response = requests.post(url, headers=headers, json=data)
        return response.json()


    def list_jobs(self, *args, **kwargs):

        url = f"{self.base_url}/jobs"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.get(url, headers=headers)
        return response.json()

    def list_job_events(self, *args, **kwargs):

        url = f"{self.base_url}/jobs/{kwargs['job_id']}/events"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.get(url, headers=headers)
        return response.json()


    def get_job_details(self, *args, **kwargs):

        url = f"{self.base_url}/jobs/{kwargs['job_id']}?log_type={kwargs.get('type', 'EVENTS')}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.get(url, headers=headers)
        return response.json()

    def cancel_job(self, *args, **kwargs):
        
        url = f"{self.base_url}/jobs/{kwargs['job_id']}/cancel"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.post(url, headers=headers)
        return response.json()
