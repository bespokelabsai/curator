import os
import typing as t

import requests

from ..log import logger
from .base_backend import BaseModelsBackend


class BespokeModelsBackend(BaseModelsBackend):
    """Client for interacting with Bespoke models API."""

    def __init__(self, base_url: str = None, api_key: str = None):
        """Initialize the Bespoke models client.

        Args:
            base_url: Base URL for the API
            api_key: API key for authentication (optional)
        """
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("BESPOKE_API_KEY")
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def list_models(self, job_id: str = None) -> t.List[dict]:
        """List available models or models related to a specific job.

        Args:
            job_id: Optional job ID to filter models by

        Returns:
            A list of model information dictionaries
        """
        url = f"{self.base_url}/v0/models/{job_id}"

        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to list models: {response.status_code}, {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []

    def deploy_model(self, model_id: str) -> bool:
        """Deploy a model to make it available for inference.

        Args:
            model_id: ID of the model to deploy

        Returns:
            True if deployment was successful, False otherwise
        """
        url = f"{self.base_url}/v0/models/{model_id}/deploy"

        try:
            response = requests.post(url, headers=self.headers)
            if response.status_code in [200, 201, 202]:
                return response.json()
            else:
                logger.error(f"Failed to deploy model: {response.status_code}, {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error deploying model: {str(e)}")
            return False

    def undeploy_model(self, model_id: str) -> bool:
        """Undeploy a model to free up resources.

        Args:
            model_id: ID of the model to undeploy

        Returns:
            True if undeployment was successful, False otherwise
        """
        url = f"{self.base_url}/v0/models/{model_id}/undeploy"

        try:
            response = requests.post(url, headers=self.headers)
            if response.status_code in [200, 202, 204]:
                return response.json()
            else:
                logger.error(f"Failed to undeploy model: {response.status_code}, {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error undeploying model: {str(e)}")
            return False
