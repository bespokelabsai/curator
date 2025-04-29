import typing as t
from abc import ABC, abstractmethod

class BaseModelsBackend(ABC):
    """Base interface for model operations."""

    @abstractmethod
    def list_models(self, job_id: str = None) -> t.List[dict]:
        """List available models or models related to a specific job."""
        pass

    @abstractmethod
    def deploy_model(self, model_id: str) -> bool:
        """Deploy a model to make it available for inference."""
        pass

    @abstractmethod
    def undeploy_model(self, model_id: str) -> bool:
        """Undeploy a model to free up resources."""
        pass

