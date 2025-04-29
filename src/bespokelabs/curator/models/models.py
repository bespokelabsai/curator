import typing as t
from .bespoke_backend import BespokeModelsBackend

class Models:
    """Factory for creating model clients based on the specified backend."""
    
    def __init__(self, backend: str = "bespoke", backend_params: dict = None):
        """Initialize the Models factory.
        
        Args:
            backend: The backend provider to use ('bespoke' supported)
            backend_params: Configuration parameters for the backend
        """
        self.backend = backend
        self.backend_params = backend_params or {}
        
        if backend == "bespoke":
            self.client = BespokeModelsBackend(
                base_url=self.backend_params.get("base_url"),
                api_key=self.backend_params.get("api_key")
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def list_models(self, job_id: str = None) -> t.List[dict]:
        """Delegate to the client's list_models method."""
        return self.client.list_models(job_id=job_id)
    
    def deploy_model(self, model_id: str) -> bool:
        """Delegate to the client's deploy_model method."""
        return self.client.deploy_model(model_id=model_id)
    
    def undeploy_model(self, model_id: str) -> bool:
        """Delegate to the client's undeploy_model method."""
        return self.client.undeploy_model(model_id=model_id)
