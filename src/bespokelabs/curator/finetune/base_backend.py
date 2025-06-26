from abc import ABC, abstractmethod


class BaseFinetuneBackend(ABC):
    """Abstract base class for finetuning backends."""

    def __init__(self, backend_params: dict):
        """Initializes the finetuning backend.

        Args:
            backend_params: A dictionary containing backend-specific parameters.
        """
        self.backend_params = backend_params

    @abstractmethod
    def create_job(self, *args, **kwargs):
        """Creates a new finetuning job."""
        pass

    @abstractmethod
    def list_jobs(self, *args, **kwargs):
        """Lists all finetuning jobs."""
        pass

    @abstractmethod
    def list_job_events(self, job_id: str):
        """Lists events for a specific finetuning job.

        Args:
            job_id: The ID of the finetuning job.
        """
        pass

    # @abstractmethod
    # def list_job_checkpoints(self, job_id: str):
    #     pass

    # @abstractmethod
    # def list_job_metrics(self, job_id: str):
    #     pass

    @abstractmethod
    def get_job_details(self, job_id: str):
        """Retrieves details for a specific finetuning job.

        Args:
            job_id: The ID of the finetuning job.
        """
        pass

    @abstractmethod
    def cancel_job(self, job_id: str):
        """Cancels a specific finetuning job.

        Args:
            job_id: The ID of the finetuning job.
        """
        pass
