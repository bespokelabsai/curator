from typing import Union

from bespokelabs.curator.finetune.bespoke_backend import BespokeFinetuneBackend
from bespokelabs.curator.finetune.openai_backend import OpenAIFinetuneBackend
from bespokelabs.curator.finetune.types import FinetuneBackendParams


class Finetune:
    """A class to manage finetuning jobs using different backends."""

    def __init__(self, backend: str, backend_params: Union[FinetuneBackendParams, dict]):
        """Initializes the Finetune class with a specified backend.

        Args:
            backend: The name of the backend to use (e.g., "bespoke", "openai").
            backend_params: Parameters for the backend.

        Raises:
            ValueError: If an invalid backend is specified.
        """
        self.backend_params = backend_params
        if backend == "bespoke":
            self._backend = BespokeFinetuneBackend(backend_params)
        elif backend == "openai":
            self._backend = OpenAIFinetuneBackend(backend_params)
        else:
            raise ValueError(f"Invalid backend: {backend}")

    def create_job(self, *args, **kwargs):
        """Creates a finetuning job using the configured backend."""
        return self._backend.create_job(*args, **kwargs)

    def list_jobs(self, *args, **kwargs):
        """Lists finetuning jobs using the configured backend."""
        return self._backend.list_jobs(*args, **kwargs)

    def list_job_events(self, *args, **kwargs):
        """Lists job events using the configured backend."""
        return self._backend.list_job_events(*args, **kwargs)

    def get_job_details(self, *args, **kwargs):
        """Gets job details using the configured backend."""
        return self._backend.get_job_details(*args, **kwargs)

    def cancel_job(self, *args, **kwargs):
        """Cancels a job using the configured backend."""
        return self._backend.cancel_job(*args, **kwargs)
