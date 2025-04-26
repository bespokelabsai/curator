from abc import ABC

from bespokelabs.curator.finetune.bespoke_backend import BespokeFinetuneBackend
from bespokelabs.curator.finetune.openai_backend import OpenAIFinetuneBackend


class Finetune(ABC):
    def __init__(self, backend: str, backend_params: dict):
        self.backend_params = backend_params
        if backend == "bespoke":
            self._backend = BespokeFinetuneBackend(backend_params)
        elif backend == "openai":
            self._backend = OpenAIFinetuneBackend(backend_params)
        else:
            raise ValueError(f"Invalid backend: {backend}")

    def create_job(self, *args, **kwargs):
        return self._backend.create_job(*args, **kwargs)

    def list_jobs(self, *args, **kwargs):
        return self._backend.list_jobs(*args, **kwargs)

    def list_job_events(self, *args, **kwargs):
        return self._backend.list_job_events(*args, **kwargs)

    def get_job_details(self, *args, **kwargs):
        return self._backend.get_job_details(*args, **kwargs)

    def cancel_job(self, *args, **kwargs):
        return self._backend.cancel_job(*args, **kwargs)
