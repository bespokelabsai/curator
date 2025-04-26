from abc import ABC, abstractmethod

class BaseFinetuneBackend(ABC):

    def __init__(self, backend_params: dict):
        self.backend_params = backend_params

    @abstractmethod
    def create_job(self, model_name: str, dataset_id: str, method: dict):
        pass

    @abstractmethod
    def list_jobs(self):
        pass

    @abstractmethod
    def list_job_events(self, job_id: str):
        pass

    # @abstractmethod
    # def list_job_checkpoints(self, job_id: str):
    #     pass

    # @abstractmethod
    # def list_job_metrics(self, job_id: str):
    #     pass
    

    @abstractmethod
    def get_job_details(self, job_id: str):
        pass

    @abstractmethod
    def cancel_job(self, job_id: str):
        pass
