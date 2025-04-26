import os
from bespokelabs.curator.finetune.base_backend import BaseFinetuneBackend
from openai import OpenAI

class OpenAIFinetuneBackend(BaseFinetuneBackend):

    def __init__(self, backend_params: dict):
        super().__init__(backend_params)
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def create_job(self, *args, **kwargs):
        job_id = self.client.fine_tuning.jobs.create(
            training_file=kwargs["dataset_id"],
            model=kwargs["model_name"],
            method=kwargs["method"]
        )
        return job_id
    

    def list_jobs(self, *args, **kwargs):
        jobs = self.client.fine_tuning.jobs.list()
        return jobs
    
    def list_job_events(self, *args, **kwargs):
        events = self.client.fine_tuning.jobs.list_events(job_id=kwargs["job_id"])
        return events
    
    def get_job_details(self, *args, **kwargs):
        job = self.client.fine_tuning.jobs.retrieve(job_id=kwargs["job_id"])
        return job
    
    def cancel_job(self, *args, **kwargs):
        self.client.fine_tuning.jobs.cancel(job_id=kwargs["job_id"])
