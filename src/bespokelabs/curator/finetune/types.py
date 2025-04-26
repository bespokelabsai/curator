from pydantic import BaseModel

class FinetuneMethod(BaseModel):
    type: str
    hyperparameters: dict

class FinetuneRequest(BaseModel): 
    job_name: str
    dataset_id: str
    model_name: str
    seed: int
    suffix: str
    method: FinetuneMethod

class FinetuneResponse(BaseModel):
    job_id: str
    job_name: str
    dataset_id: str
    model_name: str
    seed: int
    suffix: str
    method: FinetuneMethod
    status: str