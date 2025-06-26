from typing import Optional

from pydantic import BaseModel


class FinetuneMethod(BaseModel):
    """Pydantic model for specifying the finetuning method and its hyperparameters."""

    type: str
    hyperparameters: dict


class FinetuneRequest(BaseModel):
    """Pydantic model for a finetuning job request."""

    job_name: str
    dataset_id: str
    model_name: str
    seed: int
    suffix: str
    num_gpus: int
    method: FinetuneMethod


class FinetuneResponse(BaseModel):
    """Pydantic model for a finetuning job response."""

    job_id: str
    job_name: str
    dataset_id: str
    model_name: str
    seed: int
    suffix: str
    method: FinetuneMethod
    status: str


class FinetuneBackendParams(BaseModel):
    """Pydantic model for backend parameters."""

    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    dataset_id: Optional[str] = None
    method: Optional[FinetuneMethod] = None
    seed: Optional[int] = None
    suffix: Optional[str] = None
    num_gpus: Optional[int] = None
    hyperparameters: Optional[dict] = None
    validation_file: Optional[str] = None
