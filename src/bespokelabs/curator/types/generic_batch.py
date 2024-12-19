import datetime
from typing import Literal
from pydantic import BaseModel


class GenericBatchRequestCounts(BaseModel):
    total: int
    failed: int
    succeeded: int
    raw_request_counts_object: dict


class GenericBatch(BaseModel):
    request_file: str
    id: str
    created_at: datetime.datetime
    finished_at: datetime.datetime
    status: Literal["submitted", "finished", "downloaded"]
    api_key_suffix: str
    request_counts: GenericBatchRequestCounts
    raw_status: str
    raw_batch: dict
