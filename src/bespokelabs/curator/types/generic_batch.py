import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class GenericBatchStatus(Enum):
    SUBMITTED = "submitted"
    FINISHED = "finished"
    DOWNLOADED = "downloaded"


class GenericBatchRequestCounts(BaseModel):
    total: int
    failed: int
    succeeded: int
    raw_request_counts_object: dict


class GenericBatch(BaseModel):
    request_file: str
    id: str
    created_at: datetime.datetime
    finished_at: Optional[datetime.datetime]
    status: GenericBatchStatus
    api_key_suffix: str
    request_counts: GenericBatchRequestCounts
    raw_status: str
    raw_batch: dict
