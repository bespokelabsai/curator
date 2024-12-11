from typing import Literal
import datetime

from pydantic import BaseModel

from openai.types.batch import Batch
from anthropic.types.beta.messages import BetaMessageBatch
from anthropic.types.beta.messages import BetaMessageBatchRequestCounts
from openai.types.batch_request_counts import BatchRequestCounts


class GenericBatchRequestCounts(BaseModel):
    total: int
    failed: int
    succeeded: int
    raw_request_counts_object: dict

    # https://github.com/anthropics/anthropic-sdk-python/blob/e7c5fd1cf9226d73122870d07906664696da3ab8/src/anthropic/types/beta/messages/beta_message_batch_request_counts.py#L9
    # Request Counts (Anthropic): "processing", "cancelled", "errored", "expired", "succeeded"
    # total: "processing", "cancelled", "errored", "expired", "succeeded"
    # failed: "cancelled", "errored", "expired"
    # succeeded: "succeeded"

    # https://github.com/openai/openai-python/blob/6e1161bc3ed20eef070063ddd5ac52fd9a531e88/src/openai/types/batch_request_counts.py#L9
    # Request Counts (OpenAI): "completed", "failed", "total"
    # total: "total"
    # failed: "failed"
    # succeeded: "completed"


class GenericBatchObject(BaseModel):
    request_file: str
    id: str
    output: str
    created_at: datetime.datetime
    finished_at: datetime.datetime
    status: Literal["submitted", "finished", "downloaded"]
    raw_batch_object: dict

    # https://github.com/anthropics/anthropic-sdk-python/blob/e7c5fd1cf9226d73122870d07906664696da3ab8/src/anthropic/types/beta/messages/beta_message_batch.py#L53
    # Batch Status (Anthropic): "in_progress", "canceling", "ended"
    # submitted: "in_progress", "canceling"
    # finished: "ended"

    # https://github.com/openai/openai-python/blob/995cce048f9427bba4f7ac1e5fc60abbf1f8f0b7/src/openai/types/batch.py#L40C1-L41C1
    # Batch Status (OpenAI): "validating", "finalizing", "cancelling", "in_progress", "completed", "failed", "expired", "cancelled"
    # submitted: "validating", "finalizing", "cancelling", "in_progress"
    # finished: "completed", "failed", "expired", "cancelled"

    # https://github.com/openai/openai-python/blob/bb9c2de913279acc89e79f6154173a422f31de45/src/openai/types/batch.py#L27-L71
    # Timing (OpenAI): "created_at", "in_progress_at", "expires_at", "finalizing_at", "completed_at", "failed_at", "expired_at", "cancelling_at", "cancelled_at"

    # https://github.com/anthropics/anthropic-sdk-python/blob/e7c5fd1cf9226d73122870d07906664696da3ab8/src/anthropic/types/beta/messages/beta_message_batch.py#L20-L51
    # Timing (Anthropic): "created_at", "cancel_initiated_at", "archived_at", "ended_at", "expires_at"
