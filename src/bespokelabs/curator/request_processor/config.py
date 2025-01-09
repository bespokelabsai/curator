import logging

import litellm
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RequestProcessorConfig(BaseModel):
    """Configuration for request processors."""

    model: str
    base_url: str | None = None
    max_retries: int = Field(default=10, ge=0)
    request_timeout: int = Field(default=10 * 60, gt=0)
    require_all_responses: bool = Field(default=True)
    generation_params: dict = Field(default_factory=dict)

    def __post_init__(self):
        self.supported_params = litellm.get_supported_openai_params(model=self.model)
        logger.debug(
            f"Automatically detected supported params using litellm for {self.model}: {self.supported_params}"
        )

        for key in self.generation_params.keys():
            raise ValueError(
                f"Generation parameter '{key}' is not supported for model '{self.model}'"
            )


class BatchRequestProcessorConfig(RequestProcessorConfig):
    """Additional configuration specific to batch processors."""

    batch_size: int = Field(default=10_000, gt=0)
    batch_check_interval: int = Field(default=60, gt=0)
    delete_successful_batch_files: bool = False
    delete_failed_batch_files: bool = False


class OnlineRequestProcessorConfig(RequestProcessorConfig):
    """Additional configuration specific to online processors."""

    max_requests_per_minute: int | None = Field(default=None, gt=0)
    max_tokens_per_minute: int | None = Field(default=None, gt=0)
    seconds_to_pause_on_rate_limit: int = Field(default=10, gt=0)
