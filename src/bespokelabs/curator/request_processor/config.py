from dataclasses import dataclass, field
from typing import Optional
from litellm import get_supported_openai_params
import logging

logger = logging.getLogger(__name__)


def _verify_non_negative(value: Optional[int], param_name: str) -> None:
    """Helper function to verify that a value is non-negative if it exists.

    Args:
        value: The value to check
        param_name: Name of the parameter for the error message

    Raises:
        ValueError: If the value is negative
    """
    if value is not None and value < 0:
        raise ValueError(f"{param_name} must be >= 0")


@dataclass
class RequestProcessorConfig:
    """Configuration for request processors."""

    model: str
    max_retries: int = 10
    request_timeout: int = 10 * 60
    require_all_responses: bool = True
    generation_params: dict = field(default_factory=dict)
    supported_params: list[str] = field(default_factory=list)

    def __post_init__(self):
        _verify_non_negative(self.max_retries, "max_retries")

        if not self.supported_params:
            self.supported_params = get_supported_openai_params(model=self.model)
            logger.debug(
                f"Automatically detected supported params for {self.model}: {self.supported_params}"
            )

        for key in self.generation_params.keys():
            if key not in self.supported_params:
                raise ValueError(
                    f"Generation parameter '{key}' is not supported for model '{self.model}'"
                )


@dataclass
class BatchRequestProcessorConfig(RequestProcessorConfig):
    """Additional configuration specific to batch processors."""

    batch_size: int = 10_000
    batch_check_interval: int = 60
    delete_successful_batch_files: bool = False
    delete_failed_batch_files: bool = False

    def __post_init__(self):
        _verify_non_negative(self.batch_size, "batch_size")
        _verify_non_negative(self.batch_check_interval, "batch_check_interval")


@dataclass
class OnlineRequestProcessorConfig(RequestProcessorConfig):
    """Additional configuration specific to online processors."""

    max_requests_per_minute: int | None = None
    max_tokens_per_minute: int | None = None
    seconds_to_pause_on_rate_limit: int = 10

    def __post_init__(self):
        _verify_non_negative(self.max_requests_per_minute, "max_requests_per_minute")
        _verify_non_negative(self.max_tokens_per_minute, "max_tokens_per_minute")
