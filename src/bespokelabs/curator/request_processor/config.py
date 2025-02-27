import typing as t

from pydantic import BaseModel, Field, ValidationError

from bespokelabs.curator.log import logger


class RequestProcessorConfig(BaseModel):
    """Configuration for request processors.

    Base configuration class that defines common parameters used across different request processors.

    Attributes:
        model: Name/identifier of the LLM model to use
        base_url: Optional base URL for API endpoint
        max_retries: Maximum number of retry attempts for failed requests
        request_timeout: Timeout in seconds for each request
        require_all_responses: Whether to require successful responses for all requests
        generation_params: Dictionary of model-specific generation parameters
        api_key: Optional API key for authentication
        in_mtok_cost: Optional cost per million input tokens
        out_mtok_cost: Optional cost per million output tokens
        invalid_finish_reasons: List of api finish reasons which are considered failed.
    """

    model: str
    base_url: str | None = None
    max_retries: int = Field(default=10, ge=0)
    request_timeout: int = Field(default=10 * 60, gt=0)
    require_all_responses: bool = Field(default=True)
    generation_params: dict = Field(default_factory=dict)
    return_completions_object: bool = False
    api_key: str | None = None
    in_mtok_cost: int | None = None
    out_mtok_cost: int | None = None
    invalid_finish_reasons: list = Field(default_factory=lambda: ["content_filter", "length"])

    class Config:
        """BaseModel Setup class."""

        extra = "forbid"

    def __post_init__(self):
        """Post-initialization hook to validate generation parameters.

        Validates that all provided generation parameters are supported by the specified model
        using litellm's parameter validation.

        Raises:
            ValueError: If an unsupported generation parameter is provided
        """
        import litellm

        self.supported_params = litellm.get_supported_openai_params(model=self.model)
        logger.debug(f"Automatically detected supported params using litellm for {self.model}: {self.supported_params}")

        for key in self.generation_params.keys():
            raise ValueError(f"Generation parameter '{key}' is not supported for model '{self.model}'")


class BatchRequestProcessorConfig(RequestProcessorConfig):
    """Additional configuration specific to batch processors.

    Extends the base RequestProcessorConfig with batch-specific parameters.

    Attributes:
        batch_size: Maximum number of requests to process in a single batch
        batch_check_interval: Time in seconds between batch status checks
        delete_successful_batch_files: Whether to delete batch files after successful processing
        delete_failed_batch_files: Whether to delete batch files after failed processing
        completion_window: Time window to wait for batch completion
    """

    batch_size: int = Field(default=10_000, gt=0)
    batch_check_interval: int = Field(default=60, gt=0)
    delete_successful_batch_files: bool = False
    delete_failed_batch_files: bool = False
    completion_window: str = "24h"


class OnlineRequestProcessorConfig(RequestProcessorConfig):
    """Additional configuration specific to online processors.

    Extends the base RequestProcessorConfig with rate limiting and capacity parameters.

    Attributes:
        max_requests_per_minute: Maximum number of requests allowed per minute
        max_tokens_per_minute: Maximum number of tokens allowed per minute
        max_input_tokens_per_minute: Maximum number of input tokens allowed per minute
        max_output_tokens_per_minute: Maximum number of output tokens allowed per minute
        max_concurrent_requests: Maximum number of concurrent requests
        seconds_to_pause_on_rate_limit: Duration to pause when rate limit is hit
    """

    max_requests_per_minute: int | None = Field(default=None, gt=0)
    max_concurrent_requests: int | None = Field(default=None, gt=0)
    max_tokens_per_minute: int | None = Field(default=None, gt=0)
    max_input_tokens_per_minute: int | None = Field(default=None, gt=0)
    max_output_tokens_per_minute: int | None = Field(default=None, gt=0)
    seconds_to_pause_on_rate_limit: int = Field(default=10, gt=0)


class OfflineRequestProcessorConfig(RequestProcessorConfig):
    """Additional configuration specific to offline processors.

    Extends the base RequestProcessorConfig with parameters for offline/local model execution.

    Attributes:
        max_model_length: Maximum sequence length the model can handle
        max_tokens: Maximum number of tokens to generate
        min_tokens: Minimum number of tokens to generate
        enforce_eager: Whether to enforce eager execution mode
        tensor_parallel_size: Number of GPUs to use for tensor parallelism
        batch_size: Size of batches for processing
        gpu_memory_utilization: Target GPU memory utilization (0-1)
    """

    max_model_length: int = Field(default=4096, gt=0)
    max_tokens: int = Field(default=4096, gt=0)
    min_tokens: int = Field(default=1, gt=0)
    enforce_eager: bool = False
    tensor_parallel_size: int = Field(default=1, gt=0)
    batch_size: int = Field(default=256, gt=0)
    gpu_memory_utilization: float = Field(default=0.95, gt=0, le=1)

    def __post_init__(self):
        """Post-initialization hook to validate generation parameters.

        Overrides base class validation since offline models have different parameter requirements.
        """
        pass


class BaseBackendParams(t.TypedDict, total=False):
    """Base backend params TypedDict class."""

    model: t.Optional[str]
    base_url: t.Optional[str]
    max_retries: t.Optional[int]
    request_timeout: t.Optional[int]
    require_all_responses: t.Optional[bool]


class OnlineBackendParams(BaseBackendParams, total=False):
    """TypedDict for online processor."""

    max_requests_per_minute: t.Optional[int]
    max_tokens_per_minute: t.Optional[int]
    max_input_tokens_per_minute: t.Optional[int]
    max_output_tokens_per_minute: t.Optional[int]
    seconds_to_pause_on_rate_limit: t.Optional[int]


class BatchBackendParams(BaseBackendParams, total=False):
    """TypedDict for batch processor."""

    batch_size: t.Optional[int]
    batch_check_interval: t.Optional[int]
    delete_successful_batch_files: t.Optional[bool]
    delete_failed_batch_files: t.Optional[bool]


class OfflineBackendParams(BaseBackendParams, total=False):
    """TypedDict for offline processor. for example, vLLM."""

    tensor_parallel_size: t.Optional[int]
    enforce_eager: t.Optional[bool]
    max_model_length: t.Optional[int]
    max_tokens: t.Optional[int]
    min_tokens: t.Optional[int]
    gpu_memory_utilization: t.Optional[float]
    batch_size: t.Optional[int]


BackendParamsType = t.Union[OnlineBackendParams, BatchBackendParams, OfflineBackendParams]


def _validate_backend_params(params: BackendParamsType):
    validators = (
        BatchRequestProcessorConfig,
        OnlineRequestProcessorConfig,
        OfflineRequestProcessorConfig,
    )
    for validator in validators:
        try:
            validator.validate(params)
        except ValidationError:
            continue
        else:
            return validator(**params)
    raise ValueError(f"Backend params are not valid, please refer {validators} for more info on backend params.")
