from litellm import model_cost

from bespokelabs.curator.request_processor.batch.openai_batch_request_processor import OpenAIBatchRequestProcessor
from bespokelabs.curator.request_processor.config import BatchRequestProcessorConfig
from bespokelabs.curator.types.generic_request import GenericRequest


class AzureBatchRequestProcessor(OpenAIBatchRequestProcessor):
    """Azure OpenAI-specific implementation of the BatchRequestProcessor.

    Uses Azure OpenAI's OpenAI-SDK-compatible `v1` API surface
    (`https://<resource>.openai.azure.com/openai/v1/`), so file/batch lifecycle handling is
    inherited from OpenAIBatchRequestProcessor unmodified. Azure diverges from OpenAI in two ways
    this class accounts for:

    - `client.batches.create()`'s `endpoint` argument omits the "/v1" prefix ("/chat/completions",
      not "/v1/chat/completions"), even though the batch file's per-line "url" keeps it.
    - the request body's "model" field must be set to the Azure deployment name, not curator's
      `model` config (which stays the underlying model name, e.g. "gpt-4o", used for cost lookups).
    """

    _BATCH_ENDPOINT = "/chat/completions"

    def __init__(self, config: BatchRequestProcessorConfig) -> None:
        """Initialize the AzureBatchRequestProcessor."""
        if not config.base_url:
            raise ValueError("Azure OpenAI backend requires `base_url` set to your resource's v1 endpoint, e.g. https://<resource>.openai.azure.com/openai/v1/")
        if not config.azure_deployment:
            raise ValueError("Azure OpenAI backend requires `azure_deployment` set to your batch deployment name.")

        super().__init__(config, compatible_provider="azure")
        self.web_dashboard = "https://oai.azure.com/resource/batch"

    @property
    def backend(self):
        """Backend property."""
        return "azure"

    def create_api_specific_request_batch(self, generic_request: GenericRequest) -> dict:
        """Creates an Azure-specific request body, targeting the configured deployment name."""
        request = super().create_api_specific_request_batch(generic_request)
        request["body"]["model"] = self.config.azure_deployment
        return request

    def set_model_cost(self):
        """Set tracker cost information, preferring litellm's `azure/`-prefixed pricing table."""
        azure_model = f"azure/{self.prompt_formatter.model_name}"
        if azure_model in model_cost:
            self.tracker.input_cost_per_million = (model_cost[azure_model]["input_cost_per_token"] * 1_000_000) * 0.5
            self.tracker.output_cost_per_million = (model_cost[azure_model]["output_cost_per_token"] * 1_000_000) * 0.5
            self.tracker.start_tracker(self._tracker_console)
        else:
            super().set_model_cost()
