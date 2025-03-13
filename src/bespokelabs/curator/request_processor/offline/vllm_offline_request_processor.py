import contextlib
import datetime
import gc

import torch
import vllm
from pydantic import BaseModel
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.sampling_params import GuidedDecodingParams

from bespokelabs.curator.log import logger
from bespokelabs.curator.request_processor.config import OfflineRequestProcessorConfig
from bespokelabs.curator.request_processor.offline.base_offline_request_processor import BaseOfflineRequestProcessor
from bespokelabs.curator.request_processor.online.base_online_request_processor import APIRequest
from bespokelabs.curator.status_tracker.offline_status_tracker import OfflineStatusTracker
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse


class VLLMOfflineRequestProcessor(BaseOfflineRequestProcessor):
    """Offline request processor for the VLLM model.

    Args:
        config (OfflineRequestProcessorConfig): Configuration for the request processor
    """

    def __init__(
        self,
        config: OfflineRequestProcessorConfig,
    ):
        """Initialize the VLLMOfflineRequestProcessor."""
        super().__init__(
            config,
        )

    @property
    def backend(self):
        """Backend property."""
        return "vllm"

    def load_offline_model(self):
        """Load the VLLM model for offline processing."""
        self.model_class = vllm.LLM(
            self.model,
            trust_remote_code=True,
            max_model_len=self.max_model_length,
            enforce_eager=self.enforce_eager,
            tensor_parallel_size=self.tensor_parallel_size,
            max_num_seqs=self.batch_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            disable_custom_all_reduce=self.tensor_parallel_size > 1,
        )

    def format_prompts(self, prompts: list) -> list:
        """Format prompts for the VLLM model.

        Args:
            prompts (list): List of prompts to format
        """
        tokenizer = self.model_class.get_tokenizer()
        try:
            formatted_prompts = [tokenizer.apply_chat_template(conversation=prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts]
        except Exception as e:
            logger.error(f"Error formatting prompts: {e}")
            raise e

        return formatted_prompts

    def check_structured_output_support(self) -> bool:
        """Verify if the model supports structured output via instructor.

        Tests the model's capability to handle structured output by making a test request
        with a simple schema.

        Returns:
            bool: True if structured output is supported, False otherwise

        Note:
            - Uses a simple User schema as test case
            - Logs detailed information about support status
            - Required for models that will use JSON schema responses
        """
        self.load_offline_model()

        class User(BaseModel):
            name: str
            age: int

        try:
            json_schema = User.schema()
            guided_decoding_params = GuidedDecodingParams(json=json_schema)
            sampling_params = vllm.SamplingParams(guided_decoding=guided_decoding_params)
            messages = [[{"role": "user", "content": "Jason is 25 years old."}]]
            formatted_prompts = self.format_prompts(messages)
            response = self.model_class.generate(
                formatted_prompts,
                sampling_params=sampling_params,
            )
            response = response[0].outputs[0].text
            response = self.fix_json(response)
            logger.info(f"Check guided decoding structure output response: {response}")
            response = User.parse_raw(response)
            assert isinstance(response.age, int)
            assert isinstance(response.name, str)
            assert response.name == "Jason"
            assert response.age == 25
            logger.info(f"Model {self.model} supports structured output via instructor, response: {response}")
            self.support_structured_output = True
            return True
        except Exception as e:
            logger.warning(f"Model {self.model} does not support structured output via guided decoding: {e} {type(e)} {e}")
            return False

    def create_api_specific_request(self, generic_request: GenericRequest) -> dict:
        """Convert a generic request into a vLLM-compatible format.

        Checks supported parameters for the specific model and only includes
        applicable parameters.

        Args:
            generic_request (GenericRequest): The generic request to convert

        Returns:
            dict: vLLM-compatible request parameters

        Note:
            Uses vLLM's get_supported_openai_params to check parameter support
        """
        request = {
            "model": generic_request.model,
            "messages": generic_request.messages,
            "generation_params": self.generation_params,
        }

        return request

    def destroy(self):
        """Destroy the VLLM model and cleanup the environment."""
        destroy_model_parallel()
        destroy_distributed_environment()
        del self.model_class.llm_engine.model_executor
        del self.model_class
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def fix_json(self, response_message: str) -> str:
        """Fix incomplete JSON responses from vLLM guided decoding.

        When using guided decoding with vLLM, sometimes the JSON response is incomplete
        and missing the closing brace. This method adds the closing brace if needed.
        (https://github.com/vllm-project/vllm/issues/8350)

        Args:
            response_message (str): The potentially incomplete JSON response

        Returns:
            str: The JSON response with proper closing brace
        """
        if not response_message.endswith("}"):
            response_message += "}"
        return response_message

    def process_requests(self, requests: list[APIRequest], status_tracker: OfflineStatusTracker) -> list[GenericResponse]:
        """Process a list of API requests using the VLLM model.

        Args:
            requests (list[APIRequest]): List of API requests to process
            status_tracker (OfflineStatusTracker): Status tracker for the request

        Returns:
            list[GenericResponse]: List of generic responses
        """
        guided_decoding_params = None
        response_format = requests[0].generic_request.response_format
        if response_format is not None and self.support_structured_output:
            guided_decoding_params = GuidedDecodingParams(json=response_format)
        else:
            if response_format is not None:
                logger.warning(f"Model {self.model} does not support structured output via guided decoding, response_format: {response_format}")

        sampling_params = {
            "guided_decoding": guided_decoding_params,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            **self.generation_params,
        }

        formatted_prompts = self.format_prompts([request.generic_request.messages for request in requests])

        completions = self.model_class.generate(
            formatted_prompts,
            sampling_params=vllm.SamplingParams(**sampling_params),
        )
        status_tracker.time_finished = datetime.datetime.now()
        status_tracker.finished_successfully = True
        status_tracker.num_total_requests = len(requests)
        self.destroy()

        responses = []

        for completion, request in zip(completions, requests):
            response_message = completion.outputs[0].text
            if response_format is not None and self.support_structured_output:
                response_message = self.fix_json(response_message)

            raw_response = {
                "request_id": completion.request_id,
                "finished": completion.finished,
                "encoder_prompt": completion.encoder_prompt,
                "prompt": completion.prompt,
                "metrics": completion.metrics,
            }
            response = GenericResponse(
                model=self.model,
                response_message=response_message,
                raw_response=raw_response,
                created_at=status_tracker.time_started,
                finished_at=status_tracker.time_finished,
                generic_request=request.generic_request,
            )
            responses.append(response)
        return responses
