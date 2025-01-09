import logging
import os

import pytest

from bespokelabs.curator import LLM
from bespokelabs.curator.dataset import Dataset

"""
USAGE:
pytest -s tests/test_litellm_models.py
"""


@pytest.mark.cache_dir(os.path.expanduser("~/.cache/curator-tests/test-models"))
@pytest.mark.usefixtures("clear_test_cache")
class TestLiteLLMModels:
    """Test suite for validating various LLM models through the LiteLLM backend.

    This test class verifies that different language models can be accessed and used
    through curator's LLM interface using the LiteLLM backend. It tests a variety of
    models from different providers including Anthropic, OpenAI, Google, and Together.ai.
    """

    @pytest.fixture(autouse=True)
    def check_environment(self):
        """Fixture to verify required API keys are present in environment variables.

        Raises:
            AssertionError: If any required API key is missing from environment variables.
        """
        env = os.environ.copy()
        required_keys = [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "TOGETHER_API_KEY",
        ]
        for key in required_keys:
            assert key in env, f"{key} must be set"

    @pytest.mark.parametrize(
        "model",
        [
            pytest.param("claude-3-5-sonnet-20240620", id="claude-3-5-sonnet"),
            pytest.param("claude-3-5-haiku-20241022", id="claude-3-5-haiku"),
            pytest.param("claude-3-haiku-20240307", id="claude-3-haiku"),
            pytest.param("claude-3-opus-20240229", id="claude-3-opus"),
            pytest.param("claude-3-sonnet-20240229", id="claude-3-sonnet"),
            pytest.param("gpt-4o-mini", id="gpt-4-mini"),
            pytest.param("gpt-4o-2024-08-06", id="gpt-4"),
            pytest.param("gpt-4-0125-preview", id="gpt-4-preview"),
            pytest.param("gpt-3.5-turbo-1106", id="gpt-3.5"),
            pytest.param("gemini/gemini-1.5-flash", id="gemini-flash"),
            pytest.param("gemini/gemini-1.5-pro", id="gemini-pro"),
            pytest.param("together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", id="llama-8b"),
            pytest.param("together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", id="llama-70b"),
        ],
    )
    def test_model(self, model):
        """Test a specific LLM model's basic functionality.

        Tests that the model can successfully process a simple prompt and return a response
        through the curator LLM interface using LiteLLM backend.

        Args:
            model (str): The identifier/name of the model to test.
        """
        print(f"\n\n========== TESTING {model} ==========\n\n")
        logger = logging.getLogger("bespokelabs.curator")
        logger.setLevel(logging.DEBUG)

        dataset = Dataset.from_dict({"prompt": ["just say 'hi'"]})

        prompter = LLM(
            prompt_func=lambda row: row["prompt"],
            model_name=model,
            response_format=None,
            backend="litellm",
        )

        dataset = prompter(dataset)
        print(dataset.to_pandas())
