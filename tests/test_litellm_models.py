import pytest
import os
import logging
from datasets import Dataset
from bespokelabs.curator import LLM
from tests.helpers import prepare_test_cache

"""
USAGE:
pytest -s tests/test_litellm_models.py
"""


@pytest.mark.cache_dir(os.path.expanduser("~/.cache/curator-tests/test-models"))
@pytest.mark.usefixtures("prepare_test_cache")
def test_litellm_models():

    env = os.environ.copy()
    assert "ANTHROPIC_API_KEY" in env, "ANTHROPIC_API_KEY must be set"
    assert "OPENAI_API_KEY" in env, "OPENAI_API_KEY must be set"
    assert "GEMINI_API_KEY" in env, "GEMINI_API_KEY must be set"
    assert "TOGETHER_API_KEY" in env, "TOGETHER_API_KEY must be set"

    models_list = [
        "claude-3-5-sonnet-20240620",  # https://docs.litellm.ai/docs/providers/anthropic # anthropic has a different hidden param tokens structure.
        "claude-3-5-haiku-20241022",
        "claude-3-haiku-20240307",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "gpt-4o-mini",  # https://docs.litellm.ai/docs/providers/openai
        "gpt-4o-2024-08-06",
        "gpt-4-0125-preview",
        "gpt-3.5-turbo-1106",
        "gemini/gemini-1.5-flash",  # https://docs.litellm.ai/docs/providers/gemini; https://ai.google.dev/gemini-api/docs/models # 20-30 iter/s
        "gemini/gemini-1.5-pro",  # 20-30 iter/s
        "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",  # https://docs.together.ai/docs/serverless-models
        "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    ]

    for model in models_list:
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
