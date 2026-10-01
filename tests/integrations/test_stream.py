import hashlib
import os

import pytest

from bespokelabs import curator

##############################
# Online                     #
##############################


def _hash_string(input_string):
    return hashlib.sha256(input_string.encode("utf-8")).hexdigest()


_ONLINE_STREAM_REASONING_BACKENDS = [{"integration": backend} for backend in {"openai"}]


class Reasoner(curator.LLM):
    return_completions_object = True

    def prompt(self, input):  # noqa: D102
        return [{"role": "user", "content": input["question"]}]

    def parse(self, input, response):
        """Parse the LLM response to extract reasoning and solution."""
        return input


@pytest.mark.parametrize("temp_working_dir", (_ONLINE_STREAM_REASONING_BACKENDS), indirect=True)
def test_basic_reasoning(temp_working_dir, mock_reasoning_dataset):
    temp_working_dir, backend, vcr_config = temp_working_dir
    hash_book = {
        "openai": "ada3f38dafdc03168bca2f354c88da64d21686a931ca31607fcf79c1d95b2813",
    }

    with vcr_config.use_cassette("basic_qwen3_stream.yaml"):
        os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        prompter = Reasoner(
            model_name="qwen3-235b-a22b",
            backend="openai",
            generation_params={"thinking_budget": 128, "enable_thinking": True},
            batch=False,
            backend_params={"stream": True},
        )
        mock_dataset = mock_reasoning_dataset.select(range(2))

        dataset = prompter(mock_dataset, working_dir=temp_working_dir)
        dataset = dataset.dataset

        # Verify response content
        recipes = "".join([recipe[0] for recipe in dataset.to_pandas().values.tolist()])
        assert _hash_string(recipes) == hash_book[backend]
