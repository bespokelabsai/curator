import hashlib

import pytest
from datasets import Dataset
from PIL import Image

from tests.integrations import helper

##############################
# Online                     #
##############################


def _hash_string(input_string):
    return hashlib.sha256(input_string.encode("utf-8")).hexdigest()


# Mutli modal supported backends
_ONLINE_BACKENDS = [{"integration": backend} for backend in {"openai"}]


@pytest.mark.parametrize("temp_working_dir", (_ONLINE_BACKENDS), indirect=True)
def test_basic_multimodal(temp_working_dir):
    temp_working_dir, backend, vcr_config = temp_working_dir
    hash_book = {
        "openai": "d9ed8894c463276b4329d25c20efa8e3033a66475ecfeee6b583e0e345bf7377",
    }

    black_image = Image.new("RGB", (512, 512), color="black")
    dataset = Dataset.from_dict({"image": [black_image], "text": ["Describe the image"]})

    with vcr_config.use_cassette("basic_multimodal_completion.yaml"):
        prompter = helper.create_multimodal_llm()
        dataset = prompter(dataset=dataset, working_dir=temp_working_dir)
        recipes = "".join([recipe[0] for recipe in dataset.to_pandas().values.tolist()])
        assert _hash_string(recipes) == hash_book[backend]


@pytest.mark.parametrize("temp_working_dir", (_ONLINE_BACKENDS), indirect=True)
def test_basic_multimodal_image_url(temp_working_dir):
    temp_working_dir, backend, vcr_config = temp_working_dir
    hash_book = {
        "openai": "63b7f3bca97585975b8cb698956aa0b40d584b120fd1d4cf51825fc1c5d3506d",
    }

    dataset = Dataset.from_dict({"image": ["https://images.pexels.com/photos/1684880/pexels-photo-1684880.jpeg"], "text": ["Describe the image"]})

    with vcr_config.use_cassette("basic_multimodal_image_url_completion.yaml"):
        prompter = helper.create_multimodal_llm()
        dataset = prompter(dataset=dataset, working_dir=temp_working_dir)
        recipes = "".join([recipe[0] for recipe in dataset.to_pandas().values.tolist()])
        assert _hash_string(recipes) == hash_book[backend]
