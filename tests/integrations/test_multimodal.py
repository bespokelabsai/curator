import hashlib
import os

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
_ONLINE_BACKENDS = [{"integration": backend} for backend in {"litellm", "openai"}]


@pytest.mark.parametrize("temp_working_dir", (_ONLINE_BACKENDS), indirect=True)
def test_basic_multimodal(temp_working_dir):
    temp_working_dir, backend, vcr_config = temp_working_dir
    hash_book = {
        "openai": "d9ed8894c463276b4329d25c20efa8e3033a66475ecfeee6b583e0e345bf7377",
        "litellm": "45245d26a2429496e47db7c94b5df332ecddebc1ccfc275724e1c56f6ae29f1d",
    }

    black_image = Image.new("RGB", (512, 512), color="black")
    dataset = Dataset.from_dict({"image": [black_image], "text": ["Describe the image"]})

    with vcr_config.use_cassette("basic_multimodal_completion.yaml"):
        prompter = helper.create_multimodal_llm()
        dataset = prompter(dataset=dataset, working_dir=temp_working_dir).dataset
        recipes = "".join([recipe[0] for recipe in dataset.to_pandas().values.tolist()])
        assert _hash_string(recipes) == hash_book[backend]


@pytest.mark.parametrize("temp_working_dir", (_ONLINE_BACKENDS), indirect=True)
def test_basic_multimodal_image_url(temp_working_dir):
    temp_working_dir, backend, vcr_config = temp_working_dir
    hash_book = {
        "litellm": "342e75c8d52cdf6edb7deaf9a5a8621f889ce0cadf2a8129e6f029f2191640dd",
        "openai": "3204d6666cba4fc8d94411192b6224aa2e656cd6ef5f5f40c297934b5dc90efb",
    }

    dataset = Dataset.from_dict({"image": ["https://images.pexels.com/photos/1684880/pexels-photo-1684880.jpeg"], "text": ["Describe the image"]})

    with vcr_config.use_cassette("basic_multimodal_image_url_completion.yaml"):
        prompter = helper.create_multimodal_llm(model="gpt-4o-mini", backend=backend)
        dataset = prompter(dataset=dataset, working_dir=temp_working_dir).dataset
        recipes = "".join([recipe[0] for recipe in dataset.to_pandas().values.tolist()])
        assert _hash_string(recipes) == hash_book[backend]


@pytest.mark.parametrize("temp_working_dir", ([{"integration": "litellm"}]), indirect=True)
def test_basic_multimodal_file_url(temp_working_dir):
    temp_working_dir, backend, vcr_config = temp_working_dir
    hash_book = {
        "litellm": "e31aa6465350a8385d7432d95b7107b7c191eb7e26af016c08b03e4ad9d45149",
    }

    url = "https://pdfobject.com/pdf/sample.pdf"
    dataset = Dataset.from_dict({"pdf": [url], "text": ["Describe the pdf"]})

    with vcr_config.use_cassette("basic_multimodal_file_url_completion.yaml"):
        model_name = "anthropic/claude-3-5-sonnet-20241022"
        prompter = helper.create_multimodal_llm(model=model_name, backend=backend, input_type="file")
        dataset = prompter(dataset=dataset, working_dir=temp_working_dir).dataset
        recipes = "".join([recipe[0] for recipe in dataset.to_pandas().values.tolist()])
        assert _hash_string(recipes) == hash_book[backend]


@pytest.mark.parametrize("temp_working_dir", (_ONLINE_BACKENDS), indirect=True)
def test_basic_multimodal_image_url_local(temp_working_dir):
    temp_working_dir, backend, vcr_config = temp_working_dir
    hash_book = {
        "openai": "10195c4d7ce82b24ca216bde881b2317d5eeafaa91d893d988362038c75434b1",
        "litellm": "84d673b735e1275304666b86bac457760b21bf94e8e27640aa2ac5bd2c14b576",
    }

    with vcr_config.use_cassette("basic_multimodal_image_url_local_completion.yaml"):
        # Test local image path
        black_image = Image.new("RGB", (512, 512), color="black")
        local_path = os.path.join(temp_working_dir, "black_image.png")
        black_image.save(local_path)
        dataset = Dataset.from_dict({"image": [local_path], "text": ["Describe the image"]})

        prompter = helper.create_multimodal_llm()
        dataset = prompter(dataset=dataset, working_dir=temp_working_dir).dataset
        recipes = "".join([recipe[0] for recipe in dataset.to_pandas().values.tolist()])
        assert _hash_string(recipes) == hash_book[backend]


@pytest.mark.skip
@pytest.mark.parametrize("temp_working_dir", ([{"integration": "litellm"}]), indirect=True)
def test_basic_multimodal_file_local_url(temp_working_dir):
    temp_working_dir, backend, vcr_config = temp_working_dir
    hash_book = {
        "litellm": "d9257e3cbf533d1e9a1f82a5d6ab286eedb46900a939e4ac3146937a621afa6c",
    }

    url = "./tests/integrations/common_fixtures/sample.pdf"
    dataset = Dataset.from_dict({"pdf": [url], "text": ["Describe the pdf"]})

    with vcr_config.use_cassette("basic_multimodal_file_local_url_completion.yaml"):
        model_name = "gemini/gemini-2.0-flash"
        prompter = helper.create_multimodal_llm(model=model_name, backend=backend, input_type="file")
        dataset = prompter(dataset=dataset, working_dir=temp_working_dir).dataset
        recipes = "".join([recipe[0] for recipe in dataset.to_pandas().values.tolist()])
        assert _hash_string(recipes) == hash_book[backend]
