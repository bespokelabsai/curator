import shutil
import tempfile

import huggingface_hub
import pytest

from bespokelabs.curator import LLM


def download_model(model_name):
    """Download a model from the Hugging Face Hub."""
    tmpdirname = tempfile.mkdtemp()
    model_path = huggingface_hub.snapshot_download(repo_id=model_name, repo_type="model", local_dir=tmpdirname)
    return model_path


@pytest.mark.skip
@pytest.mark.parametrize("model_name", ["HuggingFaceTB/SmolLM-135M-Instruct"])
def test_simple_vllm(model_name):
    model_path = download_model(model_name)

    prompter = LLM(
        prompt_func=lambda row: "write me a poem",
        model_name=model_path,
        backend="vllm",
        max_tokens=50,
        max_model_length=1024,
    )

    dataset = prompter()

    shutil.rmtree(model_path)

    assert len(dataset) == 1
    assert "response" in dataset.column_names
