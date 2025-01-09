import os
import shutil
import socket
import tempfile
import time

import huggingface_hub
import psutil
import pytest

from bespokelabs.curator import LLM


def download_model(model_name):
    """Download a model from the Hugging Face Hub."""
    tmpdirname = tempfile.mkdtemp()
    model_path = huggingface_hub.snapshot_download(repo_id=model_name, repo_type="model", local_dir=tmpdirname)
    return model_path


def start_vllm_server(model_path, host, port):
    """Start the VLLM server."""
    cmd = f"vllm serve {model_path} --host={host} --port={port} --api-key=token-abc123 &"
    try:
        os.system(cmd)
    except Exception as e:
        print(e)
        raise e


def kill_vllm_server():
    """Kill the VLLM server."""
    try:
        for p in psutil.process_iter():
            if p.name() == "vllm":
                break

        for child in p.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                continue
        p.kill()
    except Exception as e:
        print(e)
        raise e


@pytest.mark.parametrize("model_name", ["HuggingFaceTB/SmolLM-135M-Instruct"])
def test_online_vllm(model_name):
    model_path = download_model(model_name)
    host = socket.gethostname().split(".")[0]

    port = 5432

    # pid = start_vllm_server(model_path, host, port)

    time.sleep(60)

    os.environ["HOSTED_VLLM_API_KEY"] = "token-abc123"

    url = f"http://{host}:{port}/v1"

    prompter = LLM(
        prompt_func=lambda row: "write me a poem",
        model_name=f"hosted_vllm/{model_path}",
        base_url=url,
        backend="litellm",
    )

    dataset = prompter()
    kill_vllm_server()
    shutil.rmtree(model_path)
    assert dataset is not None
    assert len(dataset) == 1
    assert "response" in dataset.column_names


# if __name__ == "__main__":
#     test_online_vllm("/p/data1/mmlaion/marianna/models/Qwen/Qwen2.5-3B-Instruct")
