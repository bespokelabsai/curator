# Local Models Support
## Table of Contents

- [Online Inference with Local Models](#online-inference-with-local-models-vllm)
- [Offline Inference with Local Models](#offline-inference-with-local-models-vllm)
  - [Setup instructions](#setup-instructions)
  - [Basic usage](#basic-usage)
  - [Inference with models that do not fit in memory of one GPU (tensor parallelism)](#inference-for-models-that-dont-fit-in-one-gpus-memory-tensor-parallel)
  - [Structured output with local models](#structured-output)
  - [Batched inference](#batched-inference)
  - [Details on vLLM specific arguments](#details-on-vllm-specific-arguments)
- [Full list of local models examples](#full-list-of-vllm-examples)

## Online Inference with Local Models (vLLM)

You can use local models served with [vLLM OpenAI compatibale server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) for online data generation.
In order to do that, first start a vLLM ssrver:
```bash
vllm serve \
NousResearch/Meta-Llama-3-8B-Instruct \
--host localhost \
--port 8787 \
--api-key token-abc123
```

Note that you still need to provide a dummy API key since OpenAI client expects it (LiteLLM [uses](https://docs.litellm.ai/docs/providers/vllm#usage---litellmcompletion-calling-openai-compatible-endpoint) OpenAI client in the background).
See [here](examples/vllm-online/start_vllm_server.sh) a full example on starting a vLLM server.

Then, to use this server, provide the API endpoint URL and the dummy API key. We use LiteLLM to generate data with vLLM hosted local models so you need to set the backend to `litellm`. In order for LiteLLM to recognize proper backend, add `hosted_vllm/` prefix to your model name. Set `HOSTED_VLLM_API_KEY` environment varibale to your dummy API key. Example:
```python
from bespokelabs import curator
model_path = "hosted_vllm/NousResearch/Meta-Llama-3-8B-Instruct" # Make sure to add hosted_vllm/ prefix
PORT = 8787
HOST = "localhost"
URL = f"http://{HOST}:{PORT}/v1"
os.environ["HOSTED_VLLM_API_KEY"] = "token-abc123

poem_prompter = curator.LLM(
    model_name=model_path,
    backend="litellm",
    base_url=URL,
)
poem_prompter("Generate a poem")
```

See [here](examples/vllm-online/vllm_online.py) a full example.
LiteLLM supports structured output, see [here](examples/vllm-online/vllm_online_structured.py) an example with a vLLM hosted model.

## Offline Inference with Local Models (vLLM)

We use [vLLM](https://docs.vllm.ai/) offline LLM running engine to generate synthetic data with local models.

[Here](https://docs.vllm.ai/en/latest/models/supported_models.html#generative-models) is the full list of models that are supported by vLLM.

### Setup instructions

Install vLLM in the Python or Conda environment:

```bash
pip install vllm
```

You may also need to install Ray for multi-node inference based on Ray clusters:

```bash
pip install ray
```

Please refer to the vLLM isntallation [instructions](https://docs.vllm.ai/en/latest/getting_started/installation.html).

### Basic usage

To use curator with a local model, provide your model local path in the model argument and set the backend to `vllm`:

```python
model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
curator.LLM(
        model_name=model_path,
        backend="vllm"
    )
```

See full example [here](examples/vllm-offline/vllm_recipe.py).

### Inference for models that don't fit in one GPU's memory (tensor parallel)

To use local model that are too big to fit on one GPU, use tensor parallelism. That way you can split the model across multiple GPUs. To do that, specify `tensor_parallel_size` argument that should be equal to number of GPUs you have:

```python
model_path = "meta-llama/Meta-Llama-3.1-70B-Instruct"
curator.LLM(
        model_name=model_path,
        backend="vllm",
        tensor_parallel_size=4 # split across 4 GPUs
    )
```

### Structured output

We use [vLLM's Guided Decoding](https://docs.vllm.ai/en/latest/usage/structured_outputs.html#offline-inference) to obtain structured output from local models during offline inference:

```python
  from pydantic import BaseModel, Field

  class Cuisines(BaseModel):
      cuisines_list: List[str] = Field(description="A list of cuisines.")

  model_path = "/local/path/to/weights/meta-llama/Meta-Llama-3.1-70B-Instruct"
  cuisines_generator = curator.LLM(
      model_name=model_path,
      response_format=Cuisines,
      backend="vllm",
  )
  cuisines = cuisines_generator("Generate 10 diverse cuisines.")
```

See full example [here](examples/vllm-offline/vllm_recipe_structured.py).

### Batched inference

Offline vLLM inference support batch inference by default, the default batch size (number of sequences to process at a time) is equal to 256. Set the `batch_size` argument to change the default value:

```python
  poet = curator.LLM(
        model_name=model_path,
        backend="vllm",
        batch_size=32
    )
  poems = poet("Write a poem.")
```

### Details on vLLM specific arguments

  - `max_model_length` (int, optional): The maximum model context length. Defaults to 4096.

  - `enforce_eager` (bool, optional): Whether to enforce eager execution. Defaults to False.

  - `tensor_parallel_size` (int, optional): The tensor parallel size. Defaults to 1.

  - `gpu_memory_utilization` (float, optional): The GPU memory utilization. Defaults to 0.95.

  - `max_tokens` (int, optional): The maximum number of tokens for models to generate. Defaults to 1024.

## Full list of vLLM examples

- [Generate recipes with Meta LLama 3.1 8B offline](examples/vllm-offline/vllm_recipe.py)
- [Recipes with structured output](examples/vllm-offline/vllm_recipe_structured.py)
- [Use vLLM OpenAI compatible server](examples/vllm-online/vllm_online.py)
- [Use vLLM OpenAI compatible server with structured output](examples/vllm-online/vllm_online_structured.py)
