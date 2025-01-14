<p align="center">
  <a href="https://bespokelabs.ai/" target="_blank">
    <picture>
      <source media="(prefers-color-scheme: light)" width="80px" srcset="https://raw.githubusercontent.com/bespokelabsai/curator/main/docs/Bespoke-Labs-Logomark-Red.png">
      <img alt="Bespoke Labs Logo" width="80px" src="https://raw.githubusercontent.com/bespokelabsai/curator/main/docs/Bespoke-Labs-Logomark-Red-on-Black.png">
    </picture>
  </a>
</p>

<h1 align="center">Bespoke Curator</h1>
<h3 align="center" style="font-size: 20px; margin-bottom: 4px">Data Curation for Post-Training & Structured Data Extraction</h3>
<br/>
<p align="center">
  <a href="https://docs.bespokelabs.ai/">
    <img alt="Static Badge" src="https://img.shields.io/badge/Docs-docs.bespokelabs.ai-blue?style=flat&link=https%3A%2F%2Fdocs.bespokelabs.ai">
  </a>
  <a href="https://bespokelabs.ai/">
    <img alt="Site" src="https://img.shields.io/badge/Site-bespokelabs.ai-blue?link=https%3A%2F%2Fbespokelabs.ai"/>
  </a>
  <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/bespokelabs-curator">
  <a href="https://twitter.com/bespokelabsai">
    <img src="https://img.shields.io/twitter/follow/bespokelabsai" alt="Follow on X" />
  </a>
  <a href="https://discord.gg/KqpXvpzVBS">
    <img alt="Discord" src="https://img.shields.io/discord/1230990265867698186">
  </a>
</p>

## Overview

Bespoke Curator makes it very easy to create high-quality synthetic data at scale, which you can use to finetune models or use for structured data extraction at scale.

Bespoke Curator is an open-source project:
* That comes with a rich Python based library for generating and curating synthetic data.
* A Curator Viewer which makes it easy to view the datasets, thus aiding in the dataset creation.
* We will also be releasing high-quality datasets that should move the needle on post-training.

## Key Features

1. **Programmability and Structured Outputs**: Synthetic data generation is lot more than just using a single prompt -- it involves calling LLMs multiple times and orchestrating control-flow. Curator treats structured outputs as first class citizens and helps you design complex pipelines.
2. **Built-in Performance Optimization**: We often see calling LLMs in loops, or inefficient implementation of multi-threading. We have baked in performance optimizations so that you don't need to worry about those!
3. **Intelligent Caching and Fault Recovery**: Given LLM calls can add up in cost and time, failures are undesirable but sometimes unavoidable. We cache the LLM requests and responses so that it is easy to recover from a failure. Moreover, when working on a multi-stage pipeline, caching of stages makes it easy to iterate.
4. **Native HuggingFace Dataset Integration**: Work directly on HuggingFace Dataset objects throughout your pipeline. Your synthetic data is immediately ready for fine-tuning!
5. **Interactive Curator Viewer**: Improve and iterate on your prompts using our built-in viewer. Inspect LLM requests and responses in real-time, allowing you to iterate and refine your data generation strategy with immediate feedback.

## Installation

```bash
pip install bespokelabs-curator
```

## Usage
To run the examples below, make sure to set your OpenAI API key in
the environment variable `OPENAI_API_KEY` by running `export OPENAI_API_KEY=sk-...` in your terminal.

### Hello World with `SimpleLLM`: A simple interface for calling LLMs

```python
from bespokelabs import curator
llm = curator.SimpleLLM(model_name="gpt-4o-mini")
poem = llm("Write a poem about the importance of data in AI.")
print(poem)
# Or you can pass a list of prompts to generate multiple responses.
poems = llm(["Write a poem about the importance of data in AI.",
            "Write a haiku about the importance of data in AI."])
print(poems)
```
Note that retries and caching are enabled by default.
So now if you run the same prompt again, you will get the same response, pretty much instantly.
You can delete the cache at `~/.cache/curator`.

#### Use LiteLLM backend for calling other models
You can use the [LiteLLM](https://docs.litellm.ai/docs/providers) backend for calling other models.

```python
from bespokelabs import curator
llm = curator.SimpleLLM(model_name="claude-3-5-sonnet-20240620", backend="litellm")
poem = llm("Write a poem about the importance of data in AI.")
print(poem)
```

### Visualize in Curator Viewer
Run `curator-viewer` on the command line to see the dataset in the viewer.

You can click on a run and then click on a specific row to see the LLM request and response.
![Curator Responses](docs/curator-responses.png)
More examples below.

### `LLM`: A more powerful interface for synthetic data generation

Let's use structured outputs to generate poems.
```python
from bespokelabs import curator
from datasets import Dataset
from pydantic import BaseModel, Field
from typing import List

topics = Dataset.from_dict({"topic": [
    "Urban loneliness in a bustling city",
    "Beauty of Bespoke Labs's Curator library"
]})
```

Define a class to encapsulate a list of poems.
```python
class Poem(BaseModel):
    poem: str = Field(description="A poem.")

class Poems(BaseModel):
    poems_list: List[Poem] = Field(description="A list of poems.")
```

We define an `LLM` object that generates poems which gets applied to the topics dataset.
```python
poet = curator.LLM(
    prompt_func=lambda row: f"Write two poems about {row['topic']}.",
    model_name="gpt-4o-mini",
    response_format=Poems,
    parse_func=lambda row, poems: [
        {"topic": row["topic"], "poem": p.poem} for p in poems.poems_list
    ],
)
```
Here:
* `prompt_func` takes a row of the dataset as input and returns the prompt for the LLM.
* `response_format` is the structured output class we defined above.
* `parse_func` takes the input (`row`) and the structured output (`poems`) and converts it to a list of dictionaries. This is so that we can easily convert the output to a HuggingFace Dataset object.

Now we can apply the `LLM` object to the dataset, which reads very pythonic.
```python
poem = poet(topics)
print(poem.to_pandas())
# Example output:
#    topic                                     poem
# 0  Urban loneliness in a bustling city       In the city's heart, where the sirens wail,\nA...
# 1  Urban loneliness in a bustling city       City streets hum with a bittersweet song,\nHor...
# 2  Beauty of Bespoke Labs's Curator library  In whispers of design and crafted grace,\nBesp...
# 3  Beauty of Bespoke Labs's Curator library  In the hushed breath of parchment and ink,\nBe...
```
Note that `topics` can be created with `curator.LLM` as well,
and we can scale this up to create tens of thousands of diverse poems.
You can see a more detailed example in the [examples/poem.py](https://github.com/bespokelabsai/curator/blob/mahesh/update_doc/examples/poem.py) file,
and other examples in the [examples](https://github.com/bespokelabsai/curator/blob/mahesh/update_doc/examples) directory.

See the [docs](https://docs.bespokelabs.ai/) for more details as well as
for troubleshooting information.

## Bespoke Curator Viewer

To run the bespoke dataset viewer:

```bash
curator-viewer
```

This will pop up a browser window with the viewer running on `127.0.0.1:3000` by default if you haven't specified a different host and port.

The dataset viewer shows all the different runs you have made.
![Curator Runs](docs/curator-runs.png)

You can also see the dataset and the responses from the LLM.
![Curator Dataset](docs/curator-dataset.png)


Optional parameters to run the viewer on a different host and port:
```bash
>>> curator-viewer -h
usage: curator-viewer [-h] [--host HOST] [--port PORT] [--verbose]

Curator Viewer

options:
  -h, --help     show this help message and exit
  --host HOST    Host to run the server on (default: localhost)
  --port PORT    Port to run the server on (default: 3000)
  --verbose, -v  Enables debug logging for more verbose output
```

The only requirement for running `curator-viewer` is to install node. You can install them by following the instructions [here](https://nodejs.org/en/download/package-manager).

For example, to check if you have node installed, you can run:

```bash
node -v
```

If it's not installed, installing latest node on MacOS, you can run:

```bash
# installs nvm (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
# download and install Node.js (you may need to restart the terminal)
nvm install 22
# verifies the right Node.js version is in the environment
node -v # should print `v22.11.0`
# verifies the right npm version is in the environment
npm -v # should print `10.9.0`
```

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
os.environ["HOSTED_VLLM_API_KEY"] = "token-abc123"

poem_prompter = curator.LLM(
    model_name=model_path,
    prompt_func=lambda row: "Generate a poem",
    backend="litellm",
    base_url=URL,
)
poem_prompter()
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

To use curator with a local model, povide your model local path in the model argument and set the backend to `vllm`:

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
      prompt_func=lambda: f"Generate 10 diverse cuisines.",
      model_name=model_path,
      response_format=Cuisines,
      backend="vllm",
  )
```

See full example [here](examples/vllm-offline/vllm_recipe_structured.py).

### Batched inference

Offline vLLM inference support batch inference by default, the default batch size (number of sequences to process at a time) is equal to 256. Set the `batch_size` argument to change the default value:

```python
  curator.LLM(
        model_name=model_path,
        prompt_func=lambda row: f"write a poem",
        backend="vllm",
        batch_size=32
    )
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
- [Use vLLM OpeneAI compatible server](examples/vllm-online/vllm_online.py)
- [Use vLLM OpenAI compatible server with structured output](examples/vllm-online/vllm_online_structured.py)

## Contributing
Contributions are welcome!
