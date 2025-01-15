<p align="center">
  <a href="https://bespokelabs.ai/" target="_blank">
    <picture>
      <source media="(prefers-color-scheme: light)" width="100px" srcset="docs/Bespoke-Labs-Logomark-Red-crop.png">
      <img alt="Bespoke Labs Logo" width="100px" src="docs/Bespoke-Labs-Logomark-Red-crop.png">
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

Bespoke Curator makes it easy to create synthetic data pipelines. Whether you are training a model or extracting structure, Curator will prepare high-quality data quickly and robustly.

* Rich Python based library for generating and curating synthetic data.
* Interactive viewer to monitor data while it is being generated
* First class support for structured outputs
* Built-in performance optimizations for asynchronous operations, caching, and fault recovery at every scale
* Support for a wide range of inference options via LiteLLM, vLLM, and popular batch APIs

![CLI in action](docs/curator-cli.gif)

## Installation

```bash
pip install bespokelabs-curator
```

## Quickstart

### Using `curator.LLM`

```python
from bespokelabs import curator
llm = curator.LLM(model_name="gpt-4o-mini")
poem = llm("Write a poem about the importance of data in AI.")
print(poem['response'])
```

> [!NOTE]
> Retries and caching are enabled by default.
> So now if you run the same prompt again, you will get the same response, pretty much instantly.
> You can delete the cache at `~/.cache/curator` or disable it with `export CURATOR_DISABLE_CACHE=true`.

### Calling other models
You can also call other [LiteLLM](https://docs.litellm.ai/docs/) supported models by
changing the `model_name` argument.

```python
llm = curator.LLM(model_name="claude-3-5-sonnet-20240620")
```

In addition to a wide range of API providers, local web servers (hosted by vLLM or Ollama) are supported by litellm. For completely offline inference directly through vLLM, see the [local models](docs/local_models.md) documentation.

> [!TIP]
> Make sure to set your API keys as environment variables for the model you are calling. For example running `export OPENAI_API_KEY=sk-...` and `export ANTHROPIC_API_KEY=ant-...` will allow you to run the previous two examples. A full list of supported models and their associated environment variable names can be found [in the litellm docs](https://docs.litellm.ai/docs/providers).

### Using structured outputs

Let's use structured outputs to generate multiple poems in a single LLM call. We can define a class to encapsulate a list of poems,
and then pass it to the `LLM` class.

```python
from typing import List
from pydantic import BaseModel, Field
from bespokelabs import curator

class Poem(BaseModel):
    poem: str = Field(description="A poem.")


class Poems(BaseModel):
    poems_list: List[Poem] = Field(description="A list of poems.")


llm = curator.LLM(model_name="gpt-4o-mini", response_format=Poems)
poems = llm(["Write two poems about the importance of data in AI.", 
              "Write three haikus about the importance of data in AI."])
print(poems['responses'])
```

Note how each `Poems` object occupies a single row in the dataset. 


For more advanced use cases, you might need to define more custom parsing and prompting logic. For example, you might want to preserve the mapping between each topic and the poem being generated from it. In this case, you can define a `Poet` object that inherits from `LLM`, and define your own prompting and parsing logic:

```python
from typing import Dict, List
from datasets import Dataset
from pydantic import BaseModel, Field
from bespokelabs import curator


class Poem(BaseModel):
    poem: str = Field(description="A poem.")


class Poems(BaseModel):
    poems: List[Poem] = Field(description="A list of poems.")


class Poet(curator.LLM):
    response_format = Poems

    def prompt(self, input: Dict) -> str:
        return f"Write two poems about {input['topic']}."

    def parse(self, input: Dict, response: Poems) -> Dict:
        return [{"topic": input["topic"], "poem": p.poem} for p in response.poems]


poet = Poet(model_name="gpt-4o-mini")

topics = Dataset.from_dict({"topic": ["Urban loneliness in a bustling city", "Beauty of Bespoke Labs's Curator library"]})
poem = poet(topics)
print(poem.to_pandas())
```
```
                                      topic                                               poem
0       Urban loneliness in a bustling city  In the city’s heart, where the lights never di...
1       Urban loneliness in a bustling city  Steps echo loudly, pavement slick with rain,\n...
2  Beauty of Bespoke Labs's Curator library  In the heart of Curation’s realm,  \nWhere art...
3  Beauty of Bespoke Labs's Curator library  Step within the library’s embrace,  \nA sanctu...
```
In the `Poet` class:
* `response_format` is the structured output class we defined above.
* `prompt` takes the input (`input`) and returns the prompt for the LLM.
* `parse` takes the input (`input`) and the structured output (`response`) and converts it to a list of dictionaries. This is so that we can easily convert the output to a HuggingFace Dataset object.

Note that `topics` can be created with another `LLM` class as well,
and we can scale this up to create tens of thousands of diverse poems.
You can see a more detailed example in the [examples/poem-generation/poem.py](examples/poem-generation/poem.py) file,
and other examples in the [examples](examples) directory.

See the [docs](https://docs.bespokelabs.ai/) for more details as well as
for troubleshooting information.

## Bespoke Curator Viewer

![Viewer in action](docs/curator-viewer.gif)

To run the bespoke dataset viewer:

```bash
curator-viewer
```

This will pop up a browser window with the viewer running on `127.0.0.1:3000` by default if you haven't specified a different host and port.

The dataset viewer shows all the different runs you have made. Once a run is selected, you can see the dataset and the responses from the LLM.

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

## Contributing
Thank you to all the contributors for making this project possible!
Please follow [these instructions](CONTRIBUTING.md) on how to contribute.

## Citation
If you find Curator useful, please consider citing us!

```
@software{Curator: A Tool for Synthetic Data Creation,
  author = {Marten, Ryan and Vu, Trung and Cheng-Jie Ji, Charlie and Sharma, Kartik and Dimakis, Alex and Sathiamoorthy, Mahesh},
  month = jan,
  title = {{Curator}},
  year = {2025}
}
```