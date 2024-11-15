<p align="center">
  <a href="https://bespokelabs.ai/" target="_blank">
    <picture>
      <source media="(prefers-color-scheme: light)" width="10%" srcset="./docs/Bespoke-Labs-Logomark-Red.png">
      <img alt="Bespoke Labs Logo" width="10%" src="./docs/Bespoke-Labs-Logomark-Red-on-Black.png">
    </picture>
  </a>
</p>

<h1 align="center">Bespoke Labs Curator</h1>
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


### Installation

```bash
pip install bespokelabs-curator
```

### Usage

```python
from bespokelabs import curator
from datasets import Dataset
from pydantic import BaseModel, Field
from typing import List

# Create a dataset object for the topics you want to create the poems.
topics = Dataset.from_list([{"topic": "Dreams vs. reality"},
                            {"topic": "Urban loneliness in a bustling city"},
                            {"topic": "Beauty of Bespoke Labs's Curator library"}])

# Define a class to encapsulate a list of poems.
class Poems(BaseModel):
    poems_list: List[str] = Field(description="A list of poems.")


# We define a prompter that generates poems which gets applied to the topics dataset.
poet = curator.Prompter(
    # The prompt_func takes a row of the dataset as input.
    # The row is a dictionary with a single key 'topic' in this case.
    prompt_func=lambda row: f"Write two poems about {row['topic']}.",
    model_name="gpt-4o-mini",
    response_format=Poems,
    # `row` is the input row, and `poems` is the Poems class which 
    # is parsed from the structured output from the LLM.
    parse_func=lambda row, poems: [
        {"topic": row["topic"], "poem": p} for p in poems.poems_list
    ],
)

# We apply the prompter to the topics dataset.
poems = poet(topics)
```
Note that `topics` can be created with `curator.Prompter` as well,
and we can scale this up to create tens of thousands of diverse poems.
You can see a more detailed example in the [examples/poem.py](examples/poem.py) file,
and other examples in the [examples](examples) directory.

To run the examples, make sure to set your OpenAI API key in 
the environment variable `OPENAI_API_KEY` by running `export OPENAI_API_KEY=sk-...` in your terminal.

See the [docs](https://docs.bespokelabs.ai/) for more details as well as troubleshooting.

## Bespoke Curator Viewer

To run the bespoke dataset viewer:

```bash
curator-viewer
```

This will pop up a browser window with the viewer running on `127.0.0.1:3000` by default if you haven't specified a different host and port.


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
