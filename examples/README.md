## Examples
We have a number of examples to help you get started with `curator`.

1. [Persona Hub](./persona_hub/)
Shows how to use `curator` to use the diverse personas in [persona-hub](https://github.com/tencent-ailab/persona-hub/)
to generate diverse datasets.

2. [Ungrounded QA](./ungrounded-qa/)
Shows how to use `curator` to generate diverse question-answer pairs using techniques similar to what's in the [Camel paper](https://arxiv.org/pdf/2303.17760).

3. [Poem Generation](./poem-generation/)
Shows how to use `curator` to generate diverse poems.

4. [Recipe Generation](./litellm-recipe-generation/)
Shows how to use `curator` along with the litellm backend to generate diverse recipes using non-openai models.

5. [Reannotation](./reannotation/)
Shows how to use `curator` to take an existing dataset and reannotate it with a new model.

6. [Use a vLLM hosted local model for data generation](./vllm-online/)
Shows how to setup a vLLM OpenAI compatible server to serve a local model and use it for online generation in curator.

7. [Use a local model for offline inference via vLLM](./vllm-offline/)
Shows how to run curator with a local model offline.


## How to run

Go to the folder, and install dependencies:
```bash
pip install -r requirements.txt
```

Run the example:
```bash
python <example>.py
```
