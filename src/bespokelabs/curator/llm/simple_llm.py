from typing import List, Union

from bespokelabs.curator.llm.llm import LLM
from datasets import Dataset


class SimpleLLM:
    """A simpler interface for the LLM class.

    Usage:
      llm = SimpleLLM(model_name="gpt-4o-mini")
      llm("Do you know about the bitter lesson?")
      llm(["What is the capital of France?", "What is the capital of Germany?"])
    For more complex use cases (e.g. structured outputs and custom prompt functions), see the LLM class.
    """

    def __init__(self, model_name: str, backend: str = "openai"):
        """Initialize the SimpleLLM instance."""
        self._model_name = model_name
        self._backend = backend

    def __call__(self, prompt: Union[str, List[str]]) -> Union[str, List[str]]:
        """Call the SimpleLLM instance."""
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        dataset: Dataset = Dataset.from_dict({"prompt": prompt_list})

        llm = LLM(
            prompt_func=lambda row: row["prompt"],
            model_name=self._model_name,
            response_format=None,
            backend=self._backend,
        )
        response = llm(dataset)
        if isinstance(prompt, str):
            return response["response"][0]
        return response["response"]
