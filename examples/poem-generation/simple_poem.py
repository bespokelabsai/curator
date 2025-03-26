"""Curator example that uses the base LLM class to generate poems.

Please see the poem.py for more complex use cases.
"""

from bespokelabs import curator
import os
os.environ["CURATOR_DISABLE_CACHE"] = "1"
# Use GPT-4o-mini for this example.
class SimpleLLM(curator.LLM):
    def parse(self, row: dict, response: str) -> str:
        return [{"input": row, "response": response}]

llm = SimpleLLM(model_name="gpt-4o-mini")
poem = llm(["Write a poem about the importance of data in AI.", "Write a three line poem that always starts with letter A and ends with letter B for each line."])
print(poem.to_pandas())

