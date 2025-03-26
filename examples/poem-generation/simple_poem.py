"""Curator example that uses the base LLM class to generate poems.

Please see the poem.py for more complex use cases.
"""

from bespokelabs import curator

# Use GPT-4o-mini for this example.
llm = curator.LLM(model_name="gpt-4o-mini")
poem = llm(["Write a poem about the importance of data in AI."])
print(poem.to_pandas().iloc[0]["response"])

# Use Claude 3.5 Sonnet for this example.
llm = curator.LLM(model_name="claude-3-5-sonnet-20240620", backend="litellm")
poem = llm(["Write a poem about the importance of data in AI."])
print(poem.to_pandas().iloc[0]["response"])

# Note that we can also pass a list of prompts to generate multiple responses.
poems = llm(
    [
        "Write a sonnet about the importance of data in AI.",
        "Write a haiku about the importance of data in AI.",
    ]
)
print(poems.to_pandas()["response"].tolist())
