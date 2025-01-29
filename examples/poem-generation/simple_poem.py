"""Curator example that uses the base LLM class to generate poems.

Please see the poem.py for more complex use cases.
"""

from bespokelabs import curator

# Use GPT-4o-mini for this example.
llm = curator.LLM(model_name="gpt-4o-mini")
poem = llm(["Write a poem about the importance of data in AI."])
print(poem.to_pandas().iloc[0]["response"])


# Note that we can also pass a list of prompts to generate multiple responses.
poems = llm(
    [
        "Write a sonnet about the importance of data in AI.",
        "Write a haiku about the importance of data in AI.",
    ]
)

poems = poems.select(lambda x: x["response"].startswith("Write a sonnet"))
poems = poems.filter(lambda x: x["response"].startswith("Write a sonnet"))

# import pdb; pdb.set_trace()
poems.push_to_hub("pimpalgaonkar/poems_test")

# print(poems.to_pandas()["response"].tolist())

