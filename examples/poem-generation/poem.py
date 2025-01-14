"""Example of using the curator library to generate diverse poems.

We generate 10 diverse topics and then generate 2 poems for each topic.
"""

from typing import List

from datasets import Dataset
from pydantic import BaseModel, Field

from bespokelabs import curator


# We use Pydantic and structured outputs to define the format of the response.
# This defines a list of topics, which is the response format for the topic generator.
class Topics(BaseModel):
    """A list of topics."""

    topics_list: List[str] = Field(description="A list of topics.")


# We define a prompter that generates topics.
topic_generator = curator.LLM(
    prompt_func=lambda: "Generate 10 diverse topics that are suitable for writing poems about.",
    model_name="gpt-4o-mini",
    response_format=Topics,
    parse_func=lambda _, topics: [{"topic": t} for t in topics.topics_list],
)

# We call the prompter to generate the dataset.
# When no input dataset is provided, an "empty" dataset with a single row is used as a starting point.
topics: Dataset = topic_generator()
print(topics["topic"])


# Define a list of poems.
class Poems(BaseModel):
    """A list of poems."""

    poems_list: List[str] = Field(description="A list of poems.")


# We define an `LLM` object that generates poems which gets applied to the topics dataset.
poet = curator.LLM(
    # The prompt_func takes a row of the dataset as input.
    # The row is a dictionary with a single key 'topic' in this case.
    prompt_func=lambda row: f"Write two poems about {row['topic']}.",
    model_name="gpt-4o-mini",
    response_format=Poems,
    # `row` is the input row, and `poems` is the Poems class which is parsed from the structured output from the LLM.
    parse_func=lambda row, poems: [{"topic": row["topic"], "poem": p} for p in poems.poems_list],
)

# We apply the prompter to the topics dataset.
poems = poet(topics)
print(poems.to_pandas())

# Expected output:
#                                           topic                                               poem
# 0                            Dreams vs. reality  In the realm where dreams take flight,\nWhere ...
# 1                            Dreams vs. reality  Reality stands with open eyes,\nA weighty thro...
# 2           Urban loneliness in a bustling city  In the city's heart where shadows blend,\nAmon...
# 3           Urban loneliness in a bustling city  Among the crowds, I walk alone,\nA sea of face...
