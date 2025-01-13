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


# We define a topic generator class that inherits from LLM
class TopicGenerator(curator.LLM):
    """A topic generator that generates diverse topics for poems."""

    response_format = Topics

    @classmethod
    def prompt(cls, input: dict) -> str:
        """Generate a prompt for the topic generator."""
        return "Generate 10 diverse topics that are suitable for writing poems about."

    @classmethod
    def parse(cls, input: dict, response: Topics) -> dict:
        """Parse the model response into the desired output format."""
        return [{"topic": t} for t in response.topics_list]


# We instantiate the topic generator and call it to generate topics
topic_generator = TopicGenerator(model_name="gpt-4o-mini")
topics: Dataset = topic_generator()
print(topics["topic"])


# Define a list of poems.
class Poems(BaseModel):
    """A list of poems."""

    poems_list: List[str] = Field(description="A list of poems.")


# We define a poet class that inherits from LLM
class Poet(curator.LLM):
    """A poet that generates poems about given topics."""

    response_format = Poems

    @classmethod
    def prompt(cls, input: dict) -> str:
        """Generate a prompt using the topic."""
        return f"Write two poems about {input['topic']}."

    @classmethod
    def parse(cls, input: dict, response: Poems) -> dict:
        """Parse the model response into the desired output format."""
        return [{"topic": input["topic"], "poem": p} for p in response.poems_list]


# We instantiate the poet and apply it to the topics dataset
poet = Poet(model_name="gpt-4o-mini")
poems = poet(topics)
print(poems.to_pandas())

# Expected output:
#                                           topic                                               poem
# 0                            Dreams vs. reality  In the realm where dreams take flight,\nWhere ...
# 1                            Dreams vs. reality  Reality stands with open eyes,\nA weighty thro...
# 2           Urban loneliness in a bustling city  In the city's heart where shadows blend,\nAmon...
# 3           Urban loneliness in a bustling city  Among the crowds, I walk alone,\nA sea of face...
