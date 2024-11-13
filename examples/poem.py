"""Example of using the curator library to generate diverse poems.

We generate 10 diverse topics and then generate 2 poems for each topic.

You can do this in a loop, but that is inefficient and breaks when requests fail.
When you need to do this thousands of times (or more), you need a better abstraction.

curator.Prompter takes care of this heavy lifting.

# Key Components of Prompter

## prompt_func

Calls an LLM on each row of the input dataset in parallel. 

1. Takes a dataset row as input
2. Returns the prompt for the LLM

## parse_func

Converts LLM output into structured data by adding it back to the dataset.

1. Takes two arguments:
    - Input row
    - LLM response (in response_format)
2. Returns new rows (in list of dictionaries)


# Data Flow Example
Input Dataset: 
    Row A
    Row B
Processing by Prompter:
    Row A    →    prompt_func(A)    →    Response R1    →    parse_func(A, R1)    →    [C, D]
    Row B    →    prompt_func(B)    →    Response R2    →    parse_func(B, R2)    →    [E, F]

Output Dataset: 
    Row C
    Row D
    Row E
    Row F

In this example:

- The two input rows (A and B) are processed in parallel to prompt the LLM
- Each generates a response (R1 and R2)
- The parse function converts each response into (multiple) new rows (C, D, E, F)
- The final dataset contains all generated rows

You can chain prompters together to iteratively build up a dataset.
"""

from bespokelabs import curator
from datasets import Dataset
from pydantic import BaseModel, Field
from typing import List


# We use Pydantic and structured outputs to define the format of the response.
# This defines a list of topics, which is the response format for the topic generator.
class Topics(BaseModel):
    topics_list: List[str] = Field(description="A list of topics.")


# We define a prompter that generates topics.
topic_generator = curator.Prompter(
    prompt_func=lambda: f"Generate 10 diverse topics that are suitable for writing poems about.",
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
    poems_list: List[str] = Field(description="A list of poems.")


# We define a prompter that generates poems which gets applied to the topics dataset.
poet = curator.Prompter(
    # The prompt_func takes a row of the dataset as input.
    # The row is a dictionary with a single key 'topic' in this case.
    prompt_func=lambda row: f"Write two poems about {row['topic']}.",
    model_name="gpt-4o-mini",
    response_format=Poems,
    # `row` is the input row, and `poems` is the Poems class which is parsed from the structured output from the LLM.
    parse_func=lambda row, poems: [
        {"topic": row["topic"], "poem": p} for p in poems.poems_list
    ],
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
