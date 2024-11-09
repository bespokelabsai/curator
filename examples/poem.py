"""Example of using the curator library to generate poems."""
from bespokelabs import curator
from datasets import Dataset
from pydantic import BaseModel, Field
from typing import List

def generate_single_poem():
  # Generate a single poem.
  poet = curator.Prompter(
      prompt_func=lambda: "Write a poem about the beauty of synthetic data.",
      model_name="gpt-4o-mini",
  )
  # poem is a Dataset object with column "response" and a single row.
  poem: Dataset = poet()
  print(poem["response"][0])


def generate_many_poems(how_many: int = 10):
  # Note that we do async calls here, so it's very fast, compared to calling 
  # generate_single_poem() many times sequentially.
  poet = curator.Prompter(
      prompt_func=lambda: "Write a poem about something interesting.",
      model_name="gpt-4o-mini",
  )

  empty_dataset = Dataset.from_dict({"poem": [''] * how_many})
  poems: Dataset = poet(empty_dataset)
  print(poems["response"])

def generate_multiple_poems_single_call():
  # Generate 10 poems using a single call.
  class Poem(BaseModel):
      poem: str = Field(description="A poem.")

  class Poems(BaseModel):
      poems: List[Poem] = Field(description="A list of poems.")

  poet = curator.Prompter(
      prompt_func=lambda: f"Write 10 diverse poems.",
      model_name="gpt-4o-mini",
      response_format=Poems,
      # Parse function expects the input that was passed to 
      # the LLM as well as the output of LLM.
      # Always return a list of dictionaries.
      parse_func=lambda _, poems_obj: [{"poem": p.poem} for p in poems_obj.poems],
  )
  poems: Dataset = poet()
  print(poems["poem"])


def generate_diverse_poems(how_many: int = 10):
  # Generate diverse topics and then generate one poem per topic.
  class Topic(BaseModel):
      topic: str = Field(description="A topic.")

  class Topics(BaseModel):
      topics: List[Topic] = Field(description="A list of topics.")

  topic_generator = curator.Prompter(
      prompt_func=lambda: f"Generate {how_many} diverse topics that are suitable for writing poems about.",
      response_format=Topics,
      model_name="gpt-4o-mini",
      parse_func=lambda _, topics_obj: [{"topic": t.topic} for t in topics_obj.topics],
  )

  topics = topic_generator()
  print(topics.to_pandas())

  class Poem(BaseModel):
      poem: str = Field(description="A poem.")

  poet = curator.Prompter(
      prompt_func=lambda topic: f"Write a poem about {topic}.",
      model_name="gpt-4o-mini",
      response_format=Poem,
      parse_func=lambda _, poem_obj: [{"poem": poem_obj.poem}],
  )
  poems = poet(topics)
  print(poems['poem'])


if __name__ == "__main__":
    generate_single_poem()
    generate_multiple_poems_single_call()
    generate_many_poems(how_many=10)
    generate_diverse_poems(how_many=10)