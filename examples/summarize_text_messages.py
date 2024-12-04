"""Generate a list of personas, generate a list of conversations between two people, and summarize the conversations.

Running:
python3 -m venv .venv
source .venv/bin/activate
pip install bespokelabs-curator datasets
python3 examples/summarize_text_messages.py

And run in another terminal:
source .venv/bin/activate
curator-viewer
"""
from bespokelabs import curator
from pydantic import BaseModel, Field
import numpy as np
from typing import List
from datasets import Dataset


class Person(BaseModel):
    name: str = Field(description="The name of the person")
    persona: str = Field(description="The persona of the person")


class PersonList(BaseModel):
    person_list: List[Person] = Field(description="A list of persons.")


persona_prompter = curator.Prompter(
    prompt_func=lambda: f"Generate a diverse list of 10 personas.",
    parse_func=lambda _, personas: [{"name": persona.name, "persona": persona.persona} for persona in personas.person_list],
    model_name="gpt-4o-mini",
    response_format=PersonList,
)

personas = persona_prompter()
print(personas.to_pandas())
len_personas = len(personas)

pairings = []
length_of_conversations = ["very short", "short", "medium", "long", "very long"]
persona_related = ["very related", "somewhat related", "somewhat unrelated", "very unrelated"]

for i in range(len_personas):
    # Randomly select someone else.
    others_list = list(range(len_personas))
    others_list.pop(i)
    other_index = int(np.random.choice(others_list))
    other_persona = personas[other_index]
    # Randomly select a length of conversation.
    length_of_conversation = np.random.choice(length_of_conversations)
    # Randomly select a persona relatedness.
    persona_relatedness = np.random.choice(persona_related)
    pairings.append({"person1_name": personas[i]["name"],
                     "person1_persona": personas[i]["persona"],
                     "person2_name": other_persona["name"],
                     "person2_persona": other_persona["persona"],
                     "length_of_conversation": length_of_conversation,
                     "persona_relatedness": persona_relatedness})

pairings_dataset = Dataset.from_list(pairings)
print(pairings_dataset.to_pandas())

def message_generator_prompt_func(row):
    user_prompt = "Generate a list of text messages between two people, initiated by the first person.\n\n"
    user_prompt += f"Keep the conversation {row['persona_relatedness']} to the personas.\n\n"
    user_prompt += f'The length of the conversation should be {row["length_of_conversation"]}.\n\n'
    user_prompt += f'The first person is {row["person1_name"]} with persona "{row["person1_persona"]}".\nThe second person is {row["person2_name"]} with persona "{row["person2_persona"]}".'
    return user_prompt

message_generator = curator.Prompter(
    prompt_func=message_generator_prompt_func,
    model_name="gpt-4o-mini",
)

text_summarizer = curator.Prompter(
    prompt_func=lambda row: f"Summarize the following conversations between two people into a very short summary:\n{row['response']}. The summary should be one sentence.",
    model_name="gpt-4o-mini",
)

messages = message_generator(pairings_dataset)
print(messages.to_pandas())

summaries = text_summarizer(messages)
print(summaries.to_pandas())
