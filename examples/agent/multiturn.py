import asyncio
import concurrent.futures
import os
from typing import Dict, List

os.environ["CURATOR_DISABLE_RICH_DISPLAY"] = "1"

import pandas as pd

from bespokelabs.curator.agent.agent import Agent, MultiTurnAgents


class Client(Agent):
    def prompt(self, text: str):
        text = text["prompt"]
        return [
            {
                "role": "user",
                "content": text,
            },
        ]


class Advisor(Client):
    pass


with open("./advisor-prompt.txt", "r") as f:
    adisor_sys_prompt = f.read()

with open("./client-prompt.txt", "r") as f:
    client_sys_prompt = f.read()

client = Client(
    name="client",
    model_name="gemini/gemini-2.0-flash",
    backend="litellm",
    system_prompt=client_sys_prompt,
)
advisor = Advisor(
    name="advisor",
    model_name="gemini/gemini-2.0-flash",
    backend="litellm",
    system_prompt=adisor_sys_prompt,
)


def read_seed_messages(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    seed_messages = df["scenario"].tolist() + df["scenario"].tolist()
    return seed_messages


def simulate_conversation(seed_message: str, client: Agent, advisor: Agent, max_length: int = 5) -> Dict:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    simulator = MultiTurnAgents(client, advisor, max_length=max_length, seed_message=seed_message)
    return loop.run_until_complete(simulator())


def run_simulations(seed_messages: List[str], client: Agent, advisor: Agent, max_workers: int = 50, max_length: int = 20):
    all_datasets = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_seed = {executor.submit(simulate_conversation, msg, client, advisor, max_length): msg for msg in seed_messages}

        for future in concurrent.futures.as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                dataset = future.result()
                all_datasets.append(dataset)
                print(f"Completed simulation for seed: {seed[:30]}...")
            except Exception as exc:
                print(f"Simulation for seed {seed[:30]}... generated an exception: {exc}")

    if all_datasets:
        combined_data = []
        for scenario, dataset in zip(seed_messages, all_datasets):
            conversation = {}
            conversation["conversation"] = dataset.to_list()
            conversation["scenario"] = scenario
            combined_data.append(conversation)

        combined_dataset = pd.DataFrame(combined_data)
        return combined_dataset
    return None


csv_path = "scenarios.csv"
seed_messages = read_seed_messages(csv_path)
combined_dataset = run_simulations(seed_messages=seed_messages, client=client, advisor=advisor, max_workers=50, max_length=20)

combined_dataset.to_csv("scenarios-conversations.csv")
print("Saved to scenarios-conversations.csv")
