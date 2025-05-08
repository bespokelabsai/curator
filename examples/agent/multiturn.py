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
    advisor_sys_prompt = f.read()

with open("./client-prompt.txt", "r") as f:
    client_sys_prompt = f.read()


def read_seed_messages(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    seed_messages = df["scenario"].tolist() + df["scenario"].tolist()
    return seed_messages


def create_agents(client_model: str, advisor_model: str) -> tuple[Client, Advisor]:
    """Create new instances of client and advisor agents."""
    client = Client(
        name="client",
        model_name=client_model,
        backend="litellm",
        system_prompt=client_sys_prompt,
    )
    advisor = Advisor(
        name="advisor",
        model_name=advisor_model,
        backend="litellm",
        system_prompt=advisor_sys_prompt,
    )
    return client, advisor


def simulate_conversation(seed_message: str, client_model: str, advisor_model: str, max_length: int = 5) -> Dict:
    """Run a single conversation simulation asynchronously."""
    client, advisor = create_agents(client_model, advisor_model)
    simulator = MultiTurnAgents(client, advisor, max_length=max_length, seed_message=seed_message)
    return simulator()


def run_simulations(seed_messages: List[str], max_workers: int = 50, max_length: int = 20):
    all_datasets = []

    client_model = "gemini/gemini-2.5-pro-preview-05-06"
    advisor_model = "gemini/gemini-2.5-pro-preview-05-06"

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(simulate_conversation, msg, client_model, advisor_model, max_length): msg for msg in seed_messages}

        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
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


if __name__ == "__main__":
    csv_path = "scenarios.csv"
    seed_messages = read_seed_messages(csv_path)
    combined_dataset = run_simulations(seed_messages=seed_messages, max_workers=5, max_length=2)

    combined_dataset.to_csv("scenarios-conversations.csv")
    print("Saved to scenarios-conversations.csv")
