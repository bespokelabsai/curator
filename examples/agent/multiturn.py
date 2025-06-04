"""This script demonstrates a simple conversation between a patient and a doctor using Curator's agent framework."""

import os

os.environ["CURATOR_DISABLE_RICH_DISPLAY"] = "1"

from bespokelabs.curator.agent.agent import Agent, MultiTurnAgents


class Patient(Agent):
    """A patient agent that represents a person seeking medical advice."""

    def prompt(self, text: str):
        """Convert input text into a patient message."""
        return [{"role": "user", "content": text["prompt"]}]

    def is_completed(self, response: str) -> bool:
        """Check if the conversation is complete based on the response."""
        return response.strip().lower() in ["done", "done."]


class Doctor(Patient):
    """A doctor agent that provides medical advice."""


# System prompts for the agents
DOCTOR_PROMPT = """You are a knowledgeable and empathetic doctor. Your role is to:
1. Listen carefully to the patient's concerns
2. Ask relevant follow-up questions
3. Provide clear, professional medical advice
4. Show empathy while maintaining professionalism
5. Recommend seeing a doctor in person for serious concerns

Remember: You are an AI assistant and cannot provide actual medical diagnosis or treatment.

Please reply with "Done" when you are finished with the conversation."""

PATIENT_PROMPT = """You are a patient seeking medical advice. Your role is to:
1. Describe your symptoms clearly
2. Answer the doctor's questions honestly
3. Ask relevant questions about your condition
4. Express any concerns you have
5. Be cooperative and respectful

Important: Please reply with "Done" when you are finished with the conversation.
"""


def main():
    """Main function to run the conversation between the patient and the doctor."""
    # Create the patient and doctor agents
    patient = Patient(
        name="patient",
        model_name="gpt-4o-mini",  # You can change this to any supported model
        backend="openai",
        system_prompt=PATIENT_PROMPT,
    )

    doctor = Doctor(
        name="doctor",
        model_name="gpt-4o-mini",  # You can change this to any supported model
        backend="openai",
        system_prompt=DOCTOR_PROMPT,
    )

    # Create the conversation simulator
    simulator = MultiTurnAgents(
        patient,
        doctor,
        max_length=10,  # Maximum 10 turns in the conversation
        seed_message="I've been having a persistent headache for the past 3 days.",
    )

    # Run the conversation
    result = simulator()
    breakpoint()

    # Print the conversation
    print("\nConversation History:")
    print("====================")
    for message in result.dataset.to_list():
        role = message["role"].capitalize()
        content = message["content"]
        print(f"\n{role}: {content}")


if __name__ == "__main__":
    main()
