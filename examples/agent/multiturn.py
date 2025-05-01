from bespokelabs.curator.agent.agent import Agent, MultiTurnAgents


class Doctor(Agent):
    def prompt(self, text: str):
        text = text['prompt']
        return [
            {
                "role": "system",
                "content": "You are a doctor. You are in conversation with a patient. Respond in a way that is helpful to the patient.",
            },
            {"role": "user", "content": text},
        ]


class Patient(Agent):
    def prompt(self, text: str):
        text = text['prompt']
        return [
            {
                "role": "system",
                "content": "You are a Patient. You are in conversation with a doctor. Respond by asking questions to the doctor.",
            },
            {
                "role": "user",
                "content": text,
            },
        ]


doctor = Doctor(name="Doctor", model_name="gpt-4o-mini")
patient = Patient(name="Patient", model_name="gpt-4o-mini")
simulator = MultiTurnAgents(doctor, patient, total_steps=10, seed_message="Hello, how are you?")
ds = simulator()
print(ds)
