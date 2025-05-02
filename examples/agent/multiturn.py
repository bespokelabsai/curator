from bespokelabs.curator.agent.agent import Agent, MultiTurnAgents


class Doctor(Agent):
    def prompt(self, text: str):
        text = text["prompt"]
        return [
            {"role": "user", "content": text},
        ]


class Patient(Agent):
    def prompt(self, text: str):
        text = text["prompt"]
        return [
            {
                "role": "user",
                "content": text,
            },
        ]


# TODO:
# 1. Converstaion history support
# 3. Cache support
# 2. Stop condition support
# 4. Dataset support
# 5. Ratelimiting support
# 6. Status tracker support for cost tracking

doctor = Doctor(
    name="Doctor",
    model_name="gpt-4o-mini",
    system_prompt="You are a doctor. You are in conversation with a patient. Respond in a way that is helpful to the patient.",
)
patient = Patient(
    name="Patient",
    model_name="gpt-4o-mini",
    system_prompt="You are a Patient. You are visiting a doctor because you have fever. Doctor will ask you questions as user role.",
)
simulator = MultiTurnAgents(doctor, patient, max_length=5, seed_message="Hello, how are you?")
ds = simulator()
for i in ds:
    print(i)
breakpoint()


"""
# 
#

"""
