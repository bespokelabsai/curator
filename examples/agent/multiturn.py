from bespokelabs.curator.agent.agent import MultiTurnAgents, Agent

class Doctor(Agent):
    def prompt(self, text: str):
        return [{'role': 'system', 'content': 'You are a doctor. You are in conversation with a patient. You need to ask and answer questions to help the patient. You are an agent for generating synthetic converations data.'}, {'role': 'user', 'content': f"Talk to the patient given the following text: {text}"}]

class Patient(Agent):
    def prompt(self, text: str):
        return [{'role': 'system', 'content': 'You are a patient. You are in conversation with a doctor. You need to answer the questions to help the doctor. You are an agent for generating synthetic converations data.'}, {'role': 'user', 'content': f"Talk to the doctor given the following text: {text}, Note: dont response by saying how can I assit you etc, try to generate your query"}]


doctor = Doctor(name="Doctor", model_name="gpt-4o-mini")
patient = Patient(name="Patient", model_name="gpt-4o-mini")
simulator = MultiTurnAgents(doctor, patient, total_steps=10, seed_message="Hello, how are you?")
ds = simulator()
print(ds)