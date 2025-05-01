from bespokelabs import curator
from bespokelabs.curator.agent.processer import MultiTurnAgenticProcessor
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop
from bespokelabs.curator.llm.llm import _CURATOR_DEFAULT_CACHE_DIR
import typing as t
import os

class Agent(curator.LLM):
    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        super().__init__(*args, **kwargs)

    def __str__(self):
        return f"Agent(name={self.name})"
    
    def __repr__(self):
        return self.__str__()
    

class MultiTurnAgents:
    def __init__(self, seeder: Agent, partner: Agent, total_steps: int, seed_message: str):
        self.seeder = seeder
        self.partner = partner
        self.total_steps = total_steps
        self.seed_message = seed_message
        self._processor = MultiTurnAgenticProcessor(self.seeder, self.partner, self.total_steps, self.seed_message)

    def __call__(self, working_dir: t.Optional[str] = None):
        if working_dir is None:
            working_dir= os.environ.get(
                "CURATOR_CACHE_DIR",
                os.path.expanduser(_CURATOR_DEFAULT_CACHE_DIR),
            )
        disable_cache = os.getenv("CURATOR_DISABLE_CACHE", "").lower() in ["true", "1"]
        fingerprint = self.seeder._hash_fingerprint(disable_cache=disable_cache)
        fingerprint += self.partner._hash_fingerprint(disable_cache=disable_cache)
        working_dir = os.path.join(working_dir, fingerprint)
        os.makedirs(working_dir, exist_ok=True)
        print(f"Running multi turn simulation, find results in {working_dir}")
        return run_in_event_loop(self._processor.run(working_dir=working_dir))