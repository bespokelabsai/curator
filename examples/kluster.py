import os

from bespokelabs import curator

api_key = os.getenv("KLUSTERAI_API_KEY")
print(api_key)


llm = curator.LLM(model_name="klusterai/Meta-Llama-3.1-8B-Instruct-Turbo", batch=True, backend_params={"api_key": api_key, "completion_window": "1h"})

response = llm("What is the capital of France?")
print(response["response"])
