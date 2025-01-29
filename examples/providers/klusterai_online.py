from bespokelabs import curator

llm = curator.LLM(model_name="klusterai/Meta-Llama-3.1-8B-Instruct-Turbo", backend="klusterai", backend_params={"max_retries": 1})

response = llm("What is the capital of France?")
print(response["response"])

llm = curator.LLM(model_name="deepseek-ai/DeepSeek-R1", backend="klusterai", backend_params={"max_retries": 1})

response = llm("What is the capital of Italy?")
print(response["response"])
