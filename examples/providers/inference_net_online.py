from bespokelabs.curator import LLM

# Get an API key at https://inference.net
# Export your API key as INFERENCE_API_KEY
llm = LLM(
    model_name="meta-llama/llama-3.1-8b-instruct/fp-16",  # View all available models at https://docs.inference.net/resources/models
    backend="inference.net",
    backend_params={"max_retries": 1},
)

response = llm("Write a short story about a dog and a cat.")
print(response.dataset[0]["response"])
