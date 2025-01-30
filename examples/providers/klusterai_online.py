from bespokelabs import curator
import litellm

litellm.suppress_debug_info = True

litellm.register_model(
    {
        "klusterai/Meta-Llama-3.1-8B-Instruct-Turbo": {
            "max_tokens": 8192, 
            "input_cost_per_token": 0.18 / 1e6,  # source: https://www.kluster.ai/#pricing
            "output_cost_per_token": 0.18 / 1e6,
            "litellm_provider": "openai",
        },
        "deepseek-ai/DeepSeek-R1": {
            "max_tokens": 8192,
            "input_cost_per_token": 2.00 / 1e6,  # source: https://www.kluster.ai/#pricing
            "output_cost_per_token": 2.00 / 1e6,
            "litellm_provider": "openai",
        }
    }
)

llm = curator.LLM(model_name="klusterai/Meta-Llama-3.1-8B-Instruct-Turbo", backend="klusterai", backend_params={"max_retries": 1})

response = llm("What is the capital of France?")
print(response["response"])

llm = curator.LLM(model_name="deepseek-ai/DeepSeek-R1", backend="klusterai", backend_params={"max_retries": 1})

response = llm("What is the capital of Italy?")
print(response["response"])
