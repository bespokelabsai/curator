import os

from datasets import Dataset

from bespokelabs import curator


class Reasoner(curator.LLM):
    """Curator class for reasoning."""

    return_completions_object = True

    def prompt(self, input):
        """Create a prompt for the LLM to reason about the problem."""
        return [{"role": "user", "content": input["problem"]}]

    def parse(self, input, response):
        """Parse the LLM response to extract reasoning and solution."""
        return [
            {
                "problem": input["problem"],
                "deepseek_reasoning": response["choices"][0]["message"]["reasoning"],
                "deepseek_solution": response["choices"][0]["message"]["content"],
            }
        ]


llm = Reasoner(
    model_name="deepseek/deepseek-r1",  # https://openrouter.ai/deepseek/deepseek-r1
    backend="openai",  # Generic openai API compatible backend
    generation_params={
        "include_reasoning": True,  # https://openrouter.ai/announcements/reasoning-tokens-for-thinking-models
        "max_tokens": 40000,  # Matching DeepSeek's official API that has 32k CoT and 8k max answer tokens
        "provider": {  # https://openrouter.ai/docs/provider-routing
            "order": ["Fireworks", "Kluster"],  # https://openrouter.ai/deepseek/deepseek-r1/providers
            "allow_fallbacks": False,  # Strictly require providers given, which have the highest output tokens
            "sort": "throughput",  # Use the highest throughput provider
            "require_parameters": True,  # Require include_reasoning
        },
    },
    backend_params={
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "max_retries": 5,  # Some requests fail due to provider rate limiting or other issues
        "max_requests_per_minute": 20,  # Based on rate limiting of the providers
        "max_tokens_per_minute": 10000000000000,  # Set high to only rate limit on requests
        "request_timeout": 60 * 30,  # Reasoning models have lots of tokens and can take a while to respond
    },
)

dataset = Dataset.from_dict({"problem": ["Find the sum of all integer bases $b > 9$ for which $17_b$ is a divisor of $97_b$."]})  # The answer is 70 (AIME25)

response = llm(dataset)
print("REASONING: ", response.dataset["deepseek_reasoning"])
print("\n\nSOLUTION: ", response.dataset["deepseek_solution"])
