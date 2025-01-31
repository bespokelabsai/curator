from bespokelabs import curator

# https://openrouter.ai/deepseek/deepseek-r1

# information about limits are here
# https://openrouter.ai/docs/limits
# https://openrouter.ai/credits

# Using openai backed - gives a litellm error when using generation_params
# llm = curator.LLM(model_name="deepseek/deepseek-r1",
#                    backend="openai",
#                 #    generation_params={"include_reasoning": True},
#                    backend_params={"base_url": "https://openrouter.ai/api/v1",
#                                   "api_key": os.environ.get("OPENROUTER_API_KEY"),
#                                   "max_retries": 1})

# Using litellm backend
llm = curator.LLM(model_name="openrouter/deepseek/deepseek-r1", generation_params={"include_reasoning": True}, backend_params={"max_retries": 1})

# to get the reasoning traces we need to do extra_body={"include_reasoning": True} above
# without it we get
# ['2 + 2 equals 4. This is a fundamental result in arithmetic where the sum of the integers 2 and 2 is 4.']
# with it we get
#
# 101.71s
response = llm("What is 2 + 2?")
print(response["response"])
