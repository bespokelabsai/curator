# Get a key from https://console.mistral.ai/api-keys. Please note Mistral's experimental keys do not work with batch mode. Choose pay per use.
# Set environment variable to MISTRAL_API_KEY='<ENTER_API_KEY>'

from bespokelabs import curator

# To visualize the dataset on Curator viewer, you can set CURATOR_VIEWER=1 environment variable, or set it here:
# import os
# os.environ["CURATOR_VIEWER"]="1"


llm = curator.LLM(model_name="mistral-tiny", batch=True)
questions = [
    "What is the capital of Montana?",
    "Who wrote the novel 'Pride and Prejudice'?",
    "What is the largest planet in our solar system?",
    "In what year did World War II end?",
    "What is the chemical symbol for gold?",
]
ds = llm(questions)

# Check the first response
print(ds.dataset[0])
