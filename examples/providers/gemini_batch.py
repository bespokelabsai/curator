import os

from bespokelabs import curator

# For more information checkout https://docs.bespokelabs.ai/bespoke-curator/how-to-guides/using-gemini-for-batch-inference


# To visualize the dataset on Curator viewer, you can set HOSTED_CURATOR_VIEWER=1 environment variable, or set it here:
# os.environ["HOSTED_CURATOR_VIEWER"]="1"

# os.environ["GOOGLE_CLOUD_PROJECT"] = "<project-id>"
# os.environ["GEMINI_BUCKET_NAME"] = "<bucket-name>"
os.environ["GOOGLE_CLOUD_REGION "] = "us-central1"  # us-central1 is default

llm = curator.LLM(model_name="gemini-1.5-flash-001", backend="gemini", batch=True)
questions = [
    "What is the capital of Montana?",
    "Who wrote the novel 'Pride and Prejudice'?",
    "What is the largest planet in our solar system?",
    "In what year did World War II end?",
    "What is the chemical symbol for gold?",
]
ds = llm(questions)
# Check the first response
print(ds[0])
