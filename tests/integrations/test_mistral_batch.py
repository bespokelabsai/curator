import pytest
from dotenv import load_dotenv

from bespokelabs import curator

load_dotenv(dotenv_path=".env", verbose=True)


def generate_test_prompts():
    """Generate a batch of 4 test prompts to test batch processing."""
    return [
        "Describe the impact of quantum computing on AI in 2 sentences.",
        "Give  a creative business idea for a startup in 2025 in 1 sentence.",
        "What is 2*10?",
        "Who is the current prime-minister of India?",
    ]


@pytest.mark.skip()
def test_mistral_batch_response():
    """Test sending a batch of requests to the Mistral AI model via Curator."""
    llm = curator.LLM(model_name="mistral-tiny", batch=True)  # Enable batch processing

    prompts = generate_test_prompts()
    responses = llm(prompts)  # Send batch request

    df_responses = responses.to_pandas()

    for i, prompt in enumerate(prompts):
        response_text = df_responses.iloc[i]["response"]
        assert isinstance(response_text, str), f"Response for prompt {i} should be a string"
        assert len(response_text.strip()) > 0, f"Response for prompt {i} should not be empty"
        print(f"\nPrompt: {prompt}\nResponse: {response_text}\n{'-'*50}")

    print("\nBatch job completed successfully.")


if __name__ == "__main__":
    test_mistral_batch_response()
