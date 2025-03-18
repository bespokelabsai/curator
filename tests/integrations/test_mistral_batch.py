import pytest
from dotenv import load_dotenv

from bespokelabs import curator

load_dotenv(dotenv_path=".env", verbose=True)


@pytest.fixture
def llm():
    """Fixture to initialize the LLM model for batch processing."""
    return curator.LLM(model_name="mistral-tiny", batch=True)


def generate_test_prompts():
    """Generate a batch of test prompts to validate batch processing."""
    return [
        "Describe the impact of quantum computing on AI in 2 sentences.",
        "Give a creative business idea for a startup in 2025 in 1 sentence.",
        "What is 2*10?",
        "Who is the current prime minister of India?",
    ]


@pytest.mark.integration
def test_mistral_batch_response(llm):
    """Test sending a batch of requests to the Mistral AI model via Curator."""

    prompts = generate_test_prompts()
    responses = llm(prompts)  # Send batch request
    df_responses = responses.to_pandas()

    assert len(df_responses) == len(prompts), "Number of responses should match number of prompts."

    for i, prompt in enumerate(prompts):
        response_text = df_responses.iloc[i].get("response", "")

        assert isinstance(response_text, str), f"Response for prompt {i} should be a string."
        assert response_text.strip(), f"Response for prompt {i} should not be empty."
        assert "error" not in response_text.lower(), f"Response for prompt {i} contains an error message."

        print(f"\nPrompt: {prompt}\nResponse: {response_text}\n{'-'*50}")

    print("\nBatch job completed successfully.")
