from pydantic import BaseModel


class TokenUsage(BaseModel):
    """Token usage information for an API request.

    Attributes:
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total number of tokens used
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
