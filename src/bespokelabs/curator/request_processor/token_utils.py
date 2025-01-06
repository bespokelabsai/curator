"""Token counting utilities for consistent token counting across different request processors."""

import tiktoken
from typing import List, Dict, Union, Optional


def get_encoding_for_model(model: str) -> tiktoken.Encoding:
    """Get the appropriate tiktoken encoding for a given model.
    
    Args:
        model: The model identifier (e.g., "gpt-4", "claude-2")
        
    Returns:
        The appropriate tiktoken encoding for the model
    """
    try:
        # First try to get exact encoding for the model
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Fall back to cl100k_base for newer models
        return tiktoken.get_encoding("cl100k_base")


def unified_token_count(model: str, messages: List[Dict[str, str]]) -> int:
    """Calculate token count for a list of messages consistently across all models.
    
    This implementation follows OpenAI's token counting rules but provides
    consistent counting across all model types. It uses tiktoken for accurate
    token counting and handles message formatting consistently.
    
    Args:
        model: The model identifier (e.g., "gpt-4", "claude-2")
        messages: List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        Total number of tokens in the messages
    """
    encoding = get_encoding_for_model(model)
    
    num_tokens = 0
    for message in messages:
        # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        
        for key, value in message.items():
            try:
                # Encode the value and count tokens
                num_tokens += len(encoding.encode(str(value)))
            except TypeError as e:
                # Fallback for any values that can't be directly encoded
                # Use character count divided by 4 as a conservative estimate
                num_tokens += len(str(value)) // 4
                
            # If there's a name, the role is omitted
            if key == "name":
                num_tokens -= 1  # role is always required and always 1 token
    
    # Every reply is primed with <im_start>assistant
    num_tokens += 2
    
    return num_tokens
