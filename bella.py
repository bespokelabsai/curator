from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Optional, Type

import instructor
from litellm import completion as litellm_completion
from pydantic import BaseModel
from tqdm import tqdm


def completion(
    model: str = "gpt-4",
    response_model: Optional[Type[BaseModel]] = None,
):
    """Decorator that creates a prompter from a function.
    
    Args:
        model: The model to use for completion
        response_format: Optional Pydantic model for response validation
    
    Returns:
        A decorator that converts a function into a prompter
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the docstring to use as system prompt
            system_prompt = func.__doc__ or "You are a helpful AI assistant."
            
            # Get the user prompt by calling the function
            user_prompt = func(*args, **kwargs)
            client = instructor.from_litellm(litellm_completion)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=response_model,
            )
            return response
        return wrapper
    return decorator

def parallel(func, rpm: int = 30_000):
    """Decorator that executes a function in parallel across list inputs using ThreadPoolExecutor.
    If the function has no input arguments, executes once.
    
    Args:
        rpm: Number of requests per minute to process in parallel
    """
    def wrapper(*args, **kwargs):
        # If no arguments, just execute the function once
        if not args:
            return func(*args, **kwargs)
        
        # If first argument is not a list, just execute once
        if not isinstance(args[0], (list, tuple)):
            return func(*args, **kwargs)
        
        input_list = args[0]
        other_args = args[1:]
        
        # Calculate max workers based on RPM (minimum of 1)
        max_workers = max(1, int(rpm / 60))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(func, item, *other_args, **kwargs)
                for item in input_list
            ]
            # Add tqdm progress bar
            return [
                future.result() 
                for future in tqdm(futures, total=len(futures), desc=f"Processing {func.__name__}")
            ]
        
    return wrapper
