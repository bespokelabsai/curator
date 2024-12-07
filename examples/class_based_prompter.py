"""Example of using the class-based Prompter approach."""

from typing import Dict, Any, Optional
from pydantic import BaseModel

from bespokelabs.curator import Prompter
from bespokelabs.curator.prompter.base_prompter import BasePrompter


class ResponseFormat(BaseModel):
    """Example response format."""

    answer: str
    confidence: float


class MathPrompter(BasePrompter):
    """Example custom prompter for math problems."""

    def prompt_func(self, row: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Generate prompts for math problems.

        Args:
            row: Optional dictionary containing 'question' key.
                If None, generates a default prompt.

        Returns:
            Dict containing user_prompt and system_prompt.
        """
        if row is None:
            return {
                "user_prompt": "What is 2 + 2?",
                "system_prompt": "You are a math tutor. Provide clear, step-by-step solutions.",
            }

        return {
            "user_prompt": row["question"],
            "system_prompt": "You are a math tutor. Provide clear, step-by-step solutions.",
        }

    def parse_func(self, row: Dict[str, Any], response: Dict[str, Any]) -> ResponseFormat:
        """Parse LLM response into structured format.

        Args:
            row: Input row that generated the response
            response: Raw response from the LLM

        Returns:
            ResponseFormat containing answer and confidence
        """
        # Extract answer and add confidence score
        return ResponseFormat(
            answer=response["message"],
            confidence=0.95 if "step" in response["message"].lower() else 0.7,
        )


def main():
    """Example usage of class-based prompter."""
    # Create instance of custom prompter
    math_prompter = MathPrompter(
        model_name="gpt-4o-mini",
        response_format=ResponseFormat,
    )

    # Single completion without input
    result = math_prompter()
    print(f"Single completion result: {result}")

    # Process multiple questions
    questions = [
        {"question": "What is 5 * 3?"},
        {"question": "Solve x + 2 = 7"},
    ]
    results = math_prompter(questions)
    print("\nBatch processing results:")
    for q, r in zip(questions, results):
        print(f"Q: {q['question']}")
        print(f"A: {r.answer} (confidence: {r.confidence})")


if __name__ == "__main__":
    main()
