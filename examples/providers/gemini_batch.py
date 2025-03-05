import json
import os

from datasets import load_dataset

from bespokelabs import curator


class MathTechniqueClassifier(curator.LLM):
    """A classifier for mathematical solution techniques using Gemini."""

    def prompt(self, input: dict) -> str:
        """Generate a prompt to analyze mathematical solution approaches."""
        prompt = f"""Analyze the following mathematical problem and solution approach:

        PROBLEM:
        {input['question']}

        SOLUTION APPROACH:
        {input['attempt']}

        Please identify the mathematical techniques used in this solution and provide a short analysis in JSON format:
        {{
            "primary_technique": "The main mathematical technique used (e.g., Trigonometric Identities, Complex Analysis, Algebraic Manipulation, etc.)",
            "secondary_techniques": ["List of other techniques used"],
            "key_insights": ["1-3 key mathematical insights from the solution"],
            "difficulty_level": "A rating from 1-5 where 5 is extremely difficult",
            "elegance_rating": "A rating from 1-5 where 5 is extremely elegant",
            "completeness": "Is the solution complete? (Yes/No/Partial)"
        }}

        Be specific about the mathematical techniques.
        For example, instead of just saying "Calculus", specify "Integration by Parts" or "L'Hôpital's Rule" if applicable.
        """
        return prompt

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input parameters."""
        response = response.strip()
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1

        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx]
            analysis_data = json.loads(json_str)
        else:
            raise ValueError("Failed to parse the analysis data from the response.")

        if isinstance(analysis_data.get("difficulty_level"), str):
            try:
                analysis_data["difficulty_level"] = int(analysis_data["difficulty_level"])
            except (ValueError, TypeError):
                analysis_data["difficulty_level"] = 0

        if isinstance(analysis_data.get("elegance_rating"), str):
            try:
                analysis_data["elegance_rating"] = int(analysis_data["elegance_rating"])
            except (ValueError, TypeError):
                analysis_data["elegance_rating"] = 0

        result = {
            "problem_id": input.get("id", "Unknown"),
            "cot_type": input.get("cot_type", "Unknown"),
            "source_type": input.get("source_type", "Unknown"),
            "analysis": analysis_data,
            "problem_preview": input["question"][:100] + "..." if len(input["question"]) > 100 else input["question"],
        }
        return result


def unroll_gemini_trajectory(example):
    """Unroll the thinking trajectory and attempt into separate columns."""
    example["gemini_thinking_trajectory"] = example["thinking_trajectories"][0]
    example["gemini_attempt"] = example["attempt"]
    return example


ds = load_dataset("simplescaling/s1K", split="train")
ds = ds.map(unroll_gemini_trajectory, num_proc=os.cpu_count())

technique_classifier = MathTechniqueClassifier(
    model_name="gemini-1.5-flash-001",
    backend="gemini",
    batch=True,
    backend_params={"require_all_responses": False},
)
analysis_results = technique_classifier(ds.take(10))
