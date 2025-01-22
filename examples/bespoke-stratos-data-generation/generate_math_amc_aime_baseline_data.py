"""Curate data using curator on the Numina dataset."""

from datasets import concatenate_datasets, load_dataset
from pydantic import BaseModel
from util.prompt import SKY_T1_SYSTEM_PROMPT
from util.testing.math import extract_answer, strip_answer_string

from bespokelabs import curator


def extract_boxed_answer(text):
    """Extract the boxed answer from the text."""
    text = strip_answer_string(text)
    return extract_answer(text)


class JudgeResult(BaseModel):
    """Result of the judge's evaluation."""

    correct: bool
    reasoning: str


class Judge(curator.LLM):
    """Curator class for processing Numina dataset."""

    response_format = JudgeResult

    def prompt(self, input):
        """Create a prompt for the judge to evaluate the correctness of a solution."""
        return f"""
        You are a judge that evaluates the correctness of a solution.
        You will be given a solution and a ground truth solution.
        You will need to determine if the solution is correct.
        Answers are in the format of \\boxed{{}}.

        SOLUTION: {input["deepseek_solution"]}
        GROUND TRUTH SOLUTION: {input["ground_truth_solution"]}
        """

    def parse(self, input, response):
        """Parse the judge's response to extract correctness and reasoning."""
        return {**input, "correct": response.correct, "judge_reasoning": response.reasoning}


class Reasoner(curator.LLM):
    """Curator class for processing Numina dataset."""

    return_completions_object = True

    def prompt(self, input):
        """Create a prompt for the LLM to reason about the problem."""
        return [
            {"role": "system", "content": SKY_T1_SYSTEM_PROMPT},
            {"role": "user", "content": input["problem"]},
        ]

    def parse(self, input, response):
        """Parse the LLM response to extract reasoning and solution."""
        return [
            {
                "problem": input["problem"],
                "reasoning": response["choices"][0]["message"]["reasoning_content"],
                "deepseek_solution": response["choices"][0]["message"]["content"],
                "ground_truth_solution": input["solution"],
                "deepseek_final_answer": extract_boxed_answer(response["choices"][0]["message"]["content"]),
                "ground_truth_final_answer": extract_boxed_answer(input["solution"]),
            }
        ]


# amc_aime
numina_amc_aime_4k_problems = load_dataset("AI-MO/NuminaMath-CoT", trust_remote_code=True)["train"].filter(
    lambda x: x["source"] == "amc_aime"
)
numina_amc_aime_4k_problems = numina_amc_aime_4k_problems.select(range(5))
llm = Reasoner(
    model_name="deepseek-reasoner",
    generation_params={"temp": 0.0},
    backend_params={
        "max_requests_per_minute": 1500,
        "max_tokens_per_minute": 100_000_000,
    },
)
numina_amc_aime_4k_response = llm(numina_amc_aime_4k_problems)
numina_amc_aime_4k_response.push_to_hub("hbXNov/numina_amc_aime_deepseek_r1_responses")

# amc_aime
numina_math_7k_problems = load_dataset("AI-MO/NuminaMath-CoT", trust_remote_code=True)["train"].filter(
    lambda x: x["source"] == "math"
)
numina_math_7k_problems = numina_math_7k_problems.select(range(5))
llm = Reasoner(
    model_name="deepseek-reasoner",
    generation_params={"temp": 0.0},
    backend_params={
        "max_requests_per_minute": 1500,
        "max_tokens_per_minute": 100_000_000,
    },
)
numina_math_7k_response = llm(numina_math_7k_problems)
numina_math_7k_response.push_to_hub("hbXNov/numina_math_deepseek_r1_responses")