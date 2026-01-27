"""Example: Optimizing a Curator LLM with GEPA using an LLM judge.

This example demonstrates how to use GEPA to optimize the prompts of a Curator LLM
that generates math word problems. An LLM judge evaluates the quality of generated
problems on clarity, correctness, and grade-appropriateness.

Requirements:
    pip install gepa

Usage:
    python examples/optimizer/gepa_example.py
"""

import gepa
from datasets import Dataset
from pydantic import BaseModel, Field

from bespokelabs import curator
from bespokelabs.curator.blocks.gepa import CuratorAdapter, EvaluationResult

# =============================================================================
# Step 1: Define the Generator LLM to optimize
# =============================================================================


class MathProblemGenerator(curator.LLM):
    """Generates math word problems for students.

    This is the LLM we want to optimize - GEPA will evolve its
    system prompt and prompt template to produce better problems.

    Define prompts as class attributes so GEPA can extract and optimize them.
    """

    # Seed prompts - GEPA will evolve these
    system_prompt = "You are a math teacher creating word problems for students."
    prompt_template = "Create a math word problem about {topic} for {grade_level} students."

    def prompt(self, input: dict) -> str:
        """Use the prompt_template (will be optimized by GEPA)."""
        return self.prompt_template.format(**input)

    def parse(self, input: dict, response: str) -> dict:
        """Parse the response into the output format."""
        return {
            "topic": input["topic"],
            "grade_level": input["grade_level"],
            "problem": response,
        }


# =============================================================================
# Step 2: Define the LLM Judge for scoring
# =============================================================================


class ProblemScore(BaseModel):
    """Structured output for the LLM judge."""

    clarity: int = Field(description="How clear and well-written is the problem (0-10)")
    correctness: int = Field(description="Is the math correct and solvable (0-10)")
    appropriateness: int = Field(description="Is it appropriate for the grade level (0-10)")
    extra_comments: bool = Field(description="Is there any extraneous text in the response other than the problem? (Yes/No)")
    reasoning: str = Field(description="Brief explanation of the scores")

    @property
    def overall_score(self) -> float:
        """Calculate overall score normalized to 0-1."""
        return (self.clarity + self.correctness + self.appropriateness + (0 if self.extra_comments else 1) * 10) / 40.0


class MathProblemJudge(curator.LLM):
    """LLM Judge that evaluates the quality of generated math problems."""

    response_format = ProblemScore

    def prompt(self, input: dict) -> str:
        """Prompt for evaluating a math problem."""
        return f"""Evaluate this math word problem on a scale of 0-10 for each criterion.

Topic: {input['topic']}
Target Grade Level: {input['grade_level']}
Problem: {input['problem']}

Rate the problem on:
1. Clarity: Is the problem well-written and easy to understand?
2. Correctness: Is the math correct? Can it be solved with a definite answer?
3. Appropriateness: Is the difficulty appropriate for {input['grade_level']} students?
4. Extra Comments: Is there any extraneous text in the response other than the problem (e.g., greetings, explanations, solutions, etc.)?
                   We want ONLY the problem.

Provide brief reasoning for your scores."""

    def parse(self, input: dict, response: ProblemScore) -> dict:
        """Parse the judge's response."""
        return {
            "topic": input["topic"],
            "grade_level": input["grade_level"],
            "problem": input["problem"],
            "clarity": response.clarity,
            "correctness": response.correctness,
            "appropriateness": response.appropriateness,
            "extra_comments": response.extra_comments,
            "overall_score": response.overall_score,
            "reasoning": response.reasoning,
        }


judge = MathProblemJudge(model_name="gpt-4o-mini")

# =============================================================================
# Step 3: Create the metric function using the LLM judge
# =============================================================================


def metric(inputs: list[dict], outputs: list[dict]) -> list[EvaluationResult]:
    """Evaluate the generated problems using the LLM judge.

    Args:
        inputs: List of original inputs to the generator
        outputs: List of outputs from the generator

    Returns:
        List of EvaluationResult with scores and natural language feedback
    """
    # Combine inputs and outputs for the judge
    judge_inputs = [
        {
            "topic": inp["topic"],
            "grade_level": inp["grade_level"],
            "problem": out.get("problem", ""),
        }
        for inp, out in zip(inputs, outputs)
    ]

    # Run the judge on all examples at once
    result = judge(judge_inputs)

    # Build evaluation results with scores and detailed feedback
    eval_results: list[EvaluationResult] = []
    for row in result.dataset:
        score = float(row["overall_score"])
        # Include the judge's reasoning and scores as feedback for GEPA's reflection
        feedback = (
            f"Score: {score:.2f}/1.0. "
            f"Clarity: {row['clarity']}/10, Correctness: {row['correctness']}/10, "
            f"Appropriateness: {row['appropriateness']}/10, Extra Comments: {row['extra_comments']}. "
            f"Reasoning: {row['reasoning']}"
        )
        eval_results.append({"score": score, "feedback": feedback})

    return eval_results


# =============================================================================
# Step 4: Prepare training and validation data
# =============================================================================


def create_datasets():
    """Create training and validation datasets.

    Returns:
        Tuple of (trainset, valset) as lists of dicts
    """
    # Topics and grade levels for math problems
    topics = [
        "fractions",
        "percentages",
        "ratios",
        "area and perimeter",
        "money and shopping",
        "time and scheduling",
        "speed and distance",
        "probability",
        "averages",
        "algebra basics",
    ]

    grade_levels = [
        "3rd grade",
        "4th grade",
        "5th grade",
        "6th grade",
        "7th grade",
        "8th grade",
    ]

    # Create examples by combining topics and grade levels
    examples = []
    for topic in topics:
        for grade in grade_levels:
            examples.append({"topic": topic, "grade_level": grade})

    # Split into train (80%) and val (20%)
    split_idx = int(len(examples) * 0.8)
    trainset = examples[:split_idx]
    valset = examples[split_idx:]

    return trainset, valset


# =============================================================================
# Main: Run GEPA optimization
# =============================================================================


def main():
    """Run GEPA optimization on the math problem generator."""
    print("=" * 60)
    print("GEPA Prompt Optimization for Curator LLM")
    print("=" * 60)

    # Create datasets
    trainset, valset = create_datasets()
    print(f"\nDataset sizes: train={len(trainset)}, val={len(valset)}")

    # Create the Curator adapter
    adapter = CuratorAdapter(
        llm_class=MathProblemGenerator,
        metric=metric,
        model_name="gpt-4o-mini",  # Model for the generator
    )

    # Extract seed candidate from the LLM class
    seed_candidate = adapter.get_seed_candidate()

    # Run GEPA optimization
    print("\nStarting GEPA optimization...")

    result = gepa.optimize(
        adapter=adapter,
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        reflection_lm="openai/gpt-4o",  # Model for reflection (use stronger model)
        max_metric_calls=15,  # Budget for optimization (increase for better results)
    )

    print("GEPA Optimization Complete!")

    print("\nBest Candidate Found:")
    best = result.best_candidate
    print(f"  System Prompt: {best.get('system_prompt', 'N/A')}")
    print(f"  Prompt Template: {best.get('prompt_template', 'N/A')}")

    # Test the optimized prompts
    print("\n" + "-" * 60)
    print("Testing optimized generator on a sample problem...")
    print("-" * 60)

    optimized_llm = adapter._create_optimized_llm(best)
    test_input = Dataset.from_list([{"topic": "fractions", "grade_level": "5th grade"}])
    test_result = optimized_llm(test_input)

    print(f"\nGenerated Problem:\n{test_result.dataset[0]['problem']}")

    return result


if __name__ == "__main__":
    main()
