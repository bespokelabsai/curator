"""Example: Optimizing Code Generation with GEPA using Test Execution.

This example demonstrates GEPA's ability to improve prompts for code generation.
"""

import os

import gepa
from code_generation_dataset import (
    create_datasets,
    deserialize_test_cases,
)
from code_generation_evaluation import (
    analyze_code,
    run_tests,
)

from bespokelabs import curator
from bespokelabs.curator.blocks.gepa import CuratorAdapter, EvaluationResult

os.environ["CURATOR_DISABLE_CACHE"] = "true"  # required since Curator doesn't add system prompt to the cache key

# =============================================================================
# Step 1: Define the Code Generator LLM
# =============================================================================


class CodeGenerator(curator.LLM):
    """Generates Python code solutions for programming problems.

    GEPA will optimize the system_prompt to improve the reward on
    test cases.
    """

    def prompt(self, input: dict) -> str:
        """Use the prompt template."""
        return "{description}".format(**input)

    def parse(self, input: dict, response: str) -> dict:
        """Parse the response, extracting the code."""
        return {
            "problem_id": input["problem_id"],
            "description": input["description"],
            "function_name": input["function_name"],
            "test_cases": input["test_cases"],
            "generated_code": response,
        }


# =============================================================================
# Step 2: Code Evaluation
# =============================================================================


def metric(inputs: list[dict], outputs: list[dict]) -> list[EvaluationResult]:
    """Evaluate generated code by running test cases.

    This metric:
    1. Extracts the generated code from each output
    2. Runs test cases and style analysis against the code
    3. Returns a score (pass rate) and feedback for GEPA reflection

    Args:
        inputs: List of original inputs (programming problems)
        outputs: List of generated outputs (code solutions)

    Returns:
        List of EvaluationResult with scores and feedback
    """
    results: list[EvaluationResult] = []

    for inp, out in zip(inputs, outputs):
        generated_code = out.get("generated_code", "")
        if not generated_code.startswith("```"):
            results.append({"score": 0.0, "feedback": f"Expected markdown code block starting with ```, got: {generated_code[:40]}..."})
            continue
        if not generated_code.endswith("```"):
            results.append({"score": 0.0, "feedback": f"Expected markdown code block ending with ```, got: ...{generated_code[-40:]}"})
            continue

        function_name = inp["function_name"]
        test_cases = deserialize_test_cases(inp["test_cases"])
        passed, total, feedback = run_tests(generated_code, function_name, test_cases)
        violations = analyze_code(generated_code, function_name)

        # Calculate score with weights for functional correctness and style
        functional_score = passed / total if total > 0 else 0.0
        # Penalty of 0.5 per violation to penalize stylistic violations
        quality_score = max(0.0, 1.0 - 0.5 * len(violations))

        # Final score: 50% functional, 50% style
        score = 0.5 * functional_score + 0.5 * quality_score

        results.append(
            {
                "score": score,
                "feedback": (f"Pass rate: {passed}/{total}. {feedback}. " f"Style violations: {', '.join(violations) if violations else 'none'}"),
            }
        )

    return results


# =============================================================================
# Step 3: Run GEPA optimization
# =============================================================================


def main():
    """Run GEPA optimization on the code generator."""
    print("=" * 70)
    print("GEPA Prompt Optimization for Code Generation")
    print("=" * 70)

    # Create datasets (deterministic shuffle for a balanced train/val split)
    trainset, valset = create_datasets(seed=42)
    print(f"\nDataset: {len(trainset)} training problems, {len(valset)} validation problems")

    # Create the LLM instance with all desired parameters
    code_generator = CodeGenerator(model_name="gpt-4o-mini", generation_params={"temperature": 0.2}, system_prompt="You are a Python programmer.")

    adapter = CuratorAdapter(
        llm=code_generator,
        metric=metric,
    )

    seed_candidate = adapter.get_seed_candidate()
    print(f"\nSeed System Prompt: {seed_candidate.get('system_prompt', 'N/A')}")

    # Run GEPA optimization
    print("\n" + "-" * 70)
    print("Starting GEPA Optimization...")
    print("-" * 70)

    result = gepa.optimize(
        adapter=adapter,
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        reflection_lm="openai/gpt-5-mini",  # Use stronger model for reflection
        max_metric_calls=15,  # Increase for better results
    )

    print("\n" + "=" * 70)
    print("GEPA Optimization Complete!")
    print("=" * 70)

    # Show optimized prompt
    best = result.best_candidate
    print("\nOptimized System Prompt:")
    print(f"  {best.get('system_prompt', 'N/A')}")

    # Evaluate optimized prompts
    optimized_eval = adapter.evaluate(valset, best)
    optimized_score = sum(optimized_eval.scores) / len(optimized_eval.scores) if optimized_eval.scores else 0
    breakpoint()
    print("\n" + "-" * 70)
    print("Results Summary")
    print("-" * 70)
    print(f"Optimized validation score: {optimized_score:.2%}")


if __name__ == "__main__":
    main()
