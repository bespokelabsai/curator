"""Math problem auto-bencher using curator.Prompter."""

from typing import List, Optional
from datasets import Dataset

from math_solver import MathProblem, MathResult
from math_prompter import MathPrompter


class MathAutoBencher:
    """Automated math problem solver and benchmarker."""

    def __init__(
        self, model_name: str = "gpt-4o-mini", batch_size: int = 20, temperature: float = 0.2
    ):
        """Initialize the auto-bencher with specified parameters."""
        self.prompter = MathPrompter(
            model_name=model_name, batch=True, batch_size=batch_size, temperature=temperature
        )

    def run_benchmark(self, problems: List[dict], output_file: Optional[str] = None) -> Dataset:
        """
        Run benchmark on a list of math problems.

        Args:
            problems: List of problem dictionaries with 'question' and optional 'expected_answer'
            output_file: Optional path to save results

        Returns:
            Dataset containing benchmark results
        """
        # Convert problems to dataset
        dataset = Dataset.from_list(problems)

        # Process all problems
        results = self.prompter(dataset)

        if output_file:
            # Save results to file
            results.to_json(output_file)

        return results

    def analyze_results(self, results: Dataset) -> dict:
        """
        Analyze benchmark results.

        Args:
            results: Dataset containing benchmark results

        Returns:
            Dictionary with analysis metrics
        """
        total = len(results)
        correct = sum(1 for result in results if result.get("is_correct", False))
        errors = sum(1 for result in results if result.get("error") is not None)

        return {
            "total_problems": total,
            "correct_answers": correct,
            "accuracy": correct / total if total > 0 else 0,
            "errors": errors,
            "error_rate": errors / total if total > 0 else 0,
        }


def main():
    """Example usage of MathAutoBencher."""
    # Example problems
    problems = [
        {"question": "What is 15 + 27?", "expected_answer": 42},
        {"question": "If x = 5 and y = 3, what is x * y?", "expected_answer": 15},
        {
            "question": "Calculate the area of a rectangle with width 8 and height 6.",
            "expected_answer": 48,
        },
    ]

    # Initialize and run benchmark
    bencher = MathAutoBencher()
    results = bencher.run_benchmark(problems, output_file="benchmark_results.json")

    # Analyze results
    analysis = bencher.analyze_results(results)
    print("\nBenchmark Analysis:")
    print(f"Total Problems: {analysis['total_problems']}")
    print(f"Correct Answers: {analysis['correct_answers']}")
    print(f"Accuracy: {analysis['accuracy']:.2%}")
    print(f"Errors: {analysis['errors']}")
    print(f"Error Rate: {analysis['error_rate']:.2%}")


if __name__ == "__main__":
    main()
