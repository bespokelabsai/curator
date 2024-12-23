"""Math problem solver using curator.Prompter with safe code execution."""

from typing import Dict, Any, Optional, Union

from bespokelabs import curator
from math_solver import MathProblem, MathSolution, MathSolutions, MathResult
from math_executor import execute_math_code


class MathPrompter(curator.Prompter):
    """Prompter specialized for math problem solving with code execution."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.2,
        batch: bool = False,
        batch_size: Optional[int] = None,
    ):
        """Initialize MathPrompter with specialized prompt and parse functions."""

        def prompt_func(problem: Union[Dict[str, Any], MathProblem]) -> Dict[str, str]:
            """Format the math problem for the LLM."""
            if isinstance(problem, dict):
                question = problem.get("question", "")
            else:
                question = problem.question

            return {
                "role": "user",
                "content": (
                    f"Solve this math problem by writing Python code. The code should print the final answer.\n\n"
                    f"Problem: {question}\n\n"
                    f"Requirements:\n"
                    f"1. Write clear, simple Python code that solves the problem\n"
                    f"2. The code must print only the final answer\n"
                    f"3. Include a brief explanation of your solution approach\n"
                    f"4. Do not use any imports\n"
                    f"5. Only use basic Python operations\n"
                ),
            }

        def parse_func(
            problem: Union[Dict[str, Any], MathProblem], response: MathSolutions
        ) -> MathResult:
            """Execute the solution code and validate results."""
            try:
                if isinstance(problem, dict):
                    question = problem.get("question", "")
                    expected = problem.get("expected_answer")
                else:
                    question = problem.question
                    expected = problem.expected_answer

                # Handle empty or invalid responses
                if not response or not response.solutions:
                    return MathResult(
                        question=question,
                        solution=MathSolution(
                            python_code="", answer="N/A", explanation="Failed to generate solution"
                        ),
                        is_correct=False,
                        error="No valid solution generated",
                    )

                solution = response.solutions[0]  # We only generate one solution per problem

                # Execute the code
                result, error = execute_math_code(solution.python_code)

                if error:
                    return MathResult(
                        question=question,
                        solution=MathSolution(
                            python_code=solution.python_code,
                            answer="N/A",
                            explanation=solution.explanation,
                        ),
                        is_correct=False,
                        error=error,
                    )

                # Update the solution with the executed result
                try:
                    # Try to convert string result to number if possible
                    try:
                        if "." in result:
                            result = float(result)
                        else:
                            result = int(result)
                    except (ValueError, TypeError):
                        # Keep as string if conversion fails
                        pass

                    solution.answer = result
                except Exception as e:
                    return MathResult(
                        question=question,
                        solution=MathSolution(
                            python_code=solution.python_code,
                            answer="N/A",
                            explanation=solution.explanation,
                        ),
                        is_correct=False,
                        error=f"Failed to process result: {str(e)}",
                    )

                # Check if the answer matches expected (if provided)
                is_correct = None
                if expected is not None:
                    try:
                        # Convert both to same type for comparison
                        if isinstance(expected, (int, float)):
                            computed = float(str(result))
                            is_correct = abs(computed - float(expected)) < 1e-6
                        else:
                            is_correct = str(result).strip() == str(expected).strip()
                    except (ValueError, TypeError):
                        is_correct = False

                return MathResult(question=question, solution=solution, is_correct=is_correct)

            except Exception as e:
                # Catch-all error handler
                return MathResult(
                    question=question,
                    solution=MathSolution(
                        python_code="", answer="N/A", explanation="Error occurred during processing"
                    ),
                    is_correct=False,
                    error=f"Unexpected error: {str(e)}",
                )

        super().__init__(
            model_name=model_name,
            prompt_func=prompt_func,
            parse_func=parse_func,
            response_format=MathSolutions,
            batch=batch,
            batch_size=batch_size,
            temperature=temperature,
        )

    def solve(self, problems: Union[str, Dict[str, Any], MathProblem, list]) -> MathResult:
        """
        Solve one or more math problems.

        Args:
            problems: A single problem (as string, dict, or MathProblem) or list of problems

        Returns:
            MathResult or Dataset containing results
        """
        # Convert string to dict format
        if isinstance(problems, str):
            problems = {"question": problems}

        # Convert single problem to list
        if not isinstance(problems, list):
            problems = [problems]

        # Process problems using curator Dataset
        dataset = curator.Dataset.from_list(problems)
        results = self(dataset)

        # Return single result if only one problem
        if len(problems) == 1:
            return results[0]

        return results
