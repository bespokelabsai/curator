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
numina_162k_amc_aime_problems = load_dataset("NovaSky-AI/labeled_numina_difficulty_162K", trust_remote_code=True)["train"].filter(
    lambda x: x["source"] == "amc_aime"
)
llm = Reasoner(
    model_name="deepseek-reasoner",
    generation_params={"temp": 0.0},
    backend_params={
        "max_requests_per_minute": 1500,
        "max_tokens_per_minute": 100_000_000,
    },
)
numina_162k_amc_aime_problems_response = llm(numina_162k_amc_aime_problems)
numina_162k_amc_aime_problems_response.push_to_hub("bespokelabs/sky-t1-numina-amc-aime-subset-unfiltered", private=True)

# math
numina_162k_math_problems = load_dataset("NovaSky-AI/labeled_numina_difficulty_162K", trust_remote_code=True)["train"].filter(lambda x: x["source"] == "math")
numina_162k_math_problems_response = llm(numina_162k_math_problems)
numina_162k_math_problems_response.push_to_hub("bespokelabs/sky-t1-numina-math-subset-unfiltered", private=True)

# olympiads
numina_162k_olympiads_problems = (
    load_dataset("NovaSky-AI/labeled_numina_difficulty_162K", trust_remote_code=True)["train"].filter(lambda x: x["source"] == "olympiads")
).take(20_000)
numina_162k_olympiads_problems_response = llm(numina_162k_olympiads_problems)
numina_162k_olympiads_problems_response.push_to_hub("bespokelabs/sky-t1-numina-olympiads-subset-unfiltered", private=True)


all_subsets = concatenate_datasets([numina_162k_amc_aime_problems_response, numina_162k_math_problems_response, numina_162k_olympiads_problems_response])

# FIRST APPROACH: Filter correct answers and calculate accuracy based on string matching. This gives ~25% accuracy.
# rejected_sampled_subsets = all_subsets.filter(lambda x: math_equal(x["ground_truth_final_answer"], x["deepseek_final_answer"]), num_proc=30)
# total_questions = len(all_subsets)
# correct_count = len(rejected_sampled_subsets)
# accuracy = correct_count / total_questions

# rejected_sampled_subsets.push_to_hub("bespokelabs/sky-t1-numina-rejection-sampled", private=True)
# print(f"Accuracy: {correct_count}/{total_questions} ({accuracy:.2%})")

# SECOND APPROACH: Filter correct answers and calculate accuracy based on judge. This gives ~73% accuracy.
judge = Judge(model_name="gpt-4o-mini")
judge_results = judge(all_subsets)
correct_answers = judge_results.filter(lambda x: x["correct"])
total_questions = len(judge_results)
correct_count = len(correct_answers)
accuracy = correct_count / total_questions
judge_results.push_to_hub("bespokelabs/sky-t1-numina-rejection-sampled", private=True)
print(f"Accuracy: {correct_count}/{total_questions} ({accuracy:.2%})")

# FOR DEBUGGING: Print some answers for inspection
# for i, item in enumerate(judge_results.filter(lambda x: x["correct"] and (x["deepseek_final_answer"] != x["ground_truth_final_answer"])).take(10)):
#     print(f"\nQuestion {i+1}:")
#     print(f"Problem: {item['problem']}")
#     # print(f"DeepSeek Reasoning: {item['reasoning']}")
#     print(f"DeepSeek Solution: {item['deepseek_solution']}")
#     print(f"Ground Truth Solution: {item['ground_truth_solution']}")
#     print(f"Re-extracted DeepSeek Answer: {extract_boxed_answer(item['deepseek_solution'])}")
#     print(f"Re-extracted Ground Truth Answer: {extract_boxed_answer(item['ground_truth_solution'])}")
#     print(f"DeepSeek Answer: {item['deepseek_final_answer']}")
#     print(f"Ground Truth: {item['ground_truth_final_answer']}")
#     print(f"Correct: {'✓' if item['correct'] else '✗'}")
#     print(f"Judge Reasoning: {item['judge_reasoning']}")
