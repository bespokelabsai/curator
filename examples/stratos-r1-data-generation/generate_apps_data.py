# Import required libraries
import argparse
import json

from datasets import load_dataset
from util.code_execution_apps import process_dataset_parallel
from util.prompt import SKY_T1_SYSTEM_PROMPT, generate_prompt

from bespokelabs import curator

# Uncomment to set DeepSeek API key
# import os
# os.environ['DEEPSEEK_API_KEY'] = ''


class APPSCurator(curator.LLM):
    """Curator class for processing APPS (Automated Programming Problems Solutions) dataset."""

    return_completions_object = True

    def prompt(self, problem):
        """Parse test cases and starter code from problem to create a prompt for the LLM."""
        test_case = json.loads(problem["input_output"])
        starter_code = problem["starter_code"]
        prompt_text = generate_prompt(test_case, problem["question"], starter_code)
        return [{"role": "system", "content": SKY_T1_SYSTEM_PROMPT}, {"role": "user", "content": prompt_text}]

    def parse(self, input, response):
        """Parse the LLM response to extract reasoning and solution."""
        input["reasoning"] = response["choices"][0]["message"]["reasoning_content"]
        input["deepseek_solution"] = response["choices"][0]["message"]["content"]
        return input


if __name__ == "__main__":
    # Set up command line arguments
    args = argparse.ArgumentParser()
    args.add_argument("--push_to_hub", action="store_true")
    args.add_argument("--split", type=str, default="test")
    args.add_argument("--dataset_name", type=str, default="bespokelabs/sky-t1-apps")
    args = args.parse_args()

    # Initialize curator with DeepSeek model and parameters
    curator = APPSCurator(
        model_name="deepseek-reasoner",
        backend_params={
            "max_requests_per_minute": 10000,
            "max_tokens_per_minute": 10000000,
            "request_timeout": 30 * 60,
        },
    )

    # Load APPS dataset based on specified split
    apps = load_dataset("codeparrot/apps", trust_remote_code=True)[args.split]

    # Generate solutions using curator
    curated_dataset = curator(apps)

    # Push unfiltered results to hub if specified
    if args.push_to_hub:
        curated_dataset.push_to_hub(f"{args.dataset_name}-{args.split}-unfiltered", private=True)

    # Run rejection sampling to filter results
    curated_dataset = process_dataset_parallel(curated_dataset)

    # Push rejection sampled results to hub if specified
    if args.push_to_hub:
        curated_dataset.push_to_hub(f"{args.dataset_name}-{args.split}-rejection-sampled", private=True)

    print("Done")
