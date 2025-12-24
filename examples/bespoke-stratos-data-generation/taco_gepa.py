"""Curate data using curator on the APPS dataset."""

import argparse
import copy
import json
import multiprocessing
import re
import resource
from multiprocessing import Manager

import dspy
import numpy as np
from datasets import load_dataset
from util.code_execution_taco import process_dataset_parallel, process_single_row
from util.prompt import SKY_T1_SYSTEM_PROMPT, generate_prompt

from bespokelabs import curator


class TACOCurator(curator.LLM):
    """Curator class for processing TACO (Testing Algorithmic Coding prOblems) dataset.

    Handles prompting the LLM and parsing responses for code generation.
    """

    return_completions_object = True

    def prompt(self, problem):
        """Parse test cases and starter code from problem to create a prompt for the LLM."""
        test_case = json.loads(problem["input_output"])
        starter_code = problem["starter_code"]
        # Generate prompt text using test case, question and starter code
        prompt_text = generate_prompt(test_case, problem["question"], starter_code)
        return [{"role": "user", "content": prompt_text}]

    def parse(self, input, response):
        """Parse the LLM response to extract reasoning and solution."""
        input["deepseek_solution"] = response["choices"][0]["message"]["content"]
        return input


def metric_fn(example, pred, trace=None, pred_name=None, pred_trace=None):
    """Metric function to evaluate correctness of generated code."""
    try:
        example["row"]["deepseek_solution"] = pred.output
        code_check = process_single_row(example["row"])
        score = int(code_check["correctness"])
        return dspy.Prediction(score=score, feedback=code_check.get("reason", ""))
    except Exception as e:
        print(f"Error in metric_fn: {e}")
        return False


if __name__ == "__main__":
    # Set up command line arguments
    args = argparse.ArgumentParser()
    args.add_argument("--split", type=str, default="test")
    args.add_argument("--optimize_prompt", action="store_true")
    args = args.parse_args()

    # Initialize curator with GPT-4o model
    curator = TACOCurator(model_name="gpt-4o-2024-08-06", system_prompt=SKY_T1_SYSTEM_PROMPT)

    # Load and filter TACO dataset based on split
    if args.split == "train":
        # For training split, only use MEDIUM difficulty problems
        taco_dataset = load_dataset("BAAI/TACO", trust_remote_code=True)["train"].filter(lambda x: x["difficulty"] == "MEDIUM")
    else:
        taco_dataset = load_dataset("BAAI/TACO", trust_remote_code=True)[args.split]

    # Select a subset of examples to experiment with.
    taco_dataset = taco_dataset.shuffle(seed=79).select(range(200))

    if args.optimize_prompt:
        # Choose small subset of training data for prompt optimization.
        train_dataset = [dspy.Example(prompt=curator.prompt(row), row=row).with_inputs("prompt") for row in taco_dataset.select(range(50))]

        # Optimize system prompt using GEPA.
        # These arguments are meant for DSPY's GEPA implementation.
        curator.optimize_system_prompt(
            metric=metric_fn,
            train_dataset=train_dataset,
            max_metric_calls=150,
            num_threads=32,
            track_stats=True,
            reflection_minibatch_size=25,
            reflection_lm=dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000),  # strong reflection model
        )

    # Generate solutions using curator
    taco_dataset = curator(taco_dataset)

    # Increase file limit for parallel processing
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

    # Run rejection sampling to filter results
    taco_dataset = process_dataset_parallel(taco_dataset.dataset)

    df = taco_dataset.to_pandas()
    df.to_parquet(f"taco_dataset_{'optimized' if args.optimize_prompt else 'default'}_{args.split}.parquet")
