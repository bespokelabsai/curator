import multiprocessing
import re
from multiprocessing import Pool

import numpy as np
import timeout_decorator
from datasets import Dataset
from tqdm import tqdm
from util.testing.taco import run_test as taco_run_test


@timeout_decorator.timeout(10)
def run_test_with_timeout(problem, generation):
    """Run the test with a timeout."""
    try:
        result = taco_run_test(problem, test=generation, debug=False)
        return bool(result and np.all(result))
    except Exception as e:
        print(f"Error in run_test: {e}")
        return False


def check_correctness(problem, generation):
    """Check if the code is correct."""
    try:
        return run_test_with_timeout(problem, generation)
    except timeout_decorator.TimeoutError:
        print("Test execution timed out")
        return False
    except Exception as e:
        print(f"Error in check_correctness: {e}")
        return False


def has_code(response: str) -> list:
    """Check if the response contains code blocks.

    Args:
        response (str): The text response to check

    Returns:
        list: List of code blocks found in the response
    """
    pattern = r"```(?:[a-zA-Z]*)\n(.*?)```"
    return re.findall(pattern, response, re.DOTALL)


def process_single_row(row: dict) -> dict:
    """Process a single row of the dataset.

    Args:
        row (dict): Dataset row containing solution and metadata

    Returns:
        dict: Processed row with correctness evaluation
    """
    try:
        code_blocks = has_code(row.get("deepseek_solution", ""))

        if not code_blocks:
            return {**row, "correctness": False, "reason": "Does not contain code component."}

        last_code = code_blocks[-1]
        if check_correctness(row, last_code):
            row["correctness"] = True
            row["reason"] = ""
        else:
            row["correctness"] = False
            row["reason"] = "Code is incorrect."

        return row

    except Exception as e:
        return {**row, "correctness": False, "reason": f"Processing error: {str(e)}"}


def process_dataset_parallel(df: Dataset, num_cpus: int = None, batch_size: int = 1024) -> Dataset:
    """Process the dataset in parallel using multiple CPUs.

    Args:
        df (Dataset): Input dataset to process
        num_cpus (int, optional): Number of CPUs to use. Defaults to max CPUs - 1
        batch_size (int, optional): Size of each processing batch. Defaults to 1024

    Returns:
        Dataset: Processed dataset with correctness evaluations
    """
    if num_cpus is None:
        num_cpus = max(1, multiprocessing.cpu_count() - 1)

    data = df.to_list()
    total_rows = len(data)
    print(f"Processing {total_rows} rows using {num_cpus} CPUs...")

    all_results = []
    for i in range(0, total_rows, batch_size):
        batch = data[i : i + batch_size]
        with Pool(processes=num_cpus) as pool:
            batch_results = list(tqdm(pool.map(process_single_row, batch), total=len(batch), desc=f"Processing batch {i//batch_size + 1}"))

        all_results.extend(batch_results)

        # Calculate and print statistics for this batch
        batch_correct = sum(1 for r in batch_results if r.get("correctness", False))
        print(f"\nBatch {i//batch_size + 1} Results:")
        print(f"Processed examples: {len(all_results)}/{total_rows}")
        print(f"Correct in this batch: {batch_correct}/{len(batch_results)} ({batch_correct/len(batch_results)*100:.2f}%)")
        print(f"Total correct so far: {sum(1 for r in all_results if r.get('correctness', False))}/{len(all_results)}\n")

    return Dataset.from_list(all_results)
