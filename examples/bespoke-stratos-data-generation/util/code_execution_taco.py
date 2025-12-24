"""Code execution for TACO dataset.

Code from https://github.com/NovaSky-AI/SkyThought/blob/e855aad095f4eeee00ba6a909dfe4300faf6d853/skythought/tools/util/task_handlers.py
"""

import multiprocessing
import re
import traceback
from multiprocessing import Pool

import numpy as np
from datasets import Dataset
from tqdm import tqdm
from util.testing.taco import run_test as taco_run_test


# 'fork' copies the parent process memory, so functions don't need to be pickled
# and child processes can use signals (each child is main thread of its process)
_mp_context = multiprocessing.get_context('fork')


def _run_test_worker(args):
    """Worker function for process pool. Takes tuple (problem, generation)."""
    problem, generation = args
    try:
        result = taco_run_test(problem, test=generation, debug=False)
        is_correct = bool(result and np.all(result))
        if is_correct:
            return True, ""
        else:
            return False, f"Code is incorrect. Result: {result}"
    except Exception as e:
        return False, f"Exception: {str(e)}\n{traceback.format_exc()}"


def check_correctness(problem, generation, timeout=10):
    """Run the test with a timeout using a separate process.
    
    Uses fork context so the child process can use signals internally.
    Works from both main thread and worker threads (e.g., GEPA optimization).
    """
    try:
        with _mp_context.Pool(processes=1) as pool:
            async_result = pool.apply_async(_run_test_worker, ((problem, generation),))
            try:
                return async_result.get(timeout=timeout)
            except multiprocessing.TimeoutError:
                pool.terminate()
                return False, "Test execution timed out"
    except Exception as e:
        return False, f"Process error: {str(e)}\n{traceback.format_exc()}"


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
    
    Called from Pool workers in process_dataset_parallel.
    Since Pool workers are forked processes, we call _run_test_worker directly
    (taco_run_test has its own signal-based timeout).

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
        # Call _run_test_worker directly - we're in a forked process, signals work,
        # and taco_run_test has its own timeout handling
        row["correctness"], row["reason"] = _run_test_worker((row, last_code))
        return row

    except Exception as e:
        return {**row, "correctness": False, "reason": f"Processing error: {str(e)}"}


def process_dataset_parallel(df: Dataset, num_cpus: int = None, batch_size: int = 2048) -> Dataset:
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
            batch_results = list(tqdm(pool.map(process_single_row, batch), total=len(batch), desc=f"Processing batch {i // batch_size + 1}"))

        all_results.extend(batch_results)

        # Calculate and print statistics for this batch
        batch_correct = sum(1 for r in batch_results if r.get("correctness", False))
        print(f"\nBatch {i // batch_size + 1} Results:")
        print(f"Processed examples: {len(all_results)}/{total_rows}")
        print(f"Correct in this batch: {batch_correct}/{len(batch_results)} ({batch_correct / len(batch_results) * 100:.2f}%)")
        print(f"Total correct so far: {sum(1 for r in all_results if r.get('correctness', False))}/{len(all_results)}\n")

    return Dataset.from_list(all_results)
