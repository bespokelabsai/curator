"""Code execution for APPS dataset.

Code from https://github.com/NovaSky-AI/SkyThought/blob/e855aad095f4eeee00ba6a909dfe4300faf6d853/skythought/tools/util/task_handlers.py
"""

import copy
import json
import multiprocessing
import re
from multiprocessing import Manager

import numpy as np
from util.testing.apps import run_test as apps_run_test


def has_code(response):
    """Extract code blocks from a text response.

    Args:
        response (str): Text that may contain code blocks

    Returns:
        list: List of code block contents
    """
    pattern = r"```(?:[a-zA-Z]*)\n(.*?)```"
    # Use re.DOTALL to match multiline content inside backticks
    matches = re.findall(pattern, response, re.DOTALL)
    return matches


def check_correctness(problem, generation, timeout=10):
    """Check if generated code passes test cases.

    Args:
        problem (dict): Problem specification
        generation (str): Generated code to test
        timeout (int): Maximum execution time in seconds

    Returns:
        bool: True if all tests pass, False otherwise
    """

    def _temp_run(problem, generation, debug, result):
        try:
            result.append(apps_run_test(problem=problem, test=generation, debug=debug))
        except Exception:
            pass

    manager = Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(problem, generation, False, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    return bool(result and np.all(result[0]))


def process_single_row(row):
    """Process a single dataset row containing a code generation response.

    Args:
        row (dict): Dataset row containing response and test cases

    Returns:
        dict: Processing results including correctness and reason
    """
    response = row.get("deepseek_solution", "")
    response_entry = {
        "content": response,
        "correctness": None,
        "reason": None,
    }

    code_filter_result = has_code(response)
    if len(code_filter_result) == 0:
        response_entry["correctness"] = False
        response_entry["reason"] = "Does not contain code component."
    else:
        last_code = code_filter_result[-1]
        problem_to_check = copy.deepcopy(row)
        problem_to_check["input_output"] = json.loads(row["input_output"])
        try:
            problem_to_check["solutions"] = json.loads(row["solutions"])
        except Exception as e:
            problem_to_check["solutions"] = ""
            print(f"Empty solution from the dataset: {e}")
        curr_res = check_correctness(problem_to_check, generation=last_code)
        if curr_res:
            response_entry["correctness"] = True
            response_entry["reason"] = ""
        else:
            response_entry["correctness"] = False
            response_entry["reason"] = "Code is incorrect."

    return response_entry


def process_dataset_parallel(df):
    """Process the dataset in parallel using multiple CPUs.

    Args:
        df (pandas.DataFrame): Dataset to process

    Returns:
        pandas.Series: Processing results for each row
    """
    return df.map(process_single_row)
