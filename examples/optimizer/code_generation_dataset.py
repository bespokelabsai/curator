"""Programming problems and dataset helpers for GEPA code generation example."""

import json
import random
from typing import Any

# Programming problems with test cases
# Mix of simple and more challenging problems to ensure room for prompt improvement
PROGRAMMING_PROBLEMS: list[dict[str, Any]] = [
    {
        "problem_id": "fibonacci",
        "description": "Write a Python function `fibonacci(n)` that returns the nth Fibonacci number (0-indexed). F(0)=0, F(1)=1, F(n)=F(n-1)+F(n-2).",
        "function_name": "fibonacci",
        "test_cases": [
            {"input": (0,), "expected": 0},
            {"input": (1,), "expected": 1},
            {"input": (2,), "expected": 1},
            {"input": (5,), "expected": 5},
            {"input": (10,), "expected": 55},
        ],
    },
    {
        "problem_id": "is_prime",
        "description": "Write a Python function `is_prime(n)` that returns True if n is a prime number, False otherwise. Assume n >= 0.",
        "function_name": "is_prime",
        "test_cases": [
            {"input": (0,), "expected": False},
            {"input": (1,), "expected": False},
            {"input": (2,), "expected": True},
            {"input": (17,), "expected": True},
            {"input": (18,), "expected": False},
            {"input": (49,), "expected": False},
            {"input": (97,), "expected": True},
        ],
    },
    {
        "problem_id": "factorial",
        "description": "Write a Python function `factorial(n)` that returns n! (n factorial). Assume n >= 0.",
        "function_name": "factorial",
        "test_cases": [
            {"input": (0,), "expected": 1},
            {"input": (1,), "expected": 1},
            {"input": (5,), "expected": 120},
            {"input": (7,), "expected": 5040},
            {"input": (3,), "expected": 6},
            {"input": (6,), "expected": 720},
        ],
    },
    {
        "problem_id": "count_vowels",
        "description": "Write a Python function `count_vowels(s)` that returns the count of vowels (a, e, i, o, u, case-insensitive) in the string.",
        "function_name": "count_vowels",
        "test_cases": [
            {"input": ("hello",), "expected": 2},
            {"input": ("AEIOU",), "expected": 5},
            {"input": ("xyz",), "expected": 0},
            {"input": ("Python Programming",), "expected": 4},
            {"input": ("Rhythm",), "expected": 0},
        ],
    },
    {
        "problem_id": "sum_digits",
        "description": "Write a Python function `sum_digits(n)` that returns the sum of digits of a non-negative integer n.",
        "function_name": "sum_digits",
        "test_cases": [
            {"input": (0,), "expected": 0},
            {"input": (123,), "expected": 6},
            {"input": (9999,), "expected": 36},
            {"input": (1001,), "expected": 2},
        ],
    },
    {
        "problem_id": "list_intersection",
        "description": (
            "Write a Python function `list_intersection(lst1, lst2)` that returns a sorted list of " "elements present in both lists (no duplicates in output)."
        ),
        "function_name": "list_intersection",
        "test_cases": [
            {"input": ([1, 2, 3], [2, 3, 4]), "expected": [2, 3]},
            {"input": ([1, 2], [3, 4]), "expected": []},
            {"input": ([1, 1, 2], [1, 2, 2]), "expected": [1, 2]},
            {"input": ([3, 3, 2, 1], [3, 1, 1]), "expected": [1, 3]},
        ],
    },
    {
        "problem_id": "run_length_encode",
        "description": (
            "Write a Python function `run_length_encode(s)` that performs run-length encoding. For "
            "consecutive repeated characters, output the character followed by the count. "
            "Example: 'aaabbc' -> 'a3b2c1'."
        ),
        "function_name": "run_length_encode",
        "test_cases": [
            {"input": ("aaabbc",), "expected": "a3b2c1"},
            {"input": ("",), "expected": ""},
            {"input": ("abc",), "expected": "a1b1c1"},
            {"input": ("aaa",), "expected": "a3"},
            {"input": ("aabbaa",), "expected": "a2b2a2"},
        ],
    },
    {
        "problem_id": "second_largest",
        "description": (
            "Write a Python function `second_largest(lst)` that returns the second largest unique value "
            "in a list. If there's no second largest (list has fewer than 2 unique values), return None."
        ),
        "function_name": "second_largest",
        "test_cases": [
            {"input": ([1, 2, 3, 4],), "expected": 3},
            {"input": ([5, 5, 5],), "expected": None},
            {"input": ([1, 1, 2, 2],), "expected": 1},
            {"input": ([10],), "expected": None},
            {"input": ([-1, -2, -3],), "expected": -2},
            {"input": ([2, 1],), "expected": 1},
        ],
    },
    {
        "problem_id": "balanced_parens",
        "description": (
            "Write a Python function `balanced_parens(s)` that returns True if the string has balanced "
            "parentheses '()', '[]', '{}', False otherwise. Other characters should be ignored."
        ),
        "function_name": "balanced_parens",
        "test_cases": [
            {"input": ("()",), "expected": True},
            {"input": ("([{}])",), "expected": True},
            {"input": ("([)]",), "expected": False},
            {"input": ("",), "expected": True},
            {"input": ("hello (world)",), "expected": True},
            {"input": ("(((",), "expected": False},
            {"input": ("{[()]}",), "expected": True},
            {"input": ("{[(])}",), "expected": False},
        ],
    },
    {
        "problem_id": "merge_intervals",
        "description": (
            "Write a Python function `merge_intervals(intervals)` that merges overlapping intervals. "
            "Input is a list of tuples (start, end). Return sorted merged intervals. "
            "Example: [(1,3), (2,6), (8,10)] -> [(1,6), (8,10)]."
        ),
        "function_name": "merge_intervals",
        "test_cases": [
            {"input": ([(1, 3), (2, 6), (8, 10)],), "expected": [(1, 6), (8, 10)]},
            {"input": ([(1, 4), (4, 5)],), "expected": [(1, 5)]},
            {"input": ([],), "expected": []},
            {"input": ([(1, 2), (3, 4), (5, 6)],), "expected": [(1, 2), (3, 4), (5, 6)]},
            {"input": ([(1, 10), (2, 3), (4, 5)],), "expected": [(1, 10)]},
            {"input": ([(5, 7), (1, 3), (2, 4)],), "expected": [(1, 4), (5, 7)]},
        ],
    },
    {
        "problem_id": "word_frequency",
        "description": (
            "Write a Python function `word_frequency(text)` that returns a dictionary mapping each word "
            "(lowercased, alphanumeric only) to its frequency. Use str.split() for tokenization."
        ),
        "function_name": "word_frequency",
        "test_cases": [
            {"input": ("hello world hello",), "expected": {"hello": 2, "world": 1}},
            {"input": ("",), "expected": {}},
            {"input": ("The the THE",), "expected": {"the": 3}},
            {"input": ("Hi, hi!!!",), "expected": {"hi": 2}},
            {"input": ("A b a B",), "expected": {"a": 2, "b": 2}},
        ],
    },
    {
        "problem_id": "longest_consecutive",
        "description": (
            "Write a Python function `longest_consecutive(nums)` that returns the length of the longest "
            "consecutive sequence in an unsorted list. Example: [100, 4, 200, 1, 3, 2] -> 4 (sequence: 1,2,3,4)."
        ),
        "function_name": "longest_consecutive",
        "test_cases": [
            {"input": ([100, 4, 200, 1, 3, 2],), "expected": 4},
            {"input": ([],), "expected": 0},
            {"input": ([1],), "expected": 1},
            {"input": ([1, 2, 0, 1],), "expected": 3},
            {"input": ([9, 1, 4, 7, 3, -1, 0, 5, 8, -1, 6],), "expected": 7},
        ],
    },
    {
        "problem_id": "binary_search",
        "description": "Write a Python function `binary_search(nums, target)` that returns the index of target in a sorted list, or -1 if not found.",
        "function_name": "binary_search",
        "test_cases": [
            {"input": ([1, 3, 5, 7, 9], 7), "expected": 3},
            {"input": ([1, 3, 5, 7, 9], 2), "expected": -1},
            {"input": ([], 1), "expected": -1},
            {"input": ([1, 2], 1), "expected": 0},
        ],
    },
    {
        "problem_id": "reverse_words",
        "description": (
            "Write a Python function `reverse_words(s)` that returns a string with the word order reversed. "
            "Words are separated by spaces; preserve single spaces in the output."
        ),
        "function_name": "reverse_words",
        "test_cases": [
            {"input": ("hello world",), "expected": "world hello"},
            {"input": ("single",), "expected": "single"},
            {"input": ("a b c",), "expected": "c b a"},
            {"input": ("  spaced out  ",), "expected": "out spaced"},
        ],
    },
    {
        "problem_id": "gcd",
        "description": "Write a Python function `gcd(a, b)` that returns the greatest common divisor of two non-negative integers.",
        "function_name": "gcd",
        "test_cases": [
            {"input": (54, 24), "expected": 6},
            {"input": (0, 5), "expected": 5},
            {"input": (10, 0), "expected": 10},
            {"input": (13, 13), "expected": 13},
            {"input": (48, 18), "expected": 6},
        ],
    },
    {
        "problem_id": "two_sum_exists",
        "description": (
            "Write a Python function `two_sum_exists(nums, target)` that returns True if any two "
            "distinct numbers in the list sum to target, False otherwise."
        ),
        "function_name": "two_sum_exists",
        "test_cases": [
            {"input": ([2, 7, 11, 15], 9), "expected": True},
            {"input": ([3, 2, 4], 6), "expected": True},
            {"input": ([3, 3], 7), "expected": False},
            {"input": ([], 1), "expected": False},
            {"input": ([1, 5, 3], 4), "expected": True},
        ],
    },
    {
        "problem_id": "remove_duplicates",
        "description": "Write a Python function `remove_duplicates(lst)` that returns a new list with duplicates removed, preserving the original order.",
        "function_name": "remove_duplicates",
        "test_cases": [
            {"input": ([1, 2, 2, 3, 1],), "expected": [1, 2, 3]},
            {"input": ([],), "expected": []},
            {"input": ([5, 5, 5],), "expected": [5]},
            {"input": ([1, 1, 2, 1],), "expected": [1, 2]},
        ],
    },
    {
        "problem_id": "max_subarray_sum",
        "description": "Write a Python function `max_subarray_sum(nums)` that returns the maximum sum of any contiguous subarray. Assume nums is non-empty.",
        "function_name": "max_subarray_sum",
        "test_cases": [
            {"input": ([1, -3, 2, 1, -1],), "expected": 3},
            {"input": ([-2, -1, -3],), "expected": -1},
            {"input": ([5],), "expected": 5},
            {"input": ([4, -1, 2, 1],), "expected": 6},
        ],
    },
    {
        "problem_id": "valid_palindrome",
        "description": (
            "Write a Python function `valid_palindrome(s)` that returns True if s is a palindrome after "
            "removing non-alphanumeric characters and ignoring case."
        ),
        "function_name": "valid_palindrome",
        "test_cases": [
            {"input": ("A man, a plan, a canal: Panama",), "expected": True},
            {"input": ("race a car",), "expected": False},
            {"input": ("",), "expected": True},
            {"input": ("No 'x' in Nixon",), "expected": True},
        ],
    },
]


def serialize_test_cases(problems: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialize test_cases to JSON to avoid PyArrow schema inference issues."""
    return [{**p, "test_cases": json.dumps(p["test_cases"])} for p in problems]


def deserialize_test_cases(test_cases: str | list) -> list:
    """Deserialize test_cases from JSON string if needed."""
    if isinstance(test_cases, str):
        return json.loads(test_cases)
    return test_cases


def create_datasets(
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create training and validation datasets from programming problems.

    Returns:
        Tuple of (trainset, valset) as lists of dicts
    """
    problems = list(PROGRAMMING_PROBLEMS)
    rng = random.Random(seed)
    rng.shuffle(problems)

    split_idx = int(len(problems) * train_ratio)
    trainset = serialize_test_cases(problems[:split_idx])
    valset = serialize_test_cases(problems[split_idx:])

    return trainset, valset
