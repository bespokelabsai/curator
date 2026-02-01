"""Utilities for evaluating LLM-generated code in GEPA optimization.

This module provides functions to extract, analyze, and test Python code generated
during the GEPA (Generalized Evolutionary Prompt Adaptation) process. It includes
static analysis for style constraints (type hints, docstrings, restricted APIs)
and dynamic execution for functional correctness (unit tests).
"""

import ast
import traceback
from typing import Any, Callable, Dict


def extract_code(response: str) -> str:
    """Extract code from markdown blocks if present."""
    code = response
    if "```python" in response:
        try:
            code = response.split("```python")[1].split("```")[0].strip()
        except IndexError:
            pass
    elif "```" in response:
        try:
            code = response.split("```")[1].split("```")[0].strip()
        except IndexError:
            pass
    return code


def analyze_code(code: str, function_name: str) -> list[str]:
    """Check style constraints and return a list of violations."""
    violations: list[str] = []
    try:
        module = ast.parse(extract_code(code))
    except SyntaxError:
        return ["syntax_error"]

    # Check for docstring
    if ast.get_docstring(module) is None:
        # Also check inside the function definition
        has_docstring = False
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                if ast.get_docstring(node):
                    has_docstring = True
                break
        if not has_docstring:
            violations.append("missing_docstring")

    allowed_toplevel = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Expr)
    for node in module.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if not isinstance(node, allowed_toplevel):
            violations.append("extra_top_level_code")
            break

    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    target_funcs = [fn for fn in functions if fn.name == function_name]
    if len(functions) != 1 or len(target_funcs) != 1:
        violations.append("function_definition_mismatch")
    else:
        fn = target_funcs[0]
        args = list(fn.args.posonlyargs) + list(fn.args.args) + list(fn.args.kwonlyargs)
        if fn.args.vararg:
            args.append(fn.args.vararg)
        if fn.args.kwarg:
            args.append(fn.args.kwarg)
        if any(arg.annotation is None for arg in args) or fn.returns is None:
            violations.append("missing_type_hints")

    for node in ast.walk(module):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            violations.append("imports_not_allowed")
            break

    banned = {"print", "input", "open", "eval", "exec"}
    for node in ast.walk(module):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in banned:
                violations.append(f"banned_call:{node.func.id}")
                break
            if isinstance(node.func, ast.Attribute) and node.func.attr in banned:
                violations.append(f"banned_call:{node.func.attr}")
                break

    return violations


def run_tests(code: str, function_name: str, test_cases: list) -> tuple[int, int, str]:
    """Execute code and run test cases.

    Returns:
        Tuple of (passed_count, total_count, feedback_message)
    """
    passed = 0
    total = len(test_cases)
    feedback_parts = []

    try:
        # Create isolated namespace and execute the code
        namespace: Dict[str, Any] = {}
        exec(extract_code(code), namespace)

        if function_name not in namespace:
            return 0, total, f"Function '{function_name}' not found in generated code"

        func: Callable[..., Any] = namespace[function_name]

        # Run each test case
        for i, test in enumerate(test_cases):
            try:
                result = func(*test["input"])
                if result == test["expected"]:
                    passed += 1
                else:
                    feedback_parts.append(f"Test {i + 1}: input={test['input']}, expected={test['expected']}, got={result}")
            except Exception as e:
                feedback_parts.append(f"Test {i + 1}: Runtime error - {str(e)}")
    except SyntaxError as e:
        return 0, total, f"Syntax error in generated code. Provide ONLY the markdown code block, no other text: {e}"
    except Exception as e:
        return 0, total, f"Execution error: {e}\n{traceback.format_exc()}"

    if passed == total:
        feedback = "All tests passed!"
    else:
        feedback = f"Passed {passed}/{total}. Failures: " + "; ".join(feedback_parts)

    return passed, total, feedback
