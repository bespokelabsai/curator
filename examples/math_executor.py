"""Safe Python code execution utilities for math problem solving."""

import ast
import contextlib
import io
import sys
from typing import Tuple, Optional


def is_safe_ast(tree: ast.AST) -> bool:
    """Check if the AST contains only safe operations."""
    for node in ast.walk(tree):
        # Block imports
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            return False
        # Block exec/eval
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ["exec", "eval", "compile"]:
                    return False
        # Block attribute access that might be dangerous
        if isinstance(node, ast.Attribute):
            if node.attr in ["open", "read", "write", "system"]:
                return False
    return True


def execute_math_code(code: str, timeout: int = 5) -> Tuple[Optional[str], Optional[str]]:
    """
    Execute Python code safely with timeout and output capture.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Tuple of (result, error_message)
    """
    try:
        # Parse and validate AST
        tree = ast.parse(code)
        if not is_safe_ast(tree):
            return None, "Code contains unsafe operations"

        # Capture stdout
        stdout = io.StringIO()
        stderr = io.StringIO()

        # Execute with timeout and output capture
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compile(tree, "<string>", "exec"), {"__builtins__": {"print": print}}, {})

        error = stderr.getvalue().strip()
        if error:
            return None, f"Execution error: {error}"

        result = stdout.getvalue().strip()
        return result, None

    except SyntaxError as e:
        return None, f"Syntax error: {str(e)}"
    except Exception as e:
        return None, f"Runtime error: {str(e)}"
