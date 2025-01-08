from datasets import Dataset

from bespokelabs import curator


def test_same_value_caching(tmp_path):
    """Test that using the same value multiple times uses cache."""
    values = []

    # Test with same value multiple times
    for _ in range(3):

        def prompt_func():
            return f"Say '1'. Do not explain."

        prompter = curator.LLM(
            prompt_func=prompt_func,
            model_name="gpt-4o-mini",
        )
        result = prompter(working_dir=str(tmp_path))
        values.append(result.to_pandas().iloc[0]["response"])

    # Count cache directories, excluding metadata.db
    cache_dirs = [d for d in tmp_path.glob("*") if d.name != "metadata.db"]
    assert len(cache_dirs) == 1, f"Expected 1 cache directory but found {len(cache_dirs)}"
    assert values == ["1", "1", "1"], "Same value should produce same results"


def test_different_values_caching(tmp_path):
    """Test that using different values creates different cache entries."""
    values = []

    # Test with different values
    for x in [1, 2, 3]:

        def prompt_func():
            return f"Say '{x}'. Do not explain."

        prompter = curator.LLM(
            prompt_func=prompt_func,
            model_name="gpt-4o-mini",
        )
        result = prompter(working_dir=str(tmp_path))
        values.append(result.to_pandas().iloc[0]["response"])

    # Count cache directories, excluding metadata.db
    cache_dirs = [d for d in tmp_path.glob("*") if d.name != "metadata.db"]
    assert len(cache_dirs) == 3, f"Expected 3 cache directories but found {len(cache_dirs)}"
    assert values == ["1", "2", "3"], "Different values should produce different results"


def test_same_dataset_caching(tmp_path):
    """Test that using the same dataset multiple times uses cache."""
    dataset = Dataset.from_list([{"instruction": "Say '1'. Do not explain."}])
    prompter = curator.LLM(
        prompt_func=lambda x: x["instruction"],
        model_name="gpt-4o-mini",
    )

    result = prompter(dataset=dataset, working_dir=str(tmp_path))
    assert result.to_pandas().iloc[0]["response"] == "1"

    result = prompter(dataset=dataset, working_dir=str(tmp_path))
    assert result.to_pandas().iloc[0]["response"] == "1"

    # Count cache directories, excluding metadata.db
    cache_dirs = [d for d in tmp_path.glob("*") if d.name != "metadata.db"]
    assert len(cache_dirs) == 1, f"Expected 1 cache directory but found {len(cache_dirs)}"


def test_different_dataset_caching(tmp_path):
    """Test that using different datasets creates different cache entries."""
    dataset1 = Dataset.from_list([{"instruction": "Say '1'. Do not explain."}])
    dataset2 = Dataset.from_list([{"instruction": "Say '2'. Do not explain."}])
    prompter = curator.LLM(
        prompt_func=lambda x: x["instruction"],
        model_name="gpt-4o-mini",
    )

    result = prompter(dataset=dataset1, working_dir=str(tmp_path))
    assert result.to_pandas().iloc[0]["response"] == "1"

    result = prompter(dataset=dataset2, working_dir=str(tmp_path))
    assert result.to_pandas().iloc[0]["response"] == "2"

    # Count cache directories, excluding metadata.db
    cache_dirs = [d for d in tmp_path.glob("*") if d.name != "metadata.db"]
    assert len(cache_dirs) == 2, f"Expected 2 cache directory but found {len(cache_dirs)}"


def test_nested_call_caching(tmp_path):
    """Test that changing a nested upstream function invalidates the cache."""

    def value_generator():
        return 1

    def prompt_func():
        return f"Say '{value_generator()}'. Do not explain."

    prompter = curator.LLM(
        prompt_func=prompt_func,
        model_name="gpt-4o-mini",
    )
    result = prompter(working_dir=str(tmp_path))
    assert result.to_pandas().iloc[0]["response"] == "1"

    def value_generator():
        return 2

    result = prompter(working_dir=str(tmp_path))
    assert result.to_pandas().iloc[0]["response"] == "2"

    # Count cache directories, excluding metadata.db
    cache_dirs = [d for d in tmp_path.glob("*") if d.name != "metadata.db"]
    assert len(cache_dirs) == 2, f"Expected 2 cache directory but found {len(cache_dirs)}"


def test_overwrite_cache_env_var(tmp_path, monkeypatch, caplog):
    """Test that setting CURATOR_OVERWRITE_CACHE overwrites cache directory."""
    # Set environment variable to overwrite cache
    monkeypatch.setenv("CURATOR_OVERWRITE_CACHE", "true")

    # Create a simple dataset with one instruction
    dataset = Dataset.from_list([{"instruction": "Say 'test'. Do not explain."}])

    def prompt_func(x):
        return x["instruction"]

    prompter = curator.LLM(
        prompt_func=prompt_func,
        model_name="gpt-4o-mini",
    )

    # First run with cache overwrite enabled
    with caplog.at_level("INFO"):
        result = prompter(dataset=dataset, working_dir=str(tmp_path))
        response = result.to_pandas().iloc[0]["response"].lower().rstrip(".")
        assert response == "test", f"Expected 'test', got '{response}'"
        
        # Verify cache directory exists and overwrite message is logged
        cache_dirs = [d for d in tmp_path.glob("*") if d.name != "metadata.db"]
        assert len(cache_dirs) > 0, "Cache directory should be created"
        assert "Cache will be overwritten" in caplog.text, "Cache overwrite should be logged"
        caplog.clear()

    # Second run to verify cache is overwritten
    with caplog.at_level("INFO"):
        result = prompter(dataset=dataset, working_dir=str(tmp_path))
        response = result.to_pandas().iloc[0]["response"].lower().rstrip(".")
        assert response == "test", f"Expected 'test', got '{response}'"
        
        # Verify cache directory still exists and is overwritten
        cache_dirs = [d for d in tmp_path.glob("*") if d.name != "metadata.db"]
        assert len(cache_dirs) > 0, "Cache directory should still exist"
        assert "Cache will be overwritten" in caplog.text, "Cache overwrite should be logged"
        assert "Using cached" not in caplog.text, "Cache should not be reused when overwrite is enabled"


def test_function_hash_dir_change():
    """Test that identical functions in different directories but same base filename produce the same hash."""
    import logging
    import os
    import sys
    import tempfile
    from pathlib import Path

    from bespokelabs.curator.llm.llm import _get_function_hash

    # Set up logging to write to a file in the current directory
    debug_log = Path("function_debug.log")
    logging.basicConfig(
        level=logging.DEBUG, format="%(message)s", filename=str(debug_log), filemode="w"
    )
    logger = logging.getLogger(__name__)

    def dump_function_details(func, prefix):
        """Helper to dump all function details."""
        print(f"\n{prefix} details:")  # Print to stdout as well
        logger.debug(f"\n{prefix} details:")
        # Basic attributes
        details = {
            "__name__": func.__name__,
            "__module__": func.__module__,
            "__qualname__": func.__qualname__,
            "__code__.co_filename": func.__code__.co_filename,
            "__code__.co_name": func.__code__.co_name,
            "__code__.co_firstlineno": func.__code__.co_firstlineno,
            "__code__.co_consts": func.__code__.co_consts,
            "__code__.co_names": func.__code__.co_names,
            "__code__.co_varnames": func.__code__.co_varnames,
            "__code__.co_code": func.__code__.co_code.hex(),
            "__code__.co_flags": func.__code__.co_flags,
            "__code__.co_stacksize": func.__code__.co_stacksize,
            "__code__.co_freevars": func.__code__.co_freevars,
            "__code__.co_cellvars": func.__code__.co_cellvars,
            "__globals__ keys": sorted(func.__globals__.keys()),
            "__closure__": func.__closure__,
            "__defaults__": func.__defaults__,
            "__kwdefaults__": func.__kwdefaults__,
        }

        for key, value in details.items():
            msg = f"  {key}: {value}"
            print(msg)  # Print to stdout
            logger.debug(msg)  # Log to file

    def create_function(name, tmp_path):
        # Create a temporary file with a function definition
        path = tmp_path / f"{name}.py"
        with open(path, "w") as f:
            f.write(
                """
def test_func():
    x = 42  # Add a constant
    y = "Hello"  # Add a string constant
    z = [1, 2, 3]  # Add a list constant
    return f"{y}, {x}! {z}"  # Use all constants
"""
            )

        # Import the function from the file
        import importlib.util

        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.test_func

    # Create two identical functions in different files
    with tempfile.TemporaryDirectory() as tmp_dir:
        func1 = create_function("module1", Path(tmp_dir))
        func2 = create_function("module1", Path(tmp_dir))

        # Dump detailed information about both functions
        dump_function_details(func1, "Function 1")
        dump_function_details(func2, "Function 2")

        # Both should produce the same hash
        hash1 = _get_function_hash(func1)
        hash2 = _get_function_hash(func2)
        print(f"\nHash comparison:")  # Print to stdout
        print(f"  hash1: {hash1}")
        print(f"  hash2: {hash2}")
        logger.debug(f"\nHash comparison:")
        logger.debug(f"  hash1: {hash1}")
        logger.debug(f"  hash2: {hash2}")

        assert hash1 == hash2, "Identical functions should produce the same hash"
