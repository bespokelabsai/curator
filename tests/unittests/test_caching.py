import os

import pytest
from datasets import Dataset

from bespokelabs import curator

_ONLINE_BACKENDS = [{"integration": backend, "vcr_path": "tests/unittests/cassettes/"} for backend in {"openai"}]


@pytest.mark.parametrize("temp_working_dir", (_ONLINE_BACKENDS), indirect=True)
def test_same_value_caching(tmp_path, temp_working_dir):
    _, _, vcr_config = temp_working_dir
    """Test that using the same value multiple times uses cache."""
    values = []

    with vcr_config.use_cassette("basic_diff_cache.yaml"):
        for _ in range(3):
            prompter = curator.LLM(model_name="gpt-4o-mini")
            result = prompter(["Say '1'. Do not explain."], working_dir=str(tmp_path)).dataset
            values.append(result.to_pandas().iloc[0]["response"])

        # Count cache directories, excluding metadata.db
        cache_dirs = [d for d in tmp_path.glob("*") if d.name != "metadata.db"]
        assert len(cache_dirs) == 1, f"Expected 1 cache directory but found {len(cache_dirs)}"
        assert values == ["1", "1", "1"], "Same value should produce same results"


@pytest.mark.parametrize("temp_working_dir", (_ONLINE_BACKENDS), indirect=True)
def test_same_dataset_caching(tmp_path, temp_working_dir):
    """Test that using the same dataset multiple times uses cache."""
    _, _, vcr_config = temp_working_dir

    with vcr_config.use_cassette("basic_diff_cache.yaml"):
        dataset = ["Say '1'. Do not explain."]
        prompter = curator.LLM(model_name="gpt-4o-mini")

        result = prompter(dataset, working_dir=str(tmp_path)).dataset
        assert result.to_pandas().iloc[0]["response"] == "1"

        result = prompter(dataset, working_dir=str(tmp_path)).dataset
        assert result.to_pandas().iloc[0]["response"] == "1"

        # Count cache directories, excluding metadata.db
        cache_dirs = [d for d in tmp_path.glob("*") if d.name != "metadata.db"]
        assert len(cache_dirs) == 1, f"Expected 1 cache directory but found {len(cache_dirs)}"


@pytest.mark.parametrize("temp_working_dir", (_ONLINE_BACKENDS), indirect=True)
def test_different_dataset_caching(tmp_path, temp_working_dir):
    """Test that using different datasets creates different cache entries."""
    _, _, vcr_config = temp_working_dir

    with vcr_config.use_cassette("basic_diff_cache.yaml"):
        dataset1 = Dataset.from_list([{"my_prompt": "Say '1'. Do not explain."}])
        dataset2 = Dataset.from_list([{"my_prompt": "Say '2'. Do not explain."}])

        class Prompter(curator.LLM):
            def prompt(self, input: str):
                return input["my_prompt"]

        prompter = Prompter(model_name="gpt-4o-mini")

        result = prompter(dataset1, working_dir=str(tmp_path)).dataset
        assert result.to_pandas().iloc[0]["response"] == "1"

        result = prompter(dataset2, working_dir=str(tmp_path)).dataset
        assert result.to_pandas().iloc[0]["response"] == "2"

        # Count cache directories, excluding metadata.db
        cache_dirs = [d for d in tmp_path.glob("*") if d.name != "metadata.db"]
        assert len(cache_dirs) == 2, f"Expected 2 cache directory but found {len(cache_dirs)}"


@pytest.mark.parametrize("temp_working_dir", (_ONLINE_BACKENDS), indirect=True)
def test_nested_call_caching(tmp_path, temp_working_dir):
    """Test that changing a nested upstream function invalidates the cache."""
    _, _, vcr_config = temp_working_dir

    def value_generator():
        return 1

    with vcr_config.use_cassette("basic_diff_cache.yaml"):
        prompter = curator.LLM(model_name="gpt-4o-mini")
        result = prompter([f"Say '{value_generator()}'. Do not explain."], working_dir=str(tmp_path)).dataset
        assert result.to_pandas().iloc[0]["response"] == "1"

        def value_generator():  # noqa: F811
            return 2

        result = prompter([f"Say '{value_generator()}'. Do not explain."], working_dir=str(tmp_path)).dataset
        assert result.to_pandas().iloc[0]["response"] == "2"

        # Count cache directories, excluding metadata.db
        cache_dirs = [d for d in tmp_path.glob("*") if d.name != "metadata.db"]
        assert len(cache_dirs) == 2, f"Expected 2 cache directory but found {len(cache_dirs)}"


def test_function_hash_dir_change():
    """Test that identical functions in different directories but same base filename produce the same hash."""
    import logging
    import tempfile
    from pathlib import Path

    from bespokelabs.curator.llm.llm import _get_function_hash

    # Set up logging to write to a file in the current directory
    debug_log = Path("function_debug.log")
    logging.basicConfig(level=logging.DEBUG, format="%(message)s", filename=str(debug_log), filemode="w")
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
        print("\nHash comparison:")  # Print to stdout
        print(f"  hash1: {hash1}")
        print(f"  hash2: {hash2}")
        logger.debug("\nHash comparison:")
        logger.debug(f"  hash1: {hash1}")
        logger.debug(f"  hash2: {hash2}")

        assert hash1 == hash2, "Identical functions should produce the same hash"


@pytest.mark.parametrize("temp_working_dir", (_ONLINE_BACKENDS), indirect=True)
def test_disable_cache(tmp_path, temp_working_dir):
    """Test that disabling cache creates different directories for each run."""
    _, _, vcr_config = temp_working_dir

    os.environ["CURATOR_DISABLE_CACHE"] = "true"

    try:
        prompter = curator.LLM(model_name="gpt-4o-mini")

        # Run twice and store results
        with vcr_config.use_cassette("basic_diff_cache.yaml"):
            result1 = prompter(["Say '1'. Do not explain."], working_dir=str(tmp_path)).dataset

        with vcr_config.use_cassette("basic_diff_cache.yaml"):
            result2 = prompter(["Say '1'. Do not explain."], working_dir=str(tmp_path)).dataset

        # Verify both runs produced the expected output
        assert result1.to_pandas().iloc[0]["response"] == "1"
        assert result2.to_pandas().iloc[0]["response"] == "1"

        # Check cache directory, excluding metadata.db
        cache_dirs = [d for d in tmp_path.glob("*") if d.name != "metadata.db"]

        # Should have exactly 2 different cache directories
        assert len(cache_dirs) == 2
        # All directories should be different (no duplicates)
        assert len({str(d) for d in cache_dirs}) == 2
    finally:
        os.environ.pop("CURATOR_DISABLE_CACHE", None)
