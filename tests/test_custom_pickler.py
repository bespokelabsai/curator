import os
import pytest
from io import BytesIO
from typing import List
from pydantic import BaseModel

from bespokelabs.curator.utils.custom_pickler import CustomPickler, dumps, loads


class TestModel(BaseModel):
    value: str
    items: List[int]


def test_custom_pickler_type_annotations():
    """Test CustomPickler handles type annotations correctly."""

    def func(x: TestModel) -> List[int]:
        return x.items

    # Test pickling and unpickling
    pickled = dumps(func)
    unpickled = loads(pickled)

    # Test function still works
    test_input = TestModel(value="test", items=[1, 2, 3])
    assert unpickled(test_input) == [1, 2, 3]


def test_custom_pickler_path_normalization():
    """Test CustomPickler normalizes paths in function source."""
    import tempfile
    from pathlib import Path

    def create_test_function(pkg_dir: Path):
        """Create a test function in a specific package directory."""
        # Create a module file in the package directory
        module_path = pkg_dir / "test_module.py"
        with open(module_path, "w") as f:
            f.write("""
def func():
    path = os.path.join("/home", "user", "file.txt")
    return path
""")

        # Import the function from the file
        import importlib.util
        from types import ModuleType
        
        spec = importlib.util.spec_from_file_location("test_module", str(module_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {module_path}")
            
        module: ModuleType = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, "func"):
            raise AttributeError(f"Module {module_path} does not have 'func' attribute")
            
        return module.func

    # Create two identical functions in different Ray-like package directories
    with tempfile.TemporaryDirectory() as tmp_dir1, tempfile.TemporaryDirectory() as tmp_dir2:
        # Simulate Ray package paths
        ray_pkg_dir1 = Path(tmp_dir1) / "ray" / "ray_pkg_123"
        ray_pkg_dir2 = Path(tmp_dir2) / "ray" / "ray_pkg_456"
        ray_pkg_dir1.mkdir(parents=True, exist_ok=True)
        ray_pkg_dir2.mkdir(parents=True, exist_ok=True)

        # Create and pickle functions from different directories
        func1 = create_test_function(ray_pkg_dir1)
        func2 = create_test_function(ray_pkg_dir2)

        # Get hashes for both functions
        pickled1 = dumps(func1)
        pickled2 = dumps(func2)

        # Hashes should match despite different Ray package paths
        assert pickled1 == pickled2, "Function hashes should match regardless of Ray package path"


def test_custom_pickler_hybrid_serialization():
    """Test CustomPickler falls back to cloudpickle for type annotations."""

    def func(x: TestModel, items: List[int]) -> List[int]:
        return [i for i in items if i > int(x.value)]

    # Test pickling with both type annotations and path-dependent code
    file = BytesIO()
    pickler = CustomPickler(file, recurse=True)
    pickler.dump(func)

    # Test unpickling
    file.seek(0)
    unpickled = loads(file.getvalue())

    # Test function works
    test_input = TestModel(value="2", items=[1, 2, 3])
    assert unpickled(test_input, [1, 2, 3, 4]) == [3, 4]
