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
    def func():
        path = os.path.join("/home", "user", "file.txt")
        return path
    
    # Pickle in one directory
    original_dir = os.getcwd()
    try:
        os.chdir("/tmp")
        pickled1 = dumps(func)
        
        # Pickle in another directory
        os.chdir("/home")
        pickled2 = dumps(func)
        
        # Hashes should match despite different directories
        assert pickled1 == pickled2
    finally:
        os.chdir(original_dir)

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
