import pytest
from pydantic import BaseModel, ValidationError, ConfigDict
from typing import List, Optional, Dict, Any
from bespokelabs.curator.request_processor.generic_request import GenericRequest

class TestResponseFormat(BaseModel):
    text: str
    confidence: float

class ComplexResponseFormat(BaseModel):
    title: str
    items: List[str]
    metadata: dict
    optional_field: Optional[int] = None
    nested: Optional["NestedFormat"] = None

class NestedFormat(BaseModel):
    id: int
    name: str

# Enable Pydantic's strict mode for response format models
ComplexResponseFormat.model_rebuild()
TestResponseFormat.model_rebuild()

def test_generic_request_basic_creation():
    """Test basic creation of GenericRequest with minimal required fields."""
    request = GenericRequest(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        original_row={},
        original_row_idx=0
    )
    assert request.model == "gpt-4"
    assert len(request.messages) == 1
    assert request.messages[0]["role"] == "user"
    assert request.messages[0]["content"] == "Hello"
    assert request.original_row == {}
    assert request.original_row_idx == 0
    assert request.response_format is None

def test_generic_request_with_response_format():
    """Test GenericRequest creation with a response format."""
    request = GenericRequest(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        original_row={},
        original_row_idx=0,
        response_format=TestResponseFormat.model_json_schema()
    )
    assert request.response_format == TestResponseFormat.model_json_schema()

def test_generic_request_with_complex_response_format():
    """Test GenericRequest with a complex nested response format."""
    request = GenericRequest(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        original_row={},
        original_row_idx=0,
        response_format=ComplexResponseFormat.model_json_schema()
    )
    schema = request.response_format
    assert "title" in schema["properties"]
    assert "items" in schema["properties"]
    assert "metadata" in schema["properties"]
    assert "optional_field" in schema["properties"]
    assert "nested" in schema["properties"]

def test_generic_request_with_invalid_response_format():
    """Test GenericRequest with invalid response format."""
    # Test with non-dict response format
    with pytest.raises(ValidationError):
        GenericRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            original_row={},
            original_row_idx=0,
            response_format=123  # Not a dict
        )
    
    with pytest.raises(ValidationError):
        GenericRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            original_row={},
            original_row_idx=0,
            response_format=["not", "a", "dict"]  # Not a dict
        )

def test_generic_request_with_complex_messages():
    """Test GenericRequest with multiple messages in conversation."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"}
    ]
    request = GenericRequest(
        model="gpt-4",
        messages=messages,
        original_row={"input": "test"},
        original_row_idx=1
    )
    assert len(request.messages) == 4
    assert request.messages == messages
    assert request.original_row == {"input": "test"}

def test_generic_request_with_complex_data():
    """Test GenericRequest with complex original_row data."""
    complex_data = {
        "nested": {"key": "value"},
        "list": [1, 2, 3],
        "number": 42,
        "boolean": True,
        "null": None
    }
    request = GenericRequest(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        original_row=complex_data,
        original_row_idx=0
    )
    assert request.original_row == complex_data

def test_generic_request_message_content():
    """Test message content edge cases."""
    # Empty string content should be valid
    request = GenericRequest(
        model="gpt-4",
        messages=[{"role": "user", "content": ""}],
        original_row={},
        original_row_idx=0
    )
    assert request.messages[0]["content"] == ""

    # Special characters in content should be preserved
    special_content = "Hello\n\tWorld! 👋 🌍"
    request = GenericRequest(
        model="gpt-4",
        messages=[{"role": "user", "content": special_content}],
        original_row={},
        original_row_idx=0
    )
    assert request.messages[0]["content"] == special_content

def test_generic_request_original_row_idx():
    """Test original_row_idx validation and edge cases."""
    # Test with zero index (first row)
    request = GenericRequest(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        original_row={},
        original_row_idx=0
    )
    assert request.original_row_idx == 0

    # Test with large positive index (large dataset)
    request = GenericRequest(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        original_row={},
        original_row_idx=999999
    )
    assert request.original_row_idx == 999999

    # Test with negative index (special cases like retry or error)
    request = GenericRequest(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        original_row={},
        original_row_idx=-1
    )
    assert request.original_row_idx == -1

def test_generic_request_state_isolation():
    """Test that GenericRequest objects maintain state isolation."""
    # Test that source data modifications don't affect request
    source_messages = [{"role": "user", "content": "Hello"}]
    source_row = {"data": "test"}
    
    request = GenericRequest(
        model="gpt-4",
        messages=source_messages,
        original_row=source_row,
        original_row_idx=0
    )
    
    # Modify source data
    source_messages[0]["content"] = "Modified"
    source_row["data"] = "modified"
    
    # Verify request data is unchanged
    assert request.messages[0]["content"] == "Hello"
    assert request.original_row["data"] == "test"

    # Test that response format modifications don't affect request
    schema = TestResponseFormat.model_json_schema()
    request = GenericRequest(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        original_row={},
        original_row_idx=0,
        response_format=schema
    )
    
    # Modify schema after request creation
    schema["title"] = "Modified"
    
    # Verify request schema is unchanged
    assert request.response_format["title"] != "Modified"

def test_generic_request_with_non_string_messages():
    """Test GenericRequest with messages containing non-string values."""
    messages = [
        {"role": "user", "content": {"key": "value"}},  # Dict content
        {"role": "assistant", "content": 42},  # Number content
        {"role": "user", "content": ["list", "of", "items"]}  # List content
    ]
    request = GenericRequest(
        model="gpt-4",
        messages=messages,
        original_row={},
        original_row_idx=0
    )
    assert request.messages == messages

def test_generic_request_response_format_types():
    """Test GenericRequest with various valid response format types."""
    # Test with empty dict
    request = GenericRequest(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        original_row={},
        original_row_idx=0,
        response_format={}
    )
    assert request.response_format == {}

    # Test with dict containing non-string values
    complex_format = {
        "number": 42,
        "list": [1, 2, 3],
        "nested": {"key": "value"}
    }
    request = GenericRequest(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        original_row={},
        original_row_idx=0,
        response_format=complex_format
    )
    assert request.response_format == complex_format
