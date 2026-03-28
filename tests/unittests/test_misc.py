from bespokelabs.curator.misc import safe_model_dump


def test_safe_model_dump_with_pydantic():
    from pydantic import BaseModel

    class MyModel(BaseModel):
        name: str
        value: int

    model = MyModel(name="test", value=42)
    result = safe_model_dump(model)
    assert result == {"name": "test", "value": 42}


def test_safe_model_dump_fallback():
    class FakeModel:
        def __init__(self):
            self.name = "test"
            self.value = 42

        def model_dump(self):
            raise TypeError("not a real pydantic model")

    result = safe_model_dump(FakeModel())
    assert result["name"] == "test"
    assert result["value"] == 42
