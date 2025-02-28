from pydantic import BaseModel, Field


class _TokenUsage(BaseModel):
    input: int | None = Field(default=0, description="Number of input tokens")
    output: int | None = Field(default=0, description="Number of output tokens")
    _total: int | None = Field(default=None, exclude=True)

    @property
    def total(self) -> int | None:
        if self._total is not None:
            return self._total
        elif self.input is None or self.output is None:
            return None
        return self.input + self.output

    @total.setter
    def total(self, value: int | None) -> None:
        self._total = value
