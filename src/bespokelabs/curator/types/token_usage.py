from pydantic import BaseModel, Field


class _TokenUsage(BaseModel):
    input: int | None = Field(default=0, description="Number of input tokens")
    output: int | None = Field(default=0, description="Number of output tokens")
    total: int | None = Field(default=None, description="Total number of tokens")

    def model_post_init(self, __context):
        # Only set total if not explicitly provided
        if self.total is None and self.input is not None and self.output is not None:
            self.total = self.input + self.output
