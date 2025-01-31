# Description: Pydantic models for multimodal prompts.
from pydantic import BaseModel, Field


class BaseType(BaseModel):
    """A class to represent the base type for multimodal prompts."""

    type: str = Field(..., description="The type of the multimodal prompt.")


class Image(BaseModel):
    """A class to represent an image for multimodal prompts."""

    url: str = Field(None, description="The URL of the image.")
    content: str = Field(None, description="Base64-encoded image content.")
    type = "image"

    def __post_init__(self):
        # assert url or content is provided
        assert self.url or self.content, "Either 'url' or 'content' must be provided."


class File(BaseModel):
    """A class to represent a file for multimodal prompts."""

    url: str = Field(..., description="The URL of the file.")
    type = "file"


class _MultiModalPrompt(BaseType):
    """A class to represent a multimodal prompt."""

    texts: str = Field(None, description="The text of the prompt.")
    images: Image = Field(None, description="The image of the prompt.")
    files: File = Field(None, description="The file of the prompt.")

    @classmethod
    def load(cls, messages):
        prompt = {}
        for msg in messages:
            if isinstance(msg, BaseType):
                if msg.type == "image":
                    prompt["images"] = msg
                elif msg.type == "file":
                    prompt["files"] = msg
            else:
                prompt["text"] = msg
        return cls(**prompt)
