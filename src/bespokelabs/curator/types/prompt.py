# Description: Pydantic models for multimodal prompts.
import base64
import typing as t

from pydantic import BaseModel, Field, root_validator


class BaseType(BaseModel):
    """A class to represent the base type for multimodal prompts."""

    type: t.ClassVar[str] = Field(..., description="The type of the multimodal prompt.")


class Image(BaseType):
    """A class to represent an image for multimodal prompts."""

    url: str = Field("", description="The URL of the image.")
    content: bytes = Field(b"", description="Image content bytes.")
    type: t.ClassVar[str] = "image"

    def serialize(self) -> str:
        """Convert image content to base64."""
        if self.url:
            return self.url
        assert self.content, "Image content is not provided."
        return base64.b64encode(self.content).decode("utf-8")

    @root_validator(pre=True)
    def check_url_or_content(self, values):
        """Ensure that only one of url or content is provided."""
        url = values.get("url")
        content = values.get("content")

        if bool(url) == bool(content):
            raise ValueError("Only one of 'url' or 'content' must be provided.")
        return values

    def __post_init__(self):
        """Post init."""
        # assert url or content is provided
        assert self.url or self.content, "Either 'url' or 'content' must be provided."


class File(BaseType):
    """A class to represent a file for multimodal prompts."""

    url: str = Field(..., description="The URL of the file.")
    type: t.ClassVar[str] = "file"


class _MultiModalPrompt(BaseType):
    """A class to represent a multimodal prompt."""

    texts: t.List[str] = Field(default_factory=list, description="The text of the prompt.")
    images: t.List[Image] = Field(default_factory=list, description="The image of the prompt.")
    files: t.List[File] = Field(default_factory=list, description="The file of the prompt.")

    @classmethod
    def load(cls, messages):
        prompt = {"texts": [], "images": [], "files": []}
        for msg in messages:
            if isinstance(msg, BaseType):
                if msg.type == "image":
                    prompt["images"].append(msg)
                elif msg.type == "file":
                    prompt["files"].append(msg)
            else:
                prompt["texts"].append(msg)
        return cls(**prompt)
