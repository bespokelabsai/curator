# Description: Pydantic models for multimodal prompts.
import base64
import mimetypes
import os
import typing as t
from io import BytesIO

from PIL import Image as PIL_Image
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator


class BaseType(BaseModel):
    """A class to represent the base type for multimodal prompts."""

    url: str = Field("", description="The URL of the file.")
    type: t.ClassVar[str] = Field(..., description="The type of the multimodal prompt.")

    @staticmethod
    def _is_local_uri(path):
        return os.path.exists(path) and os.path.isfile(path)

    @staticmethod
    def _load_file_as_b64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @property
    def is_local(self):
        """Check if the image is a local file."""
        if self.url:
            return self._is_local_uri(self.url)
        return False


def _pil_image_to_bytes(image: PIL_Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _pil_to_base64(image: PIL_Image.Image) -> str:
    b = _pil_image_to_bytes(image)
    return base64.b64encode(b).decode("utf-8")


class Image(BaseType):
    """A class to represent an image for multimodal prompts."""

    content: bytes | PIL_Image.Image | str = Field("", description="Image content bytes.")
    detail: str = Field("auto", description="Details about the image. Note 'auto' is only supported for OpenAI client.")
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: t.ClassVar[str] = "image"

    def serialize(self) -> str:
        """Convert image content to base64."""
        if self.url:
            if self.is_local:
                return self._load_file_as_b64(self.url)
            return self.url

        assert self.content, "Image content is not provided."
        if isinstance(self.content, PIL_Image.Image):
            self.content = _pil_image_to_bytes(self.content)
        elif isinstance(self.content, str):
            return self.content
        return base64.b64encode(self.content).decode("utf-8")

    @field_serializer("content")
    def serialize_content(self, content, _info) -> str:
        """Convert image content to base64."""
        if isinstance(content, PIL_Image.Image):
            return _pil_to_base64(content)
        return content

    @model_validator(mode="before")
    def check_url_or_content(self):  # noqa: N805
        """Ensure that only one of url or content is provided."""
        if bool(self.get("url")) == bool(self.get("content")):
            raise ValueError("Only one of 'url' or 'content' must be provided.")
        return self


class File(BaseType):
    """A class to represent a file for multimodal prompts."""

    mime_type: str | None = Field(None, description="The MIME type of the file.")
    type: t.ClassVar[str] = "file"

    def __post_init__(self):
        """Post init."""
        if not self.mime_type:
            mime_type, _ = mimetypes.guess_type(self.url)
            self.mime_type = mime_type.lower()

    def serialize(self) -> str:
        """Convert file to base64."""
        if self.is_local:
            return self._load_file_as_b64(self.url)
        return self.url


class _MultiModalPrompt(BaseType):
    """A class to represent a multimodal prompt."""

    texts: t.List[str] = Field(default_factory=list, description="The texts of the prompt.")
    images: t.List[Image] = Field(default_factory=list, description="The images of the prompt.")
    files: t.List[File] = Field(default_factory=list, description="The files of the prompt.")

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
