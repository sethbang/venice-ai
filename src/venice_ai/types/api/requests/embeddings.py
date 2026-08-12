"""
Embeddings generation request models for Venice.ai API.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ...identifiers import ModelId

# ============================================================================
# Embeddings Request Models
# ============================================================================


class EmbeddingsRequest(BaseModel):
    """Embeddings generation request"""

    input: str | list[str] | list[int] | list[list[int]] = Field(
        ..., description="Text(s) to embed"
    )
    model: ModelId = Field(..., description="Embedding model to use")

    # Optional parameters
    dimensions: int | None = Field(
        None, ge=1, description="Number of dimensions for output embeddings"
    )
    encoding_format: Literal["float", "base64"] | None = Field(
        None, description="Format to return embeddings (server default is 'float' when omitted)"
    )
    user: str | None = Field(None, description="User identifier (compatibility only)")

    @field_validator("input")
    @classmethod
    def validate_input(cls, v: Any) -> Any:
        if isinstance(v, str) and len(v) == 0:
            raise ValueError("Input string cannot be empty")
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("Input array cannot be empty")
            if len(v) > 2048:
                raise ValueError("Input array cannot exceed 2048 items")
            if isinstance(v[0], str) and any(len(s) == 0 for s in v if isinstance(s, str)):
                raise ValueError("Input strings cannot be empty")
        return v


# ============================================================================
# Export All Models
# ============================================================================

__all__ = [
    "EmbeddingsRequest",
]
