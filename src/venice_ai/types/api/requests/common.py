"""
Shared request components and utility models for Venice.ai API requests.

Most types are re-exported from the canonical definitions in
``venice_ai.core.models.common`` (which use ``VeniceBaseModel`` with strict
validation).  Only ``DateRangeParams`` is defined locally because the API
serialisation uses camelCase field names (``startDate`` / ``endDate``) whereas
the core model uses snake_case.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator

# ---------------------------------------------------------------------------
# Re-exports from canonical location (core/models/common)
# ---------------------------------------------------------------------------
from venice_ai.core.models.common import (
    AudioContent,
    AudioContentParam,
    ConsumptionLimit,
    FileContent,
    FileContentParam,
    FileObject,
    FileObjectParam,
    ImageContent,
    ImageContentParam,
    ImageUrl,
    ImageUrlParam,
    JSONObjectFormat,
    JSONSchemaFormat,
    MessageContentPart,
    MessageContentPartParam,
    PaginationParams,
    SpecificToolChoice,
    StreamOptions,
    TextContent,
    TextContentParam,
    TextResponseFormat,
    Tool,
    ToolChoiceFunction,
    ToolFunction,
    VeniceParameters,
    VideoContent,
    VideoContentParam,
)

# ---------------------------------------------------------------------------
# Reasoning controls (shared by chat + completion requests)
# ---------------------------------------------------------------------------

ReasoningEffortLevel = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
"""Effort tier for reasoning-capable models.

Mirrors the `/chat/completions` spec: higher levels allow more thinking tokens
but cost more. ``"max"`` unlocks the full ceiling on supported models.
"""

ReasoningSummary = Literal["auto", "concise", "detailed"]
"""Requested reasoning-summary verbosity."""


class ReasoningConfig(BaseModel):
    """Nested configuration for reasoning behavior on supported models.

    Corresponds to the ``reasoning`` object on ``/chat/completions``. The
    top-level ``reasoning_effort`` field takes precedence over
    ``reasoning.effort`` when both are provided.
    """

    effort: ReasoningEffortLevel | None = Field(default=None, description="Reasoning effort tier")
    summary: ReasoningSummary | None = Field(
        default=None, description="Requested reasoning summary style"
    )


# ============================================================================
# DateRangeParams — API-facing (camelCase) version
# ============================================================================


class DateRangeParams(BaseModel):
    """Date range filtering parameters"""

    startDate: datetime | None = Field(None, description="Start date")
    endDate: datetime | None = Field(None, description="End date")

    @field_validator("endDate")
    @classmethod
    def validate_date_range(cls, v: Any, info: ValidationInfo) -> Any:
        start_date = info.data.get("startDate")
        if start_date and v and v <= start_date:
            raise ValueError("endDate must be after startDate")
        return v


# ============================================================================
# Export All Models
# ============================================================================

__all__ = [
    # Content types
    "TextContent",
    "ImageUrl",
    "ImageContent",
    "FileObject",
    "FileContent",
    "AudioContent",
    "VideoContent",
    "MessageContentPart",
    "MessageContentPartParam",
    "TextContentParam",
    "ImageUrlParam",
    "ImageContentParam",
    "FileObjectParam",
    "FileContentParam",
    "AudioContentParam",
    "VideoContentParam",
    # Stream and tool components
    "StreamOptions",
    "ToolFunction",
    "Tool",
    "ToolChoiceFunction",
    "SpecificToolChoice",
    # Response format components
    "JSONSchemaFormat",
    "JSONObjectFormat",
    "TextResponseFormat",
    # Venice-specific components
    "VeniceParameters",
    # Reasoning controls
    "ReasoningEffortLevel",
    "ReasoningSummary",
    "ReasoningConfig",
    # Utility models
    "PaginationParams",
    "DateRangeParams",
    "ConsumptionLimit",
]
