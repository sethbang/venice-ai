"""Typed model-capability descriptors for v2.0.0 discovery API.

Returned by :meth:`venice_ai.resources.models.Models.get_capabilities`. The
underlying API model uses camelCase field names (``supportsVision`` etc.) and
splits feature flags between ``ModelCapabilities`` (text-only) and
``*ModelConstraints`` (one shape per non-text type). This module exposes a
**polymorphic** snake_case view discriminated by ``type``, so callers can
write ::

    caps = await client.models.get_capabilities(model_id)
    match caps:
        case ChatCapabilities(supports_function_calling=True): ...
        case VideoCapabilities(supports_audio=True, durations=durs): ...

instead of inspecting the raw ``ModelSpec`` and remembering which type's
flags live where.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ChatCapabilities(BaseModel):
    """Capabilities for text / chat models.

    Sourced from :class:`ModelCapabilities` plus a few fields from
    :class:`ModelSpec` that are conceptually capabilities (privacy,
    context_window). All flags are translated from camelCase wire format
    to snake_case Python idiom.
    """

    type: Literal["chat"] = "chat"
    context_window: int | None = Field(
        None, description="Max input tokens the model accepts (``None`` if unspecified)."
    )
    supports_function_calling: bool
    supports_vision: bool
    supports_reasoning: bool
    supports_response_schema: bool
    supports_web_search: bool
    supports_logprobs: bool
    supports_audio_input: bool
    supports_video_input: bool
    supports_multiple_images: bool
    supports_reasoning_effort: bool
    supports_tee_attestation: bool
    supports_e2ee: bool
    supports_x_search: bool
    optimized_for_code: bool
    # Plain str (not a Literal) so a new server-side quantization value never
    # crashes get_capabilities() — mirrors ModelCapabilities.quantization on the
    # wire model.
    quantization: str
    privacy: Literal["private", "anonymized"] | None = None


class ImageCapabilities(BaseModel):
    """Capabilities for image-generation models.

    Sourced from :class:`ImageModelConstraints` and
    :class:`ModelSpec.supportsWebSearch`.
    """

    type: Literal["image"] = "image"
    prompt_character_limit: int | None = Field(
        default=None, description="Max prompt length in characters (``None`` if unspecified)."
    )
    width_height_divisor: int | None = Field(
        default=None,
        description="Output dimensions must be divisible by this (``None`` if unspecified).",
    )
    supports_web_search: bool = False


class VideoCapabilities(BaseModel):
    """Capabilities for video models (text-to-video, image-to-video, upscale).

    Sourced from :class:`VideoModelConstraints`. Empty lists for
    ``resolutions`` / ``durations`` / ``aspect_ratios`` mean the model
    derives those from its inputs (typical for image-to-video).
    """

    type: Literal["video"] = "video"
    # Open str (not a Literal) to match the relaxed VideoModelConstraints.model_type
    # it is sourced from, so a new server-side model_type never crashes parsing.
    model_type: str
    supports_audio: bool
    audio_configurable: bool
    accepts_video_input: bool
    resolutions: list[str] = Field(default_factory=list)
    durations: list[str] = Field(default_factory=list)
    aspect_ratios: list[str] = Field(default_factory=list)


class InpaintCapabilities(BaseModel):
    """Capabilities for inpainting models, sourced from :class:`InpaintModelConstraints`."""

    type: Literal["inpaint"] = "inpaint"
    prompt_character_limit: int | None = None
    combine_images: bool = False


class GenericCapabilities(BaseModel):
    """Catch-all for model types with no current feature taxonomy.

    Used for ``embedding``, ``tts``, ``asr``, ``music``, and ``upscale`` —
    these resources don't expose feature flags today. Future versions may
    promote any of them to a dedicated ``*Capabilities`` shape.
    """

    type: Literal["embedding", "tts", "asr", "music", "upscale"]
    privacy: Literal["private", "anonymized"] | None = None


Capabilities = Annotated[
    ChatCapabilities
    | ImageCapabilities
    | VideoCapabilities
    | InpaintCapabilities
    | GenericCapabilities,
    Field(discriminator="type"),
]
"""Discriminated union returned by :meth:`Models.get_capabilities`."""


__all__ = [
    "ChatCapabilities",
    "ImageCapabilities",
    "VideoCapabilities",
    "InpaintCapabilities",
    "GenericCapabilities",
    "Capabilities",
]
