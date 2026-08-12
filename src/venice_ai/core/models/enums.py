"""
Enumerations for the Venice AI SDK.

This module defines all ``StrEnum`` types used across the SDK.  It has
**no** dependencies on sibling submodules.

Note:
    ``ModelType`` is the canonical definition.  It is re-exported by
    ``venice_ai.types.enums`` for public-API convenience.
"""

from __future__ import annotations

from enum import StrEnum


class ModelType(StrEnum):
    """Supported model types."""

    EMBEDDING = "embedding"
    IMAGE = "image"
    TEXT = "text"
    TTS = "tts"
    UPSCALE = "upscale"
    INPAINT = "inpaint"
    VIDEO = "video"
    ASR = "asr"
    MUSIC = "music"


class APIKeyType(StrEnum):
    """API key types."""

    INFERENCE = "INFERENCE"
    ADMIN = "ADMIN"


class Currency(StrEnum):
    """Supported currencies."""

    USD = "USD"
    DIEM = "DIEM"


class FinishReason(StrEnum):
    """Completion finish reasons."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"


class MessageRole(StrEnum):
    """Message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


__all__ = [
    "APIKeyType",
    "Currency",
    "FinishReason",
    "MessageRole",
    "ModelType",
]
