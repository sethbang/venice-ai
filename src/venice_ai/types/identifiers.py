"""
Semantic type definitions for Venice AI identifiers.

This module provides typed, normalized identifiers for various entities
used throughout the Venice AI API, improving type safety and IDE support.

Design Philosophy:
    These types are intentionally lenient. The API is the authoritative
    source for valid identifiers. These types provide:
    1. Semantic clarity (distinguishing model IDs from other strings)
    2. Whitespace trimming (case is PRESERVED — Venice identifiers are
       case-sensitive; the API rejects a mis-cased model ID)
    3. Basic format hints for IDE autocomplete

    They do NOT strictly validate - that's the API's job.
"""

from typing import Annotated

from pydantic import AfterValidator, Field


def normalize_model_id(value: str) -> str:
    """
    Normalize a Venice AI model ID.

    Trims surrounding whitespace only. **Case is preserved** — Venice model IDs
    are case-sensitive: the inference endpoints reject ``wai-illustrious`` and
    accept only ``wai-Illustrious``, so the SDK must relay the id exactly as the
    API returns (or the caller supplies) it. Does not strictly validate the
    format — the API is the authoritative source.

    Args:
        value: The model ID string to normalize

    Returns:
        The model ID with surrounding whitespace stripped (case preserved)

    Raises:
        ValueError: If the value is empty

    Examples:
        >>> normalize_model_id("  wai-Illustrious  ")
        'wai-Illustrious'
        >>> normalize_model_id("llama-3.3-70b")
        'llama-3.3-70b'
    """
    if not value or not value.strip():
        raise ValueError("Model ID cannot be empty")

    return value.strip()


def normalize_queue_id(value: str) -> str:
    """
    Normalize a video queue ID.

    Queue IDs are UUIDs returned by the video queue endpoint.

    Args:
        value: The queue ID string to normalize

    Returns:
        The queue ID with surrounding whitespace stripped (case preserved)

    Raises:
        ValueError: If the value is empty
    """
    if not value or not value.strip():
        raise ValueError("Queue ID cannot be empty")

    return value.strip()


# Type Aliases with Normalization
ModelId = Annotated[
    str,
    AfterValidator(normalize_model_id),
    Field(description="Venice AI model identifier (e.g., 'llama-3.3-70b', 'venice-sd35')"),
]

QueueId = Annotated[
    str,
    AfterValidator(normalize_queue_id),
    Field(description="Video generation queue identifier (UUID format)"),
]


__all__ = [
    "ModelId",
    "QueueId",
    "normalize_model_id",
    "normalize_queue_id",
]
