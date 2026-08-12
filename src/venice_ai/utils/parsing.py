"""
Shared parsing helpers for safe type conversions.

These functions consolidate the duplicated ``_parse_int`` / ``_parse_float``
patterns that previously existed in:

* ``venice_ai.exceptions`` (as ``_safe_int_parse`` / ``_safe_float_parse``)
* ``venice_ai.core.models.base.VeniceBaseModel`` (as instance methods)

All call-sites now delegate to these canonical implementations.
"""

from __future__ import annotations


def safe_int(value: str | None) -> int | None:
    """Safely parse a string to *int*, returning ``None`` on failure.

    Args:
        value: The string to convert, or ``None``.

    Returns:
        The parsed integer, or ``None`` if *value* is ``None`` or cannot be
        converted.
    """
    if value is None:
        return None
    try:
        # Use int(float(...)) to handle "50.0"-style strings from HTTP headers.
        return int(float(value))
    except (ValueError, TypeError, OverflowError):
        return None


def safe_float(value: str | None) -> float | None:
    """Safely parse a string to *float*, returning ``None`` on failure.

    Args:
        value: The string to convert, or ``None``.

    Returns:
        The parsed float, or ``None`` if *value* is ``None`` or cannot be
        converted.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def ms_epoch_to_seconds(value: float | None) -> float | None:
    """Normalize a possibly-millisecond-epoch numeric to seconds.

    Several Venice rate-limit reset headers (e.g.
    ``x-ratelimit-reset-requests`` / ``x-ratelimit-reset-tokens``) arrive as
    13-digit *absolute Unix epoch milliseconds* (e.g. ``1780580108941``).
    Values whose magnitude is at or above ``1e12`` are treated as milliseconds
    and divided by 1000; smaller values (already in seconds, e.g. a 10-digit
    epoch) pass through untouched.

    This mirrors :meth:`venice_ai.core.models.base.VeniceBaseModel._ms_to_seconds`
    so ms/seconds handling is symmetric across every reset-header consumer.

    Args:
        value: The numeric value to normalize, or ``None``.

    Returns:
        The seconds-epoch value, or ``None`` if *value* is ``None``.
    """
    if value is None:
        return None
    if abs(value) >= 1e12:
        return value / 1000
    return value


__all__ = [
    "ms_epoch_to_seconds",
    "safe_float",
    "safe_int",
]
