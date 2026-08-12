"""
Form-data serialization helpers.

Exports:
    * :func:`serialize_form_value` — convert Python values to API-friendly strings
"""

from __future__ import annotations

from typing import Any


def serialize_form_value(value: Any) -> str:
    """
    Serialize a value for form data submission.

    Handles proper serialization of booleans to lowercase strings
    as expected by the Venice AI API.

    Args:
        value: The value to serialize

    Returns:
        String representation suitable for form data

    Example:
        >>> serialize_form_value(True)
        'true'
        >>> serialize_form_value(False)
        'false'
        >>> serialize_form_value(42)
        '42'
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
