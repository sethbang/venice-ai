"""
Date validation utilities for Venice AI SDK.

This module provides reusable date validation functions to ensure consistent
date handling across the SDK. It supports ISO 8601 date formats and provides
clear error messages for validation failures.
"""

from datetime import datetime


def validate_date_string(date_str: str, param_name: str) -> str:
    """
    Validate ISO 8601 date format string.

    Performs minimal validation (checks for non-empty string) and defers to the API
    for comprehensive format validation and detailed error messages. Client-side
    validation is intentionally kept minimal; the API performs full validation.

    Supports multiple ISO 8601 formats including:
    - YYYY-MM-DD (e.g., '2025-01-01')
    - YYYY-MM-DDTHH:MM:SSZ (UTC, e.g., '2025-01-01T00:00:00Z')
    - YYYY-MM-DDTHH:MM:SS.fffZ (UTC with milliseconds)
    - YYYY-MM-DDTHH:MM:SS±HH:MM (with timezone offset)
    - YYYY-MM-DDTHH:MM:SS.fff±HH:MM (with milliseconds and timezone)

    Args:
        date_str: Date string to validate in ISO 8601 format
        param_name: Name of parameter for error messages

    Returns:
        The validated date string (unchanged)

    Raises:
        ValueError: If date_str is empty or not a string

    Examples:
        >>> validate_date_string("2025-01-01", "startDate")
        '2025-01-01'

        >>> validate_date_string("2025-01-01T00:00:00Z", "endDate")
        '2025-01-01T00:00:00Z'

        >>> validate_date_string("", "date")
        ValueError: date must be a non-empty string in ISO 8601 format
    """
    # Minimal validation - just check that the string is not empty
    # Let the API handle detailed validation and return appropriate errors
    if not date_str or not isinstance(date_str, str):
        raise ValueError(f"{param_name} must be a non-empty string in ISO 8601 format")

    return date_str


def validate_date_range(
    start_date: str,
    end_date: str,
    start_param: str = "start_date",
    end_param: str = "end_date",
) -> tuple[str, str]:
    """
    Validate date range ensuring start date is before or equal to end date.

    Both dates must be valid ISO 8601 format strings. This function validates
    each date individually and then ensures the date range is valid (start <= end).

    Args:
        start_date: Start date in ISO 8601 format
        end_date: End date in ISO 8601 format
        start_param: Name of start parameter for error messages
        end_param: Name of end parameter for error messages

    Returns:
        Tuple of (start_date, end_date) unchanged if valid

    Raises:
        ValueError: If either date is invalid or end_date is before start_date

    Examples:
        >>> validate_date_range("2025-01-01", "2025-12-31")
        ('2025-01-01', '2025-12-31')

        >>> validate_date_range("2025-12-31", "2025-01-01")
        ValueError: end_date (2025-01-01) cannot be before start_date (2025-12-31)

        >>> validate_date_range("2025-01-01T00:00:00Z", "2025-12-31T23:59:59Z")
        ('2025-01-01T00:00:00Z', '2025-12-31T23:59:59Z')
    """
    # Validate both dates individually first
    validate_date_string(start_date, start_param)
    validate_date_string(end_date, end_param)

    # Parse dates for comparison (supports multiple ISO 8601 formats)
    try:
        start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
    except ValueError as e:
        # This should rarely happen since we validated above, but handle it
        raise ValueError(
            f"Invalid date format. Both {start_param} and {end_param} must be in "
            f"ISO 8601 format (e.g., '2025-01-01' or '2025-01-01T00:00:00Z'). Error: {e}"
        ) from e

    # Validate that end date is not before start date (allow equal for single-day queries)
    if end_dt < start_dt:
        raise ValueError(f"{end_param} ({end_date}) cannot be before {start_param} ({start_date})")

    return start_date, end_date


def validate_expires_at(expires_at: str, param_name: str = "expires_at") -> str:
    """
    Validate expiration date format for API keys.

    Accepts empty string, YYYY-MM-DD format, or full ISO 8601 format.
    Empty string means no expiration.

    Note: For YYYY-MM-DD format, only the format structure is validated (length and dash
    positions), not the semantic validity of the date. This allows the API to handle
    detailed validation and provide appropriate error messages.

    Args:
        expires_at: Expiration date string (can be empty, YYYY-MM-DD, or ISO format)
        param_name: Name of parameter for error messages

    Returns:
        The validated expires_at string (unchanged)

    Raises:
        ValueError: If format is invalid

    Examples:
        >>> validate_expires_at("")
        ''

        >>> validate_expires_at("2025-12-31")
        '2025-12-31'

        >>> validate_expires_at("2025-12-31T23:59:59Z")
        '2025-12-31T23:59:59Z'

        >>> validate_expires_at("invalid")
        ValueError: expires_at must be empty string, YYYY-MM-DD, or ISO format
    """
    # Allow empty string (no expiration)
    if expires_at == "":
        return expires_at

    # For non-empty values, validate format
    if expires_at:
        # Check for YYYY-MM-DD format (only format structure, not semantic validity)
        if len(expires_at) == 10 and expires_at[4] == "-" and expires_at[7] == "-":
            return expires_at

        # Try full ISO format with semantic validation
        try:
            datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            return expires_at
        except ValueError as e:
            raise ValueError(f"{param_name} must be empty string, YYYY-MM-DD, or ISO format") from e

    return expires_at
