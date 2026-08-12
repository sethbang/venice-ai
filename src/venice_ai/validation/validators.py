"""Input validation utilities for Venice AI SDK."""

import re
from datetime import datetime


def validate_model_id(model_id: str | None, param_name: str = "model_id") -> None:
    """
    Validate model ID format.

    Args:
        model_id: Model identifier to validate
        param_name: Name of parameter for error messages

    Raises:
        TypeError: If model_id is not a string
        ValueError: If model_id is empty or invalid format
    """
    if model_id is None:
        raise ValueError(f"{param_name} cannot be None")

    if not isinstance(model_id, str):
        raise TypeError(f"{param_name} must be a string, got {type(model_id).__name__}")

    if not model_id.strip():
        raise ValueError(f"{param_name} cannot be empty or whitespace")

    # Model IDs should not contain certain characters
    invalid_chars = ["\n", "\r", "\t", "\0"]
    for char in invalid_chars:
        if char in model_id:
            raise ValueError(f"{param_name} contains invalid character: {char!r}")


def validate_positive_number(
    value: float,
    param_name: str,
    min_value: float = 0.0,
    max_value: float | None = None,
) -> None:
    """
    Validate numeric parameter is positive and within bounds.

    Args:
        value: Number to validate
        param_name: Name of parameter for error messages
        min_value: Minimum allowed value (default: 0.0)
        max_value: Maximum allowed value (default: None = no limit)

    Raises:
        TypeError: If value is not a number
        ValueError: If value is out of bounds
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{param_name} must be a number, got {type(value).__name__}")

    if value < min_value:
        raise ValueError(f"{param_name} must be >= {min_value}, got {value}")

    if max_value is not None and value > max_value:
        raise ValueError(f"{param_name} must be <= {max_value}, got {value}")


def validate_ttl(
    value: int,
    min_val: int = 1,
    max_val: int = 604800,
    param_name: str = "ttl",
) -> int:
    """
    Validate cache TTL (Time To Live) parameters.

    Args:
        value: TTL value to validate (in seconds)
        min_val: Minimum allowed value (default: 1 second)
        max_val: Maximum allowed value (default: 604800 = 7 days)
        param_name: Name of parameter for error messages

    Returns:
        The validated TTL value

    Raises:
        TypeError: If value is not an integer
        ValueError: If value is out of bounds
    """
    if not isinstance(value, int):
        raise TypeError(f"{param_name} must be an integer, got {type(value).__name__}")

    if value < min_val or value > max_val:
        raise ValueError(f"{param_name} must be between {min_val} and {max_val}, got {value}")

    return value


def validate_priority(
    priority: int,
    min_val: int = 0,
    max_val: int = 10,
) -> int:
    """
    Validate request priority values.

    Args:
        priority: Priority value to validate
        min_val: Minimum allowed priority (default: 0)
        max_val: Maximum allowed priority (default: 10)

    Returns:
        The validated priority value

    Raises:
        TypeError: If priority is not an integer
        ValueError: If priority is out of bounds
    """
    if not isinstance(priority, int):
        raise TypeError(f"priority must be an integer, got {type(priority).__name__}")

    if priority < min_val or priority > max_val:
        raise ValueError(f"priority must be between {min_val} and {max_val}, got {priority}")

    return priority


def validate_collection_size(
    size: int,
    min_val: int = 1,
    max_val: int = 1000000,
    param_name: str = "size",
) -> int:
    """
    Validate collection size limits.

    Args:
        size: Collection size to validate
        min_val: Minimum allowed size (default: 1)
        max_val: Maximum allowed size (default: 1000000)
        param_name: Name of parameter for error messages

    Returns:
        The validated size value

    Raises:
        TypeError: If size is not an integer
        ValueError: If size is out of bounds
    """
    if not isinstance(size, int):
        raise TypeError(f"{param_name} must be an integer, got {type(size).__name__}")

    if size < min_val or size > max_val:
        raise ValueError(f"{param_name} must be between {min_val} and {max_val}, got {size}")

    return size


def validate_timeout(
    value: float,
    min_val: float = 0.1,
    max_val: float = 300.0,
    param_name: str = "timeout",
) -> float:
    """
    Validate timeout values.

    Args:
        value: Timeout value to validate (in seconds)
        min_val: Minimum allowed timeout (default: 0.1 seconds)
        max_val: Maximum allowed timeout (default: 300 seconds)
        param_name: Name of parameter for error messages

    Returns:
        The validated timeout value

    Raises:
        TypeError: If value is not a number
        ValueError: If value is out of bounds
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{param_name} must be a number, got {type(value).__name__}")

    if value < min_val or value > max_val:
        raise ValueError(f"{param_name} must be between {min_val} and {max_val}, got {value}")

    return float(value)


def validate_api_key(api_key: str, param_name: str = "api_key") -> None:
    """
    Validate API key format for Venice AI.

    Venice AI API keys typically follow the pattern:
    - Start with a prefix (e.g., 'vn_', 'sk_')
    - Followed by alphanumeric characters and underscores
    - Minimum length of 32 characters for security

    Args:
        api_key: API key to validate
        param_name: Name of parameter for error messages

    Raises:
        TypeError: If api_key is not a string
        ValueError: If api_key is empty or has invalid format
    """
    if not isinstance(api_key, str):
        raise TypeError(f"{param_name} must be a string, got {type(api_key).__name__}")

    if not api_key.strip():
        raise ValueError(f"{param_name} cannot be empty or whitespace")

    # Check minimum length for security
    if len(api_key) < 32:
        raise ValueError(
            f"{param_name} is too short ({len(api_key)} chars). "
            f"Valid API keys should be at least 32 characters"
        )

    # Check for valid characters (alphanumeric, dash, underscore)
    if not re.match(r"^[a-zA-Z0-9_-]+$", api_key):
        raise ValueError(
            f"{param_name} contains invalid characters. "
            f"API keys should only contain letters, numbers, hyphens, and underscores"
        )

    # Warn about common mistakes
    if api_key.startswith((" ", "\t")) or api_key.endswith((" ", "\t")):
        raise ValueError(
            f"{param_name} contains leading or trailing whitespace. Please remove any extra spaces"
        )


def validate_date_range(
    start_date: str,
    end_date: str,
    max_range_days: int | None = None,
    start_param: str = "start_date",
    end_param: str = "end_date",
) -> None:
    """
    Validate date range parameters.

    Args:
        start_date: Start date in ISO 8601 format
        end_date: End date in ISO 8601 format
        max_range_days: Maximum allowed range in days (optional, no limit if None)
        start_param: Name of start parameter for error messages
        end_param: Name of end parameter for error messages

    Raises:
        ValueError: If dates are invalid or range exceeds maximum
    """
    if not isinstance(start_date, str) or not start_date.strip():
        raise ValueError(f"{start_param} must be a non-empty string")

    if not isinstance(end_date, str) or not end_date.strip():
        raise ValueError(f"{end_param} must be a non-empty string")

    try:
        # Try to parse dates - supports multiple ISO 8601 formats
        start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError(
            f"Invalid date format. Both {start_param} and {end_param} must be in "
            f"ISO 8601 format (e.g., '2025-01-01' or '2025-01-01T00:00:00Z'). Error: {e}"
        ) from e

    # Validate that end date is not before start date (allow equal dates for single-day queries)
    if end_dt < start_dt:
        raise ValueError(f"{end_param} ({end_date}) cannot be before {start_param} ({start_date})")

    # Validate maximum range (only if max_range_days is specified)
    if max_range_days is not None:
        date_range = end_dt - start_dt
        if date_range.days > max_range_days:
            raise ValueError(
                f"Date range ({date_range.days} days) exceeds maximum allowed "
                f"({max_range_days} days). Please use a smaller date range"
            )


def validate_text_length(
    text: str,
    min_length: int = 1,
    max_length: int = 5000,
    param_name: str = "text",
) -> None:
    """
    Validate text length constraints.

    Args:
        text: Text to validate
        min_length: Minimum allowed length (default: 1)
        max_length: Maximum allowed length (default: 5000)
        param_name: Name of parameter for error messages

    Raises:
        TypeError: If text is not a string
        ValueError: If text length is out of bounds
    """
    if not isinstance(text, str):
        raise TypeError(f"{param_name} must be a string, got {type(text).__name__}")

    text_len = len(text)

    if text_len < min_length:
        # Provide backward-compatible error message for empty text
        if text_len == 0:
            raise ValueError("Input text cannot be empty for speech generation")
        raise ValueError(
            f"{param_name} is too short ({text_len} chars). "
            f"Minimum length is {min_length} characters"
        )

    if text_len > max_length:
        raise ValueError(
            f"Text length ({text_len} chars) exceeds maximum allowed "
            f"({max_length} chars). Please split into smaller chunks."
        )


def validate_cache_size(
    size: int,
    min_val: int = 1,
    max_val: int = 10000,
    param_name: str = "cache_size",
) -> int:
    """
    Validate cache size parameters.

    Args:
        size: Cache size to validate
        min_val: Minimum allowed size (default: 1)
        max_val: Maximum allowed size (default: 10000)
        param_name: Name of parameter for error messages

    Returns:
        The validated cache size

    Raises:
        TypeError: If size is not an integer
        ValueError: If size is out of bounds
    """
    if not isinstance(size, int):
        raise TypeError(f"{param_name} must be an integer, got {type(size).__name__}")

    if size < min_val or size > max_val:
        raise ValueError(f"{param_name} must be between {min_val} and {max_val}, got {size}")

    return size


def validate_interval(
    value: float,
    min_val: float = 0.1,
    max_val: float = 86400.0,
    param_name: str = "interval",
) -> float:
    """
    Validate interval values.

    Args:
        value: Interval value to validate (in seconds)
        min_val: Minimum allowed interval (default: 0.1 seconds)
        max_val: Maximum allowed interval (default: 86400 seconds = 1 day)
        param_name: Name of parameter for error messages

    Returns:
        The validated interval value

    Raises:
        TypeError: If value is not a number
        ValueError: If value is out of bounds
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{param_name} must be a number, got {type(value).__name__}")

    if value < min_val or value > max_val:
        raise ValueError(f"{param_name} must be between {min_val} and {max_val}, got {value}")

    return float(value)


def validate_percentage(
    value: float,
    min_val: float = 0.0,
    max_val: float = 100.0,
    param_name: str = "percentage",
) -> float:
    """
    Validate percentage values.

    Args:
        value: Percentage value to validate
        min_val: Minimum allowed percentage (default: 0.0)
        max_val: Maximum allowed percentage (default: 100.0)
        param_name: Name of parameter for error messages

    Returns:
        The validated percentage value

    Raises:
        TypeError: If value is not a number
        ValueError: If value is out of bounds
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{param_name} must be a number, got {type(value).__name__}")

    if value < min_val or value > max_val:
        raise ValueError(f"{param_name} must be between {min_val} and {max_val}, got {value}")

    return float(value)
