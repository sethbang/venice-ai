"""Authentication utilities for Venice AI API."""

from __future__ import annotations


def create_auth_headers(api_key: str) -> dict[str, str]:
    """Create authentication headers from an API key."""
    return {"Authorization": f"Bearer {api_key.strip()}"}


def validate_api_key_format(api_key: str) -> bool:
    """Basic validation of API key format.

    Venice API keys are non-empty strings. This function performs
    basic format validation including a minimum length check.
    """
    if not api_key or not isinstance(api_key, str):
        return False
    api_key = api_key.strip()
    return len(api_key) > 10  # Basic length check
