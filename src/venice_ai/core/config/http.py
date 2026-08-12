"""
HTTP client configuration for Venice AI.

Core feature: Used by all SDK clients for every API request.
"""

from pydantic import BaseModel, ConfigDict, Field


def _default_user_agent() -> str:
    # Lazy import to avoid a circular import (venice_ai.__init__ imports the
    # client, which imports this config module).
    from venice_ai import __version__

    return f"VeniceAI-Python-SDK/{__version__}"


class HttpClientConfig(BaseModel):
    """Configuration for HTTP client operations.

    Core feature: Used by all SDK clients for every API request.
    """

    model_config = ConfigDict(extra="forbid")

    # Connection settings
    timeout: float = Field(default=30.0, gt=0, description="Default request timeout in seconds")

    max_connections: int = Field(default=100, ge=1, description="Maximum HTTP connections in pool")

    max_keepalive_connections: int = Field(
        default=20, ge=1, description="Maximum keepalive connections"
    )

    # Retry configuration
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")

    retry_backoff_factor: float = Field(
        default=2.0, ge=1.0, description="Backoff factor for retries"
    )

    # Headers
    user_agent: str = Field(default_factory=_default_user_agent, description="User agent string")


__all__ = [
    "HttpClientConfig",
]
