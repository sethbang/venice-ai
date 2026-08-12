"""
Production-ready configuration presets for Venice AI SDK.

This module provides pre-configured settings for different deployment environments,
making it easy to set up the SDK with battle-tested configurations.

Available Presets:
    - Production: Optimized for production deployments with Redis backend
    - Development: Fast iteration with memory backend
    - Testing: Optimized for test environments with relaxed rate limits

Example:
    >>> from venice_ai.presets import create_production_config
    >>> config = create_production_config(redis_url="redis://localhost:6379")
    >>> client = VeniceClient(config=config, api_key=api_key)
"""

from .development import create_development_config
from .production import create_production_config
from .testing import create_testing_config

__all__ = [
    "create_production_config",
    "create_development_config",
    "create_testing_config",
]
