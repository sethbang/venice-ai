"""
Unified Configuration System for Venice AI.

Core configs (``HttpClientConfig``, ``VeniceAIConfig``) are required for basic SDK
operation.  Enterprise configs (``BackendConfig``, ``StateConfig``, ``SchedulerConfig``,
etc.) are optional and only needed for advanced deployments.

Quick start::

    from venice_ai.core.config import create_minimal_config
    cfg = create_minimal_config(api_key="your-key")

Environment variables use the ``VENICE_`` prefix with ``__`` as the nested delimiter.

This package re-exports every public symbol for full backward compatibility.
All existing ``from venice_ai.core.config import X`` statements continue to work.
"""

# Re-export rate limiter types for convenience
from ...rate_limiting.config import RateLimiterConfig, RateLimiterMode
from .backends import BackendConfig, MemoryBackendConfig, RedisBackendConfig
from .enterprise import CircuitBreakerConfig, MetricsConfig, SchedulerConfig, StateConfig
from .enums import BackendType, CachePolicy, SchedulerMode
from .http import HttpClientConfig
from .main import VeniceAIConfig, create_minimal_config

__all__ = [
    # Root configuration
    "VeniceAIConfig",
    # Module-level helpers
    "create_minimal_config",
    # Core component configs
    "HttpClientConfig",
    # Enterprise component configs
    "SchedulerConfig",
    "StateConfig",
    "BackendConfig",
    "RedisBackendConfig",
    "MemoryBackendConfig",
    "CircuitBreakerConfig",
    "MetricsConfig",
    # Rate limiter (re-exported for convenience)
    "RateLimiterConfig",
    # Enums
    "CachePolicy",
    "SchedulerMode",
    "BackendType",
    "RateLimiterMode",
]
