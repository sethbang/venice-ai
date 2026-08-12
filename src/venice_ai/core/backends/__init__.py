"""
Backend Architecture for Venice AI

This package provides backend implementations for Venice AI including
Redis-based and in-memory backends.

Backend Options:
- MemoryBackend: In-memory backend for single-instance deployments
- RedisBackend: Redis backend for distributed deployments

For advanced distributed features (proactive rate limiting, Redis coordination),
use the adaptive-rate-limiter package.
"""

from .base import AccountBackend, HealthCheckResult
from .memory import MemoryBackend

__all__ = [
    # Base classes
    "AccountBackend",
    "HealthCheckResult",
    # Implementations
    "MemoryBackend",
]

# RedisBackend is only available when the 'redis' package is installed.
try:
    from .redis import RedisBackend as RedisBackend

    __all__.append("RedisBackend")
except ImportError:
    pass  # redis not installed; RedisBackend unavailable
