"""
Enums for the Venice AI configuration system.

These enums have no internal dependencies and are imported by all other
config submodules.
"""

from enum import Enum


class CachePolicy(Enum):
    """Cache write policies for state management.

    - **WRITE_THROUGH**: Durable dual writes — recommended for production.
    - **WRITE_BACK**: Fast batched writes, data-loss risk — dev/test only.
    - **WRITE_AROUND**: Bypass cache, write directly to backend — bulk writes.
    """

    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"


class SchedulerMode(Enum):
    """Scheduler operation modes."""

    BASIC = "basic"
    INTELLIGENT = "intelligent"
    ACCOUNT = "account"


class BackendType(Enum):
    """Supported backend types."""

    REDIS = "redis"
    MEMORY = "memory"  # For testing


__all__ = [
    "CachePolicy",
    "SchedulerMode",
    "BackendType",
]
