"""Rate limiter configuration."""

from dataclasses import dataclass
from enum import StrEnum


class RateLimiterMode(StrEnum):
    """Rate limiter mode selection."""

    SIMPLE = "simple"  # Default, in-memory, reactive
    ADAPTIVE = "adaptive"  # Requires adaptive-rate-limiter package, proactive
    DISABLED = "disabled"  # No rate limiting (testing only, NOT recommended for production)


@dataclass
class RateLimiterConfig:
    """
    Configuration for rate limiting behavior.

    For SimpleRateLimiter (mode=SIMPLE):
        - Reactive rate limiting (responds to 429s with backoff)
        - Single-process only
        - No Redis required

    For AdaptiveScheduler (mode=ADAPTIVE):
        - Proactive rate limiting (prevents 429s)
        - Multi-process/worker coordination via Redis
        - Token prediction and cold-start protection
        - Requires: pip install 'venice-py[adaptive]'
        - Requires: redis_url configuration
    """

    mode: RateLimiterMode = RateLimiterMode.SIMPLE

    # SimpleRateLimiter configuration
    min_backoff: float = 1.0
    max_backoff: float = 60.0
    failure_threshold: int = 20
    failure_window: float = 30.0
    block_duration: float = 30.0
    max_models: int = 1000
    stale_threshold: float = 3600.0
    max_retries: int = 3

    # AdaptiveScheduler configuration (requires adaptive package)
    redis_url: str | None = None
    account_id: str | None = None  # Required for key scoping in adaptive mode
