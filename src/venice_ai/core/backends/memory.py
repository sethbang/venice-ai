"""
Simplified MemoryBackend for VeniceAccount operations.

This in-memory backend provides basic rate limit tracking and failure management
for single-process applications. For distributed deployments, use Redis backend.
"""

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any

from .base import AccountBackend, HealthCheckResult

logger = logging.getLogger(__name__)


class MemoryBackend(AccountBackend):
    """
    Simplified in-memory backend for VeniceAccount.

    Suitable for:
    - Testing and development
    - Single-process applications
    - Scenarios where distributed state is not needed

    Not suitable for:
    - Multi-process applications
    - Distributed systems
    - Production with high availability requirements
    """

    def __init__(self, namespace: str = "venice_ai") -> None:
        """
        Initialize the in-memory backend.

        Args:
            namespace: Namespace for key isolation
        """
        super().__init__(namespace)

        # Failure tracking
        self._failures: list[tuple[float, str, str]] = []  # (timestamp, type, message)
        self._circuit_broken_until: float | None = None

        # Rate limit state (from response headers)
        self._rate_limits: dict[str, dict[str, Any]] = {}

        # Request tracking
        self._request_counts: dict[str, int] = defaultdict(int)
        self._token_counts: dict[str, int] = defaultdict(int)

        # Async lock for thread safety
        self._lock = asyncio.Lock()

        logger.debug(f"Initialized MemoryBackend with namespace '{namespace}'")

    # === Health and Monitoring ===

    async def health_check(self) -> HealthCheckResult:
        """Perform a health check on the backend."""
        async with self._lock:
            return HealthCheckResult(
                healthy=True,
                backend_type="memory",
                namespace=self.namespace,
                metadata={
                    "rate_limits_count": len(self._rate_limits),
                    "failures_count": len(self._failures),
                    "circuit_broken": self._circuit_broken_until is not None
                    and time.time() < self._circuit_broken_until,
                },
            )

    async def get_all_stats(self) -> dict[str, Any]:
        """Get all statistics from the backend."""
        async with self._lock:
            cutoff = time.time() - 30
            recent_failures = sum(1 for ts, _, _ in self._failures if ts > cutoff)

            return {
                "rate_limits_count": len(self._rate_limits),
                "total_failures": len(self._failures),
                "recent_failures_30s": recent_failures,
                "circuit_broken": self._circuit_broken_until is not None
                and time.time() < self._circuit_broken_until,
                "request_counts": dict(self._request_counts),
                "token_counts": dict(self._token_counts),
            }

    async def cleanup(self) -> None:
        """Clean up all backend resources."""
        async with self._lock:
            self._failures.clear()
            self._rate_limits.clear()
            self._request_counts.clear()
            self._token_counts.clear()
            self._circuit_broken_until = None
            logger.debug("MemoryBackend cleanup completed")

    # === Rate Limit Management ===

    async def check_capacity(self, model: str, request_type: str = "default") -> tuple[bool, float]:
        """
        Check if there's capacity for a request.

        For memory backend, we use cached rate limits from headers for guidance
        but don't enforce strict limits (SimpleRateLimiter handles that).

        Args:
            model: Model identifier
            request_type: Type of request (ignored in memory backend)

        Returns:
            Tuple of (can_proceed, wait_seconds)
        """
        del request_type  # interface-only; memory backend ignores request_type
        async with self._lock:
            if model in self._rate_limits:
                limits = self._rate_limits[model]
                rpm_remaining = limits.get("rpm_remaining", 1)
                if rpm_remaining <= 0:
                    wait_time = self._calculate_wait_time(limits)
                    if wait_time > 0:
                        return False, wait_time
            return True, 0.0

    async def update_rate_limits(self, model: str, headers: dict[str, str]) -> None:
        """
        Update rate limits from response headers.

        Args:
            model: Model identifier
            headers: Response headers containing rate limit info
        """
        async with self._lock:
            parsed = self._parse_rate_limit_headers(headers)
            self._rate_limits[model] = parsed
            logger.debug(f"Updated rate limits for {model}: {parsed}")

    async def record_request(self, model: str, tokens_used: int | None = None) -> None:
        """
        Record a request for tracking.

        Args:
            model: Model identifier
            tokens_used: Optional number of tokens used
        """
        async with self._lock:
            self._request_counts[model] += 1
            if tokens_used is not None:
                self._token_counts[model] += tokens_used

    async def release_streaming_reservation(
        self,
        bucket_id: str,
        reservation_id: str,
        reserved_tokens: int,
        actual_tokens: int,
    ) -> bool:
        """Release streaming reservation with refund-based accounting.

        For MemoryBackend, this is largely a no-op since the simplified
        backend doesn't track individual reservations. Returns True to
        indicate success for protocol compatibility.
        """
        # MemoryBackend doesn't track individual reservations
        # This is a no-op implementation for protocol compatibility
        logger.debug(
            f"release_streaming_reservation: bucket={bucket_id}, "
            f"reservation={reservation_id}, refund={reserved_tokens - actual_tokens}"
        )
        return True

    # === Failure Tracking and Circuit Breaking ===

    async def record_failure(self, error_type: str, error_message: str = "") -> None:
        """
        Record a failure for tracking.

        Args:
            error_type: Type of error (e.g., "rate_limit", "server_error")
            error_message: Optional error message
        """
        async with self._lock:
            self._failures.append((time.time(), error_type, error_message))
            logger.debug(f"Recorded failure: {error_type} - {error_message}")

    async def get_failure_count(self, window_seconds: int = 30) -> int:
        """
        Get the number of failures within the specified window.

        Args:
            window_seconds: Time window in seconds (default: 30)

        Returns:
            Number of failures in the window
        """
        async with self._lock:
            cutoff = time.time() - window_seconds
            return sum(1 for ts, _, _ in self._failures if ts > cutoff)

    async def is_circuit_broken(self) -> bool:
        """
        Check if the circuit breaker is triggered.

        Returns:
            True if circuit is broken, False otherwise
        """
        async with self._lock:
            if self._circuit_broken_until is None:
                return False
            if time.time() < self._circuit_broken_until:
                return True
            # Circuit breaker expired, clear it
            self._circuit_broken_until = None
            return False

    async def clear_failures(self) -> None:
        """Clear all failure records."""
        async with self._lock:
            self._failures.clear()
            logger.debug("Cleared all failure records")

    async def force_circuit_break(self, duration: float) -> None:
        """
        Force a circuit break for the specified duration.

        Args:
            duration: Duration in seconds to keep circuit broken
        """
        async with self._lock:
            self._circuit_broken_until = time.time() + duration
            logger.debug(f"Forced circuit break for {duration} seconds")
