"""
Simplified backend for VeniceAccount operations.

This backend supports:
- Failure tracking and circuit breaking
- Basic rate limit state from response headers
- Health monitoring

For advanced rate limiting with Redis coordination,
use the adaptive-rate-limiter package.
"""

import abc
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Structured health check result for backend monitoring."""

    healthy: bool
    backend_type: str
    namespace: str
    error: str | None = None
    metadata: dict[str, Any] | None = None


class AccountBackend(abc.ABC):
    """
    Simplified backend interface for VeniceAccount.

    This interface provides the 11 methods required by VeniceAccount
    and FailureTracker for basic operation.
    """

    def __init__(self, namespace: str = "venice_ai"):
        self.namespace = namespace

    # === Health and Monitoring ===

    @abc.abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """Perform health check."""
        pass

    @abc.abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources."""
        pass

    @abc.abstractmethod
    async def get_all_stats(self) -> dict[str, Any]:
        """Get all statistics."""
        pass

    # === Rate Limit Management ===

    @abc.abstractmethod
    async def check_capacity(self, model: str, request_type: str = "default") -> tuple[bool, float]:
        """Check if capacity available. Returns (can_proceed, wait_seconds)."""
        del request_type  # reserved for subclasses; silences vulture on abstract body
        raise NotImplementedError

    @abc.abstractmethod
    async def update_rate_limits(self, model: str, headers: dict[str, str]) -> None:
        """Update rate limits from response headers."""
        pass

    @abc.abstractmethod
    async def record_request(self, model: str, tokens_used: int | None = None) -> None:
        """Record a request."""
        pass

    @abc.abstractmethod
    async def release_streaming_reservation(
        self,
        bucket_id: str,
        reservation_id: str,
        reserved_tokens: int,
        actual_tokens: int,
    ) -> bool:
        """
        Release streaming reservation with refund-based accounting.

        This method is called when a streaming response completes. It uses
        refund-based accounting: refund = reserved_tokens - actual_tokens.

        For non-streaming workloads or simplified backends, a no-op
        implementation that returns True is acceptable.

        Args:
            bucket_id: The rate limit bucket identifier
            reservation_id: The reservation identifier
            reserved_tokens: Tokens that were reserved at request start
            actual_tokens: Actual tokens consumed by the stream

        Returns:
            True if release succeeded (or was no-op), False on error

        Note:
            This method was added for compatibility with the streaming
            infrastructure. Custom AccountBackend subclasses must implement
            this method.
        """
        pass

    # === Failure Tracking and Circuit Breaking ===

    @abc.abstractmethod
    async def record_failure(self, error_type: str, error_message: str = "") -> None:
        """Record a failure."""
        pass

    @abc.abstractmethod
    async def get_failure_count(self, window_seconds: int = 30) -> int:
        """Get failure count within window."""
        pass

    @abc.abstractmethod
    async def is_circuit_broken(self) -> bool:
        """Check if circuit breaker is triggered."""
        pass

    @abc.abstractmethod
    async def clear_failures(self) -> None:
        """Clear failure records."""
        pass

    @abc.abstractmethod
    async def force_circuit_break(self, duration: float) -> None:
        """Force circuit break for duration."""
        pass

    # === Helper Methods ===

    @staticmethod
    def _coerce_int_header(value: str) -> int | None:
        """Coerce a rate-limit header value to ``int``, tolerating floats.

        Venice's ``x-ratelimit-reset-*`` headers can arrive as fractional /
        epoch-millis timestamps (e.g. ``"1750000000.123"``); a bare ``int()``
        would raise and discard the whole header set. Parse via
        ``int(float(...))`` and return ``None`` for non-numeric values so a
        single malformed header is skipped rather than fatal.
        """
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _parse_rate_limit_headers(self, headers: dict[str, str]) -> dict[str, Any]:
        """Parse rate limit information from response headers.

        Numeric values are coerced defensively (see :meth:`_coerce_int_header`):
        fractional reset timestamps parse cleanly, and a malformed value skips
        only that field instead of dropping the entire header set.
        """
        result: dict[str, Any] = {}

        field_map = {
            "x-ratelimit-limit-requests": "rpm_limit",
            "x-ratelimit-remaining-requests": "rpm_remaining",
            "x-ratelimit-reset-requests": "rpm_reset",
            "x-ratelimit-limit-tokens": "tpm_limit",
            "x-ratelimit-remaining-tokens": "tpm_remaining",
            "x-ratelimit-reset-tokens": "tpm_reset",
            "retry-after": "retry_after",
        }
        for header, key in field_map.items():
            if header in headers:
                parsed = self._coerce_int_header(headers[header])
                if parsed is not None:
                    result[key] = parsed

        result["timestamp"] = int(time.time())
        return result

    def _calculate_wait_time(self, rate_limits: dict[str, Any]) -> float:
        """Calculate wait time based on rate limit state."""
        if not rate_limits:
            return 0.0

        wait_time = 0.0
        current_time = time.time()

        rpm_remaining = rate_limits.get("rpm_remaining", 1)
        if rpm_remaining <= 0:
            rpm_reset = rate_limits.get("rpm_reset")
            if rpm_reset:
                reset_time = self._parse_reset_time(str(rpm_reset))
                wait_time = max(wait_time, reset_time - current_time)

        tpm_remaining = rate_limits.get("tpm_remaining", 1)
        if tpm_remaining <= 0:
            tpm_reset = rate_limits.get("tpm_reset")
            if tpm_reset:
                reset_time = self._parse_reset_time(str(tpm_reset))
                wait_time = max(wait_time, reset_time - current_time)

        retry_after = rate_limits.get("retry_after", 0)
        if retry_after > 0:
            wait_time = max(wait_time, retry_after)

        return max(0.0, wait_time)

    def _parse_reset_time(self, reset_str: str) -> float:
        """Parse reset time from header string."""
        try:
            val = float(reset_str)
            if val > 1e11:
                return val / 1000.0
            if val > 1e9:
                return val
            return time.time() + val
        except ValueError:
            return time.time() + 60.0
