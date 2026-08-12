"""
Comprehensive test coverage for MemoryBackend.

This test module targets specific missing lines and branches to achieve 85%+ coverage.
It focuses on:
- Lines 63-64: health_check method
- Lines 78-82: get_all_stats method with recent failures
- Lines 94-100: cleanup method
- Lines 120-128: check_capacity with rate limits and wait time
- Lines 138-141: update_rate_limits
- Lines 153-156: record_request with tokens
- Lines 168-170: record_failure
- Lines 182-184: get_failure_count
- Lines 193-200: is_circuit_broken with expiry handling
- Lines 204-206: clear_failures
- Lines 215-217: force_circuit_break
- All 12 branches for complete branch coverage
"""

import asyncio
import time

import pytest

from venice_ai.core.backends.memory import MemoryBackend


class TestMemoryBackendInitialization:
    """Test MemoryBackend initialization."""

    def test_init_default_namespace(self):
        """Test initialization with default namespace."""
        backend = MemoryBackend()
        assert backend.namespace == "venice_ai"
        assert backend._failures == []
        assert backend._circuit_broken_until is None
        assert backend._rate_limits == {}

    def test_init_custom_namespace(self):
        """Test initialization with custom namespace."""
        backend = MemoryBackend(namespace="custom_namespace")
        assert backend.namespace == "custom_namespace"


class TestHealthCheck:
    """Test health_check method (lines 63-64)."""

    @pytest.mark.asyncio
    async def test_health_check_basic(self):
        """Test basic health check returns healthy result (lines 63-64)."""
        backend = MemoryBackend(namespace="test_health")

        result = await backend.health_check()

        # Verify health check result (covers lines 63-64)
        assert result.healthy is True
        assert result.backend_type == "memory"
        assert result.namespace == "test_health"
        assert result.metadata is not None
        assert "rate_limits_count" in result.metadata
        assert "failures_count" in result.metadata
        assert "circuit_broken" in result.metadata

    @pytest.mark.asyncio
    async def test_health_check_with_rate_limits(self):
        """Test health check with rate limits set."""
        backend = MemoryBackend(namespace="test_health_limits")
        backend._rate_limits = {"model1": {"rpm_remaining": 100}}

        result = await backend.health_check()

        assert result.metadata is not None
        assert result.metadata["rate_limits_count"] == 1
        assert result.metadata["failures_count"] == 0

    @pytest.mark.asyncio
    async def test_health_check_with_failures(self):
        """Test health check with recorded failures."""
        backend = MemoryBackend(namespace="test_health_failures")
        backend._failures = [(time.time(), "error", "test error")]

        result = await backend.health_check()

        assert result.metadata is not None
        assert result.metadata["failures_count"] == 1

    @pytest.mark.asyncio
    async def test_health_check_with_circuit_broken(self):
        """Test health check when circuit is broken."""
        backend = MemoryBackend(namespace="test_health_circuit")
        backend._circuit_broken_until = time.time() + 60  # Broken for 60 seconds

        result = await backend.health_check()

        assert result.metadata is not None
        assert result.metadata["circuit_broken"] is True

    @pytest.mark.asyncio
    async def test_health_check_with_circuit_expired(self):
        """Test health check when circuit breaker has expired."""
        backend = MemoryBackend(namespace="test_health_expired")
        backend._circuit_broken_until = time.time() - 10  # Expired 10 seconds ago

        result = await backend.health_check()

        assert result.metadata is not None
        assert result.metadata["circuit_broken"] is False


class TestGetAllStats:
    """Test get_all_stats method (lines 78-82)."""

    @pytest.mark.asyncio
    async def test_get_all_stats_empty(self):
        """Test get_all_stats with no data."""
        backend = MemoryBackend(namespace="test_stats")

        result = await backend.get_all_stats()

        # Verify stats structure (covers lines 78-82)
        assert result["rate_limits_count"] == 0
        assert result["total_failures"] == 0
        assert result["recent_failures_30s"] == 0
        assert result["circuit_broken"] is False
        assert result["request_counts"] == {}
        assert result["token_counts"] == {}

    @pytest.mark.asyncio
    async def test_get_all_stats_with_recent_failures(self):
        """Test get_all_stats counts recent failures correctly (line 80)."""
        backend = MemoryBackend(namespace="test_stats_recent")
        current_time = time.time()
        # Add failures - some within 30s window, some outside
        backend._failures = [
            (current_time - 10, "error", "recent 1"),  # Within 30s
            (current_time - 25, "error", "recent 2"),  # Within 30s
            (current_time - 35, "error", "old 1"),  # Outside 30s
            (current_time - 60, "error", "old 2"),  # Outside 30s
        ]

        result = await backend.get_all_stats()

        # Verify recent failures count (covers line 80)
        assert result["total_failures"] == 4
        assert result["recent_failures_30s"] == 2

    @pytest.mark.asyncio
    async def test_get_all_stats_no_recent_failures(self):
        """Test get_all_stats when all failures are old."""
        backend = MemoryBackend(namespace="test_stats_old")
        current_time = time.time()
        backend._failures = [
            (current_time - 60, "error", "old 1"),
            (current_time - 120, "error", "old 2"),
        ]

        result = await backend.get_all_stats()

        assert result["total_failures"] == 2
        assert result["recent_failures_30s"] == 0

    @pytest.mark.asyncio
    async def test_get_all_stats_with_rate_limits(self):
        """Test get_all_stats with rate limits."""
        backend = MemoryBackend(namespace="test_stats_limits")
        backend._rate_limits = {
            "model1": {"rpm_remaining": 100},
            "model2": {"rpm_remaining": 50},
        }

        result = await backend.get_all_stats()

        assert result["rate_limits_count"] == 2

    @pytest.mark.asyncio
    async def test_get_all_stats_with_request_counts(self):
        """Test get_all_stats with request and token counts."""
        backend = MemoryBackend(namespace="test_stats_counts")
        backend._request_counts["model1"] = 10
        backend._token_counts["model1"] = 1000

        result = await backend.get_all_stats()

        assert result["request_counts"] == {"model1": 10}
        assert result["token_counts"] == {"model1": 1000}

    @pytest.mark.asyncio
    async def test_get_all_stats_circuit_broken_active(self):
        """Test get_all_stats with active circuit breaker."""
        backend = MemoryBackend(namespace="test_stats_circuit")
        backend._circuit_broken_until = time.time() + 60

        result = await backend.get_all_stats()

        assert result["circuit_broken"] is True

    @pytest.mark.asyncio
    async def test_get_all_stats_circuit_broken_expired(self):
        """Test get_all_stats when circuit breaker expired."""
        backend = MemoryBackend(namespace="test_stats_expired_circuit")
        backend._circuit_broken_until = time.time() - 10

        result = await backend.get_all_stats()

        assert result["circuit_broken"] is False


class TestCleanup:
    """Test cleanup method (lines 94-100)."""

    @pytest.mark.asyncio
    async def test_cleanup_clears_all_data(self):
        """Test cleanup clears all data structures (lines 94-100)."""
        backend = MemoryBackend(namespace="test_cleanup")

        # Populate all data structures
        backend._failures = [(time.time(), "error", "test")]
        backend._rate_limits = {"model1": {"rpm_remaining": 100}}
        backend._request_counts["model1"] = 5
        backend._token_counts["model1"] = 500
        backend._circuit_broken_until = time.time() + 60

        await backend.cleanup()

        # Verify all data cleared (covers lines 94-100)
        assert backend._failures == []
        assert backend._rate_limits == {}
        assert dict(backend._request_counts) == {}
        assert dict(backend._token_counts) == {}
        assert backend._circuit_broken_until is None

    @pytest.mark.asyncio
    async def test_cleanup_empty_backend(self):
        """Test cleanup on already empty backend."""
        backend = MemoryBackend(namespace="test_cleanup_empty")

        # Should not raise on empty backend
        await backend.cleanup()

        assert backend._failures == []
        assert backend._rate_limits == {}


class TestCheckCapacity:
    """Test check_capacity method (lines 120-128)."""

    @pytest.mark.asyncio
    async def test_check_capacity_no_rate_limits(self):
        """Test check_capacity when no rate limits exist (line 121 false branch)."""
        backend = MemoryBackend(namespace="test_capacity")

        can_proceed, wait_time = await backend.check_capacity("unknown_model")

        # Should proceed when no rate limits (covers line 128)
        assert can_proceed is True
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_check_capacity_with_remaining(self):
        """Test check_capacity when rpm_remaining > 0 (line 124 false branch)."""
        backend = MemoryBackend(namespace="test_capacity_remaining")
        backend._rate_limits["test_model"] = {"rpm_remaining": 50}

        can_proceed, wait_time = await backend.check_capacity("test_model")

        # Should proceed when remaining > 0
        assert can_proceed is True
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_check_capacity_no_remaining_no_wait(self):
        """Test check_capacity when rpm_remaining = 0 but no wait time (lines 124-128)."""
        backend = MemoryBackend(namespace="test_capacity_zero")
        # Set up rate limits with 0 remaining but no reset info that would trigger wait
        backend._rate_limits["test_model"] = {"rpm_remaining": 0}

        can_proceed, wait_time = await backend.check_capacity("test_model")

        # No wait time calculated, should proceed
        assert can_proceed is True
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_check_capacity_with_wait_time(self):
        """Test check_capacity returns wait time when capacity exhausted (lines 125-127)."""
        backend = MemoryBackend(namespace="test_capacity_wait")
        # Set up rate limits to trigger wait time
        backend._rate_limits["test_model"] = {
            "rpm_remaining": 0,
            "rpm_reset": time.time() + 30,  # Reset in 30 seconds
        }

        can_proceed, wait_time = await backend.check_capacity("test_model")

        # Should return wait time (covers lines 125-127)
        assert can_proceed is False
        assert wait_time > 0

    @pytest.mark.asyncio
    async def test_check_capacity_with_retry_after(self):
        """Test check_capacity with retry_after header."""
        backend = MemoryBackend(namespace="test_capacity_retry")
        backend._rate_limits["test_model"] = {
            "rpm_remaining": 0,
            "retry_after": 15,
        }

        can_proceed, wait_time = await backend.check_capacity("test_model")

        assert can_proceed is False
        assert wait_time >= 15

    @pytest.mark.asyncio
    async def test_check_capacity_ignores_request_type(self):
        """Test check_capacity ignores request_type parameter."""
        backend = MemoryBackend(namespace="test_capacity_type")

        # Various request types should behave the same
        for request_type in ["default", "streaming", "batch"]:
            can_proceed, wait_time = await backend.check_capacity(
                "test_model", request_type=request_type
            )
            assert can_proceed is True


class TestUpdateRateLimits:
    """Test update_rate_limits method (lines 138-141)."""

    @pytest.mark.asyncio
    async def test_update_rate_limits_basic(self):
        """Test update_rate_limits stores parsed headers (lines 138-141)."""
        backend = MemoryBackend(namespace="test_update")

        headers = {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "50",
        }

        await backend.update_rate_limits("test_model", headers)

        # Verify rate limits stored (covers lines 138-141)
        assert "test_model" in backend._rate_limits
        assert backend._rate_limits["test_model"]["rpm_limit"] == 100
        assert backend._rate_limits["test_model"]["rpm_remaining"] == 50

    @pytest.mark.asyncio
    async def test_update_rate_limits_full_headers(self):
        """Test update_rate_limits with all header types."""
        backend = MemoryBackend(namespace="test_update_full")

        headers = {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "50",
            "x-ratelimit-reset-requests": "60",
            "x-ratelimit-limit-tokens": "10000",
            "x-ratelimit-remaining-tokens": "5000",
            "x-ratelimit-reset-tokens": "120",
            "retry-after": "30",
        }

        await backend.update_rate_limits("test_model", headers)

        limits = backend._rate_limits["test_model"]
        assert limits["rpm_limit"] == 100
        assert limits["rpm_remaining"] == 50
        assert limits["rpm_reset"] == 60
        assert limits["tpm_limit"] == 10000
        assert limits["tpm_remaining"] == 5000
        assert limits["tpm_reset"] == 120
        assert limits["retry_after"] == 30

    @pytest.mark.asyncio
    async def test_update_rate_limits_empty_headers(self):
        """Test update_rate_limits with empty headers."""
        backend = MemoryBackend(namespace="test_update_empty")

        await backend.update_rate_limits("test_model", {})

        # Should still create entry with timestamp
        assert "test_model" in backend._rate_limits
        assert "timestamp" in backend._rate_limits["test_model"]

    @pytest.mark.asyncio
    async def test_update_rate_limits_overwrites_existing(self):
        """Test update_rate_limits overwrites existing model limits."""
        backend = MemoryBackend(namespace="test_update_overwrite")

        # Set initial limits
        backend._rate_limits["test_model"] = {"rpm_remaining": 100}

        # Update with new limits
        headers = {"x-ratelimit-remaining-requests": "25"}
        await backend.update_rate_limits("test_model", headers)

        assert backend._rate_limits["test_model"]["rpm_remaining"] == 25


class TestRecordRequest:
    """Test record_request method (lines 153-156)."""

    @pytest.mark.asyncio
    async def test_record_request_basic(self):
        """Test record_request increments count (line 154)."""
        backend = MemoryBackend(namespace="test_record")

        await backend.record_request("test_model")

        # Verify request count incremented (covers line 154)
        assert backend._request_counts["test_model"] == 1

    @pytest.mark.asyncio
    async def test_record_request_with_tokens(self):
        """Test record_request with token count (lines 155-156)."""
        backend = MemoryBackend(namespace="test_record_tokens")

        await backend.record_request("test_model", tokens_used=500)

        # Verify token count recorded (covers lines 155-156)
        assert backend._request_counts["test_model"] == 1
        assert backend._token_counts["test_model"] == 500

    @pytest.mark.asyncio
    async def test_record_request_without_tokens(self):
        """Test record_request without tokens (line 155 false branch)."""
        backend = MemoryBackend(namespace="test_record_no_tokens")

        await backend.record_request("test_model")

        assert backend._request_counts["test_model"] == 1
        assert backend._token_counts["test_model"] == 0

    @pytest.mark.asyncio
    async def test_record_request_multiple(self):
        """Test record_request accumulates correctly."""
        backend = MemoryBackend(namespace="test_record_multi")

        await backend.record_request("test_model", tokens_used=100)
        await backend.record_request("test_model", tokens_used=200)
        await backend.record_request("test_model")

        assert backend._request_counts["test_model"] == 3
        assert backend._token_counts["test_model"] == 300

    @pytest.mark.asyncio
    async def test_record_request_zero_tokens(self):
        """Test record_request with zero tokens."""
        backend = MemoryBackend(namespace="test_record_zero")

        await backend.record_request("test_model", tokens_used=0)

        assert backend._request_counts["test_model"] == 1
        assert backend._token_counts["test_model"] == 0


class TestRecordFailure:
    """Test record_failure method (lines 168-170)."""

    @pytest.mark.asyncio
    async def test_record_failure_basic(self):
        """Test record_failure stores failure (lines 168-170)."""
        backend = MemoryBackend(namespace="test_failure")

        await backend.record_failure("rate_limit", "Too many requests")

        # Verify failure recorded (covers lines 168-170)
        assert len(backend._failures) == 1
        ts, error_type, message = backend._failures[0]
        assert error_type == "rate_limit"
        assert message == "Too many requests"
        assert ts <= time.time()

    @pytest.mark.asyncio
    async def test_record_failure_empty_message(self):
        """Test record_failure with empty message."""
        backend = MemoryBackend(namespace="test_failure_empty")

        await backend.record_failure("server_error")

        assert len(backend._failures) == 1
        _, error_type, message = backend._failures[0]
        assert error_type == "server_error"
        assert message == ""

    @pytest.mark.asyncio
    async def test_record_failure_multiple(self):
        """Test record_failure accumulates failures."""
        backend = MemoryBackend(namespace="test_failure_multi")

        await backend.record_failure("error1", "msg1")
        await backend.record_failure("error2", "msg2")
        await backend.record_failure("error3", "msg3")

        assert len(backend._failures) == 3


class TestGetFailureCount:
    """Test get_failure_count method (lines 182-184)."""

    @pytest.mark.asyncio
    async def test_get_failure_count_empty(self):
        """Test get_failure_count with no failures."""
        backend = MemoryBackend(namespace="test_count")

        count = await backend.get_failure_count()

        assert count == 0

    @pytest.mark.asyncio
    async def test_get_failure_count_all_in_window(self):
        """Test get_failure_count when all failures in window (lines 182-184)."""
        backend = MemoryBackend(namespace="test_count_all")
        current_time = time.time()
        backend._failures = [
            (current_time - 5, "error", "msg1"),
            (current_time - 10, "error", "msg2"),
            (current_time - 15, "error", "msg3"),
        ]

        count = await backend.get_failure_count(window_seconds=30)

        # Verify count (covers lines 182-184)
        assert count == 3

    @pytest.mark.asyncio
    async def test_get_failure_count_mixed(self):
        """Test get_failure_count with mixed ages."""
        backend = MemoryBackend(namespace="test_count_mixed")
        current_time = time.time()
        backend._failures = [
            (current_time - 10, "error", "recent"),  # In 30s window
            (current_time - 25, "error", "recent"),  # In 30s window
            (current_time - 40, "error", "old"),  # Outside 30s window
            (current_time - 60, "error", "old"),  # Outside 30s window
        ]

        count = await backend.get_failure_count(window_seconds=30)

        assert count == 2

    @pytest.mark.asyncio
    async def test_get_failure_count_custom_window(self):
        """Test get_failure_count with custom window."""
        backend = MemoryBackend(namespace="test_count_custom")
        current_time = time.time()
        backend._failures = [
            (current_time - 5, "error", "msg1"),
            (current_time - 15, "error", "msg2"),
            (current_time - 25, "error", "msg3"),
        ]

        # 10 second window should only include first failure
        count = await backend.get_failure_count(window_seconds=10)

        assert count == 1

    @pytest.mark.asyncio
    async def test_get_failure_count_all_outside_window(self):
        """Test get_failure_count when all failures outside window."""
        backend = MemoryBackend(namespace="test_count_none")
        current_time = time.time()
        backend._failures = [
            (current_time - 60, "error", "old"),
            (current_time - 120, "error", "older"),
        ]

        count = await backend.get_failure_count(window_seconds=30)

        assert count == 0


class TestIsCircuitBroken:
    """Test is_circuit_broken method (lines 193-200)."""

    @pytest.mark.asyncio
    async def test_is_circuit_broken_not_set(self):
        """Test is_circuit_broken when not set (line 194-195)."""
        backend = MemoryBackend(namespace="test_circuit")

        result = await backend.is_circuit_broken()

        # Verify not broken when not set (covers lines 194-195)
        assert result is False

    @pytest.mark.asyncio
    async def test_is_circuit_broken_active(self):
        """Test is_circuit_broken when active (lines 196-197)."""
        backend = MemoryBackend(namespace="test_circuit_active")
        backend._circuit_broken_until = time.time() + 60

        result = await backend.is_circuit_broken()

        # Verify broken (covers lines 196-197)
        assert result is True

    @pytest.mark.asyncio
    async def test_is_circuit_broken_expired(self):
        """Test is_circuit_broken when expired clears state (lines 198-200)."""
        backend = MemoryBackend(namespace="test_circuit_expired")
        backend._circuit_broken_until = time.time() - 10  # Expired

        result = await backend.is_circuit_broken()

        # Verify not broken and state cleared (covers lines 198-200)
        assert result is False
        assert backend._circuit_broken_until is None

    @pytest.mark.asyncio
    async def test_is_circuit_broken_just_expired(self):
        """Test is_circuit_broken when just at boundary."""
        backend = MemoryBackend(namespace="test_circuit_boundary")
        # Set to expire at current time (boundary case)
        backend._circuit_broken_until = time.time()

        # Allow a tiny delay for the time check
        await asyncio.sleep(0.001)

        result = await backend.is_circuit_broken()

        # Should be expired now
        assert result is False


class TestClearFailures:
    """Test clear_failures method (lines 204-206)."""

    @pytest.mark.asyncio
    async def test_clear_failures_basic(self):
        """Test clear_failures removes all failures (lines 204-206)."""
        backend = MemoryBackend(namespace="test_clear")
        backend._failures = [
            (time.time(), "error1", "msg1"),
            (time.time(), "error2", "msg2"),
        ]

        await backend.clear_failures()

        # Verify cleared (covers lines 204-206)
        assert backend._failures == []

    @pytest.mark.asyncio
    async def test_clear_failures_empty(self):
        """Test clear_failures on empty list."""
        backend = MemoryBackend(namespace="test_clear_empty")

        # Should not raise
        await backend.clear_failures()

        assert backend._failures == []


class TestForceCircuitBreak:
    """Test force_circuit_break method (lines 215-217)."""

    @pytest.mark.asyncio
    async def test_force_circuit_break_basic(self):
        """Test force_circuit_break sets expiry (lines 215-217)."""
        backend = MemoryBackend(namespace="test_force")

        await backend.force_circuit_break(30.0)

        # Verify circuit broken (covers lines 215-217)
        assert backend._circuit_broken_until is not None
        assert backend._circuit_broken_until > time.time()
        assert backend._circuit_broken_until < time.time() + 35

    @pytest.mark.asyncio
    async def test_force_circuit_break_short_duration(self):
        """Test force_circuit_break with short duration."""
        backend = MemoryBackend(namespace="test_force_short")

        await backend.force_circuit_break(0.1)

        # Should be broken now
        assert await backend.is_circuit_broken() is True

        # Wait for expiry
        await asyncio.sleep(0.15)

        # Should be unbroken
        assert await backend.is_circuit_broken() is False

    @pytest.mark.asyncio
    async def test_force_circuit_break_overwrites(self):
        """Test force_circuit_break overwrites existing."""
        backend = MemoryBackend(namespace="test_force_overwrite")

        # Set initial break
        await backend.force_circuit_break(60.0)
        first_expiry = backend._circuit_broken_until

        # Overwrite with shorter duration
        await backend.force_circuit_break(10.0)
        second_expiry = backend._circuit_broken_until

        assert first_expiry is not None
        assert second_expiry is not None
        assert second_expiry < first_expiry


class TestConcurrency:
    """Test thread safety with async locks."""

    @pytest.mark.asyncio
    async def test_concurrent_record_requests(self):
        """Test concurrent record_request calls."""
        backend = MemoryBackend(namespace="test_concurrent")

        async def record():
            for _ in range(10):
                await backend.record_request("test_model", tokens_used=100)

        # Run multiple concurrent tasks
        await asyncio.gather(*[record() for _ in range(5)])

        # Should have 50 requests total
        assert backend._request_counts["test_model"] == 50
        assert backend._token_counts["test_model"] == 5000

    @pytest.mark.asyncio
    async def test_concurrent_failures(self):
        """Test concurrent record_failure calls."""
        backend = MemoryBackend(namespace="test_concurrent_failures")

        async def record_fail():
            for i in range(10):
                await backend.record_failure("error", f"msg_{i}")

        # Run multiple concurrent tasks
        await asyncio.gather(*[record_fail() for _ in range(3)])

        # Should have 30 failures total
        assert len(backend._failures) == 30


class TestBranchCoverage:
    """Additional tests specifically for branch coverage."""

    @pytest.mark.asyncio
    async def test_check_capacity_model_in_rate_limits_true_branch(self):
        """Test check_capacity when model exists in rate_limits (line 121 true branch)."""
        backend = MemoryBackend(namespace="test_branch_121_true")
        backend._rate_limits["test_model"] = {"rpm_remaining": 100}

        can_proceed, wait_time = await backend.check_capacity("test_model")

        assert can_proceed is True
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_check_capacity_model_not_in_rate_limits_false_branch(self):
        """Test check_capacity when model not in rate_limits (line 121 false branch)."""
        backend = MemoryBackend(namespace="test_branch_121_false")

        can_proceed, wait_time = await backend.check_capacity("nonexistent_model")

        assert can_proceed is True
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_check_capacity_rpm_remaining_gt_zero(self):
        """Test check_capacity when rpm_remaining > 0 (line 124 false branch)."""
        backend = MemoryBackend(namespace="test_branch_124_false")
        backend._rate_limits["test_model"] = {"rpm_remaining": 10}

        can_proceed, wait_time = await backend.check_capacity("test_model")

        assert can_proceed is True

    @pytest.mark.asyncio
    async def test_check_capacity_rpm_remaining_zero_with_wait(self):
        """Test check_capacity when rpm_remaining <= 0 with wait time (lines 124-127)."""
        backend = MemoryBackend(namespace="test_branch_124_127")
        backend._rate_limits["test_model"] = {
            "rpm_remaining": 0,
            "rpm_reset": time.time() + 30,
        }

        can_proceed, wait_time = await backend.check_capacity("test_model")

        assert can_proceed is False
        assert wait_time > 0

    @pytest.mark.asyncio
    async def test_check_capacity_wait_time_zero(self):
        """Test check_capacity when wait_time <= 0 (line 126 false branch)."""
        backend = MemoryBackend(namespace="test_branch_126_false")
        backend._rate_limits["test_model"] = {
            "rpm_remaining": 0,
            # No reset time set, so _calculate_wait_time returns 0
        }

        can_proceed, wait_time = await backend.check_capacity("test_model")

        # wait_time is 0 so can proceed
        assert can_proceed is True
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_record_request_tokens_none_branch(self):
        """Test record_request when tokens_used is None (line 155 false branch)."""
        backend = MemoryBackend(namespace="test_branch_155_none")

        await backend.record_request("test_model", tokens_used=None)

        assert backend._request_counts["test_model"] == 1
        # Token count should remain 0 (default)
        assert backend._token_counts["test_model"] == 0

    @pytest.mark.asyncio
    async def test_record_request_tokens_provided_branch(self):
        """Test record_request when tokens_used is provided (line 155 true branch)."""
        backend = MemoryBackend(namespace="test_branch_155_true")

        await backend.record_request("test_model", tokens_used=500)

        assert backend._request_counts["test_model"] == 1
        assert backend._token_counts["test_model"] == 500

    @pytest.mark.asyncio
    async def test_is_circuit_broken_until_is_none(self):
        """Test is_circuit_broken when _circuit_broken_until is None (line 194)."""
        backend = MemoryBackend(namespace="test_branch_194")
        backend._circuit_broken_until = None

        result = await backend.is_circuit_broken()

        assert result is False

    @pytest.mark.asyncio
    async def test_is_circuit_broken_time_lt_until(self):
        """Test is_circuit_broken when time < _circuit_broken_until (line 196 true)."""
        backend = MemoryBackend(namespace="test_branch_196_true")
        backend._circuit_broken_until = time.time() + 60

        result = await backend.is_circuit_broken()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_circuit_broken_time_gte_until(self):
        """Test is_circuit_broken when time >= _circuit_broken_until (line 196 false)."""
        backend = MemoryBackend(namespace="test_branch_196_false")
        backend._circuit_broken_until = time.time() - 1

        result = await backend.is_circuit_broken()

        assert result is False
        # Also clears the state
        assert backend._circuit_broken_until is None

    @pytest.mark.asyncio
    async def test_health_check_circuit_broken_evaluation(self):
        """Test health_check circuit_broken condition with both branches."""
        backend = MemoryBackend(namespace="test_branch_health_circuit")

        # Test None
        backend._circuit_broken_until = None
        result = await backend.health_check()
        assert result.metadata is not None
        assert result.metadata["circuit_broken"] is False

        # Test active
        backend._circuit_broken_until = time.time() + 60
        result = await backend.health_check()
        assert result.metadata is not None
        assert result.metadata["circuit_broken"] is True

        # Test expired
        backend._circuit_broken_until = time.time() - 10
        result = await backend.health_check()
        assert result.metadata is not None
        assert result.metadata["circuit_broken"] is False

    @pytest.mark.asyncio
    async def test_get_all_stats_circuit_broken_evaluation(self):
        """Test get_all_stats circuit_broken condition with both branches."""
        backend = MemoryBackend(namespace="test_branch_stats_circuit")

        # Test None
        backend._circuit_broken_until = None
        result = await backend.get_all_stats()
        assert result["circuit_broken"] is False

        # Test active
        backend._circuit_broken_until = time.time() + 60
        result = await backend.get_all_stats()
        assert result["circuit_broken"] is True

        # Test expired
        backend._circuit_broken_until = time.time() - 10
        result = await backend.get_all_stats()
        assert result["circuit_broken"] is False


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_check_capacity_with_tpm_limits(self):
        """Test check_capacity considers tpm limits."""
        backend = MemoryBackend(namespace="test_tpm")
        backend._rate_limits["test_model"] = {
            "rpm_remaining": 100,  # RPM OK
            "tpm_remaining": 0,  # TPM exhausted
            "tpm_reset": time.time() + 30,
        }

        can_proceed, wait_time = await backend.check_capacity("test_model")

        # Should still proceed based on RPM (memory backend is simple)
        assert can_proceed is True

    @pytest.mark.asyncio
    async def test_multiple_models(self):
        """Test handling multiple models independently."""
        backend = MemoryBackend(namespace="test_multi_model")

        # Record requests for different models
        await backend.record_request("model_a", tokens_used=100)
        await backend.record_request("model_b", tokens_used=200)
        await backend.record_request("model_a", tokens_used=150)

        # Update rate limits for different models
        await backend.update_rate_limits("model_a", {"x-ratelimit-remaining-requests": "50"})
        await backend.update_rate_limits("model_b", {"x-ratelimit-remaining-requests": "30"})

        assert backend._request_counts["model_a"] == 2
        assert backend._request_counts["model_b"] == 1
        assert backend._token_counts["model_a"] == 250
        assert backend._token_counts["model_b"] == 200
        assert backend._rate_limits["model_a"]["rpm_remaining"] == 50
        assert backend._rate_limits["model_b"]["rpm_remaining"] == 30

    @pytest.mark.asyncio
    async def test_failure_timestamp_accuracy(self):
        """Test failure timestamps are accurate."""
        backend = MemoryBackend(namespace="test_timestamp")

        before = time.time()
        await backend.record_failure("test", "msg")
        after = time.time()

        ts, _, _ = backend._failures[0]
        assert before <= ts <= after
