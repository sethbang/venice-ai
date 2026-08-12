"""Unit tests for NoOpRateLimiter.

NoOpRateLimiter is used when rate limiting is disabled. It must implement
the same interface as SimpleRateLimiter but do nothing but pass through.
"""

from unittest.mock import MagicMock

import pytest

from venice_ai.rate_limiting import NoOpRateLimiter


class TestNoOpAlwaysAllows:
    """Tests verifying NoOpRateLimiter always allows requests."""

    @pytest.mark.asyncio
    async def test_noop_always_allows(self):
        """NoOpRateLimiter always allows requests."""
        limiter = NoOpRateLimiter()

        can_proceed, wait_time = await limiter.acquire("any-model")
        assert can_proceed is True
        assert wait_time == 0

    @pytest.mark.asyncio
    async def test_acquire_multiple_models(self):
        """NoOpRateLimiter allows all models."""
        limiter = NoOpRateLimiter()

        for model in ["model-a", "model-b", "model-c"]:
            can_proceed, wait_time = await limiter.acquire(model)
            assert can_proceed is True
            assert wait_time == 0

    @pytest.mark.asyncio
    async def test_acquire_rapid_fire(self):
        """NoOpRateLimiter allows rapid-fire requests."""
        limiter = NoOpRateLimiter()

        # All should succeed immediately
        for _ in range(1000):
            can_proceed, wait_time = await limiter.acquire("stress-model")
            assert can_proceed is True
            assert wait_time == 0


class TestNoOpIgnoresFailures:
    """Tests verifying NoOpRateLimiter ignores failures and 429s."""

    @pytest.mark.asyncio
    async def test_noop_ignores_failures(self):
        """NoOpRateLimiter ignores failures."""
        limiter = NoOpRateLimiter()

        # These should not raise
        await limiter.record_failure("test-model")
        await limiter.update_from_headers("test-model", {}, status_code=429)

        # Should still allow
        can_proceed, _ = await limiter.acquire("test-model")
        assert can_proceed is True

    @pytest.mark.asyncio
    async def test_noop_ignores_rate_limit_headers(self):
        """NoOpRateLimiter ignores rate limit headers."""
        limiter = NoOpRateLimiter()

        # These rate limit headers should have no effect
        headers = {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "0",  # Exhausted!
            "x-ratelimit-reset-requests": "60",
            "retry-after": "30",
        }

        await limiter.update_from_headers("test-model", headers, status_code=429)

        # Even with exhausted limits, should still allow
        can_proceed, wait_time = await limiter.acquire("test-model")
        assert can_proceed is True
        assert wait_time == 0

    @pytest.mark.asyncio
    async def test_record_success_is_noop(self):
        """NoOpRateLimiter.record_success is a no-op."""
        limiter = NoOpRateLimiter()

        # Should not raise
        await limiter.record_success("test-model")

        # Should still allow
        can_proceed, _ = await limiter.acquire("test-model")
        assert can_proceed is True


class TestNoOpSubmitRequest:
    """Tests for NoOpRateLimiter.submit_request() - CRITICAL."""

    @pytest.mark.asyncio
    async def test_noop_submit_request_executes_directly(self):
        """
        CRITICAL: NoOpRateLimiter.submit_request must execute the request.

        Even though it doesn't rate limit, it still must implement the same
        interface as SimpleRateLimiter for VeniceClient compatibility.
        """
        limiter = NoOpRateLimiter()

        request_executed = False
        expected_result = {"data": "test"}

        async def mock_request():
            nonlocal request_executed
            request_executed = True
            return expected_result

        metadata = MagicMock()
        metadata.model_id = "test-model"

        result = await limiter.submit_request(metadata, mock_request)

        assert request_executed, "NoOpRateLimiter must execute the request"
        assert result == expected_result

    @pytest.mark.asyncio
    async def test_noop_passes_through_429_responses(self):
        """NoOpRateLimiter passes through 429 responses without retry."""
        limiter = NoOpRateLimiter()

        mock_response = MagicMock()
        mock_response.status = 429
        mock_response.headers = {}

        async def mock_request():
            return mock_response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        result = await limiter.submit_request(metadata, mock_request)

        # NoOpRateLimiter should NOT retry or create errors
        assert result.status == 429

    @pytest.mark.asyncio
    async def test_noop_no_backoff_after_429(self):
        """NoOpRateLimiter does not enter backoff after 429."""
        limiter = NoOpRateLimiter()

        mock_response = MagicMock()
        mock_response.status = 429
        mock_response.headers = {"retry-after": "30"}

        async def mock_request():
            return mock_response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        # First request returns 429
        await limiter.submit_request(metadata, mock_request)

        # Should still allow next request immediately (no backoff)
        can_proceed, wait_time = await limiter.acquire("test-model")
        assert can_proceed is True
        assert wait_time == 0

    @pytest.mark.asyncio
    async def test_submit_request_ignores_error_factory(self):
        """NoOpRateLimiter ignores error_factory parameter."""
        limiter = NoOpRateLimiter()

        error_factory_called = False

        def mock_error_factory(message, request, body, response):
            nonlocal error_factory_called
            error_factory_called = True
            return Exception(message)

        mock_response = MagicMock()
        mock_response.status = 429
        mock_response.headers = {}

        async def mock_request():
            return mock_response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        # Pass error_factory but it should be ignored
        result = await limiter.submit_request(
            metadata, mock_request, error_factory=mock_error_factory
        )

        assert not error_factory_called, "NoOpRateLimiter should not call error_factory"
        assert result.status == 429

    @pytest.mark.asyncio
    async def test_submit_request_propagates_exceptions(self):
        """NoOpRateLimiter propagates request function exceptions."""
        limiter = NoOpRateLimiter()

        async def failing_request():
            raise ValueError("Test exception")

        metadata = MagicMock()
        metadata.model_id = "test-model"

        with pytest.raises(ValueError, match="Test exception"):
            await limiter.submit_request(metadata, failing_request)


class TestNoOpLifecycle:
    """Tests for NoOpRateLimiter lifecycle management."""

    @pytest.mark.asyncio
    async def test_lifecycle_start_stop(self):
        """Test NoOpRateLimiter lifecycle."""
        limiter = NoOpRateLimiter()

        assert not limiter.is_running()

        await limiter.start()
        assert limiter.is_running()

        await limiter.stop()
        assert not limiter.is_running()

    def test_classifier_property(self):
        """Test NoOpRateLimiter classifier property."""
        limiter = NoOpRateLimiter()

        assert limiter.classifier is None

        mock_classifier = MagicMock()
        limiter.classifier = mock_classifier

        assert limiter.classifier is mock_classifier


class TestNoOpConcurrency:
    """Tests for NoOpRateLimiter concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_submit_requests(self):
        """NoOpRateLimiter handles concurrent requests trivially."""
        import asyncio

        limiter = NoOpRateLimiter()

        call_count = 0

        async def mock_request():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {}
            return mock_response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        # Submit 10 concurrent requests
        results = await asyncio.gather(
            *[limiter.submit_request(metadata, mock_request) for _ in range(10)]
        )

        assert len(results) == 10
        assert call_count == 10
        assert all(r.status == 200 for r in results)
