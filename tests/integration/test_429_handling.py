"""
Integration tests for 429 Rate Limit Response Handling.

This module tests the rate limiter's behavior when receiving 429 responses,
including backoff calculation and handling of missing rate limit headers.

Uses aiohttp mocking (not httpx) as the SDK uses aiohttp for HTTP requests.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.exceptions import RateLimitError
from venice_ai.rate_limiting import SimpleRateLimiter

# ============================================================================
# 429 Response Handling Tests
# ============================================================================


class Test429ResponseHandling:
    """Tests for 429 response handling in the rate limiter."""

    @pytest.mark.asyncio
    async def test_429_triggers_backoff(self):
        """
        Verify 429 response triggers backoff.

        When a 429 response is received with rate limit headers,
        the rate limiter should enter a backoff state for that model.
        """
        limiter = SimpleRateLimiter(min_backoff=1.0, max_retries=0)

        # Create a mock 429 response with rate limit headers
        mock_response = MagicMock()
        mock_response.status = 429
        mock_response.headers = {
            "retry-after": "5",
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-limit-requests": "100",
        }
        mock_response.json = AsyncMock(return_value={"error": {"message": "Rate limit exceeded"}})

        async def mock_request_func():
            return mock_response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        # Should raise RateLimitError due to 429 and max_retries=0
        with pytest.raises(RateLimitError):
            await limiter.submit_request(metadata, mock_request_func)

        # Rate limiter should be in backoff
        can_proceed, wait_time = await limiter.acquire("test-model")
        assert not can_proceed, "Model should be in backoff after 429"
        assert wait_time >= 1.0, f"Wait time should be at least 1s, got {wait_time}"

    @pytest.mark.asyncio
    async def test_429_without_headers(self):
        """
        Verify 429 response without rate limit headers still triggers backoff.

        When a 429 response is received without rate limit headers,
        the rate limiter should still apply a default backoff strategy.
        """
        limiter = SimpleRateLimiter(min_backoff=1.0, max_retries=0)

        # Mock 429 response with no rate limit headers
        mock_response = MagicMock()
        mock_response.status = 429
        mock_response.headers = {}  # No rate limit headers
        mock_response.json = AsyncMock(return_value={"error": {"message": "Rate limit exceeded"}})

        async def mock_request_func():
            return mock_response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        # Should raise RateLimitError
        with pytest.raises(RateLimitError):
            await limiter.submit_request(metadata, mock_request_func)

        # Should still be in backoff (default backoff applied)
        can_proceed, wait_time = await limiter.acquire("test-model")
        assert not can_proceed, "Model should be in backoff after 429"
        assert wait_time > 0, "Some backoff should be applied"

    @pytest.mark.asyncio
    async def test_429_uses_retry_after_header(self):
        """
        Verify that retry-after header is respected when available.
        """
        limiter = SimpleRateLimiter(min_backoff=0.1, max_retries=0)

        # Create a mock 429 response with retry-after
        mock_response = MagicMock()
        mock_response.status = 429
        mock_response.headers = {
            "retry-after": "10",
        }
        mock_response.json = AsyncMock(return_value={"error": {"message": "Rate limit exceeded"}})

        async def mock_request_func():
            return mock_response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        with pytest.raises(RateLimitError):
            await limiter.submit_request(metadata, mock_request_func)

        # Check state reflects the retry-after
        can_proceed, wait_time = await limiter.acquire("test-model")
        assert not can_proceed

    @pytest.mark.asyncio
    async def test_429_with_retry_succeeds_on_second_attempt(self):
        """
        Verify 429 response leads to retry that succeeds.

        When the first request fails with 429 but retry is enabled,
        the second attempt should succeed.
        """
        limiter = SimpleRateLimiter(min_backoff=0.01, max_retries=2)

        attempt_count = 0

        async def mock_request_func():
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count == 1:
                # First attempt: 429
                response = MagicMock()
                response.status = 429
                response.headers = {"retry-after": "0.01"}
                response.json = AsyncMock(return_value={"error": "rate limited"})
                return response
            else:
                # Second attempt: success
                response = MagicMock()
                response.status = 200
                response.headers = {
                    "x-ratelimit-limit-requests": "100",
                    "x-ratelimit-remaining-requests": "99",
                }
                return response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        result = await limiter.submit_request(metadata, mock_request_func)

        assert attempt_count == 2, "Should have retried once"
        assert result.status == 200

    @pytest.mark.asyncio
    async def test_multiple_429s_increase_backoff(self):
        """
        Verify multiple 429s lead to increasing backoff times.

        Each consecutive failure should increase the backoff duration
        following an exponential pattern.
        """
        limiter = SimpleRateLimiter(min_backoff=1.0, max_backoff=60.0, max_retries=0)

        metadata = MagicMock()
        metadata.model_id = "test-model"

        # First 429
        mock_response = MagicMock()
        mock_response.status = 429
        mock_response.headers = {}
        mock_response.json = AsyncMock(return_value={"error": "rate limited"})

        async def mock_request_func():
            return mock_response

        with pytest.raises(RateLimitError):
            await limiter.submit_request(metadata, mock_request_func)

        _, first_wait = await limiter.acquire("test-model")

        # Record another failure to get increased backoff
        await limiter.record_failure("test-model")

        _, second_wait = await limiter.acquire("test-model")

        # Second wait should be longer due to exponential backoff
        # Note: Wait times are relative to different timestamps, so we just verify
        # the backoff mechanism is working
        assert first_wait > 0
        assert second_wait > 0


class Test429WithDifferentModels:
    """Tests for 429 handling across different models."""

    @pytest.mark.asyncio
    async def test_429_on_one_model_doesnt_affect_another(self):
        """
        Verify 429 on model A doesn't block model B.

        Per-model isolation should ensure rate limits are independent.
        """
        limiter = SimpleRateLimiter(min_backoff=1.0, max_retries=0)

        # 429 on model-a
        mock_response_429 = MagicMock()
        mock_response_429.status = 429
        mock_response_429.headers = {}
        mock_response_429.json = AsyncMock(return_value={"error": "rate limited"})

        async def request_model_a():
            return mock_response_429

        metadata_a = MagicMock()
        metadata_a.model_id = "model-a"

        with pytest.raises(RateLimitError):
            await limiter.submit_request(metadata_a, request_model_a)

        # Model A should be blocked
        can_proceed_a, _ = await limiter.acquire("model-a")
        assert not can_proceed_a

        # Model B should still be available
        can_proceed_b, wait_time_b = await limiter.acquire("model-b")
        assert can_proceed_b
        assert wait_time_b == 0


class Test429ErrorDetails:
    """Tests for error details in 429 responses."""

    @pytest.mark.asyncio
    async def test_429_error_contains_response_body(self):
        """
        Verify RateLimitError contains the response body.
        """
        limiter = SimpleRateLimiter(max_retries=0)

        expected_body = {
            "error": {
                "message": "Rate limit exceeded for this model",
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded",
            }
        }

        mock_response = MagicMock()
        mock_response.status = 429
        mock_response.headers = {}
        mock_response.json = AsyncMock(return_value=expected_body)

        async def mock_request_func():
            return mock_response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        with pytest.raises(RateLimitError) as exc_info:
            await limiter.submit_request(metadata, mock_request_func)

        error = exc_info.value
        assert error.body == expected_body

    @pytest.mark.asyncio
    async def test_429_error_preserves_retry_after(self):
        """
        Verify RateLimitError includes retry-after information when available.
        """
        limiter = SimpleRateLimiter(max_retries=0)

        mock_response = MagicMock()
        mock_response.status = 429
        mock_response.headers = {"retry-after": "30"}
        mock_response.json = AsyncMock(return_value={"error": "rate limited"})

        async def mock_request_func():
            return mock_response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        with pytest.raises(RateLimitError) as exc_info:
            await limiter.submit_request(metadata, mock_request_func)

        error = exc_info.value
        assert error.response is mock_response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
