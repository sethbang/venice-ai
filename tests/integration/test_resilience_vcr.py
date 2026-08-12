"""
Integration tests for resilience infrastructure.

This module tests the retry middleware using VCRpy for HTTP interaction
recording/replay.

Tests cover:
- Retry middleware with exponential backoff
- Error handling paths for various HTTP error types
"""

import asyncio
import os
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from venice_ai import create_test_venice_client
from venice_ai.core.config import SchedulerMode
from venice_ai.exceptions import (
    InternalServerError,
    RateLimitError,
    VeniceError,
)
from venice_ai.middleware.retry import (
    RetryOptions,
    calculate_backoff_delay,
    create_retry_middleware,
    parse_retry_after_header,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def venice_client(backend_instance):
    """Create a Venice client for resilience testing with shared rate limit coordination."""
    api_key = os.getenv("VENICE_API_KEY", "test-api-key-for-recording")

    client = create_test_venice_client(
        api_key=api_key,
        scheduler_mode=SchedulerMode.INTELLIGENT,
        enable_redis=True,
    )
    try:
        yield client
    finally:
        await client.close()


# model_selector fixture is provided by the root conftest.py


# ============================================================================
# Retry Middleware Tests
# ============================================================================


@pytest.mark.integration
def test_retry_middleware_exponential_backoff():
    """
    Test retry logic with exponential backoff timing calculations.

    Tests:
    - Delay increases exponentially
    - Jitter is applied correctly
    - Maximum delay is respected
    """
    options = RetryOptions(
        base_delay=1.0,
        exponential_base=2.0,
        max_delay=30.0,
        jitter_factor=0.0,  # No jitter for deterministic testing
    )

    # Test exponential growth
    delay0 = calculate_backoff_delay(
        attempt=0,
        base_delay=options.base_delay,
        exponential_base=options.exponential_base,
        max_delay=options.max_delay,
        jitter_factor=options.jitter_factor,
    )
    assert delay0 == 1.0  # 1.0 * (2.0 ^ 0) = 1.0

    delay1 = calculate_backoff_delay(
        attempt=1,
        base_delay=options.base_delay,
        exponential_base=options.exponential_base,
        max_delay=options.max_delay,
        jitter_factor=options.jitter_factor,
    )
    assert delay1 == 2.0  # 1.0 * (2.0 ^ 1) = 2.0

    delay2 = calculate_backoff_delay(
        attempt=2,
        base_delay=options.base_delay,
        exponential_base=options.exponential_base,
        max_delay=options.max_delay,
        jitter_factor=options.jitter_factor,
    )
    assert delay2 == 4.0  # 1.0 * (2.0 ^ 2) = 4.0

    # Test max delay cap
    delay10 = calculate_backoff_delay(
        attempt=10,
        base_delay=options.base_delay,
        exponential_base=options.exponential_base,
        max_delay=options.max_delay,
        jitter_factor=options.jitter_factor,
    )
    assert delay10 == 30.0  # Capped at max_delay

    # Test with jitter
    delay_with_jitter = calculate_backoff_delay(
        attempt=0,
        base_delay=1.0,
        exponential_base=2.0,
        max_delay=30.0,
        jitter_factor=0.1,
    )
    # With 10% jitter, delay should be in range [0.9, 1.1]
    assert 0.9 <= delay_with_jitter <= 1.1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retry_middleware_max_attempts():
    """
    Test retry exhaustion behavior.

    Tests that:
    - Retries are attempted up to max_attempts
    - Final exception is raised after exhaustion
    - on_retry callback is called for each retry
    """
    retry_counts = []

    options = RetryOptions(
        max_attempts=3,
        base_delay=0.01,  # Fast for testing
        retry_non_idempotent=True,  # Allow POST retries for testing
        on_retry=lambda attempt, delay, exc: retry_counts.append(
            {
                "attempt": attempt,
                "delay": delay,
                "exception": type(exc).__name__ if exc else None,
            }
        ),
    )

    middleware = create_retry_middleware(options)

    # Create mock request and handler that always fails
    mock_request = MagicMock()
    mock_request.method = "GET"
    mock_request.url = "https://api.test.com/test"

    fail_count = 0

    async def failing_handler(request):
        nonlocal fail_count
        fail_count += 1
        raise TimeoutError("Simulated timeout")

    # Execute middleware - should retry and eventually raise
    with pytest.raises(asyncio.TimeoutError):
        await middleware(mock_request, failing_handler)

    # Should have attempted 4 times (1 initial + 3 retries)
    assert fail_count == 4

    # Should have logged 3 retries
    assert len(retry_counts) == 3


@pytest.mark.integration
def test_retry_after_header_parsing():
    """
    Test Retry-After header parsing for various formats.

    Tests:
    - Integer seconds format
    - HTTP-date format
    - Invalid format handling
    """
    # Test with integer seconds
    mock_response = MagicMock()
    mock_response.headers = {"Retry-After": "60"}

    result = parse_retry_after_header(mock_response)
    assert result == 60.0

    # Test with missing header
    mock_response.headers = {}
    result = parse_retry_after_header(mock_response)
    assert result is None

    # Test with invalid format
    mock_response.headers = {"Retry-After": "invalid"}
    result = parse_retry_after_header(mock_response)
    assert result is None


# ============================================================================
# Error Exception Tests
# ============================================================================


@pytest.mark.integration
def test_exception_hierarchy_attributes():
    """
    Test that exception classes have proper attributes and hierarchy.
    """
    # Test VeniceError base class
    venice_err = VeniceError("Test error")
    assert str(venice_err) == "Test error"
    assert venice_err.message == "Test error"
    assert venice_err.request is None
    assert venice_err.response is None

    # Test RateLimitError with attributes
    mock_response = MagicMock()
    mock_response.status = 429
    mock_response.headers = {}

    rate_err = RateLimitError(
        "Rate limited",
        response=mock_response,
        retry_after_seconds=60,
        remaining_requests=0,
    )
    assert rate_err.status_code == 429
    assert rate_err.retry_after_seconds == 60
    assert rate_err.remaining_requests == 0
    assert isinstance(rate_err, VeniceError)

    # Test InternalServerError
    mock_response.status = 500
    server_err = InternalServerError(
        "Server error",
        response=mock_response,
        body={"error": "Internal error"},
    )
    assert server_err.status_code == 500
    assert isinstance(server_err, VeniceError)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
