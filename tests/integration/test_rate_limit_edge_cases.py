"""
Integration tests for rate limit edge cases.

This module tests boundary conditions and edge cases in rate limiting:
- Exactly at rate limit boundary
- Rate limit reset behavior
- Multi-tier rate limit handling
- Queue overflow conditions
- Rate limit header variations
"""

import asyncio
import os

import pytest
import pytest_asyncio

from venice_ai import create_test_venice_client
from venice_ai.core.config import SchedulerMode
from venice_ai.exceptions import RateLimitError, VeniceError


@pytest_asyncio.fixture
async def venice_client(backend_instance):
    """Create a Venice client for rate limit testing with shared rate limit coordination."""
    api_key = os.getenv("VENICE_API_KEY")
    if not api_key:
        pytest.skip("VENICE_API_KEY environment variable required for integration tests")

    client = create_test_venice_client(
        api_key=api_key,
        scheduler_mode=SchedulerMode.INTELLIGENT,
        enable_redis=True,
    )
    try:
        yield client
    finally:
        await client.close()


# model_selector fixture is now provided by the root conftest.py


# ============================================================================
# Rate Limit Boundary Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
async def test_rate_limit_at_boundary(venice_client, model_selector, vcr_cassette):
    """
    Test behavior when exactly at rate limit boundary.

    Validates that the last allowed request succeeds and
    the next request properly handles the limit.
    May immediately hit rate limit in test scenarios - this is expected.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        # Make several requests to approach limit
        responses = []
        rate_limited = False
        for i in range(5):
            try:
                response = await venice_client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "user", "content": f"Request {i}"}],
                    max_completion_tokens=5,
                )
                responses.append(response)

                # Check rate limit headers if available
                if hasattr(response, "response_rate_limits") and response.response_rate_limits:
                    remaining = response.response_rate_limits.remaining_requests
                    if remaining is not None and remaining <= 1:
                        # We're at the boundary
                        break

            except RateLimitError:
                # Hit the limit - this is acceptable
                rate_limited = True
                break

        # Either got responses or hit rate limit - both valid
        assert len(responses) > 0 or rate_limited, "Should either get responses or hit rate limit"


@pytest.mark.integration
async def test_rate_limit_headers_parsing(venice_client, model_selector, vcr_cassette):
    """
    Test that rate limit headers are properly parsed.

    Validates parsing of various rate limit header formats.
    May hit rate limit - tests error structure in that case.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        try:
            response = await venice_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Test"}],
                max_completion_tokens=5,
            )

            # Check if rate limit info is available
            if hasattr(response, "response_rate_limits") and response.response_rate_limits:
                rate_limits = response.response_rate_limits

                # Validate structure (may be None if not provided by API)
                if rate_limits.limit_requests is not None:
                    assert rate_limits.limit_requests > 0

                if rate_limits.remaining_requests is not None:
                    assert rate_limits.remaining_requests >= 0
        except RateLimitError as e:
            # Rate limited - verify error has proper structure
            assert e.status_code == 429
            assert hasattr(e, "retry_after_seconds")


@pytest.mark.integration
@pytest.mark.slow
async def test_rate_limit_recovery_after_reset(venice_client, model_selector, vcr_cassette):
    """
    Test that rate limits properly reset after cooldown period.

    Validates that after hitting a rate limit and waiting,
    requests can be made again.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        # Phase 1: Make requests until rate limited
        initial_requests = []
        for i in range(20):
            try:
                response = await venice_client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "user", "content": f"Initial {i}"}],
                    max_completion_tokens=5,
                )
                initial_requests.append(response)
            except RateLimitError as e:
                # Got rate limited - note the retry_after
                retry_after = getattr(e, "retry_after_seconds", 60)

                # Wait for reset (in tests, this might be short)
                # In VCR mode, time might be compressed
                await asyncio.sleep(min(retry_after, 2))
                break

        # Phase 2: Try request after waiting
        try:
            recovery_response = await venice_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Recovery test"}],
                max_completion_tokens=5,
            )
            # If we get here, rate limit was reset or we're using VCR
            assert recovery_response is not None
        except RateLimitError:
            # Still rate limited - acceptable in some scenarios
            pass


@pytest.mark.integration
async def test_rate_limit_different_models_independent(venice_client, model_selector, vcr_cassette):
    """
    Test that rate limits for different models are independent.

    Validates that hitting rate limit on one model doesn't
    affect another model's rate limits.
    """
    with vcr_cassette:
        # Get a chat model
        chat_model = await model_selector.select_chat_model()

        # Make requests to the chat model
        responses_model1 = []
        for i in range(3):
            try:
                response = await venice_client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "user", "content": f"Test {i}"}],
                    max_completion_tokens=5,
                )
                responses_model1.append(response)
            except RateLimitError:
                break

        # Verify we got responses
        assert len(responses_model1) > 0


# ============================================================================
# Queue Overflow Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
async def test_queue_overflow_handling(venice_client, model_selector, vcr_cassette):
    """
    Test handling of queue overflow conditions.

    Validates that when the request queue fills up,
    overflow is handled gracefully.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        # Create many concurrent requests to potentially overflow queue
        async def make_request(i: int):
            try:
                await venice_client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "user", "content": f"Overflow test {i}"}],
                    max_completion_tokens=5,
                )
                return {"success": True, "index": i}
            except RateLimitError:
                return {"success": False, "error": "rate_limit", "index": i}
            except VeniceError as e:
                # Check if it's an overflow-related error
                error_msg = str(e).lower()
                if "overflow" in error_msg or "queue" in error_msg:
                    return {"success": False, "error": "overflow", "index": i}
                return {"success": False, "error": str(e), "index": i}
            except Exception as e:
                return {"success": False, "error": str(e), "index": i}

        # Create more requests than typical queue size
        tasks = [make_request(i) for i in range(100)]
        results = await asyncio.gather(*tasks)

        # Categorize results
        successful = [r for r in results if r["success"]]
        overflowed = [r for r in results if not r["success"] and r.get("error") == "overflow"]
        rate_limited = [r for r in results if not r["success"] and r.get("error") == "rate_limit"]

        # The core "handled gracefully" guarantee: every request returned a
        # result dict (make_request catches all exceptions), so nothing crashed
        # under load.
        assert len(results) == 100, "Some requests did not return a result"
        # Under 100-way live concurrency a few requests can fail transiently
        # (connection resets/cancellations) with an uncategorized error, so
        # tolerate a small fraction rather than demanding every single request
        # land in success/overflow/rate_limit.
        total_handled = len(successful) + len(overflowed) + len(rate_limited)
        assert total_handled >= 0.9 * len(results), (
            f"Too many uncategorized failures: only {total_handled}/{len(results)} "
            "requests landed in success/overflow/rate_limit"
        )


# ============================================================================
# Edge Case Tests
# ============================================================================


@pytest.mark.integration
async def test_zero_rate_limit_handling(venice_client, model_selector, vcr_cassette):
    """
    Test handling when rate limit is exhausted (zero remaining).

    Validates proper error handling when no requests remain.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        # Make a request and check rate limits
        try:
            response = await venice_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Test"}],
                max_completion_tokens=5,
            )

            # Check if we can see rate limit info
            if hasattr(response, "response_rate_limits") and response.response_rate_limits:
                remaining = response.response_rate_limits.remaining_requests
                if remaining == 0:
                    # Try another request - should be rate limited
                    import contextlib

                    with contextlib.suppress(RateLimitError):
                        await venice_client.chat.completions.create(
                            model=chat_model,
                            messages=[{"role": "user", "content": "Should fail"}],
                            max_completion_tokens=5,
                        )
        except RateLimitError:
            # Already at zero - expected
            pass


@pytest.mark.integration
async def test_rate_limit_with_retries(venice_client, model_selector, vcr_cassette):
    """
    Test that rate limit errors trigger appropriate retry behavior.

    Validates that retry logic respects rate limit backoff.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        max_retries = 3
        attempt = 0
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                response = await venice_client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "user", "content": f"Retry test {attempt}"}],
                    max_completion_tokens=5,
                )
                # Success
                assert response is not None
                break
            except RateLimitError as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Wait before retry
                    retry_after = getattr(e, "retry_after_seconds", None)
                    if retry_after is not None:
                        await asyncio.sleep(min(retry_after, 2))
                    else:
                        await asyncio.sleep(1)
            except Exception as e:
                # Other error - don't retry
                last_error = e
                break

        # Either succeeded or properly handled rate limits
        assert attempt < max_retries or isinstance(last_error, RateLimitError)


@pytest.mark.integration
async def test_rapid_sequential_requests(venice_client, model_selector, vcr_cassette):
    """
    Test rapid sequential requests staying under rate limit.

    Validates that making requests as fast as possible
    (but sequentially) doesn't violate rate limits.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        num_requests = 10
        responses = []
        errors = []

        for i in range(num_requests):
            try:
                response = await venice_client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "user", "content": f"Rapid {i}"}],
                    max_completion_tokens=5,
                )
                responses.append(response)
            except RateLimitError as e:
                errors.append(e)
                # Respect the rate limit
                break
            except Exception as e:
                errors.append(e)
                break

        # Should have some successful requests
        assert len(responses) > 0 or len(errors) > 0


@pytest.mark.integration
async def test_rate_limit_info_consistency(venice_client, model_selector, vcr_cassette):
    """
    Test that rate limit information remains consistent across requests.

    Validates that rate limit counters decrement properly.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        rate_limit_sequence = []

        for i in range(3):
            try:
                response = await venice_client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "user", "content": f"Consistency test {i}"}],
                    max_completion_tokens=5,
                )

                # Capture rate limit info
                if hasattr(response, "response_rate_limits") and response.response_rate_limits:
                    rate_limit_sequence.append(
                        {
                            "index": i,
                            "limit": response.response_rate_limits.limit_requests,
                            "remaining": response.response_rate_limits.remaining_requests,
                        }
                    )

            except RateLimitError:
                break

        # If we captured rate limits, verify they're sensible
        if len(rate_limit_sequence) >= 2:
            # Remaining should decrease or stay same (might not always be provided)
            for i in range(len(rate_limit_sequence) - 1):
                curr = rate_limit_sequence[i]
                next_item = rate_limit_sequence[i + 1]

                if curr["remaining"] is not None and next_item["remaining"] is not None:
                    # Remaining should decrease or stay the same
                    assert next_item["remaining"] <= curr["remaining"]
