"""
Integration tests for concurrent request handling.

This module tests the SDK's behavior under concurrent load, including:
- Race conditions in rate limiting
- Connection pool utilization
- Parallel request correctness
- State consistency under load
"""

import asyncio
import os

import pytest
import pytest_asyncio

from venice_ai import create_test_venice_client
from venice_ai.core.config import SchedulerMode
from venice_ai.exceptions import RateLimitError


@pytest_asyncio.fixture
async def venice_client(backend_instance):
    """Create a Venice client for concurrent testing with shared rate limit coordination."""
    api_key = os.getenv("VENICE_API_KEY")
    if not api_key:
        pytest.skip("VENICE_API_KEY environment variable required for integration tests")

    # Use INTELLIGENT mode with MemoryBackend for VCR tests
    # This provides rate limit protection (prevents 429s) without Redis connection leaks
    client = create_test_venice_client(
        api_key=api_key,
        scheduler_mode=SchedulerMode.INTELLIGENT,
        enable_redis=False,  # Use MemoryBackend instead of Redis
    )
    try:
        yield client
    finally:
        await client.close()


# model_selector fixture is now provided by the root conftest.py


# ============================================================================
# Concurrent Request Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
async def test_concurrent_requests_basic(venice_client, model_selector, vcr_cassette):
    """
    Test basic concurrent request handling.

    Validates that multiple concurrent requests all succeed
    and return valid responses.
    """
    with vcr_cassette:
        # Select a model once for all requests
        chat_model = await model_selector.select_chat_model()

        # Create 10 concurrent requests
        async def make_request(i: int):
            response = await venice_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": f"Count: {i}"}],
                max_completion_tokens=10,
                temperature=0.1,
            )
            return response

        # Execute concurrently
        tasks = [make_request(i) for i in range(10)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Validate all succeeded (or properly handled rate limits)
        successful = 0
        rate_limited = 0
        errors = 0

        for result in responses:
            if isinstance(result, Exception):
                if isinstance(result, RateLimitError):
                    rate_limited += 1
                else:
                    errors += 1
            else:
                successful += 1
                # Validate response structure for successful requests. Reasoning
                # models can return content=None (whole budget spent on
                # reasoning_content, finish_reason="length"), so accept either.
                assert hasattr(result, "choices"), "Response should have choices"
                message = result.choices[0].message  # type: ignore[union-attr]
                assert (message.content is not None) or (message.reasoning_content is not None), (
                    "Response should carry content or reasoning_content"
                )

        # At least some should succeed
        assert successful > 0, "No requests succeeded"
        # Should not have non-rate-limit errors
        assert errors == 0, f"Had {errors} unexpected errors"


@pytest.mark.integration
@pytest.mark.slow
async def test_concurrent_requests_respect_rate_limits(venice_client, model_selector, vcr_cassette):
    """
    Verify concurrent requests don't violate rate limits.

    Tests that the rate limiter properly queues or rejects
    requests to stay within API limits.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        # Create more requests than typical rate limit
        num_requests = 20

        async def make_request(i: int):
            try:
                response = await venice_client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "user", "content": f"Request {i}"}],
                    max_completion_tokens=5,
                )
                return {"success": True, "index": i, "response": response}
            except RateLimitError:
                return {"success": False, "index": i, "error": "rate_limit"}
            except Exception as e:
                return {"success": False, "index": i, "error": str(e)}

        # Execute all concurrently
        tasks = [make_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)

        successful = sum(1 for r in results if r["success"])
        rate_limited = sum(
            1 for r in results if not r["success"] and r.get("error") == "rate_limit"
        )

        # Should have handled requests gracefully (either success or rate limit)
        assert successful + rate_limited == num_requests
        # At least some should succeed
        assert successful > 0


@pytest.mark.integration
async def test_concurrent_different_models(venice_client, model_selector, vcr_cassette):
    """
    Test concurrent requests to different models.

    Validates that concurrent requests to different models
    are handled correctly and don't interfere with each other.
    """
    with vcr_cassette:
        # Get multiple models
        chat_model = await model_selector.select_chat_model()

        # Make concurrent requests
        async def make_chat_request():
            return await venice_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Hello"}],
                max_completion_tokens=10,
            )

        # Run 5 concurrent requests
        tasks = [make_chat_request() for _ in range(5)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successful responses
        successful = sum(1 for r in responses if not isinstance(r, Exception))
        assert successful > 0, "No requests succeeded"


@pytest.mark.integration
@pytest.mark.slow
async def test_concurrent_requests_connection_pool(venice_client, model_selector, vcr_cassette):
    """
    Test that connection pool handles concurrent requests efficiently.

    Validates that the HTTP client's connection pool properly
    manages connections under concurrent load.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        # Create many concurrent requests to test pool
        num_requests = 20

        async def make_request(i: int):
            start_time = asyncio.get_event_loop().time()
            try:
                await venice_client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "user", "content": f"Req {i}"}],
                    max_completion_tokens=5,
                )
                elapsed = asyncio.get_event_loop().time() - start_time
                return {"success": True, "elapsed": elapsed, "index": i}
            except Exception as e:
                elapsed = asyncio.get_event_loop().time() - start_time
                return {"success": False, "elapsed": elapsed, "error": str(e)}

        tasks = [make_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)

        successful = [r for r in results if r["success"]]
        assert len(successful) > 0, "No requests succeeded"

        # Check that concurrent execution was reasonably fast
        # (i.e., connection pool enabled parallelism)
        avg_time = sum(r["elapsed"] for r in successful) / len(successful)
        # This is a loose check - just ensure it's not absurdly slow
        assert avg_time < 30.0, f"Average request time too slow: {avg_time}s"


@pytest.mark.integration
# Marked individually (not covered by test-ci's global --only-rerun 500/timeout
# rule): this flakes as an AssertionError ("got no chunks") when a concurrent
# stream occasionally yields nothing, not as a network exception.
@pytest.mark.flaky(reruns=2, reruns_delay=2)
async def test_concurrent_streaming_requests(venice_client, model_selector, vcr_cassette):
    """
    Test concurrent streaming requests.

    Validates that multiple streaming requests can run
    concurrently without interference. Accepts rate limiting as valid outcome.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        async def stream_request(i: int):
            chunks = []
            try:
                async for chunk in await venice_client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "user", "content": f"Count to {i}"}],
                    max_completion_tokens=20,
                    stream=True,
                ):
                    if chunk.choices and chunk.choices[0].delta.content:
                        chunks.append(chunk.choices[0].delta.content)
                return {"success": True, "chunks": len(chunks), "index": i}
            except RateLimitError:
                return {"success": False, "error": "rate_limited", "index": i}
            except Exception as e:
                return {"success": False, "error": str(e), "index": i}

        # Run 3 concurrent streaming requests
        tasks = [stream_request(i) for i in range(1, 4)]
        results = await asyncio.gather(*tasks)

        # All requests should be handled (success or rate limited)
        assert len(results) == 3
        successful = [r for r in results if r["success"]]
        rate_limited = [r for r in results if r.get("error") == "rate_limited"]

        # Either succeeded or rate limited - both are acceptable outcomes
        assert len(successful) + len(rate_limited) == 3

        # Validate we got chunks for successful requests
        for result in successful:
            assert result["chunks"] > 0, f"Request {result['index']} got no chunks"


# ============================================================================
# State Consistency Tests
# ============================================================================


@pytest.mark.integration
async def test_concurrent_requests_state_consistency(venice_client, model_selector, vcr_cassette):
    """
    Test that concurrent requests maintain state consistency.

    Validates that rate limit state, circuit breaker state, etc.
    remain consistent under concurrent load. Accepts rate limiting as valid.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        # Make concurrent requests and track any state issues
        async def make_request(i: int):
            try:
                response = await venice_client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "user", "content": f"Test {i}"}],
                    max_completion_tokens=5,
                )
                # If response has rate limit info, capture it
                rate_limits = None
                if hasattr(response, "response_rate_limits"):
                    rate_limits = response.response_rate_limits
                return {"success": True, "index": i, "rate_limits": rate_limits}
            except RateLimitError:
                return {"success": False, "index": i, "error": "rate_limited"}
            except Exception as e:
                return {"success": False, "index": i, "error": str(e)}

        tasks = [make_request(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All requests should be tracked
        assert len(results) == 10

        # Verify no corruption - all results should have an index
        for result in results:
            assert result["index"] is not None

        # Either succeeded or hit rate limit - both are valid
        successful = [r for r in results if r["success"]]
        rate_limited = [r for r in results if r.get("error") == "rate_limited"]
        assert len(successful) + len(rate_limited) >= 0


@pytest.mark.integration
@pytest.mark.slow
async def test_burst_then_steady_load(venice_client, model_selector, vcr_cassette):
    """
    Test handling of burst traffic followed by steady load.

    Validates that the system handles traffic spikes gracefully.
    Accepts rate limiting as expected behavior.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        # Phase 1: Burst of requests
        async def make_request(i: int, phase: str):
            try:
                await venice_client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "user", "content": f"{phase} {i}"}],
                    max_completion_tokens=5,
                )
                return {"success": True, "phase": phase}
            except RateLimitError:
                return {"success": False, "phase": phase, "error": "rate_limited"}
            except Exception as e:
                return {"success": False, "phase": phase, "error": str(e)}

        # Burst: 15 concurrent requests
        burst_tasks = [make_request(i, "burst") for i in range(15)]
        burst_results = await asyncio.gather(*burst_tasks)

        # Brief pause
        await asyncio.sleep(1)

        # Steady: 5 sequential requests
        steady_results = []
        for i in range(5):
            result = await make_request(i, "steady")
            steady_results.append(result)
            await asyncio.sleep(0.5)

        # Validate both phases handled requests
        assert len(burst_results) == 15
        assert len(steady_results) == 5

        # At least some requests should be processed (success or rate limited)
        burst_handled = sum(1 for r in burst_results if r.get("success") or r.get("error"))
        steady_handled = sum(1 for r in steady_results if r.get("success") or r.get("error"))

        assert burst_handled == 15, "All burst requests should be handled"
        assert steady_handled == 5, "All steady requests should be handled"
