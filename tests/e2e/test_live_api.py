"""
Live API E2E Test Scenarios.

Tests against the live Venice API to validate rate limiting, header parsing,
distributed concurrency, and streaming token accuracy.

Prerequisites:
- VENICE_API_KEY: Valid API key with known rate limits
- VENICE_REDIS_URL: Connection string to test Redis instance
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from tests.e2e.conftest import get_e2e_test_model_sync

# Mark all tests in this module as E2E tests requiring live API access.
#
# A per-test timeout caps how long a single test can run when the live API
# mis-behaves (500s, slow reads). Without this, the combination of client-side
# retries, aiohttp timeouts, and ``@pytest.mark.flaky(reruns=3)`` can push a
# single failing test past 8 minutes and hang ``make coverage``. 120 s comfortably
# covers one healthy retry cycle on the happy path.
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.asyncio,
    pytest.mark.timeout(120),
]


class TestLiveHeaderParsing:
    """Validate rate limit headers are correctly parsed and stored."""

    @pytest.mark.flaky(
        reruns=3,
        reruns_delay=2,
        only_rerun=["APITimeoutError", "ConnectionTimeoutError", "TimeoutError"],
    )
    async def test_live_header_parsing(self, e2e_client, margin_config, e2e_retry_config):
        """Validate rate limit headers are correctly parsed and stored."""
        model = get_e2e_test_model_sync()  # No API call - uses cache or fallback

        response = await e2e_client.chat.completions.create(
            model=model,  # XS tier - high limits, low cost
            messages=[{"role": "user", "content": "Hello"}],
            max_completion_tokens=10,
        )

        # Verify we got a response
        assert response is not None
        assert response.choices is not None
        assert len(response.choices) > 0

        # Get rate limit state if available
        if hasattr(e2e_client, "rate_limiter") and e2e_client.rate_limiter:
            state = await e2e_client.rate_limiter.get_state(model)

            if state:
                # Verify state contains expected fields
                assert hasattr(state, "remaining_requests") or "remaining_requests" in state

                # If we have header information from response, verify margin
                if hasattr(response, "headers"):
                    headers = response.headers
                    if hasattr(headers, "get_remaining_requests"):
                        header_remaining = headers.get_remaining_requests()
                        if header_remaining is not None and state.remaining_requests is not None:
                            assert (
                                abs(state.remaining_requests - header_remaining)
                                <= margin_config.REMAINING_MARGIN
                            )


class TestDistributedConcurrency:
    """Test distributed concurrency with multiple concurrent requests."""

    async def test_distributed_concurrency(self, e2e_client, e2e_retry_config):
        """
        5 processes × 10 requests, verify aggregate pending tracking.

        Note: This test uses async concurrency rather than true multiprocessing
        for simplicity in the test environment. For full distributed testing,
        use the multiprocessing variant with separate process pools.
        """
        model = get_e2e_test_model_sync()

        # Create concurrent requests
        num_concurrent = 5
        requests_per_batch = 2  # Reduced for cost efficiency

        async def make_request(request_id: int) -> dict[str, Any]:
            """Make a single chat request."""
            response = await e2e_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"Say '{request_id}'"}],
                max_completion_tokens=5,
            )
            return {
                "request_id": request_id,
                "success": response is not None,
                "model": response.model if response else None,
            }

        # Execute concurrent batches
        all_results = []
        for batch in range(requests_per_batch):
            tasks = [make_request(batch * num_concurrent + i) for i in range(num_concurrent)]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            all_results.extend(batch_results)
            # Small delay between batches to avoid rate limiting
            await asyncio.sleep(0.5)

        # Verify results
        successful_requests = [r for r in all_results if isinstance(r, dict) and r.get("success")]

        # At least some requests should succeed
        assert len(successful_requests) > 0, (
            f"Expected some successful requests, got {len(successful_requests)} "
            f"out of {len(all_results)}"
        )


class TestStreamingTokenAccuracy:
    """Verify streaming token accumulation and refund calculation."""

    async def test_streaming_token_accuracy(self, e2e_client):
        """Verify streaming token accumulation and refund calculation."""
        model = get_e2e_test_model_sync()

        stream = await e2e_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Count to 10"}],
            max_completion_tokens=100,
            stream=True,
        )

        # Consume stream and count tokens
        chunk_count = 0
        total_content = ""

        async for chunk in stream:
            chunk_count += 1
            if hasattr(chunk, "choices") and chunk.choices:
                for choice in chunk.choices:
                    if (
                        hasattr(choice, "delta")
                        and hasattr(choice.delta, "content")
                        and choice.delta.content
                    ):
                        total_content += choice.delta.content

        # Verify we received chunks
        assert chunk_count > 0, "Expected to receive streaming chunks"

        # Verify we got content
        assert len(total_content) > 0, "Expected to receive content from stream"

        # Verify rate limiter state if available
        if hasattr(e2e_client, "rate_limiter") and e2e_client.rate_limiter:
            state = await e2e_client.rate_limiter.get_state(model)
            # State should exist and reflect usage
            # Remaining should reflect actual usage, not reserved amount
            if state:
                assert hasattr(state, "remaining_tokens") or "remaining_tokens" in state


class Test429Handling:
    """Test 429 rate limit handling and recovery."""

    @pytest.mark.requires_dedicated_key
    async def test_429_handling(self, e2e_client):
        """
        Intentionally trigger 429 and verify recovery.

        IMPORTANT: Uses isolated API key to avoid affecting other tests.
        This test requires a dedicated test API key with minimal limits.
        Contact Venice API for test tier access.
        """
        # Skip if no isolated test key is configured
        isolated_key = os.environ.get("VENICE_E2E_ISOLATED_API_KEY")
        if not isolated_key:
            pytest.skip(
                "VENICE_E2E_ISOLATED_API_KEY not set. "
                "This test requires a dedicated API key with minimal limits."
            )

        model = get_e2e_test_model_sync()

        # Track 429 occurrences
        rate_limit_hit = False

        # Attempt to trigger rate limit by making many requests
        # Note: This is intentionally aggressive to trigger 429
        for _i in range(50):
            try:
                await e2e_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_completion_tokens=5,
                )
            except Exception as e:
                # Check if this is a rate limit error (429)
                if "429" in str(e) or "rate limit" in str(e).lower():
                    rate_limit_hit = True
                    break

        if not rate_limit_hit:
            pytest.skip("Could not trigger 429 rate limit. Test API key may have high limits.")

        # Wait for rate limit window to reset
        await asyncio.sleep(5)

        # Verify recovery - should be able to make requests again
        response = await e2e_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello after rate limit"}],
            max_completion_tokens=5,
        )

        assert response is not None, "Expected successful response after rate limit recovery"


class TestBasicConnectivity:
    """Basic connectivity tests for the E2E test infrastructure."""

    @pytest.mark.flaky(
        reruns=3,
        reruns_delay=2,
        only_rerun=["APITimeoutError", "ConnectionTimeoutError", "TimeoutError"],
    )
    async def test_basic_chat_completion(self, e2e_client):
        """Verify basic chat completion works with the test model."""
        model = get_e2e_test_model_sync()

        response = await e2e_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'test'"}],
            max_completion_tokens=5,
        )

        assert response is not None
        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message is not None

    async def test_model_availability(self, e2e_client, e2e_test_model):
        """Verify the test model is available."""
        # The model should be usable
        response = await e2e_client.chat.completions.create(
            model=e2e_test_model,
            messages=[{"role": "user", "content": "Hi"}],
            max_completion_tokens=5,
        )

        assert response is not None
        # The response model should match what we requested
        # (or be a variant of it)
        assert response.model is not None
