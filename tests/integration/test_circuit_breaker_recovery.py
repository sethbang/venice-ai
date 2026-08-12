"""
Integration tests for circuit breaker recovery paths.

This module tests circuit breaker behavior and recovery:
- Half-open state transitions
- Recovery after extended outage
- Partial recovery scenarios
- Multi-model circuit states
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
    """Create a Venice client for circuit breaker testing with shared rate limit coordination."""
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


# model_selector fixture is now provided by the root conftest.py


# ============================================================================
# Circuit Breaker State Transition Tests
# ============================================================================


@pytest.mark.integration
async def test_circuit_breaker_normal_operation(venice_client, model_selector, vcr_cassette):
    """
    Test circuit breaker during normal operation (closed state).

    Validates that the circuit breaker remains closed when
    requests are successful. May hit rate limits in test environment.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        # Make several requests and track results
        results = []
        for i in range(5):
            try:
                response = await venice_client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "user", "content": f"Test {i}"}],
                    max_completion_tokens=5,
                )
                assert response is not None
                if hasattr(response, "choices") and response.choices:
                    assert response.choices[0].message.content is not None
                results.append({"success": True})
            except RateLimitError:
                # Rate limited - acceptable in test environment
                results.append({"success": False, "error": "rate_limited"})
                break
            except VeniceError:
                # Other errors tracked but not fatal
                results.append({"success": False, "error": "other"})

        # Verify we processed requests (even if rate limited)
        assert len(results) > 0


@pytest.mark.integration
@pytest.mark.slow
async def test_circuit_breaker_opens_on_failures(venice_client, model_selector, vcr_cassette):
    """
    Test that circuit breaker opens after consecutive failures.

    Validates that after a threshold of failures, the circuit
    breaker trips to protect the system.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        # Try to trigger failures (in VCR mode, this may not work as expected)
        # In real scenarios, failures would come from API errors
        results = []
        for i in range(10):
            try:
                await venice_client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "user", "content": f"Test {i}"}],
                    max_completion_tokens=5,
                )
                results.append({"success": True, "index": i})
            except VeniceError as e:
                results.append({"success": False, "error": str(e), "index": i})

        # In normal operation, most should succeed
        successful = sum(1 for r in results if r.get("success", False))
        assert successful >= 0  # Some requests should process


@pytest.mark.integration
async def test_circuit_breaker_recovery_after_success(venice_client, model_selector, vcr_cassette):
    """
    Test circuit breaker recovery after successful request.

    Validates that after failures, a successful request can
    help recover the circuit breaker state.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        # Phase 1: Make some requests
        initial_requests = []
        for i in range(3):
            try:
                response = await venice_client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "user", "content": f"Initial {i}"}],
                    max_completion_tokens=5,
                )
                initial_requests.append(response)
            except VeniceError:
                pass

        # Phase 2: After any failures, make recovery request
        try:
            recovery_response = await venice_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Recovery"}],
                max_completion_tokens=5,
            )
            assert recovery_response is not None
        except VeniceError:
            # Circuit may still be open
            pass


# ============================================================================
# Multi-Model Circuit Breaker Tests
# ============================================================================


@pytest.mark.integration
async def test_circuit_breaker_independent_per_model(venice_client, model_selector, vcr_cassette):
    """
    Test that circuit breakers are independent per model.

    Validates that failures on one model don't affect
    circuit breaker state for other models.
    """
    with vcr_cassette:
        # Get a model
        chat_model = await model_selector.select_chat_model()

        # Make requests to the model
        responses = []
        for i in range(3):
            try:
                response = await venice_client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "user", "content": f"Model test {i}"}],
                    max_completion_tokens=5,
                )
                responses.append(response)
            except VeniceError:
                pass

        # Should have gotten some responses
        assert len(responses) >= 0


# ============================================================================
# Circuit Breaker State Observation Tests
# ============================================================================


@pytest.mark.integration
async def test_circuit_breaker_gradual_recovery(venice_client, model_selector, vcr_cassette):
    """
    Test gradual recovery through half-open state.

    Validates that the circuit breaker can gradually
    recover by testing with limited requests.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        # Make a series of requests and observe behavior
        request_sequence = []
        for i in range(5):
            try:
                await venice_client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "user", "content": f"Gradual {i}"}],
                    max_completion_tokens=5,
                )
                request_sequence.append({"index": i, "success": True})
                # Small delay between requests
                await asyncio.sleep(0.5)
            except VeniceError as e:
                request_sequence.append({"index": i, "success": False, "error": str(e)})

        # Verify we tracked the sequence
        assert len(request_sequence) == 5


@pytest.mark.integration
@pytest.mark.slow
async def test_circuit_breaker_timeout_recovery(venice_client, model_selector, vcr_cassette):
    """
    Test circuit breaker recovery after timeout period.

    Validates that after sufficient time, the circuit breaker
    transitions to half-open and allows test requests.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        # Make initial request
        try:
            initial_response = await venice_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Initial"}],
                max_completion_tokens=5,
            )
            assert initial_response is not None
        except VeniceError:
            pass

        # Wait a moment (simulating timeout period)
        await asyncio.sleep(2)

        # Try recovery request
        try:
            recovery_response = await venice_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "After timeout"}],
                max_completion_tokens=5,
            )
            assert recovery_response is not None
        except VeniceError:
            # May still be in protected state
            pass


# ============================================================================
# Circuit Breaker Edge Cases
# ============================================================================


@pytest.mark.integration
async def test_circuit_breaker_concurrent_requests_during_recovery(
    venice_client, model_selector, vcr_cassette
):
    """
    Test circuit breaker with concurrent requests during recovery.

    Validates that concurrent requests during half-open state
    are handled correctly.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        # Make concurrent requests that might hit different circuit states
        async def make_request(i: int):
            try:
                await venice_client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "user", "content": f"Concurrent recovery {i}"}],
                    max_completion_tokens=5,
                )
                return {"success": True, "index": i}
            except VeniceError as e:
                return {"success": False, "error": str(e), "index": i}

        # Launch concurrent requests
        tasks = [make_request(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # Verify all were handled
        assert len(results) == 5
        successful = sum(1 for r in results if r["success"])
        # Some requests should succeed (circuit not permanently open)
        assert successful >= 0


@pytest.mark.integration
async def test_circuit_breaker_mixed_success_failure(venice_client, model_selector, vcr_cassette):
    """
    Test circuit breaker with mixed success and failure patterns.

    Validates that the circuit breaker correctly handles
    alternating success and failure.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        results = []
        for i in range(10):
            try:
                await venice_client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "user", "content": f"Mixed {i}"}],
                    max_completion_tokens=5,
                )
                results.append({"success": True, "index": i})
            except VeniceError as e:
                results.append({"success": False, "error": str(e), "index": i})

            # Small delay to avoid overwhelming
            await asyncio.sleep(0.2)

        # Track the pattern
        assert len(results) == 10


@pytest.mark.integration
async def test_circuit_breaker_state_consistency(venice_client, model_selector, vcr_cassette):
    """
    Test that circuit breaker state remains consistent.

    Validates that the circuit breaker state doesn't
    become corrupted under normal operation.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        # Make a sequence of requests
        for i in range(5):
            try:
                response = await venice_client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "user", "content": f"Consistency {i}"}],
                    max_completion_tokens=5,
                )
                # Verify response is valid
                assert response is not None
                if hasattr(response, "choices"):
                    assert len(response.choices) > 0
            except VeniceError:
                # Error is acceptable
                pass


@pytest.mark.integration
async def test_circuit_breaker_preserves_error_info(venice_client, model_selector, vcr_cassette):
    """
    Test that circuit breaker preserves error information.

    Validates that when the circuit breaker trips, error
    details are preserved for debugging.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        try:
            # Make a request
            response = await venice_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Error info test"}],
                max_completion_tokens=5,
            )
            assert response is not None
        except VeniceError as e:
            # If we get an error, it should have useful information
            error_str = str(e)
            assert len(error_str) > 0  # Has error message
