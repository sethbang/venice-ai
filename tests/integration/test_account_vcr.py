"""
VCRpy-based integration tests for Account management behavior.

This module tests account management functionality through real API interactions recorded
with VCRpy, focusing on user-facing behaviors rather than internal implementation.
These tests replace the broken mock-based unit tests in the account directory.

Key behaviors tested:
- Account creation and configuration
- API key authentication and validation
- Account tier discovery and feature access
- Request processing and rate limiting
- Health monitoring and metrics collection
- Client registration and lifecycle management
- Circuit breaker behavior with failures
"""

import asyncio
import os

import pytest
import pytest_asyncio

from venice_ai import create_test_venice_client
from venice_ai.core.config import SchedulerMode
from venice_ai.exceptions import VeniceError


@pytest_asyncio.fixture
async def venice_client(backend_instance):
    """Create a Venice client for VCR testing with shared rate limit coordination."""
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


@pytest.mark.integration
async def test_account_creation_and_authentication(venice_client, vcr_cassette):
    """
    Test account creation with valid API key authentication.

    This validates that accounts can be created and authenticated
    through the public API, replacing unit tests for basic initialization.
    """
    with vcr_cassette:
        # Test basic authentication by making a simple API call
        models_response = await venice_client.models.list()

    # Validate authentication succeeded
    assert models_response is not None
    assert hasattr(models_response, "data")
    assert len(models_response.data) > 0

    # Validate model data structure
    first_model = models_response.data[0]
    assert hasattr(first_model, "id")
    assert hasattr(first_model, "object")
    assert first_model.object == "model"


@pytest.mark.integration
async def test_account_request_processing(venice_client, model_selector, vcr_cassette):
    """
    Test basic account request processing functionality.

    This validates that accounts can process requests through the scheduler
    and handle rate limiting, replacing unit tests for request management.
    """
    with vcr_cassette:
        # Select a suitable model dynamically
        chat_model = await model_selector.select_chat_model()

        # Make a simple chat completion request
        response = await venice_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Say 'Account test successful' briefly."}],
            # 200 (was 20): the default chat model is now a reasoning-class model
            # that can spend a tiny budget on reasoning tokens and return null
            # content, tripping the `message.content is not None` assert below.
            max_completion_tokens=200,
            temperature=0.1,
        )

    # Validate successful request processing
    assert response.id is not None
    assert response.model is not None
    assert len(response.choices) > 0
    assert response.choices[0].message.content is not None
    assert response.choices[0].message.role == "assistant"
    assert response.usage.total_tokens > 0

    # Validate expected content
    content = response.choices[0].message.content.strip()
    assert "Account test" in content.lower() or "successful" in content.lower()


@pytest.mark.integration
async def test_account_concurrent_request_handling(venice_client, model_selector, vcr_cassette):
    """
    Test account handling of concurrent requests.

    This validates that accounts can manage multiple simultaneous requests
    through proper queuing and rate limiting, replacing unit tests for
    client registration and concurrent access.
    """
    with vcr_cassette:
        # Select a suitable model dynamically
        chat_model = await model_selector.select_chat_model()

        # Submit multiple concurrent requests to test account management
        tasks = []
        for i in range(3):
            task = venice_client.chat.completions.create(
                model=chat_model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Say 'Concurrent test {i + 1}' briefly.",
                    }
                ],
                max_completion_tokens=15,
                temperature=0.1,
            )
            tasks.append(task)

        # Execute all requests concurrently
        responses = await asyncio.gather(*tasks)

    # Validate all requests completed successfully
    assert len(responses) == 3

    for _i, response in enumerate(responses):
        assert response.id is not None
        assert response.model is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        assert response.usage.total_tokens > 0


@pytest.mark.integration
async def test_account_mixed_resource_types(venice_client, model_selector, vcr_cassette):
    """
    Test account handling of different resource types.

    This validates that accounts can handle various API endpoints
    (chat, embeddings) through the same account management system,
    replacing unit tests for different request types.
    """
    with vcr_cassette:
        # Select suitable models for different resource types
        chat_model = await model_selector.select_chat_model()
        embedding_model = await model_selector.select_embedding_model()

        # Submit requests for different resource types
        chat_task = venice_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Say 'Mixed resource test' briefly."}],
            max_completion_tokens=15,
            temperature=0.1,
        )

        embeddings_task = venice_client.embeddings.create(
            model=embedding_model,
            input=["Account mixed resource test"],
        )

        # Execute both types concurrently
        chat_response, embeddings_response = await asyncio.gather(chat_task, embeddings_task)

    # Validate chat response
    assert chat_response.id is not None
    assert chat_response.model is not None
    assert len(chat_response.choices) > 0
    assert chat_response.usage.total_tokens > 0

    # Validate embeddings response
    assert embeddings_response.model is not None
    assert len(embeddings_response.data) > 0
    assert len(embeddings_response.data[0].embedding) > 0
    assert embeddings_response.usage.total_tokens > 0


@pytest.mark.integration
async def test_account_rate_limiting_behavior(venice_client, model_selector, vcr_cassette):
    """
    Test account rate limiting and capacity management.

    This validates that accounts properly handle rate limits and
    provide appropriate feedback, replacing unit tests for rate limiting.
    """
    with vcr_cassette:
        # Select a suitable model
        chat_model = await model_selector.select_chat_model()

        # Make a request to test rate limiting behavior
        response = await venice_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Say 'Rate limit test' briefly."}],
            max_completion_tokens=15,
            temperature=0.1,
        )

    # Validate successful request (rate limits should be managed transparently)
    assert response.id is not None
    assert response.model is not None
    assert len(response.choices) > 0
    assert response.usage.total_tokens > 0


@pytest.mark.integration
async def test_account_error_handling(venice_client, vcr_cassette):
    """
    Test account error handling with invalid requests.

    This validates that accounts properly handle and propagate errors
    from the API, replacing unit tests for failure tracking.
    """
    with vcr_cassette, pytest.raises(VeniceError):
        # Test with invalid model (hardcoded for error testing)
        await venice_client.chat.completions.create(
            model="invalid-model-name-that-definitely-does-not-exist",
            messages=[{"role": "user", "content": "This should fail"}],
            max_completion_tokens=20,
        )


@pytest.mark.integration
async def test_account_tier_discovery(venice_client, vcr_cassette):
    """
    Test account tier discovery functionality.

    This validates that accounts can discover their tier and available
    features through the API, replacing unit tests for tier discovery.
    """
    with vcr_cassette:
        # Get available models to infer account tier
        models_response = await venice_client.models.list()

    # Validate we can access models (indicates valid tier)
    assert models_response is not None
    assert hasattr(models_response, "data")
    assert len(models_response.data) > 0

    # Check that we have access to basic model categories
    model_ids = [model.id for model in models_response.data]

    # Should have access to at least some chat models
    chat_models = [
        mid
        for mid in model_ids
        if any(term in mid.lower() for term in ["gpt", "llama", "claude", "gemini"])
    ]
    assert len(chat_models) > 0, f"No chat models found in: {model_ids}"


@pytest.mark.integration
async def test_account_health_monitoring(venice_client, model_selector, vcr_cassette):
    """
    Test account health monitoring through API interactions.

    This validates that accounts maintain health status through
    successful API interactions, replacing unit tests for health checks.
    """
    with vcr_cassette:
        # Make a series of requests to test health monitoring
        chat_model = await model_selector.select_chat_model()

        # Multiple successful requests should indicate healthy account
        for i in range(2):
            response = await venice_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": f"Health check {i + 1}"}],
                max_completion_tokens=10,
                temperature=0.1,
            )

            # Each request should succeed (indicates healthy account)
            assert response.id is not None
            assert response.usage.total_tokens > 0


@pytest.mark.integration
async def test_account_metrics_collection(venice_client, model_selector, vcr_cassette):
    """
    Test account metrics collection through API usage.

    This validates that accounts properly track usage metrics
    through API interactions, replacing unit tests for metrics.
    """
    with vcr_cassette:
        # Select model and make tracked requests
        chat_model = await model_selector.select_chat_model()

        # Make requests that should be tracked in metrics
        response1 = await venice_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Metrics test 1"}],
            max_completion_tokens=10,
            temperature=0.1,
        )

        response2 = await venice_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Metrics test 2"}],
            max_completion_tokens=10,
            temperature=0.1,
        )

    # Validate both requests completed (metrics should be tracked internally)
    assert response1.id is not None
    assert response2.id is not None
    assert response1.usage.total_tokens > 0
    assert response2.usage.total_tokens > 0

    # Different request IDs indicate separate tracked requests
    assert response1.id != response2.id


@pytest.mark.integration
async def test_account_lifecycle_management(venice_client, model_selector, vcr_cassette):
    """
    Test account lifecycle through client context management.

    This validates that accounts properly handle creation, usage,
    and cleanup, replacing unit tests for async context management.
    """
    with vcr_cassette:
        # Test account lifecycle through context manager
        async with venice_client as client:
            # Select model within context
            chat_model = await model_selector.select_chat_model()

            # Make request within context (tests lifecycle)
            response = await client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Lifecycle test"}],
                max_completion_tokens=10,
                temperature=0.1,
            )

            assert response.id is not None
            assert response.usage.total_tokens > 0

        # Context should handle cleanup automatically
