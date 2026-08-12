"""
VCRpy-based integration tests for Scheduler behavior.

This module tests scheduler functionality through real API interactions recorded
with VCRpy, focusing on user-facing behaviors rather than internal implementation.
These tests replace the broken mock-based unit tests in the scheduler directory.

Key behaviors tested:
- Request queuing and processing under load
- Rate limiting and capacity management
- Circuit breaker behavior with failing requests
- Queue overflow handling
- Concurrent request scheduling
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
async def test_scheduler_basic_request_processing(venice_client, model_selector, vcr_cassette):
    """
    Test basic scheduler request processing through public API.

    This test validates that the scheduler properly processes requests
    through the public chat completions endpoint.

    Note: Reasoning models may return empty content but provide reasoning_content.
    The test validates that the scheduler processes the request successfully.
    """
    with vcr_cassette:
        # Dynamically select a suitable chat model
        chat_model = await model_selector.select_chat_model()

        response = await venice_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Say 'Scheduler test 1' and nothing else."}],
            max_completion_tokens=50,  # Increased from 20 to allow full response generation
            temperature=0.1,
        )

    # Validate response structure
    assert response.id is not None
    assert response.model is not None
    assert len(response.choices) > 0
    # Note: content may be None for reasoning models that hit token limits
    # but the scheduler should still process the request successfully
    assert response.choices[0].message.role == "assistant"
    assert response.usage.total_tokens > 0

    # Validate response - allow for reasoning models that return reasoning_content
    message = response.choices[0].message
    has_content = bool(message.content)
    has_reasoning = hasattr(message, "reasoning_content") and bool(message.reasoning_content)

    # The scheduler test validates processing, not specific content
    # Either content or reasoning_content indicates successful processing
    assert has_content or has_reasoning, (
        f"Expected content or reasoning_content from scheduler processing, "
        f"got content='{message.content}', reasoning_content='{getattr(message, 'reasoning_content', None)}'"
    )


@pytest.mark.integration
async def test_scheduler_concurrent_request_handling(venice_client, model_selector, vcr_cassette):
    """
    Test scheduler handling of concurrent requests.

    This validates that the scheduler can properly queue and process
    multiple simultaneous requests without conflicts.
    """
    with vcr_cassette:
        # Dynamically select a suitable chat model
        chat_model = await model_selector.select_chat_model()

        # Submit multiple concurrent requests
        tasks = []
        for i in range(3):
            task = venice_client.chat.completions.create(
                model=chat_model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Say 'Concurrent test {i + 1}' and nothing else.",
                    }
                ],
                max_completion_tokens=64,
                temperature=0.1,
            )
            tasks.append(task)

        # Wait for all requests to complete
        responses = await asyncio.gather(*tasks)

    # Validate that every concurrent request was processed. These scheduler
    # tests exercise concurrency/queuing, not model output quality — a
    # reasoning model may return empty content with reasoning_content under a
    # tight token budget (see test_scheduler_basic_request_processing). Enforce
    # structural success + processing per response; check the echoed phrase only
    # when every model returned visible content (the cheap model pool is
    # non-deterministic and may include reasoning models).
    assert len(responses) == 3

    response_contents = []
    for response in responses:
        assert response.id is not None
        assert response.model is not None
        assert len(response.choices) > 0
        assert response.usage.total_tokens > 0

        message = response.choices[0].message
        content = (message.content or "").strip()
        assert content or getattr(message, "reasoning_content", None), (
            "scheduler returned neither content nor reasoning_content"
        )
        response_contents.append(content)

    # Best-effort echo check: only when all responses returned visible content.
    if all(response_contents):
        expected_texts = ["Concurrent test 1", "Concurrent test 2", "Concurrent test 3"]
        for expected in expected_texts:
            assert any(expected in content for content in response_contents), (
                f"Expected '{expected}' not found in any response: {response_contents}"
            )


@pytest.mark.integration
async def test_scheduler_different_models(venice_client, model_selector, vcr_cassette):
    """
    Test scheduler handling requests to different models.

    This validates that the scheduler properly routes requests
    to different models and manages separate queues/rate limits.
    """
    with vcr_cassette:
        # Dynamically select multiple different models for testing
        available_models = await model_selector.select_models_for_concurrency_test(count=2)

        if len(available_models) < 2:
            pytest.skip("Need at least 2 different models for this test")

        # Test with different models
        tasks = [
            venice_client.chat.completions.create(
                model=available_models[0],
                messages=[{"role": "user", "content": "Say 'Model A test' briefly."}],
                max_completion_tokens=20,
                temperature=0.1,
            ),
            venice_client.chat.completions.create(
                model=available_models[1],
                messages=[{"role": "user", "content": "Say 'Model B test' briefly."}],
                max_completion_tokens=20,
                temperature=0.1,
            ),
        ]

        responses = await asyncio.gather(*tasks)

    # This test verifies the scheduler ROUTES to two different models. Assert
    # on response.model (two distinct models handled), not on echoed content,
    # which a reasoning model may leave empty under a tight token budget. VCR
    # matches chat on method+path only, so under asyncio.gather the two
    # responses can replay in either order — but the set of models is stable.
    assert len(responses) == 2
    for response in responses:
        assert response.id is not None
        assert response.model is not None
        assert len(response.choices) > 0
        assert response.usage.total_tokens > 0

        message = response.choices[0].message
        assert (message.content or "").strip() or getattr(message, "reasoning_content", None), (
            "scheduler returned neither content nor reasoning_content"
        )

    # Two distinct models handled → routing worked.
    assert len({response.model for response in responses}) == 2


@pytest.mark.integration
async def test_scheduler_sequential_requests(venice_client, model_selector, vcr_cassette):
    """
    Test scheduler handling of sequential requests over time.

    This validates that the scheduler maintains state properly
    across multiple sequential requests.
    """
    with vcr_cassette:
        # Dynamically select a suitable chat model
        chat_model = await model_selector.select_chat_model()

        responses = []

        # Submit requests sequentially
        for i in range(3):
            response = await venice_client.chat.completions.create(
                model=chat_model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Say 'Sequential test {i + 1}' and nothing else.",
                    }
                ],
                max_completion_tokens=64,
                temperature=0.1,
            )
            responses.append(response)

    # Validate all responses. Sequential requests replay in order, so the echo
    # check stays per-index — but only when content is visible; a reasoning
    # model may return reasoning_content with empty content under a tight budget
    # (see test_scheduler_basic_request_processing).
    assert len(responses) == 3
    for i, response in enumerate(responses):
        assert response.id is not None
        assert response.model is not None
        assert len(response.choices) > 0
        assert response.usage.total_tokens > 0

        message = response.choices[0].message
        content = (message.content or "").strip()
        assert content or getattr(message, "reasoning_content", None), (
            f"request {i + 1} returned neither content nor reasoning_content"
        )
        if content:
            assert f"Sequential test {i + 1}" in content


@pytest.mark.integration
async def test_scheduler_request_with_parameters(venice_client, model_selector, vcr_cassette):
    """
    Test scheduler with various request parameters.

    This validates that the scheduler properly handles requests
    with different parameters and configurations.
    """
    with vcr_cassette:
        # Dynamically select a suitable chat model
        chat_model = await model_selector.select_chat_model()

        response = await venice_client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Parameter test' and nothing else."},
            ],
            max_completion_tokens=30,
            temperature=0.0,  # Zero temperature for determinism
            venice_parameters={
                "strip_thinking_response": True,
                "disable_thinking": True,
                "enable_web_search": "off",
                "include_venice_system_prompt": False,
            },
        )

    # Validate response
    assert response.id is not None
    assert response.model is not None
    assert len(response.choices) > 0
    assert response.usage.total_tokens > 0

    message = response.choices[0].message
    content = (message.content or "").strip()
    assert content or getattr(message, "reasoning_content", None), (
        "scheduler returned neither content nor reasoning_content"
    )
    if content:
        assert "Parameter test" in content

    # Check Venice parameters were handled
    if hasattr(response, "venice_parameters") and response.venice_parameters:
        assert response.venice_parameters.strip_thinking_response is True
        assert response.venice_parameters.enable_web_search == "off"


@pytest.mark.integration
async def test_scheduler_with_embeddings(venice_client, model_selector, vcr_cassette):
    """
    Test scheduler handling different resource types (embeddings).

    This validates that the scheduler can handle different types
    of API requests beyond just chat completions.
    """
    with vcr_cassette:
        # Dynamically select a suitable embedding model
        embedding_model = await model_selector.select_embedding_model()

        response = await venice_client.embeddings.create(
            model=embedding_model,
            input=["Scheduler embeddings test"],
        )

    # Validate embeddings response
    assert response.model is not None
    assert len(response.data) > 0
    assert len(response.data[0].embedding) > 0
    assert response.usage.total_tokens > 0


@pytest.mark.integration
async def test_scheduler_mixed_resource_types(venice_client, model_selector, vcr_cassette):
    """
    Test scheduler handling mixed resource types concurrently.

    This validates that the scheduler can properly manage
    different types of requests (chat + embeddings) simultaneously.
    """
    with vcr_cassette:
        # Dynamically select suitable models for both types
        chat_model = await model_selector.select_chat_model()
        embedding_model = await model_selector.select_embedding_model()

        # Submit mixed requests concurrently
        chat_task = venice_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Say 'Mixed test chat' briefly."}],
            max_completion_tokens=20,
            temperature=0.1,
        )

        embeddings_task = venice_client.embeddings.create(
            model=embedding_model,
            input=["Mixed test embeddings"],
        )

        chat_response, embeddings_response = await asyncio.gather(chat_task, embeddings_task)

    # Validate chat response was processed (content or reasoning_content)
    assert chat_response.id is not None
    assert len(chat_response.choices) > 0
    message = chat_response.choices[0].message
    content = (message.content or "").strip()
    assert content or getattr(message, "reasoning_content", None), (
        "scheduler returned neither content nor reasoning_content"
    )
    if content:
        assert "Mixed test chat" in content

    # Validate embeddings response
    assert embeddings_response.model is not None
    assert len(embeddings_response.data) > 0
    assert len(embeddings_response.data[0].embedding) > 0


@pytest.mark.integration
async def test_scheduler_error_handling(venice_client, vcr_cassette):
    """
    Test scheduler error handling with invalid requests.

    This validates that the scheduler properly handles
    and propagates errors from the API.
    """
    with vcr_cassette, pytest.raises(VeniceError):
        # Test with invalid model (hardcoded for error testing is acceptable)
        await venice_client.chat.completions.create(
            model="invalid-model-name-that-definitely-does-not-exist",
            messages=[{"role": "user", "content": "This should fail"}],
            max_completion_tokens=20,
        )


@pytest.mark.integration
async def test_scheduler_large_batch_processing(venice_client, model_selector, vcr_cassette):
    """
    Test scheduler handling of larger batches of requests.

    This validates scheduler behavior under higher load
    and tests queue management capabilities.
    """
    with vcr_cassette:
        # Dynamically select a suitable chat model
        chat_model = await model_selector.select_chat_model()

        # Submit a larger batch of requests
        tasks = []
        batch_size = 5

        for i in range(batch_size):
            task = venice_client.chat.completions.create(
                model=chat_model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Say 'Batch {i + 1}' and nothing else.",
                    }
                ],
                max_completion_tokens=64,
                temperature=0.1,
            )
            tasks.append(task)

        # Process all requests
        responses = await asyncio.gather(*tasks)

    # Validate the scheduler handled a larger batch. Enforce structural success
    # + that each request was processed (content or reasoning_content). The
    # strict "every batch number echoed" check only holds when all models
    # returned visible content, so make it best-effort (the cheap model pool is
    # non-deterministic and may include reasoning models).
    assert len(responses) == batch_size
    processed_numbers = set()
    all_visible = True

    for response in responses:
        assert response.id is not None
        assert response.model is not None
        assert len(response.choices) > 0
        assert response.usage.total_tokens > 0

        message = response.choices[0].message
        content = (message.content or "").strip()
        assert content or getattr(message, "reasoning_content", None), (
            "scheduler returned neither content nor reasoning_content"
        )
        if not content:
            all_visible = False
        # Extract batch number from response
        for i in range(1, batch_size + 1):
            if f"Batch {i}" in content:
                processed_numbers.add(i)
                break

    # Strict echo check only when every response returned visible content.
    if all_visible:
        assert len(processed_numbers) == batch_size
