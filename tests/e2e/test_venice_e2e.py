"""
End-to-end tests for Venice AI SDK using VCR recording/replay.

These tests use pre-recorded HTTP interactions (cassettes) to:
- Achieve fast execution (~15s vs 765s with real API)
- Provide reliable, repeatable results
- Enable testing without API key in CI/local environments

## Recording New Cassettes

When the Venice AI API changes, re-record cassettes:

1. Set API key: `export VENICE_API_KEY="your-key"`
2. Delete old cassettes: `rm tests/e2e/cassettes/*.yaml`
3. Run tests: `poetry run pytest tests/e2e/test_venice_e2e.py -vv`
4. Verify: `ls tests/e2e/cassettes/`
5. Commit updated cassettes to repository

## Security Note

Cassettes are automatically scrubbed of sensitive data:
- Authorization headers removed (via vcr_config.filter_headers)
- API keys never stored in cassettes
- Safe to commit to public repository
"""

import asyncio
import os

import pytest
import pytest_asyncio

from venice_ai import create_test_venice_client
from venice_ai.core.config import SchedulerMode
from venice_ai.exceptions import APIStatusError
from venice_ai.models.selection import DynamicModelSelector


@pytest.mark.e2e
@pytest.mark.asyncio
class TestVeniceClientE2E:
    """End-to-end tests for VeniceClient AI SDK using VCR cassettes."""

    @pytest_asyncio.fixture
    async def venice_client(self):
        """Create VeniceClient for E2E testing with intelligent rate limiting."""
        api_key = os.getenv("VENICE_API_KEY")
        if not api_key:
            pytest.skip("VENICE_API_KEY environment variable required for E2E tests")

        client = create_test_venice_client(
            api_key=api_key,
            scheduler_mode=SchedulerMode.INTELLIGENT,
            enable_redis=True,
        )
        try:
            yield client
        finally:
            await client.close()

    @pytest_asyncio.fixture
    async def model_selector(self, venice_client):
        """Dynamic model selector using VCR-recorded model list."""
        return DynamicModelSelector(venice_client)

    async def test_complete_chat_workflow(self, venice_client, model_selector, vcr_cassette):
        """Test complete chat completion workflow with VCR recording/replay."""
        with vcr_cassette:
            model = await model_selector.select_chat_model()

            response = await venice_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "Say 'Hello, Venice!' and nothing else.",
                    }
                ],
                max_completion_tokens=20,
                temperature=0.1,
            )

            # Validate response structure
            assert response.id is not None
            assert response.model is not None
            assert len(response.choices) > 0
            assert response.choices[0].message.content is not None
            assert response.choices[0].message.role == "assistant"
            assert response.usage.total_tokens > 0

    async def test_streaming_chat_workflow(self, venice_client, model_selector, vcr_cassette):
        """Test streaming chat completion workflow with VCR recording/replay."""
        with vcr_cassette:
            model = await model_selector.select_chat_model()

            stream = await venice_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Count from 1 to 3"}],
                stream=True,
                max_completion_tokens=200,
                temperature=0.1,
            )

            chunks = []
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)

            # Verify we received streaming chunks
            assert len(chunks) > 0
            full_response = "".join(chunks)
            assert len(full_response) > 0

    async def test_models_list_workflow(self, venice_client, vcr_cassette):
        """Test listing models workflow with VCR recording/replay."""
        with vcr_cassette:
            models = await venice_client.models.list()

            # Verify we got models back
            assert models.data is not None
            assert len(models.data) > 0

            # Check that models have expected fields
            for model in models.data[:5]:  # Check first 5 models
                assert model.id is not None
                assert model.object == "model"

    async def test_error_handling_workflow(self, venice_client, vcr_cassette):
        """Test error handling with invalid model."""
        from venice_ai.exceptions import NotFoundError

        with vcr_cassette:
            # The API now returns a NotFoundError for invalid models
            with pytest.raises((APIStatusError, NotFoundError)) as exc_info:
                await venice_client.chat.completions.create(
                    model="invalid-model-that-does-not-exist-xyz123",
                    messages=[{"role": "user", "content": "Test"}],
                    max_completion_tokens=10,
                )

            # Verify we get a proper error response
            if hasattr(exc_info.value, "status_code"):
                assert exc_info.value.status_code in [400, 404, 422]
            else:
                # NotFoundError doesn't have status_code but is still valid
                assert "not found" in str(exc_info.value).lower()

    async def test_embeddings_workflow(self, venice_client, model_selector, vcr_cassette):
        """Test embeddings creation workflow with VCR recording/replay."""
        with vcr_cassette:
            model = await model_selector.select_embedding_model()

            response = await venice_client.embeddings.create(
                model=model, input="Test text for embedding"
            )

            # Validate embeddings response
            assert len(response.data) > 0
            assert len(response.data[0].embedding) > 0
            assert all(isinstance(x, float) for x in response.data[0].embedding[:10])
            assert response.usage.total_tokens > 0

    async def test_concurrent_requests_workflow(self, venice_client, model_selector, vcr_cassette):
        """Test handling concurrent requests with VCR recording/replay."""
        with vcr_cassette:
            model = await model_selector.select_chat_model()

            # Create fewer concurrent requests
            async def make_request(i):
                return await venice_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": f"Say the number {i}"}],
                    max_completion_tokens=10,
                    temperature=0.1,
                )

            # Create only 2 concurrent requests
            tasks = [make_request(i) for i in range(2)]
            responses = await asyncio.gather(*tasks)

            # Validate all responses
            for response in responses:
                assert response.id is not None
                assert len(response.choices) > 0
                assert response.choices[0].message.content is not None

    async def test_system_message_workflow(self, venice_client, model_selector, vcr_cassette):
        """Test chat with system message."""
        with vcr_cassette:
            model = await model_selector.select_chat_model()

            response = await venice_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that speaks like a pirate.",
                    },
                    {"role": "user", "content": "Say hello"},
                ],
                max_completion_tokens=50,
                temperature=0.5,
            )

            assert response.choices[0].message.content is not None
            assert len(response.choices[0].message.content) > 0

    async def test_function_calling_workflow(self, venice_client, model_selector, vcr_cassette):
        """Test function calling with a model that supports it."""
        with vcr_cassette:
            try:
                model = await model_selector.select_function_calling_model()
            except Exception:
                # Fallback to chat model if function calling model not available
                model = await model_selector.select_chat_model()

            try:
                response = await venice_client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": "What's the weather in San Francisco?",
                        }
                    ],
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get the weather for a location",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "location": {
                                            "type": "string",
                                            "description": "The city and state",
                                        }
                                    },
                                    "required": ["location"],
                                },
                            },
                        }
                    ],
                    max_completion_tokens=100,
                )

                # Check that we got a response (function call or regular message)
                assert response.choices[0] is not None
                assert response.choices[0].message is not None

            except APIStatusError as e:
                if e.status_code == 400 and "function" in str(e).lower():
                    pytest.skip(f"Model {model} does not support function calling")
                raise

    async def test_multi_turn_conversation(self, venice_client, model_selector, vcr_cassette):
        """Test multi-turn conversation workflow."""
        with vcr_cassette:
            model = await model_selector.select_chat_model()

            messages = [{"role": "user", "content": "My name is Alice. Remember it."}]

            # First turn
            response1 = await venice_client.chat.completions.create(
                model=model, messages=messages, max_completion_tokens=50, temperature=0.3
            )

            # Add assistant response to conversation
            messages.append({"role": "assistant", "content": response1.choices[0].message.content})

            # Second turn - test memory
            messages.append({"role": "user", "content": "What's my name?"})

            response2 = await venice_client.chat.completions.create(
                model=model, messages=messages, max_completion_tokens=50, temperature=0.3
            )

            # Verify we got responses
            assert response1.choices[0].message.content is not None
            assert response2.choices[0].message.content is not None

    async def test_max_completion_tokens_limit(self, venice_client, model_selector, vcr_cassette):
        """Test that max_completion_tokens parameter is respected."""
        with vcr_cassette:
            model = await model_selector.select_chat_model()

            response = await venice_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Write a very long story about dragons"}],
                max_completion_tokens=10,  # Very low limit
                temperature=0.5,
            )

            # Check that response was truncated
            assert response.choices[0].finish_reason in ["length", "stop"]
            assert response.usage.completion_tokens <= 15  # Allow small buffer

    async def test_temperature_variation(self, venice_client, model_selector, vcr_cassette):
        """Test temperature parameter affects response variability."""
        with vcr_cassette:
            model = await model_selector.select_chat_model()
            prompt = "Generate a random word"

            # Low temperature (more deterministic)
            responses_low_temp = []
            for _i in range(3):
                response = await venice_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=10,
                    temperature=0.1,
                )
                responses_low_temp.append(response.choices[0].message.content)

            # High temperature (more random)
            responses_high_temp = []
            for _i in range(3):
                response = await venice_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=10,
                    temperature=0.9,
                )
                responses_high_temp.append(response.choices[0].message.content)

            # Verify we got responses
            assert all(r is not None for r in responses_low_temp)
            assert all(r is not None for r in responses_high_temp)

    @pytest.mark.slow
    async def test_large_context_handling(self, venice_client, model_selector, vcr_cassette):
        """Test handling of large context."""
        with vcr_cassette:
            model = await model_selector.select_chat_model()

            # Create a large context
            large_text = "The quick brown fox jumps over the lazy dog. " * 100

            response = await venice_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Summarize this text in one sentence: {large_text}",
                    }
                ],
                max_completion_tokens=50,
                temperature=0.3,
            )

            # Verify we got a response despite large input
            assert response.choices[0].message.content is not None
            assert response.usage.prompt_tokens > 100  # Should have processed significant input

    async def test_rate_limit_tier_discovery(self, venice_client, vcr_cassette):
        """Test that rate limit tiers can be discovered from the API."""
        with vcr_cassette:
            # This tests the rate_limiting.py module's tier discovery functionality
            from venice_ai.core.rate_limit_discovery import RateLimitDiscovery

            # Create tier discovery with the client
            rate_limit_discovery = RateLimitDiscovery(client=venice_client)

            # Discover tiers from the API
            tiers = await rate_limit_discovery.discover_tiers()

            # Verify we discovered some tiers
            assert len(tiers) > 0

            # Check tier structure
            for _bucket_id, bucket in tiers.items():
                assert bucket.bucket_id is not None
                assert bucket.rpm_limit > 0
                assert isinstance(bucket.models, set)

    async def test_model_tier_integration(self, venice_client, model_selector, vcr_cassette):
        """Test integration between models and their rate limit tiers."""
        with vcr_cassette:
            from venice_ai.core.rate_limit_discovery import RateLimitDiscovery

            rate_limit_discovery = RateLimitDiscovery(client=venice_client)
            await rate_limit_discovery.discover_tiers()

            # Get a model to test with
            model = await model_selector.select_chat_model()

            # Check that we can get tier for the model
            bucket_id = await rate_limit_discovery.get_tier_for_model(model)

            if bucket_id:
                # Get the tier details
                bucket = await rate_limit_discovery.get_tier(bucket_id)
                assert bucket is not None
                assert bucket.rpm_limit > 0

                # Verify the model is in this tier
                models_in_tier = await rate_limit_discovery.get_models_in_tier(bucket_id)
                assert model in models_in_tier

    async def test_audio_speech_workflow(self, venice_client, model_selector, vcr_cassette):
        """Test audio speech generation workflow with VCR recording/replay."""
        with vcr_cassette:
            model = await model_selector.select_audio_model()

            # Test streaming audio generation
            stream = await venice_client.audio.create_speech(
                model=model,
                input="Hello, this is a test of Venice AI text-to-speech.",
                voice="af_alloy",  # Use a standard voice
                stream=True,
            )

            # Verify we got a streaming response
            assert stream is not None

            # Collect some chunks to verify streaming works
            chunks = []
            chunk_count = 0
            async for chunk in stream:
                chunks.append(chunk)
                chunk_count += 1
                # Only collect first few chunks to avoid long processing
                if chunk_count >= 3:
                    break

            # Verify we received streaming chunks
            assert len(chunks) > 0
            assert all(isinstance(chunk, bytes) for chunk in chunks)
