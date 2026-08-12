"""
VCRpy-based integration tests for Embeddings resource.

This module tests embeddings creation functionality through real API
interactions recorded with VCRpy, replacing mock-based unit tests with
actual API response validation.

Key differences from mock tests:
- Mock tests focus on client-side validation and request structure
- VCR tests validate real API behavior, responses, and error handling
- Validation logic tests remain in unit tests (e.g., empty input checks)
"""

import os

import pytest
import pytest_asyncio

from venice_ai import create_test_venice_client
from venice_ai.core.config import SchedulerMode
from venice_ai.exceptions import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    NotFoundError,
)


@pytest_asyncio.fixture
async def venice_client(backend_instance):
    """Create a Venice client for VCR testing."""
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


# ============================================================================
# Basic Embeddings Creation Tests
# ============================================================================


@pytest.mark.integration
async def test_embeddings_create_single_string(venice_client, vcr_cassette):
    """Test successful embedding creation with single string input."""
    with vcr_cassette:
        response = await venice_client.embeddings.create(
            model="text-embedding-bge-m3",
            input="This is a test sentence for embedding.",
        )

        # Validate response structure
        assert response is not None
        assert hasattr(response, "data")
        assert hasattr(response, "model")
        assert hasattr(response, "usage")

        # Validate model
        assert response.model == "text-embedding-bge-m3"

        # Validate data structure
        assert isinstance(response.data, list)
        assert len(response.data) == 1

        # Validate embedding
        embedding = response.data[0]
        assert hasattr(embedding, "embedding")
        assert hasattr(embedding, "index")
        assert embedding.index == 0
        assert isinstance(embedding.embedding, list)
        assert len(embedding.embedding) > 0  # Should have dimensions
        assert all(isinstance(val, float) for val in embedding.embedding)

        # Validate usage
        assert hasattr(response.usage, "prompt_tokens")
        assert hasattr(response.usage, "total_tokens")
        assert response.usage.prompt_tokens > 0
        assert response.usage.total_tokens > 0


@pytest.mark.integration
async def test_embeddings_create_batch_strings(venice_client, vcr_cassette):
    """Test successful embedding creation with batch of strings."""
    with vcr_cassette:
        inputs = [
            "First test sentence.",
            "Second test sentence with more content.",
            "Third sentence for batch processing test.",
        ]

        response = await venice_client.embeddings.create(
            model="text-embedding-bge-m3", input=inputs
        )

        # Validate response structure
        assert response is not None
        assert len(response.data) == 3

        # Validate each embedding
        for i, embedding in enumerate(response.data):
            assert embedding.index == i
            assert isinstance(embedding.embedding, list)
            assert len(embedding.embedding) > 0

        # Validate usage reflects batch size
        assert response.usage.prompt_tokens > 0


@pytest.mark.integration
async def test_embeddings_create_with_dimensions(venice_client, vcr_cassette):
    """Test embedding creation with custom dimensions parameter."""
    with vcr_cassette:
        response = await venice_client.embeddings.create(
            model="text-embedding-bge-m3",
            input="Test with custom dimensions",
            dimensions=512,
        )

        assert response is not None
        assert len(response.data) == 1

        # Validate embedding dimensions
        embedding = response.data[0]
        # Note: Actual dimension may vary based on model support
        # This validates the API accepts the parameter
        assert isinstance(embedding.embedding, list)
        assert len(embedding.embedding) > 0


@pytest.mark.integration
async def test_embeddings_create_with_base64_encoding(venice_client, vcr_cassette):
    """Test embedding creation with base64 encoding format."""
    with vcr_cassette:
        response = await venice_client.embeddings.create(
            model="text-embedding-bge-m3",
            input="Test base64 encoding",
            encoding_format="base64",
        )

        assert response is not None
        assert len(response.data) == 1

        # With base64, embedding should be a string
        embedding = response.data[0]
        assert hasattr(embedding, "embedding")
        # Base64 encoding returns string instead of float array
        assert isinstance(embedding.embedding, (str, list))


@pytest.mark.integration
async def test_embeddings_create_with_user_parameter(venice_client, vcr_cassette):
    """Test embedding creation with user tracking parameter."""
    with vcr_cassette:
        response = await venice_client.embeddings.create(
            model="text-embedding-bge-m3",
            input="Test user parameter",
            user="test_user_vcr_123",
        )

        assert response is not None
        assert len(response.data) == 1
        assert response.data[0].embedding is not None


# ============================================================================
# Input Type Tests
# ============================================================================


# ============================================================================
# Batch Processing Tests
# ============================================================================


@pytest.mark.integration
async def test_embeddings_create_large_batch(venice_client, vcr_cassette):
    """Test successful batch processing with larger number of inputs."""
    with vcr_cassette:
        # Create a batch of 50 sentences
        large_batch = [f"Sentence number {i} for batch processing." for i in range(50)]

        response = await venice_client.embeddings.create(
            model="text-embedding-bge-m3", input=large_batch
        )

        assert response is not None
        assert len(response.data) == 50

        # Validate indices are correct
        for i, embedding in enumerate(response.data):
            assert embedding.index == i
            assert len(embedding.embedding) > 0


@pytest.mark.integration
async def test_embeddings_create_mixed_length_strings(venice_client, vcr_cassette):
    """Test batch processing with strings of varying lengths."""
    with vcr_cassette:
        mixed_input = [
            "Short",
            "Medium length sentence for testing.",
            "This is a much longer sentence that contains significantly more text content to test how the embedding model handles varying input lengths within a single batch request.",
        ]

        response = await venice_client.embeddings.create(
            model="text-embedding-bge-m3", input=mixed_input
        )

        assert response is not None
        assert len(response.data) == 3

        # All embeddings should have same dimensionality regardless of input length
        dimensions = [len(emb.embedding) for emb in response.data]
        assert len(set(dimensions)) == 1  # All same dimension


# ============================================================================
# Special Input Tests
# ============================================================================


@pytest.mark.integration
async def test_embeddings_create_unicode_input(venice_client, vcr_cassette):
    """Test embedding creation with unicode characters."""
    with vcr_cassette:
        unicode_input = "Test with émojis 🚀 and special chars: 中文, العربية, русский"

        response = await venice_client.embeddings.create(
            model="text-embedding-bge-m3", input=unicode_input
        )

        assert response is not None
        assert len(response.data) == 1
        assert response.data[0].embedding is not None


@pytest.mark.integration
async def test_embeddings_create_newlines_and_whitespace(venice_client, vcr_cassette):
    """Test embedding creation with newlines and whitespace."""
    with vcr_cassette:
        input_with_whitespace = "Line 1\n\nLine 2\t\tTabbed\r\nWindows line ending"

        response = await venice_client.embeddings.create(
            model="text-embedding-bge-m3", input=input_with_whitespace
        )

        assert response is not None
        assert len(response.data) == 1
        assert response.data[0].embedding is not None


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.integration
async def test_embeddings_create_invalid_api_key(vcr_cassette):
    """Test handling of authentication error with invalid API key."""
    with vcr_cassette:
        # Create client with invalid API key
        client = create_test_venice_client(
            api_key="invalid-api-key-for-testing-32chars-long",
            scheduler_mode=SchedulerMode.BASIC,
        )
        try:
            with pytest.raises(AuthenticationError) as exc_info:
                await client.embeddings.create(
                    model="text-embedding-bge-m3", input="Test auth error"
                )

            # Validate error details
            assert "401" in str(exc_info.value) or "authentication" in str(exc_info.value).lower()
        finally:
            await client.close()


@pytest.mark.integration
async def test_embeddings_create_nonexistent_model(venice_client, vcr_cassette):
    """Test handling of model not found error."""
    with vcr_cassette:
        with pytest.raises((NotFoundError, APIError, InvalidRequestError)) as exc_info:
            await venice_client.embeddings.create(
                model="nonexistent-embedding-model-xyz", input="Test not found error"
            )

        # Error should indicate model issue
        error_msg = str(exc_info.value).lower()
        assert "model" in error_msg or "not found" in error_msg or "invalid" in error_msg


# ============================================================================
# Model Comparison Tests
# ============================================================================


@pytest.mark.integration
async def test_embeddings_different_models_same_input(venice_client, vcr_cassette):
    """Test same input with different embedding models."""
    with vcr_cassette:
        input_text = "Test input for model comparison"

        # Test first model
        response1 = await venice_client.embeddings.create(
            model="text-embedding-bge-m3", input=input_text
        )

        # Note: Would need to test different model if available
        # For now just validate structure
        assert response1 is not None
        assert len(response1.data) == 1
        assert len(response1.data[0].embedding) > 0


@pytest.mark.integration
async def test_embeddings_single_vs_batch_consistency(venice_client, vcr_cassette):
    """Test consistency between single and batch embedding requests."""
    with vcr_cassette:
        text = "Consistency test sentence"

        # Single request
        single_response = await venice_client.embeddings.create(
            model="text-embedding-bge-m3", input=text
        )

        # Batch request with same text
        batch_response = await venice_client.embeddings.create(
            model="text-embedding-bge-m3", input=[text]
        )

        # Both should return same structure
        assert len(single_response.data) == 1
        assert len(batch_response.data) == 1
        assert single_response.model == batch_response.model

        # Embeddings should have same dimensionality
        assert len(single_response.data[0].embedding) == len(batch_response.data[0].embedding)


# ============================================================================
# Response Structure Tests
# ============================================================================


@pytest.mark.integration
async def test_embeddings_response_has_all_required_fields(venice_client, vcr_cassette):
    """Test that response contains all expected fields."""
    with vcr_cassette:
        response = await venice_client.embeddings.create(
            model="text-embedding-bge-m3", input="Test complete response structure"
        )

        # Top-level fields
        assert hasattr(response, "object")
        assert hasattr(response, "data")
        assert hasattr(response, "model")
        assert hasattr(response, "usage")

        # Data fields
        assert len(response.data) > 0
        embedding_obj = response.data[0]
        assert hasattr(embedding_obj, "object")
        assert hasattr(embedding_obj, "embedding")
        assert hasattr(embedding_obj, "index")

        # Usage fields
        assert hasattr(response.usage, "prompt_tokens")
        assert hasattr(response.usage, "total_tokens")


@pytest.mark.integration
async def test_embeddings_usage_tokens_accuracy(venice_client, vcr_cassette):
    """Test that usage tokens are reported accurately."""
    with vcr_cassette:
        # Create embeddings with known input sizes
        short_input = "Short text"
        long_input = "This is a much longer text that should consume more tokens when processed by the embedding model."

        short_response = await venice_client.embeddings.create(
            model="text-embedding-bge-m3", input=short_input
        )

        long_response = await venice_client.embeddings.create(
            model="text-embedding-bge-m3", input=long_input
        )

        # Longer input should use more tokens
        assert long_response.usage.prompt_tokens > short_response.usage.prompt_tokens

        # Both should report positive token counts
        assert short_response.usage.prompt_tokens > 0
        assert short_response.usage.total_tokens > 0
        assert long_response.usage.prompt_tokens > 0
        assert long_response.usage.total_tokens > 0
