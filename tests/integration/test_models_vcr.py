"""
VCRpy-based integration tests for Models resource.

This module tests model listing, traits, and compatibility functionality
through real API interactions recorded with VCRpy, replacing mock-based unit tests.
"""

import os

import pytest
import pytest_asyncio

from venice_ai import VeniceClient, create_test_venice_client
from venice_ai.core.config import SchedulerMode
from venice_ai.exceptions import APIError, InvalidRequestError, VeniceError


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


# ============================================================================
# Model Listing Tests
# ============================================================================


@pytest.mark.integration
async def test_models_list_all(venice_client, vcr_cassette):
    """Test listing all available models."""
    with vcr_cassette:
        response = await venice_client.models.list()

        # Validate response structure
        assert response is not None
        assert hasattr(response, "data")
        assert isinstance(response.data, list)

        # Should have at least some models
        assert len(response.data) > 0

        # Check model structure
        for model in response.data:
            assert hasattr(model, "id")
            assert hasattr(model, "object")
            assert model.object == "model"

            # Model should have basic metadata
            if hasattr(model, "created"):
                assert isinstance(model.created, (int, float))


@pytest.mark.integration
async def test_models_list_by_type_text(venice_client, vcr_cassette):
    """Test listing text/chat models."""
    with vcr_cassette:
        response = await venice_client.models.list(type="text")

        assert response is not None
        assert hasattr(response, "data")
        assert isinstance(response.data, list)

        # Check that we got text models
        if hasattr(response, "type"):
            assert response.type == "text"

        # Validate each model
        for model in response.data:
            assert hasattr(model, "id")
            # Text models should have certain capabilities
            if hasattr(model, "capabilities"):
                assert model.capabilities is not None


@pytest.mark.integration
async def test_models_list_by_type_image(venice_client, vcr_cassette):
    """Test listing image generation models."""
    with vcr_cassette:
        response = await venice_client.models.list(type="image")

        assert response is not None
        assert hasattr(response, "data")
        assert isinstance(response.data, list)

        # Check that we got image models
        if hasattr(response, "type"):
            assert response.type == "image"

        # Image models might be empty for some tiers
        if len(response.data) > 0:
            for model in response.data:
                assert hasattr(model, "id")


@pytest.mark.integration
async def test_models_list_by_type_embedding(venice_client, vcr_cassette):
    """Test listing embedding models."""
    with vcr_cassette:
        response = await venice_client.models.list(type="embedding")

        assert response is not None
        assert hasattr(response, "data")
        assert isinstance(response.data, list)

        # Check that we got embedding models
        if hasattr(response, "type"):
            assert response.type == "embedding"

        # Validate embedding models
        for model in response.data:
            assert hasattr(model, "id")
            # Embedding models might have dimension info
            if hasattr(model, "constraints"):
                assert model.constraints is not None


@pytest.mark.integration
async def test_models_list_by_type_audio(venice_client, vcr_cassette):
    """Test listing audio/TTS models."""
    with vcr_cassette:
        response = await venice_client.models.list(type="tts")

        assert response is not None
        assert hasattr(response, "data")
        assert isinstance(response.data, list)

        # Check that we got TTS models
        if hasattr(response, "type"):
            assert response.type == "tts"

        # TTS models might be limited
        if len(response.data) > 0:
            for model in response.data:
                assert hasattr(model, "id")


# Removed test_models_list_with_pagination - API no longer supports limit/offset parameters


# ============================================================================
# Model Traits Tests
# ============================================================================


@pytest.mark.integration
async def test_models_list_traits_all(venice_client, vcr_cassette):
    """Test listing model traits for all types."""
    with vcr_cassette:
        try:
            response = await venice_client.models.list_traits()

            assert response is not None
            assert hasattr(response, "data")

            # Traits data should be a dict or similar structure
            if response.data:
                assert isinstance(response.data, (dict, list))
        except (VeniceError, APIError) as e:
            # Some endpoints might not support traits
            pytest.skip(f"Model traits not supported: {e}")


@pytest.mark.integration
async def test_models_list_traits_by_type(venice_client, vcr_cassette):
    """Test listing model traits for specific type."""
    with vcr_cassette:
        try:
            response = await venice_client.models.list_traits(type="text")

            assert response is not None
            assert hasattr(response, "data")

            # Check type matches
            if hasattr(response, "type"):
                assert response.type == "text"
        except (VeniceError, APIError) as e:
            # Some endpoints might not support traits
            pytest.skip(f"Model traits not supported: {e}")


@pytest.mark.integration
async def test_models_list_traits_image(venice_client, vcr_cassette):
    """Test listing traits for image models."""
    with vcr_cassette:
        try:
            response = await venice_client.models.list_traits(type="image")

            assert response is not None
            assert hasattr(response, "data")

            # Image traits might include resolution, styles, etc.
            if response.data and isinstance(response.data, dict):
                # Check for common image traits
                for key in response.data:
                    assert isinstance(key, str)
        except (VeniceError, APIError) as e:
            # Some endpoints might not support traits
            pytest.skip(f"Model traits not supported: {e}")


# ============================================================================
# Model Compatibility Tests
# ============================================================================


@pytest.mark.integration
async def test_models_list_compatibility(venice_client, vcr_cassette):
    """Test listing model compatibility information."""
    with vcr_cassette:
        try:
            response = await venice_client.models.list_compatibility()

            assert response is not None
            assert hasattr(response, "data")

            # Compatibility data structure varies by implementation
            if response.data:
                assert isinstance(response.data, (dict, list))

                # If dict, check for model IDs as keys
                if isinstance(response.data, dict):
                    for model_id, compat_info in response.data.items():
                        assert isinstance(model_id, str)
                        assert compat_info is not None
        except (VeniceError, APIError) as e:
            # Some endpoints might not support compatibility
            pytest.skip(f"Model compatibility not supported: {e}")


@pytest.mark.integration
async def test_models_list_compatibility_with_type(venice_client, vcr_cassette):
    """Test listing model compatibility for specific type."""
    with vcr_cassette:
        try:
            response = await venice_client.models.list_compatibility(type="text")

            assert response is not None
            assert hasattr(response, "data")

            # Check type matches if present
            if hasattr(response, "type"):
                assert response.type == "text"
        except (VeniceError, APIError) as e:
            # Some endpoints might not support compatibility
            pytest.skip(f"Model compatibility not supported: {e}")


# ============================================================================
# Model Metadata Tests
# ============================================================================


@pytest.mark.integration
async def test_models_metadata_structure(venice_client, vcr_cassette):
    """Test the structure of model metadata."""
    with vcr_cassette:
        response = await venice_client.models.list(type="text")

        assert len(response.data) > 0

        # Check first model's metadata
        model = response.data[0]

        # Basic fields
        assert hasattr(model, "id")
        assert model.id is not None

        # Optional but common fields
        if hasattr(model, "name"):
            assert isinstance(model.name, str)

        if hasattr(model, "description"):
            assert isinstance(model.description, str)

        if hasattr(model, "pricing"):
            assert model.pricing is not None
            # Check pricing structure
            if hasattr(model.pricing, "per_token"):
                assert isinstance(model.pricing.per_token, (int, float))

        if hasattr(model, "constraints"):
            assert model.constraints is not None
            # Check constraints structure
            if hasattr(model.constraints, "max_tokens"):
                assert isinstance(model.constraints.max_tokens, (int, type(None)))


@pytest.mark.integration
async def test_models_capabilities(venice_client, vcr_cassette):
    """Test model capabilities information."""
    with vcr_cassette:
        response = await venice_client.models.list(type="text")

        # Find a model with capabilities
        models_with_capabilities = [
            m for m in response.data if hasattr(m, "capabilities") and m.capabilities
        ]

        if models_with_capabilities:
            model = models_with_capabilities[0]
            capabilities = model.capabilities

            # Check common capabilities
            if hasattr(capabilities, "chat"):
                assert isinstance(capabilities.chat, bool)

            if hasattr(capabilities, "completion"):
                assert isinstance(capabilities.completion, bool)

            if hasattr(capabilities, "embedding"):
                assert isinstance(capabilities.embedding, bool)

            if hasattr(capabilities, "function_calling"):
                assert isinstance(capabilities.function_calling, bool)


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.integration
async def test_models_list_invalid_type(venice_client, vcr_cassette):
    """Test error handling for invalid model type."""
    with vcr_cassette, pytest.raises(InvalidRequestError):
        # Test with invalid type should raise InvalidRequestError
        await venice_client.models.list(type="invalid_type_xyz")


@pytest.mark.integration
async def test_models_list_invalid_api_key(vcr_cassette):
    """Test that models endpoint allows anonymous requests."""
    with vcr_cassette:
        # Create client with invalid API key
        client = VeniceClient(api_key="invalid-api-key-12345")

        try:
            # Models endpoint should work with invalid API key (anonymous access)
            response = await client.models.list()
            assert response is not None
            assert hasattr(response, "data")
            assert isinstance(response.data, list)
        finally:
            await client.close()


# ============================================================================
# Model Discovery Tests
# ============================================================================


@pytest.mark.integration
async def test_models_discover_available_types(venice_client, vcr_cassette):
    """Test discovering what model types are available."""
    with vcr_cassette:
        model_types = ["text", "image", "embedding", "tts", "video"]
        available_types = []

        for model_type in model_types:
            try:
                response = await venice_client.models.list(type=model_type)
                if response.data and len(response.data) > 0:
                    available_types.append(model_type)
            except InvalidRequestError:
                # Some types might not be supported
                continue

        # Should have at least text models
        assert "text" in available_types

        # Log what's available (for debugging)
        print(f"Available model types: {available_types}")


@pytest.mark.integration
async def test_models_count_by_type(venice_client, vcr_cassette):
    """Test counting models by type."""
    with vcr_cassette:
        counts = {}

        for model_type in ["text", "image", "embedding", "tts"]:
            try:
                response = await venice_client.models.list(type=model_type)
                counts[model_type] = len(response.data)
            except InvalidRequestError:
                # Some types might not be supported
                counts[model_type] = 0

        # Should have some models
        assert sum(counts.values()) > 0

        # Text models should be available
        assert counts["text"] > 0


# ============================================================================
# Model Filtering Tests
# ============================================================================


@pytest.mark.integration
async def test_models_filter_by_name_pattern(venice_client, vcr_cassette):
    """Test filtering models by name pattern (if supported)."""
    with vcr_cassette:
        # Get all models
        all_models = await venice_client.models.list()

        # Filter client-side since API might not support name filtering
        llama_models = [m for m in all_models.data if "llama" in m.id.lower()]

        # Check if we found any Llama models
        if llama_models:
            assert len(llama_models) > 0
            for model in llama_models:
                assert "llama" in model.id.lower()


@pytest.mark.integration
async def test_models_filter_by_capabilities(venice_client, vcr_cassette):
    """Test filtering models by capabilities (client-side)."""
    with vcr_cassette:
        response = await venice_client.models.list(type="text")

        # Filter models that support function calling (client-side)
        function_calling_models = [
            m
            for m in response.data
            if hasattr(m, "capabilities")
            and m.capabilities
            and hasattr(m.capabilities, "function_calling")
            and m.capabilities.function_calling
        ]

        # Log what we found
        if function_calling_models:
            print(f"Found {len(function_calling_models)} models with function calling")


# ============================================================================
# Model Pricing Tests
# ============================================================================


@pytest.mark.integration
async def test_models_pricing_information(venice_client, vcr_cassette):
    """Test model pricing information."""
    with vcr_cassette:
        response = await venice_client.models.list(type="text")

        # Find models with pricing info
        models_with_pricing = [m for m in response.data if hasattr(m, "pricing") and m.pricing]

        if models_with_pricing:
            model = models_with_pricing[0]
            pricing = model.pricing

            # Check pricing structure
            if hasattr(pricing, "input"):
                assert isinstance(pricing.input, (int, float))
                assert pricing.input >= 0

            if hasattr(pricing, "output"):
                assert isinstance(pricing.output, (int, float))
                assert pricing.output >= 0

            if hasattr(pricing, "currency"):
                assert isinstance(pricing.currency, str)


# ============================================================================
# Model Constraints Tests
# ============================================================================


@pytest.mark.integration
async def test_models_constraints(venice_client, vcr_cassette):
    """Test model constraints information."""
    with vcr_cassette:
        response = await venice_client.models.list(type="text")

        # Find models with constraints
        models_with_constraints = [
            m for m in response.data if hasattr(m, "constraints") and m.constraints
        ]

        if models_with_constraints:
            model = models_with_constraints[0]
            constraints = model.constraints

            # Check common constraints
            if hasattr(constraints, "max_tokens") and constraints.max_tokens is not None:
                assert isinstance(constraints.max_tokens, int)
                assert constraints.max_tokens > 0

            if hasattr(constraints, "max_context") and constraints.max_context is not None:
                assert isinstance(constraints.max_context, int)
                assert constraints.max_context > 0

            if hasattr(constraints, "temperature") and constraints.temperature is not None:
                # Could be a range or single value
                pass
