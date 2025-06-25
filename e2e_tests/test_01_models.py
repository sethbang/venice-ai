import pytest
import pytest_asyncio
from typing import List, cast
from venice_ai import VeniceClient, AsyncVeniceClient
from venice_ai.types import Model, ModelType # Added Model, ModelType
from venice_ai import utils as venice_utils

# Functional Tests for Models API

def test_list_models_sync(venice_client: VeniceClient):
    """Tests synchronous listing of all models."""
    print("\n=== Running test_list_models_sync ===")
    models_response = venice_client.models.list()
    print(f"Response type: {type(models_response)}")
    print(f"Response preview: {str(models_response)[:200]}...") # Increased preview length
    assert isinstance(models_response, dict)
    assert "data" in models_response
    assert isinstance(models_response["data"], list)
    assert len(models_response["data"]) > 0 # Ensure some models are returned

    # Inspect the first model as an example
    first_model = models_response["data"][0]
    assert "id" in first_model
    assert "model_spec" in first_model
    model_spec = first_model["model_spec"]
    assert isinstance(model_spec, dict)
    assert "capabilities" in model_spec
    capabilities = model_spec["capabilities"]
    assert isinstance(capabilities, dict)
    
    # Check for a new capability field (example: supportsVision)
    # Its value can be True, False, or it might be absent if not applicable,
    # so we just check if the access doesn't error out or if it's present.
    # For TypedDict with total=False, .get() is safer.
    assert "supportsVision" in capabilities or capabilities.get("supportsVision") is not None or capabilities.get("supportsVision") is False
    
    # Check for the new 'beta' field in model_spec (optional)
    # The field might not be present, or if present, should be a boolean
    if "beta" in model_spec:
        assert isinstance(model_spec["beta"], bool)


@pytest.mark.asyncio
async def test_list_models_async(async_venice_client: AsyncVeniceClient):
    """Tests asynchronous listing of all models."""
    print("\n=== Running test_list_models_async ===")
    models_response = await async_venice_client.models.list()
    print(f"Response type: {type(models_response)}")
    print(f"Response preview: {str(models_response)[:200]}...") # Increased preview length
    assert isinstance(models_response, dict)
    assert "data" in models_response
    assert isinstance(models_response["data"], list)
    assert len(models_response["data"]) > 0 # Ensure some models are returned

    # Inspect the first model as an example
    first_model = models_response["data"][0]
    assert "id" in first_model
    assert "model_spec" in first_model
    model_spec = first_model["model_spec"]
    assert isinstance(model_spec, dict)
    assert "capabilities" in model_spec
    capabilities = model_spec["capabilities"]
    assert isinstance(capabilities, dict)

    assert "supportsVision" in capabilities or capabilities.get("supportsVision") is not None or capabilities.get("supportsVision") is False
    # Check for the new 'beta' field in model_spec (optional)
    # The field might not be present, or if present, should be a boolean
    if "beta" in model_spec:
        assert isinstance(model_spec["beta"], bool)

def test_list_models_with_type_filter_sync(venice_client: VeniceClient):
    """Tests synchronous listing of models filtered by type."""
    print("\n=== Running test_list_models_with_type_filter_sync ===")
    chat_models = venice_client.models.list(type="text")
    print(f"Response type: {type(chat_models)}")
    print(f"Response preview: {str(chat_models)[:100]}...")
    assert isinstance(chat_models, dict)
    assert "data" in chat_models
    assert isinstance(chat_models["data"], list)

@pytest.mark.asyncio
async def test_list_models_with_type_filter_async(async_venice_client: AsyncVeniceClient):
    """Tests asynchronous listing of models filtered by type."""
    print("\n=== Running test_list_models_with_type_filter_async ===")
    image_models = await async_venice_client.models.list(type="image")
    print(f"Response type: {type(image_models)}")
    print(f"Response preview: {str(image_models)[:100]}...")
    assert isinstance(image_models, dict)
    assert "data" in image_models
    assert isinstance(image_models["data"], list)

def test_list_model_traits_sync(venice_client: VeniceClient):
    """Tests synchronous listing of all model traits."""
    print("\n=== Running test_list_model_traits_sync ===")
    traits = venice_client.models.list_traits()
    print(f"Response type: {type(traits)}")
    print(f"Response preview: {str(traits)[:100]}...")
    assert isinstance(traits, dict)

@pytest.mark.asyncio
async def test_list_model_traits_async(async_venice_client: AsyncVeniceClient):
    """Tests asynchronous listing of all model traits."""
    print("\n=== Running test_list_model_traits_async ===")
    traits = await async_venice_client.models.list_traits()
    print(f"Response type: {type(traits)}")
    print(f"Response preview: {str(traits)[:100]}...")
    assert isinstance(traits, dict)

def test_list_model_traits_with_type_filter_sync(venice_client: VeniceClient):
    """Tests synchronous listing of model traits filtered by type."""
    print("\n=== Running test_list_model_traits_with_type_filter_sync ===")
    embedding_traits = venice_client.models.list_traits(type="embedding")
    print(f"Response type: {type(embedding_traits)}")
    print(f"Response preview: {str(embedding_traits)[:100]}...")
    assert isinstance(embedding_traits, dict)

@pytest.mark.asyncio
async def test_list_model_traits_with_type_filter_async(async_venice_client: AsyncVeniceClient):
    """Tests asynchronous listing of model traits filtered by type."""
    print("\n=== Running test_list_model_traits_with_type_filter_async ===")
    audio_traits = await async_venice_client.models.list_traits(type="tts")
    print(f"Response type: {type(audio_traits)}")
    print(f"Response preview: {str(audio_traits)[:100]}...")
    assert isinstance(audio_traits, dict)

def test_list_model_compatibility_sync(venice_client: VeniceClient):
    """Tests synchronous listing of model compatibility mappings."""
    print("\n=== Running test_list_model_compatibility_sync ===")
    compatibility = venice_client.models.list_compatibility()
    print(f"Response type: {type(compatibility)}")
    print(f"Response preview: {str(compatibility)[:100]}...")
    assert isinstance(compatibility, dict)

@pytest.mark.asyncio
async def test_list_model_compatibility_async(async_venice_client: AsyncVeniceClient):
    """Tests asynchronous listing of model compatibility mappings."""
    print("\n=== Running test_list_model_compatibility_async ===")
    compatibility = await async_venice_client.models.list_compatibility()
    print(f"Response type: {type(compatibility)}")
    print(f"Response preview: {str(compatibility)[:100]}...")
    assert isinstance(compatibility, dict)

def test_list_model_compatibility_with_type_filter_sync(venice_client: VeniceClient):
    """Tests synchronous listing of model compatibility filtered by type."""
    print("\n=== Running test_list_model_compatibility_with_type_filter_sync ===")
    chat_compatibility = venice_client.models.list_compatibility(type="text")
    print(f"Response type: {type(chat_compatibility)}")
    print(f"Response preview: {str(chat_compatibility)[:100]}...")
    assert isinstance(chat_compatibility, dict)

@pytest.mark.asyncio
async def test_list_model_compatibility_with_type_filter_async(async_venice_client: AsyncVeniceClient):
    """Tests asynchronous listing of model compatibility filtered by type."""
    print("\n=== Running test_list_model_compatibility_with_type_filter_async ===")
    image_compatibility = await async_venice_client.models.list_compatibility(type="image")
    print(f"Response type: {type(image_compatibility)}")
    print(f"Response preview: {str(image_compatibility)[:100]}...")
    assert isinstance(image_compatibility, dict)

# Tests for venice_ai.utils.get_filtered_models

def test_get_filtered_models_sync_by_capability(venice_client: VeniceClient):
    """Tests synchronous filtering of models by a specific new capability."""
    print("\n=== Running test_get_filtered_models_sync_by_capability ===")
    # Example: Filter for models that support vision
    # Note: This relies on the actual API data having models with this capability.
    # Adjust capability and expected outcome based on live API data.
    vision_models_result = venice_utils.get_filtered_models(
        client=venice_client,
        supports_vision=True
    )
    vision_models = cast(List[Model], vision_models_result)
    print(f"Found {len(vision_models)} vision models (sync).")
    assert isinstance(vision_models, list)
    if vision_models:
        for model in vision_models:
            model_spec = model.get("model_spec", {})
            capabilities = model_spec.get("capabilities", {})
            assert capabilities.get("supportsVision") is True

    # Example: Filter by quantization (e.g., "fp16")
    fp16_models_result = venice_utils.get_filtered_models(
        client=venice_client,
        quantization="fp16" # Assuming some models use fp16
    )
    fp16_models = cast(List[Model], fp16_models_result)
    print(f"Found {len(fp16_models)} fp16 models (sync).")
    assert isinstance(fp16_models, list)
    if fp16_models:
        for model in fp16_models:
            model_spec = model.get("model_spec", {})
            capabilities = model_spec.get("capabilities", {})
            assert capabilities.get("quantization") == "fp16"

@pytest.mark.asyncio
async def test_get_filtered_models_async_by_capability(async_venice_client: AsyncVeniceClient):
    """Tests asynchronous filtering of models by a specific new capability."""
    print("\n=== Running test_get_filtered_models_async_by_capability ===")
    # Example: Filter for models that support reasoning
    reasoning_models = await venice_utils.get_filtered_models(  # type: ignore
        client=async_venice_client,
        supports_reasoning=True
    )
    print(f"Found {len(reasoning_models)} reasoning models (async).")
    assert isinstance(reasoning_models, list)
    if reasoning_models:
        for model in reasoning_models:
            model_spec = model.get("model_spec", {})
            capabilities = model_spec.get("capabilities", {})
            assert capabilities.get("supportsReasoning") is True
            
    # Example: Filter for non-beta models
    non_beta_models = await venice_utils.get_filtered_models(  # type: ignore
        client=async_venice_client,
        is_beta=False
    )
    print(f"Found {len(non_beta_models)} non-beta models (async).")
    assert isinstance(non_beta_models, list)
    if non_beta_models:
        for model in non_beta_models:
            model_spec = model.get("model_spec", {})
            assert model_spec.get("beta") is False or "beta" not in model_spec


def test_get_filtered_models_sync_by_trait(venice_client: VeniceClient):
    """Tests synchronous filtering of models by a specific trait."""
    print("\n=== Running test_get_filtered_models_sync_by_trait ===")
    # Example: Filter for models with "default" trait
    # This assumes "default" is a common trait.
    default_trait_models_result = venice_utils.get_filtered_models(
        client=venice_client,
        has_trait="default"
    )
    default_trait_models = cast(List[Model], default_trait_models_result)
    print(f"Found {len(default_trait_models)} models with 'default' trait (sync).")
    assert isinstance(default_trait_models, list)
    if default_trait_models:
        for model in default_trait_models:
            model_spec = model.get("model_spec", {})
            assert "default" in model_spec.get("traits", [])

@pytest.mark.asyncio
async def test_get_filtered_models_async_combined_filters(async_venice_client: AsyncVeniceClient):
    """Tests asynchronous filtering with combined criteria."""
    print("\n=== Running test_get_filtered_models_async_combined_filters ===")
    # Example: Text models that support function calling and are not beta
    filtered_models = await venice_utils.get_filtered_models(  # type: ignore
        client=async_venice_client,
        model_type="text",
        supports_function_calling=True,
        is_beta=False
    )
    print(f"Found {len(filtered_models)} text, function-calling, non-beta models (async).")
    assert isinstance(filtered_models, list)
    if filtered_models:
        for model in filtered_models:
            assert model.get("type") == "text"
            model_spec = model.get("model_spec", {})
            capabilities = model_spec.get("capabilities", {})
            assert capabilities.get("supportsFunctionCalling") is True
            assert model_spec.get("beta") is False or "beta" not in model_spec
