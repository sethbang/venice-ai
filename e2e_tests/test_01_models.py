import pytest
import pytest_asyncio
from venice_ai import VeniceClient, AsyncVeniceClient

# Functional Tests for Models API

def test_list_models_sync(venice_client: VeniceClient):
    """Tests synchronous listing of all models."""
    print("\n=== Running test_list_models_sync ===")
    models = venice_client.models.list()
    print(f"Response type: {type(models)}")
    print(f"Response preview: {str(models)[:100]}...")
    assert isinstance(models, dict)
    assert "data" in models
    assert isinstance(models["data"], list)

@pytest.mark.asyncio
async def test_list_models_async(async_venice_client: AsyncVeniceClient):
    """Tests asynchronous listing of all models."""
    print("\n=== Running test_list_models_async ===")
    models = await async_venice_client.models.list()
    print(f"Response type: {type(models)}")
    print(f"Response preview: {str(models)[:100]}...")
    assert isinstance(models, dict)
    assert "data" in models
    assert isinstance(models["data"], list)

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
