import pytest
import httpx
from typing import List, Literal, Optional, TypedDict

from venice_ai import AsyncVeniceClient
from venice_ai.types.models import ModelList, Model, ModelType
from venice_ai.exceptions import APIError, AuthenticationError

# Define a minimal mock Model and ModelList structure for testing
class MockModelPricingDetail(TypedDict):
    unit: str
    cost: float

class MockModelPricing(TypedDict):
    input: MockModelPricingDetail
    output: MockModelPricingDetail

class MockModelCapabilities(TypedDict):
    streaming: bool
    async_: bool
    max_tokens: int
    supports_functions: bool

class MockModelConstraintsTemperature(TypedDict):
    default: float
    min: float
    max: float

class MockModelConstraintsTopP(TypedDict):
    default: float
    min: float
    max: float

class MockModelConstraints(TypedDict):
    temperature: MockModelConstraintsTemperature
    top_p: MockModelConstraintsTopP

class MockModelSpec(TypedDict):
    input_format: str
    output_format: str

class MockModel(TypedDict):
    id: str
    object: Literal["model"]
    created: int
    owned_by: str
    name: str
    description: str
    type: str
    pricing: MockModelPricing
    capabilities: MockModelCapabilities
    constraints: MockModelConstraints
    spec: MockModelSpec

class MockModelList(TypedDict):
    object: Literal["list"]
    data: List[MockModel]
    type: Optional[ModelType]


@pytest.mark.asyncio
async def test_models_list_success_async(httpx_mock):
    """Tests successful asynchronous retrieval of the models list."""
    mock_response_data: MockModelList = {
        "object": "list",
        "data": [
            {
                "id": "model-1",
                "object": "model",
                "created": 1678888888,
                "owned_by": "organization-1",
                "name": "Model One",
                "description": "Description of Model One",
                "type": "text",
                "pricing": {"input": {"unit": "token", "cost": 0.001}, "output": {"unit": "token", "cost": 0.002}},
                "capabilities": {"streaming": True, "async_": True, "max_tokens": 4096, "supports_functions": False},
                "constraints": {
                    "temperature": {"default": 0.7, "min": 0.0, "max": 1.0},
                    "top_p": {"default": 1.0, "min": 0.0, "max": 1.0},
                },
                "spec": {"input_format": "text", "output_format": "text"},
            },
            {
                "id": "model-2",
                "object": "model",
                "created": 1678888889,
                "owned_by": "organization-2",
                "name": "Model Two",
                "description": "Description of Model Two",
                "type": "image",
                "pricing": {"input": {"unit": "pixel", "cost": 0.0001}, "output": {"unit": "pixel", "cost": 0.0002}},
                 "capabilities": {"streaming": False, "async_": True, "max_tokens": 1024, "supports_functions": False},
                "constraints": {
                    "temperature": {"default": 0.5, "min": 0.0, "max": 1.0},
                    "top_p": {"default": 0.9, "min": 0.0, "max": 1.0},
                },
                "spec": {"input_format": "image_url", "output_format": "image_url"},
            },
        ],
        "type": None,
    }

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/models?type=all",
        json=mock_response_data,
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        models_list = await client.models.list()

    assert isinstance(models_list, dict) # httpx returns dict by default
    assert models_list["object"] == "list"
    assert isinstance(models_list["data"], list)
    assert len(models_list["data"]) == 2
    assert models_list["data"][0]["id"] == "model-1"
    assert models_list["data"][1]["type"] == "image"
    assert models_list.get("type") is None


@pytest.mark.asyncio
async def test_models_list_with_type_filter_async(httpx_mock):
    """Tests asynchronous retrieval of models list with a type filter."""
    mock_response_data: MockModelList = {
        "object": "list",
        "data": [
            {
                "id": "model-1",
                "object": "model",
                "created": 1678888888,
                "owned_by": "organization-1",
                "name": "Model One",
                "description": "Description of Model One",
                "type": "text",
                "pricing": {"input": {"unit": "token", "cost": 0.001}, "output": {"unit": "token", "cost": 0.002}},
                "capabilities": {"streaming": True, "async_": True, "max_tokens": 4096, "supports_functions": False},
                "constraints": {
                    "temperature": {"default": 0.7, "min": 0.0, "max": 1.0},
                    "top_p": {"default": 1.0, "min": 0.0, "max": 1.0},
                },
                "spec": {"input_format": "text", "output_format": "text"},
            },
        ],
        "type": "text",
    }

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/models?type=text",
        json=mock_response_data,
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        models_list = await client.models.list(type="text")

    assert isinstance(models_list, dict)
    assert models_list["object"] == "list"
    assert isinstance(models_list["data"], list)
    assert len(models_list["data"]) == 1
    assert models_list["data"][0]["id"] == "model-1"
    assert models_list["type"] == "text"


@pytest.mark.asyncio
async def test_models_list_api_error_async(httpx_mock):
    """Tests asynchronous API error handling for models list retrieval."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/models?type=all",
        status_code=401,
        json={"error": {"message": "Invalid API key", "type": "authentication_error"}},
    )

    async with AsyncVeniceClient(api_key="invalid-key") as client:
        with pytest.raises(AuthenticationError) as excinfo:
            await client.models.list()

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 401
    assert "Invalid API key" in str(excinfo.value)


@pytest.mark.asyncio
async def test_models_list_traits_success_async(httpx_mock):
    """Tests successful asynchronous retrieval of model traits."""
    mock_response_data = {
        "object": "list",
        "data": {
            "default": "model-1",
            "fastest": "model-2",
            "most_capable": "model-3"
        },
        "type": None,
    }

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/models/traits",
        json=mock_response_data,
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        traits_list = await client.models.list_traits()

        assert isinstance(traits_list, dict)
        assert traits_list["object"] == "list"
        assert isinstance(traits_list["data"], dict)
        assert len(traits_list["data"]) == 3
        assert traits_list["data"]["default"] == "model-1"
        assert traits_list["data"]["fastest"] == "model-2"
        assert traits_list.get("type") is None


@pytest.mark.asyncio
async def test_models_list_traits_with_type_filter_async(httpx_mock):
    """Tests asynchronous retrieval of model traits with a type filter."""
    mock_response_data = {
        "object": "list",
        "data": {
            "default": "model-1",
            "fastest": "model-2",
        },
        "type": "text",
    }

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/models/traits?type=text",
        json=mock_response_data,
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        traits_list = await client.models.list_traits(type="text")

    assert isinstance(traits_list, dict)
    assert traits_list["object"] == "list"
    assert isinstance(traits_list["data"], dict)
    assert len(traits_list["data"]) == 2
    assert traits_list["data"]["default"] == "model-1"
    assert traits_list["data"]["fastest"] == "model-2"
    assert traits_list["type"] == "text"


@pytest.mark.asyncio
async def test_models_list_traits_api_error_async(httpx_mock):
    """Tests asynchronous API error handling for model traits retrieval."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/models/traits",
        status_code=401,
        json={"error": {"message": "Invalid API key", "type": "authentication_error"}},
    )

    async with AsyncVeniceClient(api_key="invalid-key") as client:
        with pytest.raises(AuthenticationError) as excinfo:
            await client.models.list_traits()

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 401
    assert "Invalid API key" in str(excinfo.value)


@pytest.mark.asyncio
async def test_models_list_compatibility_success_async(httpx_mock):
    """Tests successful asynchronous retrieval of model compatibility mapping."""
    mock_response_data = {
        "object": "list",
        "data": {
            "gpt-4o": "model-1",
            "gpt-4": "model-2",
            "gpt-3.5-turbo": "model-3"
        },
        "type": None,
    }

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/models/compatibility_mapping",
        json=mock_response_data,
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        compatibility_list = await client.models.list_compatibility()

    assert isinstance(compatibility_list, dict)
    assert compatibility_list["object"] == "list"
    assert isinstance(compatibility_list["data"], dict)
    assert len(compatibility_list["data"]) == 3
    assert compatibility_list["data"]["gpt-4o"] == "model-1"
    assert compatibility_list["data"]["gpt-4"] == "model-2"
    assert compatibility_list.get("type") is None


@pytest.mark.asyncio
async def test_models_list_compatibility_with_type_filter_async(httpx_mock):
    """Tests asynchronous retrieval of model compatibility mapping with a type filter."""
    mock_response_data = {
        "object": "list",
        "data": {
            "gpt-4o": "model-1",
            "gpt-4": "model-2",
        },
        "type": "text",
    }

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/models/compatibility_mapping?type=text",
        json=mock_response_data,
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        compatibility_list = await client.models.list_compatibility(type="text")

    assert isinstance(compatibility_list, dict)
    assert compatibility_list["object"] == "list"
    assert isinstance(compatibility_list["data"], dict)
    assert len(compatibility_list["data"]) == 2
    assert compatibility_list["data"]["gpt-4o"] == "model-1"
    assert compatibility_list["data"]["gpt-4"] == "model-2"
    assert compatibility_list["type"] == "text"


@pytest.mark.asyncio
async def test_models_list_compatibility_api_error_async(httpx_mock):
    """Tests asynchronous API error handling for model compatibility mapping retrieval."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/models/compatibility_mapping",
        status_code=401,
        json={"error": {"message": "Invalid API key", "type": "authentication_error"}},
    )

    async with AsyncVeniceClient(api_key="invalid-key") as client:
        with pytest.raises(AuthenticationError) as excinfo:
            await client.models.list_compatibility()

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 401
    assert "Invalid API key" in str(excinfo.value)


@pytest.mark.asyncio
async def test_async_models_list_type_chat_maps_to_text():
    """Tests that when type='chat' is provided to AsyncModels.list, it maps to api_type='text' in the API call."""
    from unittest.mock import MagicMock, AsyncMock
    from venice_ai import AsyncVeniceClient
    from venice_ai.resources.models import AsyncModels

    # Set up mock client
    mock_async_client = MagicMock(spec=AsyncVeniceClient)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json = AsyncMock(return_value={"object": "list", "data": [], "type": "text"})
    mock_response.aread = AsyncMock()
    mock_response.aclose = AsyncMock()
    mock_async_client.get = AsyncMock(return_value=mock_response)

    # Create the resource with the mock client
    async_models_resource = AsyncModels(mock_async_client)
    
    # Call the list method with type="chat"
    await async_models_resource.list(type="chat")  # type: ignore[arg-type]
    
    # Verify the get method was called with the correct parameters
    mock_async_client.get.assert_called_once()
    args, kwargs = mock_async_client.get.call_args
    assert args[0] == "models"
    assert kwargs["params"] == {"type": "text"}


@pytest.mark.asyncio
async def test_async_models_list_type_audio_maps_to_tts():
    """Tests that when type='audio' is provided to AsyncModels.list, it maps to api_type='tts' in the API call."""
    from unittest.mock import MagicMock, AsyncMock
    from venice_ai import AsyncVeniceClient
    from venice_ai.resources.models import AsyncModels

    # Set up mock client
    mock_async_client = MagicMock(spec=AsyncVeniceClient)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json = AsyncMock(return_value={"object": "list", "data": [], "type": "tts"})
    mock_response.aread = AsyncMock()
    mock_response.aclose = AsyncMock()
    mock_async_client.get = AsyncMock(return_value=mock_response)

    # Create the resource with the mock client
    async_models_resource = AsyncModels(mock_async_client)
    
    # Call the list method with type="audio"
    await async_models_resource.list(type="audio")  # type: ignore[arg-type]
    
    # Verify the get method was called with the correct parameters
    mock_async_client.get.assert_called_once()
    args, kwargs = mock_async_client.get.call_args
    assert args[0] == "models"
    assert kwargs["params"] == {"type": "tts"}


@pytest.mark.asyncio
async def test_async_models_list_other_types_pass_through():
    """Tests that other type values pass through directly in the AsyncModels.list method."""
    from unittest.mock import MagicMock, AsyncMock
    from venice_ai import AsyncVeniceClient
    from venice_ai.resources.models import AsyncModels

    # Set up mock client
    mock_async_client = MagicMock(spec=AsyncVeniceClient)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json = AsyncMock(return_value={"object": "list", "data": [], "type": "text"})
    mock_response.aread = AsyncMock()
    mock_response.aclose = AsyncMock()
    mock_async_client.get = AsyncMock(return_value=mock_response)

    # Create the resource with the mock client
    async_models_resource = AsyncModels(mock_async_client)
    
    # Test with type="text" (should pass through as is)
    await async_models_resource.list(type="text")
    
    # Verify the get method was called with the correct parameters
    mock_async_client.get.assert_called_once()
    args, kwargs = mock_async_client.get.call_args
    assert args[0] == "models"
    assert kwargs["params"] == {"type": "text"}
    
    # Reset the mock for the next test
    mock_async_client.get.reset_mock()
    
    # Test with type="tts" (should pass through as is)
    await async_models_resource.list(type="tts")
    
    # Verify the get method was called with the correct parameters
    mock_async_client.get.assert_called_once()
    args, kwargs = mock_async_client.get.call_args
    assert args[0] == "models"
    assert kwargs["params"] == {"type": "tts"}
    
    # Reset the mock for the next test
    mock_async_client.get.reset_mock()
    
    # Test with type=None (should result in type="all")
    await async_models_resource.list(type=None)
    
    # Verify the get method was called with the correct parameters
    mock_async_client.get.assert_called_once()
    args, kwargs = mock_async_client.get.call_args
    assert args[0] == "models"
    assert kwargs["params"] == {"type": "all"}