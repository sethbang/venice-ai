import pytest
import httpx
from typing import List, Literal, TypedDict

from venice_ai import AsyncVeniceClient
from venice_ai.types.embeddings import EmbeddingList
from venice_ai.exceptions import APIError, AuthenticationError

# Define a minimal mock structure for testing
class MockEmbedding(TypedDict):
    embedding: List[float]
    index: int
    object: Literal["embedding"]

class MockEmbeddingUsage(TypedDict):
    prompt_tokens: int
    total_tokens: int

class MockEmbeddingList(TypedDict):
    data: List[MockEmbedding]
    model: str
    object: Literal["list"]
    usage: MockEmbeddingUsage


@pytest.mark.asyncio
async def test_embeddings_create_success_async(httpx_mock):
    """Tests successful asynchronous creation of embeddings."""
    # Mock response data
    mock_response_data: MockEmbeddingList = {
        "object": "list",
        "data": [
            {
                "embedding": [0.0023064255, -0.009327292, 0.015797347, -0.007435346],
                "index": 0,
                "object": "embedding"
            },
            {
                "embedding": [0.008299528, 0.012940704, -0.01048945, -0.01997849],
                "index": 1,
                "object": "embedding"
            }
        ],
        "model": "venice-embed-v1",
        "usage": {
            "prompt_tokens": 8,
            "total_tokens": 8
        }
    }

    # Mock the HTTP response
    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/embeddings",
        json=mock_response_data,
        status_code=200,
    )

    # Create client and call the create method asynchronously
    async with AsyncVeniceClient(api_key="test-key") as client:
        embeddings_list = await client.embeddings.create(
            model="venice-embed-v1",
            input=["Hello world", "How are you?"]
        )

    # Assert the response meets our expectations
    assert isinstance(embeddings_list, dict)
    assert embeddings_list["object"] == "list"
    assert isinstance(embeddings_list["data"], list)
    assert len(embeddings_list["data"]) == 2
    assert embeddings_list["model"] == "venice-embed-v1"
    assert embeddings_list["usage"]["prompt_tokens"] == 8
    assert embeddings_list["usage"]["total_tokens"] == 8
    assert isinstance(embeddings_list["data"][0]["embedding"], list)
    assert embeddings_list["data"][0]["index"] == 0
    assert embeddings_list["data"][1]["index"] == 1


@pytest.mark.asyncio
async def test_embeddings_create_with_optional_params_async(httpx_mock):
    """Tests creating embeddings asynchronously with optional parameters."""
    # Mock response data
    mock_response_data: MockEmbeddingList = {
        "object": "list",
        "data": [
            {
                "embedding": [0.0023064255, -0.009327292],
                "index": 0,
                "object": "embedding"
            }
        ],
        "model": "venice-embed-v1",
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 5
        }
    }

    # Mock the HTTP response
    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/embeddings",
        json=mock_response_data,
        status_code=200,
    )

    # Create client and call the create method with optional parameters
    async with AsyncVeniceClient(api_key="test-key") as client:
        embeddings_list = await client.embeddings.create(
            model="venice-embed-v1",
            input="Hello world",
            dimensions=2,
            encoding_format="float",
            user="test-user-123"
        )

    # Assert the response meets our expectations
    assert isinstance(embeddings_list, dict)
    assert embeddings_list["object"] == "list"
    assert len(embeddings_list["data"]) == 1
    assert len(embeddings_list["data"][0]["embedding"]) == 2  # Verify dimensions


@pytest.mark.asyncio
async def test_embeddings_create_api_error_async(httpx_mock):
    """Tests asynchronous API error handling for embeddings creation."""
    # Mock an authentication error
    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/embeddings",
        status_code=401,
        json={"error": {"message": "Invalid API key", "type": "authentication_error"}},
    )

    # Create client and attempt to call the create method
    async with AsyncVeniceClient(api_key="invalid-key") as client:
        # Assert that the correct exception is raised
        with pytest.raises(AuthenticationError) as excinfo:
            await client.embeddings.create(
                model="venice-embed-v1",
                input=["Hello world"]
            )

    # Verify the exception details
    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 401
    assert "Invalid API key" in str(excinfo.value)