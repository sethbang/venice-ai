import pytest
import pytest_asyncio
import httpx
from unittest.mock import patch, MagicMock, AsyncMock
import json
import numpy as np

from venice_ai import AsyncVeniceClient
from venice_ai.resources.embeddings import Embeddings, AsyncEmbeddings
from venice_ai.exceptions import (
    InvalidRequestError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    RateLimitError,
    APIError
)
from tests.conftest import create_mock_response

# Mock data for successful embedding response
BASE_SUCCESS_RESPONSE = {
    "object": "list",
    "data": [
        {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
        {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1}
    ],
    "model": "venice-embed-v1",
    "usage": {"prompt_tokens": 10, "total_tokens": 10}
}

# Test synchronous Embeddings class with enhanced coverage
class TestEmbeddingsSyncEnhanced:
    @pytest.fixture
    def embeddings(self, mocker):
        client_mock = mocker.Mock()
        return Embeddings(client_mock)

    def test_create_empty_input(self, embeddings, mocker):
        """Test create with empty input."""
        with pytest.raises(InvalidRequestError, match="input cannot be empty"):
            embeddings.create(model="venice-embed-v1", input="")

    def test_create_empty_list_input(self, embeddings, mocker):
        """Test create with empty list input."""
        with pytest.raises(InvalidRequestError, match="input cannot be empty"):
            embeddings.create(model="venice-embed-v1", input=[])

    def test_create_with_dimensions(self, embeddings, mocker):
        """Test create with dimensions parameter."""
        # Create a modified response with dimensions
        response = BASE_SUCCESS_RESPONSE.copy()
        response["data"] = [
            {"object": "embedding", "embedding": [0.1] * 64, "index": 0},
            {"object": "embedding", "embedding": [0.2] * 64, "index": 1}
        ]
        
        mocker.patch.object(embeddings._client, 'post', return_value=response)
        result = embeddings.create(model="venice-embed-v1", input=["Hello", "world"], dimensions=64)
        
        assert result["object"] == "list"
        assert len(result["data"]) == 2
        assert len(result["data"][0]["embedding"]) == 64
        embeddings._client.post.assert_called_once_with("embeddings", json_data={
            "model": "venice-embed-v1",
            "input": ["Hello", "world"],
            "dimensions": 64
        })

    def test_create_with_encoding_format_base64(self, embeddings, mocker):
        """Test create with encoding_format parameter set to base64."""
        # Create a modified response with base64 encoded embeddings
        response = {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": "SGVsbG8=", "index": 0},
                {"object": "embedding", "embedding": "d29ybGQ=", "index": 1}
            ],
            "model": "venice-embed-v1",
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        }
        
        mocker.patch.object(embeddings._client, 'post', return_value=response)
        result = embeddings.create(model="venice-embed-v1", input=["Hello", "world"], encoding_format="base64")
        
        assert result["object"] == "list"
        assert len(result["data"]) == 2
        assert result["data"][0]["embedding"] == "SGVsbG8="
        embeddings._client.post.assert_called_once_with("embeddings", json_data={
            "model": "venice-embed-v1",
            "input": ["Hello", "world"],
            "encoding_format": "base64"
        })

    def test_create_with_user_parameter(self, embeddings, mocker):
        """Test create with user parameter."""
        mocker.patch.object(embeddings._client, 'post', return_value=BASE_SUCCESS_RESPONSE)
        result = embeddings.create(model="venice-embed-v1", input="Hello world", user="test-user-123")
        
        assert result["object"] == "list"
        embeddings._client.post.assert_called_once_with("embeddings", json_data={
            "model": "venice-embed-v1",
            "input": "Hello world",
            "user": "test-user-123"
        })

    def test_create_with_large_batch_input(self, embeddings, mocker):
        """Test create with a large batch of inputs."""
        # Create a response with many embeddings
        large_batch_size = 50
        large_response = {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": i}
                for i in range(large_batch_size)
            ],
            "model": "venice-embed-v1",
            "usage": {"prompt_tokens": 500, "total_tokens": 500}
        }
        
        large_input = [f"Text {i}" for i in range(large_batch_size)]
        
        mocker.patch.object(embeddings._client, 'post', return_value=large_response)
        result = embeddings.create(model="venice-embed-v1", input=large_input)
        
        assert result["object"] == "list"
        assert len(result["data"]) == large_batch_size
        embeddings._client.post.assert_called_once()
        # Verify the first argument is correct
        call_args = embeddings._client.post.call_args
        assert call_args[0][0] == "embeddings"
        # Verify the input list length in the json_data
        assert len(call_args[1]["json_data"]["input"]) == large_batch_size

    def test_create_with_float_input(self, embeddings, mocker):
        """Test create with float values in input."""
        mocker.patch.object(embeddings._client, 'post', return_value=BASE_SUCCESS_RESPONSE)
        float_input = [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]
        
        result = embeddings.create(model="venice-embed-v1", input=float_input)
        
        assert result["object"] == "list"
        embeddings._client.post.assert_called_once_with("embeddings", json_data={
            "model": "venice-embed-v1",
            "input": float_input
        })

    def test_create_with_mixed_input_types(self, embeddings, mocker):
        """Test create with mixed input types."""
        mocker.patch.object(embeddings._client, 'post', return_value=BASE_SUCCESS_RESPONSE)
        mixed_input = ["text", [1, 2, 3], "more text"]
        
        result = embeddings.create(model="venice-embed-v1", input=mixed_input)
        
        assert result["object"] == "list"
        embeddings._client.post.assert_called_once_with("embeddings", json_data={
            "model": "venice-embed-v1",
            "input": mixed_input
        })

    @pytest.mark.parametrize("error_class,status_code,message", [
        (InvalidRequestError, 400, "Invalid input for embeddings"),
        (AuthenticationError, 401, "Invalid API key"),
        (PermissionDeniedError, 403, "Permission denied"),
        (NotFoundError, 404, "Model not found"),
        (RateLimitError, 429, "Rate limit exceeded")
    ])
    def test_create_error_handling_with_messages(self, embeddings, mocker, error_class, status_code, message):
        """Test error handling with specific error messages."""
        mock_response = create_mock_response(
            status_code=status_code,
            json_data={"error": {"message": message, "code": status_code}}
        )
        error_data = {"error": {"message": message, "code": status_code}}
        mocker.patch.object(
            embeddings._client,
            'post',
            side_effect=error_class(message, response=mock_response, body=error_data)
        )
        
        with pytest.raises(error_class, match=message):
            embeddings.create(model="venice-embed-v1", input="Hello world")

    def test_create_model_validation(self, embeddings, mocker):
        """Test validation of the model parameter."""
        with pytest.raises(InvalidRequestError, match="model parameter is required"):
            embeddings.create(model="", input="Hello world")

        with pytest.raises(InvalidRequestError, match="model parameter is required"):
            embeddings.create(model=None, input="Hello world")

# Test asynchronous AsyncEmbeddings class with enhanced coverage
@pytest.mark.asyncio
class TestEmbeddingsAsyncEnhanced:
    @pytest_asyncio.fixture
    async def embeddings(self, mocker):
        client_mock = mocker.AsyncMock(spec=AsyncVeniceClient) # Use AsyncMock for the client
        client_mock.post = mocker.AsyncMock() # Ensure post is also an AsyncMock
        client_mock._base_url = httpx.URL("https://api.venice.ai/api/v1/") # Add _base_url attribute
        return AsyncEmbeddings(client_mock)

    async def test_create_empty_input(self, embeddings, mocker):
        """Test create with empty input."""
        with pytest.raises(InvalidRequestError, match="input cannot be empty"):
            await embeddings.create(model="venice-embed-v1", input="")

    async def test_create_empty_list_input(self, embeddings, mocker):
        """Test create with empty list input."""
        with pytest.raises(InvalidRequestError, match="input cannot be empty"):
            await embeddings.create(model="venice-embed-v1", input=[])

    async def test_create_with_dimensions(self, embeddings, mocker):
        """Test create with dimensions parameter."""
        # Create a modified response with dimensions
        response = BASE_SUCCESS_RESPONSE.copy()
        response["data"] = [
            {"object": "embedding", "embedding": [0.1] * 64, "index": 0},
            {"object": "embedding", "embedding": [0.2] * 64, "index": 1}
        ]
        
        embeddings._client.post.return_value = response
        result = await embeddings.create(model="venice-embed-v1", input=["Hello", "world"], dimensions=64)
        
        assert result["object"] == "list"
        assert len(result["data"]) == 2
        assert len(result["data"][0]["embedding"]) == 64
        embeddings._client.post.assert_awaited_once_with("embeddings", json_data={
            "model": "venice-embed-v1",
            "input": ["Hello", "world"],
            "dimensions": 64
        })

    async def test_create_with_encoding_format_base64(self, embeddings, mocker):
        """Test create with encoding_format parameter set to base64."""
        # Create a modified response with base64 encoded embeddings
        response = {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": "SGVsbG8=", "index": 0},
                {"object": "embedding", "embedding": "d29ybGQ=", "index": 1}
            ],
            "model": "venice-embed-v1",
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        }
        
        embeddings._client.post.return_value = response
        result = await embeddings.create(model="venice-embed-v1", input=["Hello", "world"], encoding_format="base64")
        
        assert result["object"] == "list"
        assert len(result["data"]) == 2
        assert result["data"][0]["embedding"] == "SGVsbG8="
        embeddings._client.post.assert_awaited_once_with("embeddings", json_data={
            "model": "venice-embed-v1",
            "input": ["Hello", "world"],
            "encoding_format": "base64"
        })

    async def test_create_with_user_parameter(self, embeddings, mocker):
        """Test create with user parameter."""
        embeddings._client.post.return_value = BASE_SUCCESS_RESPONSE
        result = await embeddings.create(model="venice-embed-v1", input="Hello world", user="test-user-123")
        
        assert result["object"] == "list"
        embeddings._client.post.assert_awaited_once_with("embeddings", json_data={
            "model": "venice-embed-v1",
            "input": "Hello world",
            "user": "test-user-123"
        })

    async def test_create_with_numpy_array_input(self, embeddings, mocker):
        """Test create with NumPy array input."""
        embeddings._client.post.return_value = BASE_SUCCESS_RESPONSE
        
        # Create NumPy array input
        np_input = np.array([[1, 2, 3], [4, 5, 6]])
        
        # Call with numpy arrays converted to lists
        result = await embeddings.create(model="venice-embed-v1", input=np_input.tolist())
        
        assert result["object"] == "list"
        embeddings._client.post.assert_awaited_once()
        call_args = embeddings._client.post.call_args
        assert call_args[1]["json_data"]["input"] == [[1, 2, 3], [4, 5, 6]]

    async def test_create_model_validation(self, embeddings, mocker):
        """Test validation of the model parameter."""
        with pytest.raises(InvalidRequestError, match="model parameter is required"):
            await embeddings.create(model="", input="Hello world")

        with pytest.raises(InvalidRequestError, match="model parameter is required"):
            await embeddings.create(model=None, input="Hello world")

    @pytest.mark.parametrize("error_class,status_code,message", [
        (InvalidRequestError, 400, "Invalid input for embeddings"),
        (AuthenticationError, 401, "Invalid API key"),
        (PermissionDeniedError, 403, "Permission denied"),
        (NotFoundError, 404, "Model not found"),
        (RateLimitError, 429, "Rate limit exceeded")
    ])
    async def test_create_error_handling_with_messages(self, embeddings, mocker, error_class, status_code, message):
        """Test error handling with specific error messages."""
        mock_response = create_mock_response(
            status_code=status_code,
            json_data={"error": {"message": message, "code": status_code}}
        )
        error_data = {"error": {"message": message, "code": status_code}}
        embeddings._client.post.side_effect = error_class(message, response=mock_response, body=error_data)
        
        with pytest.raises(error_class, match=message):
            await embeddings.create(model="venice-embed-v1", input="Hello world")