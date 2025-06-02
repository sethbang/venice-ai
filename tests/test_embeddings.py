import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock
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
SUCCESS_RESPONSE = {
    "object": "list",
    "data": [
        {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
        {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1}
    ],
    "model": "venice-embed-v1",
    "usage": {"prompt_tokens": 10, "total_tokens": 10}
}

# Test synchronous Embeddings class
class TestEmbeddingsSync:
    @pytest.fixture
    def embeddings(self, mocker):
        client_mock = mocker.Mock()
        return Embeddings(client_mock)

    def test_create_single_string_input(self, embeddings, mocker):
        mocker.patch.object(embeddings._client, 'post', return_value=SUCCESS_RESPONSE)
        result = embeddings.create(model="venice-embed-v1", input="Hello world")
        assert result["object"] == "list"
        assert len(result["data"]) == 2
        embeddings._client.post.assert_called_once_with("embeddings", json_data={
            "model": "venice-embed-v1",
            "input": "Hello world"
        })

    def test_create_list_string_input(self, embeddings, mocker):
        mocker.patch.object(embeddings._client, 'post', return_value=SUCCESS_RESPONSE)
        result = embeddings.create(model="venice-embed-v1", input=["Hello", "world"])
        assert result["object"] == "list"
        assert len(result["data"]) == 2
        embeddings._client.post.assert_called_once_with("embeddings", json_data={
            "model": "venice-embed-v1",
            "input": ["Hello", "world"]
        })

    def test_create_list_int_input(self, embeddings, mocker):
        mocker.patch.object(embeddings._client, 'post', return_value=SUCCESS_RESPONSE)
        result = embeddings.create(model="venice-embed-v1", input=[1, 2, 3])
        assert result["object"] == "list"
        embeddings._client.post.assert_called_once_with("embeddings", json_data={
            "model": "venice-embed-v1",
            "input": [1, 2, 3]
        })

    def test_create_list_list_int_input(self, embeddings, mocker):
        mocker.patch.object(embeddings._client, 'post', return_value=SUCCESS_RESPONSE)
        result = embeddings.create(model="venice-embed-v1", input=[[1, 2], [3, 4]])
        assert result["object"] == "list"
        embeddings._client.post.assert_called_once_with("embeddings", json_data={
            "model": "venice-embed-v1",
            "input": [[1, 2], [3, 4]]
        })

    def test_create_with_all_parameters(self, embeddings, mocker):
        mocker.patch.object(embeddings._client, 'post', return_value=SUCCESS_RESPONSE)
        result = embeddings.create(
            model="venice-embed-v1",
            input="Hello world",
            dimensions=128,
            encoding_format="base64",
            user="test-user"
        )
        assert result["object"] == "list"
        embeddings._client.post.assert_called_once_with("embeddings", json_data={
            "model": "venice-embed-v1",
            "input": "Hello world",
            "dimensions": 128,
            "encoding_format": "base64",
            "user": "test-user"
        })

    @pytest.mark.parametrize("error_class,status_code", [
        (InvalidRequestError, 400),
        (AuthenticationError, 401),
        (PermissionDeniedError, 403),
        (NotFoundError, 404),
        (RateLimitError, 429)
    ])
    def test_create_error_handling(self, embeddings, mocker, error_class, status_code):
        mock_response = create_mock_response(
            status_code=status_code,
            json_data={"error": {"message": "Error message", "code": status_code}}
        )
        error_data = {"error": {"message": "Error message", "code": status_code}}
        mocker.patch.object(
            embeddings._client,
            'post',
            side_effect=error_class("Error message", response=mock_response, body=error_data)
        )
        with pytest.raises(error_class):
            embeddings.create(model="venice-embed-v1", input="Hello world")

# Test asynchronous AsyncEmbeddings class
@pytest.mark.asyncio
class TestEmbeddingsAsync:
    @pytest_asyncio.fixture
    async def embeddings(self, mocker):
        client_mock = mocker.Mock()
        client_mock.post = AsyncMock()
        return AsyncEmbeddings(client_mock)

    async def test_create_single_string_input(self, embeddings, mocker):
        embeddings._client.post.return_value = SUCCESS_RESPONSE
        result = await embeddings.create(model="venice-embed-v1", input="Hello world")
        assert result["object"] == "list"
        assert len(result["data"]) == 2
        embeddings._client.post.assert_awaited_once_with("embeddings", json_data={
            "model": "venice-embed-v1",
            "input": "Hello world"
        })

    async def test_create_list_string_input(self, embeddings, mocker):
        embeddings._client.post.return_value = SUCCESS_RESPONSE
        result = await embeddings.create(model="venice-embed-v1", input=["Hello", "world"])
        assert result["object"] == "list"
        assert len(result["data"]) == 2
        embeddings._client.post.assert_awaited_once_with("embeddings", json_data={
            "model": "venice-embed-v1",
            "input": ["Hello", "world"]
        })

    async def test_create_list_int_input(self, embeddings, mocker):
        embeddings._client.post.return_value = SUCCESS_RESPONSE
        result = await embeddings.create(model="venice-embed-v1", input=[1, 2, 3])
        assert result["object"] == "list"
        embeddings._client.post.assert_awaited_once_with("embeddings", json_data={
            "model": "venice-embed-v1",
            "input": [1, 2, 3]
        })

    async def test_create_list_list_int_input(self, embeddings, mocker):
        embeddings._client.post.return_value = SUCCESS_RESPONSE
        result = await embeddings.create(model="venice-embed-v1", input=[[1, 2], [3, 4]])
        assert result["object"] == "list"
        embeddings._client.post.assert_awaited_once_with("embeddings", json_data={
            "model": "venice-embed-v1",
            "input": [[1, 2], [3, 4]]
        })

    async def test_create_with_all_parameters(self, embeddings, mocker):
        embeddings._client.post.return_value = SUCCESS_RESPONSE
        result = await embeddings.create(
            model="venice-embed-v1",
            input="Hello world",
            dimensions=128,
            encoding_format="base64",
            user="test-user"
        )
        assert result["object"] == "list"
        embeddings._client.post.assert_awaited_once_with("embeddings", json_data={
            "model": "venice-embed-v1",
            "input": "Hello world",
            "dimensions": 128,
            "encoding_format": "base64",
            "user": "test-user"
        })

    @pytest.mark.parametrize("error_class,status_code", [
        (InvalidRequestError, 400),
        (AuthenticationError, 401),
        (PermissionDeniedError, 403),
        (NotFoundError, 404),
        (RateLimitError, 429)
    ])
    async def test_create_error_handling(self, embeddings, mocker, error_class, status_code):
        mock_response = create_mock_response(
            status_code=status_code,
            json_data={"error": {"message": "Error message", "code": status_code}}
        )
        error_data = {"error": {"message": "Error message", "code": status_code}}
        embeddings._client.post.side_effect = error_class("Error message", response=mock_response, body=error_data)
        with pytest.raises(error_class):
            await embeddings.create(model="venice-embed-v1", input="Hello world")