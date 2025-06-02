import pytest
import pytest_asyncio
import httpx # Import httpx
from unittest.mock import MagicMock, AsyncMock

from venice_ai.resources.embeddings import Embeddings, AsyncEmbeddings
from venice_ai.exceptions import InvalidRequestError, APIError

# Mock data for successful embedding response
SUCCESS_RESPONSE = {
    "object": "list",
    "data": [
        {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
    ],
    "model": "venice-embed-v1",
    "usage": {"prompt_tokens": 5, "total_tokens": 5}
}

# Tests specifically targeting coverage gaps in embeddings.py
class TestEmbeddingsCoverageGaps:
    @pytest.fixture
    def embeddings(self, mocker):
        client_mock = mocker.Mock()
        client_mock.post.return_value = SUCCESS_RESPONSE
        return Embeddings(client_mock)
    
    def test_create_with_explicit_encoding_format(self, embeddings):
        """Test create with explicit encoding_format parameter to cover line 94."""
        result = embeddings.create(
            model="venice-embed-v1", 
            input="Hello world",
            encoding_format="base64"  # Explicitly providing encoding_format
        )
        
        # Verify the result
        assert result == SUCCESS_RESPONSE
        
        # Verify encoding_format was added to request body
        called_args = embeddings._client.post.call_args
        assert called_args[1]["json_data"]["encoding_format"] == "base64"
    
    def test_create_with_explicit_user(self, embeddings):
        """Test create with explicit user parameter to cover line 97."""
        result = embeddings.create(
            model="venice-embed-v1", 
            input="Hello world",
            user="test_user"  # Explicitly providing user
        )
        
        # Verify the result
        assert result == SUCCESS_RESPONSE
        
        # Verify user was added to request body
        called_args = embeddings._client.post.call_args
        assert called_args[1]["json_data"]["user"] == "test_user"
    
    def test_create_with_all_optional_params(self, embeddings):
        """Test create with all optional parameters."""
        result = embeddings.create(
            model="venice-embed-v1", 
            input="Hello world",
            dimensions=1024,
            encoding_format="base64",
            user="test_user"
        )
        
        # Verify the result
        assert result == SUCCESS_RESPONSE
        
        # Verify all parameters were added to request body
        called_args = embeddings._client.post.call_args
        json_data = called_args[1]["json_data"]
        assert json_data["model"] == "venice-embed-v1"
        assert json_data["input"] == "Hello world"
        assert json_data["dimensions"] == 1024
        assert json_data["encoding_format"] == "base64"
        assert json_data["user"] == "test_user"


# Specific tests for AsyncEmbeddings implementation
@pytest.mark.asyncio
class TestAsyncEmbeddingsCoverageGaps:
    @pytest_asyncio.fixture
    async def async_embeddings(self, mocker):
        client_mock = mocker.Mock()
        client_mock.post = AsyncMock(return_value=SUCCESS_RESPONSE)
        return AsyncEmbeddings(client_mock)
    
    async def test_async_create_basic(self, async_embeddings):
        """Test the basic async create method to cover the implementation (lines 171-187)."""
        result = await async_embeddings.create(
            model="venice-embed-v1",
            input="Hello world"
        )
        
        # Verify the result
        assert result == SUCCESS_RESPONSE
        
        # Verify the request was made properly
        async_embeddings._client.post.assert_awaited_once()
        called_args = async_embeddings._client.post.call_args
        assert called_args[0][0] == "embeddings"
        assert called_args[1]["json_data"]["model"] == "venice-embed-v1"
        assert called_args[1]["json_data"]["input"] == "Hello world"
    
    async def test_async_create_with_dimensions(self, async_embeddings):
        """Test async create with dimensions parameter."""
        result = await async_embeddings.create(
            model="venice-embed-v1",
            input="Hello world",
            dimensions=512
        )
        
        # Verify the result
        assert result == SUCCESS_RESPONSE
        
        # Verify dimensions was added to request body
        called_args = async_embeddings._client.post.call_args
        assert called_args[1]["json_data"]["dimensions"] == 512
    
    async def test_async_create_with_encoding_format(self, async_embeddings):
        """Test async create with encoding_format parameter to cover line 181."""
        result = await async_embeddings.create(
            model="venice-embed-v1",
            input="Hello world",
            encoding_format="base64"
        )
        
        # Verify the result
        assert result == SUCCESS_RESPONSE
        
        # Verify encoding_format was added to request body
        called_args = async_embeddings._client.post.call_args
        assert called_args[1]["json_data"]["encoding_format"] == "base64"
    
    async def test_async_create_with_user(self, async_embeddings):
        """Test async create with user parameter to cover line 184."""
        result = await async_embeddings.create(
            model="venice-embed-v1",
            input="Hello world",
            user="test_user"
        )
        
        # Verify the result
        assert result == SUCCESS_RESPONSE
        
        # Verify user was added to request body
        called_args = async_embeddings._client.post.call_args
        assert called_args[1]["json_data"]["user"] == "test_user"
    
    async def test_async_create_with_all_optional_params(self, async_embeddings):
        """Test async create with all optional parameters."""
        result = await async_embeddings.create(
            model="venice-embed-v1",
            input="Hello world",
            dimensions=1024,
            encoding_format="base64",
            user="test_user"
        )
        
        # Verify the result
        assert result == SUCCESS_RESPONSE
        
        # Verify all parameters were added to request body
        called_args = async_embeddings._client.post.call_args
        json_data = called_args[1]["json_data"]
        assert json_data["model"] == "venice-embed-v1"
        assert json_data["input"] == "Hello world"
        assert json_data["dimensions"] == 1024
        assert json_data["encoding_format"] == "base64"
        assert json_data["user"] == "test_user"
    
    async def test_async_create_with_error(self, async_embeddings, mocker):
        """Test async create with an error response."""
        # Configure the mock to raise an exception
        error_message = "API error occurred"
        # Create a mock httpx.Response
        mock_http_response = MagicMock(spec=httpx.Response)
        mock_http_response.status_code = 500 # Example status code
        mock_http_response.request = MagicMock(spec=httpx.Request)
        mock_http_response.text = error_message # Example text
        mock_http_response.json = MagicMock(return_value={"error": {"message": error_message}}) # Example JSON

        async_embeddings._client.post.side_effect = APIError(error_message, response=mock_http_response)
        
        # Test that the exception is propagated
        with pytest.raises(APIError, match=error_message):
            await async_embeddings.create(
                model="venice-embed-v1",
                input="Hello world"
            )