import pytest
import json
from unittest.mock import MagicMock, patch
import httpx
from typing import AsyncIterator

from venice_ai import VeniceClient
from venice_ai import AsyncVeniceClient
from venice_ai.resources.embeddings import Embeddings, AsyncEmbeddings
from venice_ai.exceptions import (
    VeniceError,
    InvalidRequestError, 
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    RateLimitError
)

# Sample response data for mocking
SUCCESS_RESPONSE = {
    "object": "list",
    "data": [
        {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
        {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1}
    ],
    "model": "venice-embed-v1",
    "usage": {"prompt_tokens": 8, "total_tokens": 8}
}

# Testing import paths for TYPE_CHECKING
def test_imports_type_checking():
    """Test that the TYPE_CHECKING imports are covered."""
    # This test simply imports the modules to cover the import lines that only
    # execute during TYPE_CHECKING, which normally wouldn't be covered
    from venice_ai.resources.embeddings import AsyncEmbeddings, Embeddings
    
    # Test creating with explicit client instances to cover initialization
    client = VeniceClient(api_key="test-key")
    embeddings = Embeddings(client)
    
    async_client = AsyncVeniceClient(api_key="test-key")
    async_embeddings = AsyncEmbeddings(async_client)
    
    assert embeddings._client == client
    assert async_embeddings._client == async_client

class TestEmbeddingsAdvanced:
    @pytest.fixture
    def embeddings(self, mocker):
        client_mock = mocker.Mock()
        client_mock.post.return_value = SUCCESS_RESPONSE
        return Embeddings(client_mock)

    def test_create_with_none_dimensions(self, embeddings, mocker):
        """Test create with dimensions=None (should not be added to body)."""
        result = embeddings.create(model="venice-embed-v1", input="Hello world", dimensions=None)
        
        assert result["object"] == "list"
        # Verify dimensions was not added to request body
        called_args = embeddings._client.post.call_args
        assert "dimensions" not in called_args[1]["json_data"]

    def test_create_with_none_encoding_format(self, embeddings, mocker):
        """Test create with encoding_format=None (should not be added to body)."""
        result = embeddings.create(
            model="venice-embed-v1", 
            input="Hello world", 
            encoding_format=None
        )
        
        assert result["object"] == "list"
        # Verify encoding_format was not added to request body
        called_args = embeddings._client.post.call_args
        assert "encoding_format" not in called_args[1]["json_data"]

    def test_create_with_none_user(self, embeddings, mocker):
        """Test create with user=None (should not be added to body)."""
        result = embeddings.create(
            model="venice-embed-v1", 
            input="Hello world", 
            user=None
        )
        
        assert result["object"] == "list"
        # Verify user was not added to request body
        called_args = embeddings._client.post.call_args
        assert "user" not in called_args[1]["json_data"]

class TestAsyncEmbeddingsAdvanced:
    @pytest.fixture
    async def async_embeddings(self, mocker):
        # Use AsyncMock for the client and its post method
        client_mock = mocker.AsyncMock(spec=AsyncVeniceClient)
        # Configure the post method to be an async mock as well
        client_mock.post = mocker.AsyncMock(return_value=SUCCESS_RESPONSE)
        return AsyncEmbeddings(client_mock)

    @pytest.mark.asyncio
    async def test_create_with_none_dimensions(self, async_embeddings, mocker):
        """Test async create with dimensions=None (should not be added to body)."""
        result = await async_embeddings.create(
            model="venice-embed-v1",
            input="Hello world",
            dimensions=None
        )
        
        assert result["object"] == "list"
        # Verify dimensions was not added to request body
        called_args = async_embeddings._client.post.call_args
        assert "dimensions" not in called_args[1]["json_data"]

    @pytest.mark.asyncio
    async def test_create_with_none_encoding_format(self, async_embeddings, mocker):
        """Test async create with encoding_format=None (should not be added to body)."""
        result = await async_embeddings.create(
            model="venice-embed-v1",
            input="Hello world",
            encoding_format=None
        )
        
        assert result["object"] == "list"
        # Verify encoding_format was not added to request body
        called_args = async_embeddings._client.post.call_args
        assert "encoding_format" not in called_args[1]["json_data"]

    @pytest.mark.asyncio
    async def test_create_with_none_user(self, async_embeddings, mocker):
        """Test async create with user=None (should not be added to body)."""
        result = await async_embeddings.create(
            model="venice-embed-v1",
            input="Hello world",
            user=None
        )
        
        assert result["object"] == "list"
        # Verify user was not added to request body
        called_args = async_embeddings._client.post.call_args
        assert "user" not in called_args[1]["json_data"]