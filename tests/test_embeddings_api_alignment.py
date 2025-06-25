"""
Tests for embeddings API alignment improvements.

This module tests the new functionality added to align the SDK with the official API documentation:
- PaymentRequiredError and ServiceUnavailableError exceptions
- Input array length validation (max 2048 items)
- Updated user parameter behavior
- Support for base64 encoding format in type definitions
"""

import pytest
import httpx
from unittest.mock import MagicMock, patch
from typing import List

from venice_ai import VeniceClient, AsyncVeniceClient
from venice_ai.exceptions import (
    InvalidRequestError,
    PaymentRequiredError,
    ServiceUnavailableError,
    _make_status_error
)
from venice_ai.resources.embeddings import Embeddings, AsyncEmbeddings
from venice_ai.types.embeddings import EmbeddingList, Embedding


class TestNewExceptionClasses:
    """Test the new PaymentRequiredError and ServiceUnavailableError classes."""
    
    def test_payment_required_error_initialization(self):
        """Test PaymentRequiredError can be initialized correctly."""
        mock_request = MagicMock(spec=httpx.Request)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 402
        
        error = PaymentRequiredError(
            "Insufficient USD or VCU balance",
            request=mock_request,
            response=mock_response,
            body={"error": "Insufficient balance"}
        )
        
        assert error.message == "Insufficient USD or VCU balance"
        assert error.status_code == 402
        assert error.request_obj == mock_request
        assert error.response_obj == mock_response
        assert error.body == {"error": "Insufficient balance"}
    
    def test_service_unavailable_error_initialization(self):
        """Test ServiceUnavailableError can be initialized correctly."""
        mock_request = MagicMock(spec=httpx.Request)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 503
        
        error = ServiceUnavailableError(
            "The model is at capacity",
            request=mock_request,
            response=mock_response,
            body={"error": "Model at capacity"}
        )
        
        assert error.message == "The model is at capacity"
        assert error.status_code == 503
        assert error.request_obj == mock_request
        assert error.response_obj == mock_response
        assert error.body == {"error": "Model at capacity"}
    
    def test_make_status_error_402(self):
        """Test _make_status_error returns PaymentRequiredError for 402 status."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 402
        mock_response.headers = {}
        
        error = _make_status_error(
            "Payment required",
            response=mock_response,
            body={"error": {"message": "Insufficient balance"}}
        )
        
        assert isinstance(error, PaymentRequiredError)
        assert "Insufficient balance" in error.message
    
    def test_make_status_error_503(self):
        """Test _make_status_error returns ServiceUnavailableError for 503 status."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 503
        mock_response.headers = {}
        
        error = _make_status_error(
            "Service unavailable",
            response=mock_response,
            body={"error": {"message": "Model at capacity"}}
        )
        
        assert isinstance(error, ServiceUnavailableError)
        assert "Model at capacity" in error.message


class TestEmbeddingsInputValidation:
    """Test the new input array length validation for embeddings."""
    
    @pytest.fixture
    def embeddings(self):
        """Create an Embeddings instance with a mocked client."""
        mock_client = MagicMock()
        mock_client._base_url = httpx.URL("https://api.venice.ai/api/v1/")
        return Embeddings(mock_client)
    
    @pytest.fixture
    def async_embeddings(self):
        """Create an AsyncEmbeddings instance with a mocked client."""
        mock_client = MagicMock()
        mock_client._base_url = httpx.URL("https://api.venice.ai/api/v1/")
        return AsyncEmbeddings(mock_client)
    
    def test_create_with_array_exceeding_limit(self, embeddings):
        """Test that arrays with more than 2048 items raise InvalidRequestError."""
        # Create an input array with 2049 items
        large_input = ["text"] * 2049
        
        with pytest.raises(InvalidRequestError) as exc_info:
            embeddings.create(model="text-embedding-bge-m3", input=large_input)
        
        assert "input array must have 2048 or fewer items" in str(exc_info.value)
        assert "got 2049 items" in str(exc_info.value)
    
    def test_create_with_array_at_limit(self, embeddings, mocker):
        """Test that arrays with exactly 2048 items are accepted."""
        # Create an input array with exactly 2048 items
        max_input = ["text"] * 2048
        
        # Mock the client's post method
        mock_response = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": i} for i in range(2048)],
            "model": "text-embedding-bge-m3",
            "usage": {"prompt_tokens": 10000, "total_tokens": 10000}
        }
        mocker.patch.object(embeddings._client, 'post', return_value=mock_response)
        
        # This should not raise an error
        result = embeddings.create(model="text-embedding-bge-m3", input=max_input)
        
        # Verify the call was made
        embeddings._client.post.assert_called_once()
        call_args = embeddings._client.post.call_args
        assert call_args[0][0] == "embeddings"
        assert len(call_args[1]["json_data"]["input"]) == 2048
    
    def test_create_with_array_below_limit(self, embeddings, mocker):
        """Test that arrays with less than 2048 items work normally."""
        # Create a small input array
        small_input = ["text1", "text2", "text3"]
        
        # Mock the client's post method
        mock_response = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": i} for i in range(3)],
            "model": "text-embedding-bge-m3",
            "usage": {"prompt_tokens": 15, "total_tokens": 15}
        }
        mocker.patch.object(embeddings._client, 'post', return_value=mock_response)
        
        # This should not raise an error
        result = embeddings.create(model="text-embedding-bge-m3", input=small_input)
        
        # Verify the call was made
        embeddings._client.post.assert_called_once()
        call_args = embeddings._client.post.call_args
        assert call_args[0][0] == "embeddings"
        assert len(call_args[1]["json_data"]["input"]) == 3
    
    def test_create_with_single_string_no_validation(self, embeddings, mocker):
        """Test that single string inputs don't trigger array validation."""
        # Mock the client's post method
        mock_response = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
            "model": "text-embedding-bge-m3",
            "usage": {"prompt_tokens": 5, "total_tokens": 5}
        }
        mocker.patch.object(embeddings._client, 'post', return_value=mock_response)
        
        # This should not raise an error
        result = embeddings.create(model="text-embedding-bge-m3", input="Single text input")
        
        # Verify the call was made
        embeddings._client.post.assert_called_once()
        call_args = embeddings._client.post.call_args
        assert call_args[0][0] == "embeddings"
        assert call_args[1]["json_data"]["input"] == "Single text input"
    
    def test_create_with_nested_array_validation(self, embeddings):
        """Test that nested arrays (list of token lists) are also validated."""
        # Create a nested array with more than 2048 items
        large_nested_input = [[1, 2, 3]] * 2049
        
        with pytest.raises(InvalidRequestError) as exc_info:
            embeddings.create(model="text-embedding-bge-m3", input=large_nested_input)
        
        assert "input array must have 2048 or fewer items" in str(exc_info.value)
        assert "got 2049 items" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_async_create_with_array_exceeding_limit(self, async_embeddings):
        """Test that async arrays with more than 2048 items raise InvalidRequestError."""
        # Create an input array with 2049 items
        large_input = ["text"] * 2049
        
        with pytest.raises(InvalidRequestError) as exc_info:
            await async_embeddings.create(model="text-embedding-bge-m3", input=large_input)
        
        assert "input array must have 2048 or fewer items" in str(exc_info.value)
        assert "got 2049 items" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_async_create_with_array_at_limit(self, async_embeddings, mocker):
        """Test that async arrays with exactly 2048 items are accepted."""
        # Create an input array with exactly 2048 items
        max_input = ["text"] * 2048
        
        # Mock the client's post method with an async mock
        mock_response = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": i} for i in range(2048)],
            "model": "text-embedding-bge-m3",
            "usage": {"prompt_tokens": 10000, "total_tokens": 10000}
        }
        
        # Create an async mock
        async def mock_post(*args, **kwargs):
            return mock_response
        
        mocker.patch.object(async_embeddings._client, 'post', side_effect=mock_post)
        
        # This should not raise an error
        result = await async_embeddings.create(model="text-embedding-bge-m3", input=max_input)
        
        # Verify the call was made
        async_embeddings._client.post.assert_called_once()
        call_args = async_embeddings._client.post.call_args
        assert call_args[0][0] == "embeddings"
        assert len(call_args[1]["json_data"]["input"]) == 2048

    def test_create_with_empty_model(self, embeddings):
        """Test that create raises InvalidRequestError for empty model."""
        with pytest.raises(InvalidRequestError) as exc_info:
            embeddings.create(model="", input="test input")
        assert "model parameter is required and cannot be empty" in str(exc_info.value)

    def test_create_with_empty_input_string(self, embeddings):
        """Test that create raises InvalidRequestError for empty input string."""
        with pytest.raises(InvalidRequestError) as exc_info:
            embeddings.create(model="text-embedding-bge-m3", input="")
        assert "input cannot be empty" in str(exc_info.value)

    def test_create_with_empty_input_list(self, embeddings):
        """Test that create raises InvalidRequestError for empty input list."""
        with pytest.raises(InvalidRequestError) as exc_info:
            embeddings.create(model="text-embedding-bge-m3", input=[])
        assert "input cannot be empty" in str(exc_info.value)
    
    def test_create_with_dimensions(self, embeddings, mocker):
        """Test that dimensions parameter is correctly passed."""
        mocker.patch.object(embeddings._client, 'post', return_value={
            "object": "list", "data": [], "model": "text-embedding-bge-m3", "usage": {}
        })
        embeddings.create(model="text-embedding-bge-m3", input="test", dimensions=256)
        embeddings._client.post.assert_called_once()
        call_args = embeddings._client.post.call_args
        assert call_args[1]["json_data"]["dimensions"] == 256

    @pytest.mark.asyncio
    async def test_async_create_with_empty_model(self, async_embeddings):
        """Test that async create raises InvalidRequestError for empty model."""
        with pytest.raises(InvalidRequestError) as exc_info:
            await async_embeddings.create(model="", input="test input")
        assert "model parameter is required and cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_create_with_empty_input_string(self, async_embeddings):
        """Test that async create raises InvalidRequestError for empty input string."""
        with pytest.raises(InvalidRequestError) as exc_info:
            await async_embeddings.create(model="text-embedding-bge-m3", input="")
        assert "input cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_create_with_empty_input_list(self, async_embeddings):
        """Test that async create raises InvalidRequestError for empty input list."""
        with pytest.raises(InvalidRequestError) as exc_info:
            await async_embeddings.create(model="text-embedding-bge-m3", input=[])
        assert "input cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_create_with_dimensions(self, async_embeddings, mocker):
        """Test that async dimensions parameter is correctly passed."""
        async def mock_post(*args, **kwargs):
            return {"object": "list", "data": [], "model": "text-embedding-bge-m3", "usage": {}}
        mocker.patch.object(async_embeddings._client, 'post', side_effect=mock_post)
        
        await async_embeddings.create(model="text-embedding-bge-m3", input="test", dimensions=256)
        async_embeddings._client.post.assert_called_once()
        call_args = async_embeddings._client.post.call_args
        assert call_args[1]["json_data"]["dimensions"] == 256

    @pytest.mark.asyncio
    async def test_async_create_with_encoding_format(self, async_embeddings, mocker):
        """Test that async encoding_format parameter is correctly passed."""
        async def mock_post(*args, **kwargs):
            return {"object": "list", "data": [], "model": "text-embedding-bge-m3", "usage": {}}
        mocker.patch.object(async_embeddings._client, 'post', side_effect=mock_post)

        await async_embeddings.create(model="text-embedding-bge-m3", input="test", encoding_format="base64")
        async_embeddings._client.post.assert_called_once()
        call_args = async_embeddings._client.post.call_args
        assert call_args[1]["json_data"]["encoding_format"] == "base64"

    @pytest.mark.asyncio
    async def test_async_create_with_user(self, async_embeddings, mocker):
        """Test that async user parameter is correctly passed."""
        async def mock_post(*args, **kwargs):
            return {"object": "list", "data": [], "model": "text-embedding-bge-m3", "usage": {}}
        mocker.patch.object(async_embeddings._client, 'post', side_effect=mock_post)

        await async_embeddings.create(model="text-embedding-bge-m3", input="test", user="test-user")
        async_embeddings._client.post.assert_called_once()
        call_args = async_embeddings._client.post.call_args
        assert call_args[1]["json_data"]["user"] == "test-user"


class TestEmbeddingTypeDefinition:
    """Test that the Embedding type definition supports both List[float] and str."""
    
    def test_embedding_with_float_list(self):
        """Test that Embedding type accepts List[float] for embedding field."""
        embedding: Embedding = {
            "object": "embedding",
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "index": 0
        }
        
        assert isinstance(embedding["embedding"], list)
        assert all(isinstance(x, float) for x in embedding["embedding"])
    
    def test_embedding_with_base64_string(self):
        """Test that Embedding type accepts str for embedding field (base64)."""
        embedding: Embedding = {
            "object": "embedding",
            "embedding": "VGVzdCBzdHJpbmcgdG8gYmUgZW5jb2RlZCBpbnRvIGJhc2U2NA==",
            "index": 0
        }
        
        assert isinstance(embedding["embedding"], str)
    
    def test_embedding_list_with_mixed_formats(self):
        """Test that EmbeddingList can contain embeddings with different formats."""
        embedding_list: EmbeddingList = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3],
                    "index": 0
                },
                {
                    "object": "embedding",
                    "embedding": "VGVzdCBzdHJpbmcgdG8gYmUgZW5jb2RlZA==",
                    "index": 1
                }
            ],
            "model": "text-embedding-bge-m3",
            "usage": {
                "prompt_tokens": 10,
                "total_tokens": 10
            }
        }
        
        # Verify both formats are present
        assert isinstance(embedding_list["data"][0]["embedding"], list)
        assert isinstance(embedding_list["data"][1]["embedding"], str)


class TestUserParameterDocumentation:
    """Test that the user parameter documentation is accurate."""
    
    def test_embeddings_create_docstring_mentions_discarded(self):
        """Test that the create method docstring mentions user parameter is discarded."""
        # Check synchronous version
        sync_doc = Embeddings.create.__doc__
        assert sync_doc is not None
        assert "discarded by" in sync_doc
        assert "does not affect the response" in sync_doc
        
        # Check asynchronous version
        async_doc = AsyncEmbeddings.create.__doc__
        assert async_doc is not None
        assert "discarded by" in async_doc
        assert "does not affect the response" in async_doc
    
    def test_user_parameter_still_sent_in_request(self, mocker):
        """Test that user parameter is still included in the request for compatibility."""
        mock_client = MagicMock()
        mock_client._base_url = httpx.URL("https://api.venice.ai/api/v1/")
        embeddings = Embeddings(mock_client)
        
        # Mock the response
        mock_response = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
            "model": "text-embedding-bge-m3",
            "usage": {"prompt_tokens": 5, "total_tokens": 5}
        }
        mocker.patch.object(embeddings._client, 'post', return_value=mock_response)
        
        # Call with user parameter
        embeddings.create(
            model="text-embedding-bge-m3",
            input="Test input",
            user="test-user-123"
        )
        
        # Verify user was included in the request
        embeddings._client.post.assert_called_once()
        call_args = embeddings._client.post.call_args
        assert call_args[1]["json_data"]["user"] == "test-user-123"


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple improvements."""
    
    def test_payment_required_with_large_batch(self, mocker):
        """Test PaymentRequiredError when trying to process a large batch."""
        mock_client = MagicMock()
        mock_client._base_url = httpx.URL("https://api.venice.ai/api/v1/")
        embeddings = Embeddings(mock_client)
        
        # Create a large but valid batch (under 2048 limit)
        large_batch = ["text"] * 2000
        
        # Mock a 402 response
        mock_request = httpx.Request("POST", "https://api.venice.ai/api/v1/embeddings")
        mock_response = httpx.Response(
            402,
            request=mock_request,
            json={"error": {"message": "Insufficient USD or VCU balance"}}
        )
        
        # Mock the post method to raise PaymentRequiredError
        mocker.patch.object(
            embeddings._client,
            'post',
            side_effect=PaymentRequiredError(
                "Insufficient USD or VCU balance",
                request=mock_request,
                response=mock_response,
                body={"error": {"message": "Insufficient USD or VCU balance"}}
            )
        )
        
        with pytest.raises(PaymentRequiredError) as exc_info:
            embeddings.create(model="text-embedding-bge-m3", input=large_batch)
        
        assert "Insufficient USD or VCU balance" in str(exc_info.value)
        assert exc_info.value.status_code == 402
    
    def test_service_unavailable_with_base64_format(self, mocker):
        """Test ServiceUnavailableError when requesting base64 format."""
        mock_client = MagicMock()
        mock_client._base_url = httpx.URL("https://api.venice.ai/api/v1/")
        embeddings = Embeddings(mock_client)
        
        # Mock a 503 response
        mock_request = httpx.Request("POST", "https://api.venice.ai/api/v1/embeddings")
        mock_response = httpx.Response(
            503,
            request=mock_request,
            json={"error": {"message": "The model is at capacity"}}
        )
        
        # Mock the post method to raise ServiceUnavailableError
        mocker.patch.object(
            embeddings._client,
            'post',
            side_effect=ServiceUnavailableError(
                "The model is at capacity",
                request=mock_request,
                response=mock_response,
                body={"error": {"message": "The model is at capacity"}}
            )
        )
        
        with pytest.raises(ServiceUnavailableError) as exc_info:
            embeddings.create(
                model="text-embedding-bge-m3",
                input="Test input",
                encoding_format="base64"
            )
        
        assert "The model is at capacity" in str(exc_info.value)
        assert exc_info.value.status_code == 503


if __name__ == "__main__":
    pytest.main([__file__, "-v"])