import pytest
import httpx
import json
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, Mock
from typing import Dict, Any, AsyncIterator, List

from venice_ai._async_client import AsyncVeniceClient
from venice_ai.exceptions import (
    VeniceError, APIError, AuthenticationError, PermissionDeniedError,
    NotFoundError, RateLimitError, InternalServerError, InvalidRequestError
)

# Helper class to create proper async iterators for mocking aiter_lines and aiter_bytes
class AsyncIteratorMock:
    """Helper for mocking async iterators in tests."""
    
    def __init__(self, items):
        self.items = items
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        try:
            return next(self.items)
        except StopIteration:
            raise StopAsyncIteration

def create_async_iterator(items):
    """Creates a proper async iterator for use in mocking."""
    return AsyncIteratorMock(iter(items))

class TestAsyncClientExtended:
    """Additional tests for AsyncVeniceClient to improve coverage."""
    
    @pytest.fixture
    async def client(self):
        """Create a client with a mocked httpx client."""
        # Use AsyncMock directly for the client instance for better async defaults
        mock_httpx_instance = AsyncMock(spec=httpx.AsyncClient)
        
        # Ensure essential attributes are also AsyncMocks or configured correctly
        mock_httpx_instance.headers = MagicMock(spec=httpx.Headers) # Use MagicMock with spec instead of real Headers
        mock_httpx_instance.request = AsyncMock()
        
        # Important: stream must NOT be an AsyncMock but instead a properly configured MagicMock
        # because we need to prevent it from returning a coroutine
        mock_httpx_instance.stream = MagicMock()
        
        # Ensure stream returns an async context manager
        mock_stream_context_manager = AsyncMock()
        mock_httpx_instance.stream.return_value = mock_stream_context_manager
        
        # Properly setup the response returned by __aenter__
        mock_response = AsyncMock(spec=httpx.Response)
        # Explicitly set common attributes that might be accessed
        mock_response.status_code = 200
        mock_response.headers = MagicMock(spec=httpx.Headers)
        mock_response.content = b""
        mock_response.text = ""
        # Ensure raise_for_status is a sync method
        mock_response.raise_for_status = MagicMock()
        
        mock_stream_context_manager.__aenter__.return_value = mock_response
        mock_stream_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_httpx_instance.aclose = AsyncMock()

        client = AsyncVeniceClient(api_key="test-key")
        client._client = mock_httpx_instance # Assign the fully configured AsyncMock instance
        return client
    
    @pytest.mark.asyncio
    async def test_request_get_removes_content_type_and_accept(self, client):
        """Test that GET requests remove Content-Type and Accept headers if not explicitly provided."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.headers = {}
        mock_response.json = AsyncMock(return_value={"result": "success"})
        mock_response.aread = AsyncMock()
        mock_response.aclose = AsyncMock()
        mock_response.raise_for_status = MagicMock() # raise_for_status is sync
        mock_response.status_code = 200  # Explicitly set status_code
        client._client.request.return_value = mock_response
        
        # Set initial headers to verify they get removed
        client._client.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer test-key"
        }
        
        result = await client._request("GET", "test/endpoint")
        
        assert result == {"result": "success"}
        
        # Verify headers were removed for the GET request
        _, kwargs = client._client.request.call_args
        assert "Content-Type" not in kwargs["headers"]
        assert "Accept" not in kwargs["headers"]
        assert "Authorization" in kwargs["headers"]
    
    @pytest.mark.asyncio
    async def test_request_get_keeps_headers_if_explicitly_provided(self, client):
        """Test that GET requests keep Content-Type and Accept if explicitly provided in headers."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.headers = {}
        mock_response.json = AsyncMock(return_value={"result": "success"})
        mock_response.aread = AsyncMock()
        mock_response.aclose = AsyncMock()
        mock_response.raise_for_status = MagicMock() # raise_for_status is sync
        mock_response.status_code = 200  # Explicitly set status_code
        client._client.request.return_value = mock_response
        
        # Set initial client headers
        client._client.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer test-key"
        }
        
        # Provide explicit headers that should override
        explicit_headers = {
            "Content-Type": "text/plain",
            "Accept": "text/plain"
        }
        
        result = await client._request("GET", "test/endpoint", headers=explicit_headers)
        
        assert result == {"result": "success"}
        
        # Verify explicit headers were kept
        _, kwargs = client._client.request.call_args
        assert kwargs["headers"]["Content-Type"] == "text/plain"
        assert kwargs["headers"]["Accept"] == "text/plain"
    
    @pytest.mark.asyncio
    async def test_request_raw_response(self, client):
        """Test requesting raw binary response instead of parsed JSON."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.headers = {}
        mock_response.content = b"binary content"
        mock_response.raise_for_status = MagicMock() # raise_for_status is sync
        mock_response.status_code = 200  # Explicitly set status_code
        client._client.request.return_value = mock_response
        
        result = await client._request("GET", "test/endpoint", raw_response=True)
        
        assert result == b"binary content"
        assert not mock_response.json.called
    
    @pytest.mark.asyncio
    async def test_request_json_decode_error(self, client):
        """Test HTTP error with invalid JSON response."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.headers = {}
        http_error = httpx.HTTPStatusError(
            "Error", request=MagicMock(method="POST", url=httpx.URL("https://api.venice.ai/api/v1/test/endpoint")), response=mock_response
        )
        mock_response.raise_for_status.side_effect = http_error
        mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
        mock_response.aread = AsyncMock()
        mock_response.aclose = AsyncMock()
        mock_response.status_code = 400
        client._client.request.return_value = mock_response
        
        with pytest.raises(InvalidRequestError):
            await client._request("POST", "test/endpoint")
    
    @pytest.mark.asyncio
    async def test_request_timeout(self, client):
        """Test handling of timeout exceptions."""
        mock_httpx_request = MagicMock(spec=httpx.Request)
        mock_httpx_request.method = "POST"
        mock_httpx_request.url = httpx.URL("https://api.venice.ai/api/v1/test/endpoint")
        timeout_error = httpx.TimeoutException("Timeout", request=mock_httpx_request)
        
        # Create an async function that raises the exception
        async def async_raise_timeout(*args, **kwargs):
            raise timeout_error
        
        # Set the side_effect to the async function
        client._client.request.side_effect = async_raise_timeout
        
        with pytest.raises(VeniceError) as excinfo:
            await client._request("POST", "test/endpoint")
        
        assert "Request timed out" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_request_network_error(self, client):
        """Test handling of network errors."""
        mock_httpx_request = MagicMock(spec=httpx.Request)
        mock_httpx_request.method = "POST"
        mock_httpx_request.url = httpx.URL("https://api.venice.ai/api/v1/test/endpoint")
        network_error = httpx.NetworkError("Network error", request=mock_httpx_request)
        client._client.request.side_effect = network_error
        
        with pytest.raises(VeniceError) as excinfo:
            await client._request("POST", "test/endpoint")
        
        assert "Request failed" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_request_multipart(self, client):
        """Test making a multipart request for file uploads."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.headers = {}
        mock_response.json = AsyncMock(return_value={"result": "success"})
        mock_response.aread = AsyncMock()
        mock_response.aclose = AsyncMock()
        mock_response.raise_for_status = MagicMock() # raise_for_status is sync
        mock_response.status_code = 200  # Explicitly set status_code
        mock_response.headers = {"Content-Type": "application/json"}  # Explicitly set headers
        client._client.request.return_value = mock_response
        
        # Set initial client headers
        client._client.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer test-key",
            "User-Agent": "Test Agent"
        }
        
        files = {"file": ("test.jpg", b"file content", "image/jpeg")}
        data = {"model": "test-model"}
        
        result = await client._request_multipart("POST", "test/endpoint", files=files, data=data)
        
        assert result == {"result": "success"}
        
        # Verify the request was made with the right parameters
        client._client.request.assert_called_once()
        args, kwargs = client._client.request.call_args
        
        assert kwargs["method"] == "POST"
        assert "files" in kwargs
        assert kwargs["files"] == files
        assert kwargs["data"] == data
        
        # Verify headers were set correctly
        assert "Authorization" in kwargs["headers"]
        assert "User-Agent" in kwargs["headers"]
        assert "Content-Type" not in kwargs["headers"]  # Should be set by httpx for multipart
        assert kwargs["headers"]["Accept"] == "*/*"
    
    @pytest.mark.asyncio
    async def test_stream_request_empty_lines(self, client):
        """Test handling of empty lines in SSE stream."""
        # Define the lines to be yielded
        lines = ["", "data: {\"chunk\": 1}", "", "data: {\"chunk\": 2}", "data: [DONE]"]
        
        # Mock the response properly with status_code and other required attributes
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.headers = {}
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        
        # Set up the async iterator for lines
        mock_response.aiter_lines = MagicMock(side_effect=lambda: create_async_iterator(lines))
        mock_response.aread = AsyncMock()
        mock_response.aclose = AsyncMock()
        
        # Set up the context manager correctly
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_response
        mock_ctx.__aexit__.return_value = None
        
        # Use MagicMock for stream to avoid returning a coroutine
        client._client.stream = MagicMock(return_value=mock_ctx)
        
        chunks = []
        async for chunk in client._stream_request("POST", "test/endpoint"):
            chunks.append(chunk)
        
        assert len(chunks) == 2
        assert chunks[0] == {"chunk": 1}
        assert chunks[1] == {"chunk": 2}
    
    @pytest.mark.asyncio
    async def test_stream_request_invalid_json(self, client):
        """Test handling of invalid JSON in stream."""
        # Define the lines with invalid JSON to be yielded
        lines = [
            "data: {\"valid\": true}",
            "data: {invalid json}",  # This one should be skipped
            "data: {\"also_valid\": true}"
        ]
        
        # Mock the response properly with status_code and other required attributes
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.headers = {}
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        
        # Set up the async iterator for lines
        mock_response.aiter_lines = MagicMock(side_effect=lambda: create_async_iterator(lines))
        mock_response.aread = AsyncMock()
        mock_response.aclose = AsyncMock()
        
        # Set up the context manager correctly
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_response
        mock_ctx.__aexit__.return_value = None
        
        # Use MagicMock for stream to avoid returning a coroutine
        client._client.stream = MagicMock(return_value=mock_ctx)
        
        chunks = []
        async for chunk in client._stream_request("POST", "test/endpoint"):
            chunks.append(chunk)
        
        # We should only get the valid JSON chunks
        assert len(chunks) == 2
        assert chunks[0] == {"valid": True}
        assert chunks[1] == {"also_valid": True}
    
    @pytest.mark.asyncio
    async def test_stream_request_http_error(self, client):
        """Test handling of HTTP errors in stream requests."""
        # Mock the response properly with status_code and other required attributes
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 401  # Unauthorized
        
        # Create the HTTP error
        http_error = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(method="POST", url=httpx.URL("https://api.venice.ai/api/v1/test/endpoint")), response=mock_response
        )
        mock_response.raise_for_status = MagicMock(side_effect=http_error)
        
        # Set up the context manager correctly
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_response
        mock_ctx.__aexit__.return_value = None
        
        # Use MagicMock for stream to avoid returning a coroutine
        client._client.stream = MagicMock(return_value=mock_ctx)
        
        with pytest.raises(AuthenticationError):
            async for chunk in client._stream_request("POST", "test/endpoint"):
                pass
    
    @pytest.mark.asyncio
    async def test_stream_request_raw(self, client):
        """Test streaming raw binary data."""
        # Define the binary chunks to be yielded
        chunks = [b"chunk1", b"", b"chunk2"]  # Empty chunk should be skipped
        
        # Mock the response properly with status_code and other required attributes
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.headers = {}
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        
        # Set up the async iterator for bytes
        mock_response.aiter_bytes = MagicMock(side_effect=lambda: create_async_iterator(chunks))
        mock_response.aread = AsyncMock()
        mock_response.aclose = AsyncMock()
        
        # Set up the context manager correctly
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_response
        mock_ctx.__aexit__.return_value = None
        
        # Use MagicMock for stream to avoid returning a coroutine
        client._client.stream = MagicMock(return_value=mock_ctx)
        
        received_chunks = []
        async for chunk in client._stream_request_raw("POST", "test/endpoint"):
            received_chunks.append(chunk)
        
        # Empty chunk should be skipped
        assert len(received_chunks) == 2
        assert received_chunks[0] == b"chunk1"
        assert received_chunks[1] == b"chunk2"
    
    @pytest.mark.asyncio
    async def test_close_and_context_manager(self):
        """Test client close method and context manager functionality."""
        client = AsyncVeniceClient(api_key="test-key")
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        client._client = mock_client
        
        # Test close method
        await client.close()
        mock_client.aclose.assert_called_once()
        
        # Test context manager
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        with patch("httpx.AsyncClient", return_value=mock_client):
            async with AsyncVeniceClient(api_key="test-key") as client:
                assert isinstance(client, AsyncVeniceClient)
                
            # Verify client was closed on exit
            mock_client.aclose.assert_called_once()