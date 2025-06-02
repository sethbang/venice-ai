import pytest
import httpx
import json
import io
from typing import AsyncIterator, cast, Dict, Any
from unittest.mock import patch, MagicMock, AsyncMock
from venice_ai.exceptions import APIError # Import APIError
import tempfile
from pathlib import Path

from venice_ai._client import VeniceClient
from venice_ai._async_client import AsyncVeniceClient
from venice_ai.exceptions import VeniceError, AuthenticationError, InvalidRequestError

class TestVeniceClientEnhanced:
    """Enhanced tests for VeniceClient to improve code coverage."""

    def test_client_with_external_http_client(self):
        """Test initializing client with external HTTP client."""
        external_client = httpx.Client(timeout=30.0)
        client = VeniceClient(api_key="test-api-key", http_client=external_client)
        
        # Should use the provided client
        assert client._client == external_client
    
    def test_close_multiple_times(self):
        """Test calling close() multiple times."""
        client = VeniceClient(api_key="test-api-key")
        
        # First close should work
        client.close()
        
        # Second close should not raise an error
        client.close()
    
    def test_client_with_http_transport_options(self):
        """Test client with custom HTTP transport options."""
        with patch('httpx.HTTPTransport') as mock_transport, patch('httpx.Client') as mock_client:
            # Configure the mock HTTP transport
            mock_transport_instance = MagicMock()
            mock_transport.return_value = mock_transport_instance
            
            client = VeniceClient(
                api_key="test-api-key",
                base_url="https://custom.api.com",
                timeout=15.0,
                max_retries=3
            )
            
            # Validate HTTPTransport was created with expected retries
            mock_transport.assert_called_once_with(retries=3)
            
            # Validate client was created with expected args
            mock_client.assert_called_once()
            _, kwargs = mock_client.call_args
            
            assert kwargs["timeout"].read == 15.0
            assert kwargs["transport"] == mock_transport_instance
    
    def test_request_with_null_json(self):
        """Test _request with None json_data."""
        with patch('httpx.Client') as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {"status": "success"}
            mock_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            result = client._request("GET", "test/path", json_data=None)
            
            # Should successfully make request
            assert result == {"status": "success"}
            mock_request = cast(MagicMock, client._client.request)
            mock_request.assert_called_once()
            _, kwargs = mock_request.call_args
            assert kwargs["json"] is None
    
    def test_request_with_get_content_type_handling(self):
        """Test GET request with automatic content-type header handling."""
        with patch('httpx.Client') as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {"status": "success"}
            mock_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            
            # Set default headers including Content-Type
            client._client.headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer test-api-key"
            }
            
            # Make GET request without explicit headers
            result = client._request("GET", "test/path")
            
            # Should remove Content-Type for GET request
            assert result == {"status": "success"}
            cast(MagicMock, client._client.request).assert_called_once()
            _, kwargs = cast(MagicMock, client._client.request).call_args
            headers = kwargs["headers"]
            # Content-Type should be removed for GET
            assert "Content-Type" not in headers
            assert "Accept" not in headers
    
    def test_request_with_custom_headers_overriding_defaults(self):
        """Test request with custom headers overriding defaults."""
        with patch('httpx.Client') as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {"status": "success"}
            mock_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            
            # Set default headers
            client._client.headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer test-api-key",
                "User-Agent": "VeniceClient/1.0"
            }
            
            # Custom headers should override defaults
            custom_headers = {
                "Content-Type": "application/xml",
                "Accept": "application/xml",
                "X-Custom": "value"
            }
            
            result = client._request("POST", "test/path", headers=custom_headers)
            
            assert result == {"status": "success"}
            cast(MagicMock, client._client.request).assert_called_once()
            _, kwargs = cast(MagicMock, client._client.request).call_args
            headers = kwargs["headers"]
            assert headers["Content-Type"] == "application/xml"
            assert headers["Accept"] == "application/xml"
            assert headers["X-Custom"] == "value"
            assert headers["Authorization"] == "Bearer test-api-key"
            assert headers["User-Agent"] == "VeniceClient/1.0"
    
    def test_request_raw_response(self):
        """Test _request with raw_response=True."""
        with patch('httpx.Client') as mock_client:
            mock_response = MagicMock()
            mock_response.content = b"binary data"
            mock_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            result = client._request("GET", "file/download", raw_response=True)
            
            assert result == b"binary data"
            # Should not call json() on response
            mock_response.json.assert_not_called()
    
    def test_request_json_decode_error(self):
        """Test _request with JSON decode error."""
        with patch('httpx.Client') as mock_client:
            mock_response = MagicMock()
            # Simulate JSON decode error
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_response.text = "Not JSON"
            mock_response.status_code = 200  # Add status_code
            mock_response.headers = {}  # Add headers attribute to prevent AttributeError
            mock_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            
            # Should propagate JSONDecodeError
            with pytest.raises(json.JSONDecodeError):
                client._request("POST", "test/path")
    
    def test_request_multipart_handles_error_response_non_json(self):
        """Test _request_multipart with error response that's not JSON."""
        with patch('httpx.Client') as mock_client:
            # Create mock response with 400 error
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 400
            mock_response.request = MagicMock(spec=httpx.Request)
            # Add method and url attributes to the mock request
            mock_response.request.method = "POST"
            mock_response.request.url = httpx.URL("https://api.venice.ai/api/v1/upload")
            mock_response.text = "Error: Bad Request"  # Not JSON
            mock_response.headers = {}  # Add headers attribute to prevent AttributeError
            
            # Make response.raise_for_status raise exception
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                message="HTTP Error 400: Bad Request",
                request=mock_response.request,
                response=mock_response
            )
            
            # Make response.json raise JSONDecodeError
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            
            mock_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            
            # Call _request_multipart which should handle the error
            with pytest.raises(InvalidRequestError):
                files = {"file": ("test.jpg", b"image data", "image/jpeg")}
                client._request_multipart("POST", "upload", files=files)
    
    def test_request_multipart_with_pathlib_path(self):
        """Test _request_multipart with pathlib.Path file path."""
        with patch('httpx.Client') as mock_client, tempfile.NamedTemporaryFile(suffix='.jpg') as tmp_file:
            # Write test data to temp file
            tmp_file.write(b"test image data")
            tmp_file.flush()
            
            # Setup mock response
            mock_response = MagicMock()
            mock_response.json.return_value = {"status": "success"}
            mock_response.status_code = 200  # Add status_code
            mock_response.headers = {}  # Add headers attribute to prevent AttributeError
            mock_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            
            # Use pathlib.Path instead of string
            file_path = Path(tmp_file.name)
            result = client._request_multipart(
                "POST",
                "images/upload",
                files={"file": (file_path.name, file_path.read_bytes(), "image/jpeg")}
            )
            
            assert result == {"status": "success"}
            cast(MagicMock, client._client.request).assert_called_once()
    
    def test_stream_request_with_empty_response(self):
        """Test _stream_request with empty response."""        
        with patch('httpx.Client') as mock_client:
            # Create a mock response with lines iterator that returns empty sequence
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.iter_lines.return_value = []
            mock_response.status_code = 200  # Add status_code
            mock_response.headers = {}  # Add headers attribute to prevent AttributeError
            
            # Setup for context manager mock
            mock_stream_context = MagicMock()
            mock_stream_context.__enter__.return_value = mock_response
            mock_stream_context.__exit__.return_value = None
            mock_client.return_value.stream.return_value = mock_stream_context
            
            client = VeniceClient(api_key="test-api-key")
            
            # Should handle empty response
            chunks = list(client._stream_request("POST", "chat/completions", json_data={"model": "venice-1"}))
            
            assert len(chunks) == 0
            cast(MagicMock, client._client.stream).assert_called_once()
    
    def test_stream_request_with_malformed_sse(self):
        """Test _stream_request with malformed SSE data."""
        with patch('httpx.Client') as mock_client:
            # Create a mock response with lines that don't follow SSE format
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 200  # Add status_code
            mock_response.headers = {}  # Add headers attribute to prevent AttributeError
            mock_response.iter_lines.return_value = [
                b"not a valid SSE line",
                b"data: malformed JSON {",
                b"data: {\"valid\": \"json\"}"
            ]
            
            # Setup for context manager mock
            mock_stream_context = MagicMock()
            mock_stream_context.__enter__.return_value = mock_response
            mock_stream_context.__exit__.return_value = None
            mock_client.return_value.stream.return_value = mock_stream_context
            
            client = VeniceClient(api_key="test-api-key")
            
            # Should skip malformed lines
            chunks = list(client._stream_request("POST", "chat/completions", json_data={"model": "venice-1"}))
            
            assert len(chunks) == 1
            assert chunks[0] == {"valid": "json"}


@pytest.mark.asyncio
class TestAsyncVeniceClientEnhanced:
    """Enhanced tests for AsyncVeniceClient to improve code coverage."""

    async def test_client_with_external_http_client(self):
        """Test initializing client with external HTTP client."""
        external_client = httpx.AsyncClient(timeout=30.0)
        client = AsyncVeniceClient(api_key="test-api-key", http_client=external_client)
        
        # Should use the provided client
        assert client._client == external_client
        
        # Clean up
        await external_client.aclose()
    
    async def test_close_multiple_times(self):
        """Test calling close() multiple times."""
        with patch('httpx.AsyncClient'):
            client = AsyncVeniceClient(api_key="test-api-key")
            client._client.aclose = AsyncMock()
            
            # Debug logs to validate the diagnosis
            print(f"DEBUG: Initial _is_closed value: {client._is_closed}")
            
            # First close should work
            await client.close()
            client._client.aclose.assert_called_once()
            print(f"DEBUG: _is_closed after first close: {client._is_closed}")
            
            # Reset mock to check second call
            client._client.aclose.reset_mock()
            
            # Second close should not raise an error
            await client.close()
            print(f"DEBUG: _is_closed after second close: {client._is_closed}")
            client._client.aclose.assert_not_called()  # Should not call aclose again if already closed
    
    async def test_client_with_http_transport_options(self):
        """Test client with custom HTTP transport options."""
        with patch('httpx.AsyncHTTPTransport') as mock_transport, patch('httpx.AsyncClient') as mock_client:
            # Configure the mock HTTP transport
            mock_transport_instance = MagicMock()
            mock_transport.return_value = mock_transport_instance
            
            client = AsyncVeniceClient(
                api_key="test-api-key",
                base_url="https://custom.api.com",
                timeout=15.0,
                max_retries=3
            )
            
            # Validate AsyncHTTPTransport was created with expected retries
            mock_transport.assert_called_once_with(retries=3)
            
            # Validate client was created with expected args
            mock_client.assert_called_once()
            _, kwargs = mock_client.call_args
            
            # Check the timeout was set correctly
            assert kwargs["timeout"].read == 15.0
            
            # Check that our mocked transport instance was passed to the client
            assert kwargs["transport"] == mock_transport_instance
    
    async def test_request_with_null_json(self):
        """Test _request with None json_data."""
        with patch('httpx.AsyncClient') as mock_client:
            # Create a proper mock setup for async response
            mock_response = MagicMock()
            # Configure json method to return a properly awaitable response
            mock_response.json = AsyncMock()
            mock_response.json.return_value = {"status": "success"}
            
            # Configure the client's request method
            mock_client.return_value.request = AsyncMock()
            mock_client.return_value.request.return_value = mock_response
            
            client = AsyncVeniceClient(api_key="test-api-key")
            result = await client._request("GET", "test/path", json_data=None)
            
            # Should successfully make request
            assert result == {"status": "success"}
            mock_client.return_value.request.assert_awaited_once()
            _, kwargs = mock_client.return_value.request.call_args
            assert kwargs["json"] is None
    
    async def test_request_with_get_content_type_handling(self):
        """Test GET request with automatic content-type header handling."""
        with patch('httpx.AsyncClient') as mock_client:
            # Configure mock response with proper async behavior
            mock_response = MagicMock()
            mock_response.json = AsyncMock()
            mock_response.json.return_value = {"status": "success"}
            
            # Configure request mock properly
            mock_client.return_value.request = AsyncMock()
            mock_client.return_value.request.return_value = mock_response
            
            client = AsyncVeniceClient(api_key="test-api-key")
            
            # Set default headers including Content-Type
            client._client.headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer test-api-key"
            }
            
            # Make GET request without explicit headers
            result = await client._request("GET", "test/path")
            
            # Should remove Content-Type for GET request
            assert result == {"status": "success"}
            mock_client.return_value.request.assert_awaited_once()
            _, kwargs = mock_client.return_value.request.call_args
            headers = kwargs["headers"]
            # Content-Type should be removed for GET
            assert "Content-Type" not in headers
            assert "Accept" not in headers
    
    async def test_stream_request_with_malformed_sse(self):
        """Test _stream_request with malformed SSE data."""
        # Mock the async context manager
        mock_response = AsyncMock()
        mock_response.aiter_lines = AsyncMock()
        
        # Create an async iterator for the lines
        async def mock_aiter_lines():
            lines = [
                b"not a valid SSE line",
                b"data: malformed JSON {",
                b"data: {\"valid\": \"json\"}"
            ]
            for line in lines:
                yield line
                
        # Debug logs to validate the diagnosis
        print(f"DEBUG: Type of mock_aiter_lines: {type(mock_aiter_lines)}")
        print(f"DEBUG: Type of mock_aiter_lines(): {type(mock_aiter_lines())}")
        # Correctly assign the function itself as the attribute
        mock_response.aiter_lines = mock_aiter_lines
        mock_response.raise_for_status = MagicMock()  # Synchronous method
        
        # Setup context manager for stream
        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_context.__aexit__ = AsyncMock(return_value=None)
        
        client = AsyncVeniceClient(api_key="test-api-key")
        client._client.stream = MagicMock(return_value=mock_stream_context)
        
        # Collect chunks from the stream
        chunks = []
        async for chunk in client._stream_request("POST", "chat/completions", json_data={"model": "venice-1"}):
            chunks.append(chunk)
        
        # Should only get the one valid JSON chunk
        assert len(chunks) == 1
        assert chunks[0] == {"valid": "json"}
        
    async def test_stream_request_raw_with_empty_chunks(self):
        """Test _stream_request_raw with empty chunks."""
        # Mock the async context manager
        mock_response = AsyncMock()
        
        # Create an async iterator for bytes
        async def mock_aiter_bytes():
            chunks = [b"", b"chunk1", b"", b"chunk2", b""]
            for chunk in chunks:
                yield chunk
                
        # Debug logs to validate the diagnosis
        print(f"DEBUG: Type of mock_aiter_bytes: {type(mock_aiter_bytes)}")
        print(f"DEBUG: Type of mock_aiter_bytes(): {type(mock_aiter_bytes())}")
        # Correctly assign the function itself as the attribute
        mock_response.aiter_bytes = mock_aiter_bytes
        mock_response.raise_for_status = MagicMock()  # Synchronous method
        
        # Setup context manager for stream
        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_context.__aexit__ = AsyncMock(return_value=None)
        
        client = AsyncVeniceClient(api_key="test-api-key")
        client._client.stream = MagicMock(return_value=mock_stream_context)
        
        # Collect chunks from the stream
        chunks = []
        async for chunk in client._stream_request_raw("POST", "audio/speech", json_data={"text": "Hello"}):
            chunks.append(chunk)
        
        # Should only get the non-empty chunks
        assert len(chunks) == 2
        assert chunks[0] == b"chunk1"
        assert chunks[1] == b"chunk2"
    
    async def test_request_multipart_with_io_bytesio(self):
        """Test _request_multipart with io.BytesIO file object."""
        with patch('httpx.AsyncClient') as mock_client:
            # Create file-like object
            file_obj = io.BytesIO(b"test image data")
            
            # Setup mock response with proper async behavior
            mock_response = MagicMock()
            mock_response.json = AsyncMock()
            mock_response.json.return_value = {"status": "success"}
            
            # Configure the client's request method directly
            mock_client.return_value.request = AsyncMock()
            mock_client.return_value.request.return_value = mock_response
            
            client = AsyncVeniceClient(api_key="test-api-key")
            
            # Use file-like object
            result = await client._request_multipart(
                "POST",
                "images/upload",
                files={"file": ("test.jpg", file_obj, "image/jpeg")}
            )
            
            assert result == {"status": "success"}
            mock_client.return_value.request.assert_awaited_once()
            
            # Verify file was included in the request
            _, kwargs = mock_client.return_value.request.call_args
            assert "files" in kwargs
            assert "file" in kwargs["files"]
    
    async def test_stream_request_with_headers_and_params(self):
        """Test _stream_request with custom headers and URL parameters."""
        # Mock the response
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        
        # Create a proper async iterator for aiter_lines
        async def mock_aiter_lines() -> AsyncIterator[bytes]:
            # Empty async generator that yields bytes
            if False:  # This ensures the generator is properly typed but never runs
                yield b""
            return
        
        # Set the aiter_lines attribute to our async generator function
        mock_response.aiter_lines = mock_aiter_lines
        
        # Setup context manager
        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_context.__aexit__ = AsyncMock(return_value=None)
        
        client = AsyncVeniceClient(api_key="test-api-key")
        client._client.stream = MagicMock(return_value=mock_stream_context)
        
        # Custom headers and params
        headers = {"X-Custom": "value"}
        params = {"param1": "value1"}
        
        # Call with custom values
        chunks = []
        async for chunk in client._stream_request(
            "POST",
            "chat/completions",
            json_data={"model": "venice-1"},
            headers=headers,
            params=params
        ):
            chunks.append(chunk)
        
        # Verify the stream was called with our custom values
        client._client.stream.assert_called_once()
        _, kwargs = client._client.stream.call_args
        assert headers.items() <= kwargs["headers"].items()  # Check if custom headers are a subset
        assert kwargs["params"] == params