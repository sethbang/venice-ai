"""
Additional targeted tests for AsyncVeniceClient to improve code coverage.

This module focuses on edge cases and specific scenarios that might
help improve test coverage for the async client specifically.
"""

import pytest
import httpx
import json
import logging
from unittest.mock import patch, MagicMock, AsyncMock
from typing import AsyncIterator, Any

from venice_ai._async_client import AsyncVeniceClient
from venice_ai.exceptions import VeniceError


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

def create_async_iterator_mock(items):
    """Creates a proper async iterator for use in mocking."""
    return AsyncIteratorMock(iter(items))


@pytest.mark.asyncio
class TestAsyncClientCoverageAdditional:
    """Additional tests targeting specific coverage gaps in AsyncVeniceClient."""

    async def test_initialization_with_external_client_attributes(self):
        """Test that client attributes are properly handled with external client."""
        external_client = httpx.AsyncClient(
            base_url="https://custom-base.api/",
            timeout=30.0,
            headers={"User-Agent": "CustomAgent/1.0"}
        )
        
        client = AsyncVeniceClient(
            api_key="test-key", 
            http_client=external_client
        )
        
        # Should use the external client's attributes
        assert client._client == external_client
        assert client._client.headers["Authorization"] == f"Bearer test-key"
        
        # Clean up
        await external_client.aclose()

    async def test_request_json_decode_error(self):
        """Test handling of JSONDecodeError in request."""
        original_httpx_async_client = httpx.AsyncClient # Store original
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncClientClass:
            # Setup mock client
            mock_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original
            MockAsyncClientClass.return_value = mock_client_instance
            mock_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_client_instance.aclose = AsyncMock()
            mock_client = mock_client_instance # maintain variable name for minimal diff
            
            # Setup mock response
            mock_response = MagicMock(spec=httpx.Response) # Add spec
            mock_response.status_code = 400 # Set status_code
            mock_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
                "Error", request=MagicMock(method="GET", url=httpx.URL("https://api.venice.ai/api/v1/test/endpoint")), response=mock_response
            ))
            mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
            mock_response.aread = AsyncMock()
            mock_response.aclose = AsyncMock()
            mock_response.text = "Not JSON"
            mock_client.request.return_value = mock_response
            
            client = AsyncVeniceClient(api_key="test-key")
            
            # Test that JSONDecodeError is handled
            with pytest.raises(VeniceError):
                await client._request("GET", "test/endpoint")

    async def test_request_timeout_exception(self):
        """Test handling of TimeoutException in request."""
        original_httpx_async_client = httpx.AsyncClient # Store original
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncClientClass:
            # Setup mock client
            mock_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original
            MockAsyncClientClass.return_value = mock_client_instance
            mock_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_client_instance.aclose = AsyncMock()
            mock_client = mock_client_instance # maintain variable name for minimal diff
            
            # Setup timeout exception with proper request attribute
            # 1. Create a mock httpx.Request object
            mock_httpx_request = MagicMock(spec=httpx.Request)
            mock_httpx_request.method = "GET"
            mock_httpx_request.url = httpx.URL("https://api.venice.ai/api/v1/test/endpoint")
            
            # 2. Instantiate the TimeoutException
            timeout_exception_instance = httpx.TimeoutException("Request timed out")
            
            # 3. Manually set the internal _request attribute
            timeout_exception_instance._request = mock_httpx_request
            
            # 4. Assign the configured exception as the side_effect
            mock_client.request.side_effect = timeout_exception_instance
            
            client = AsyncVeniceClient(api_key="test-key")
            
            # Test that TimeoutException is handled
            with pytest.raises(VeniceError) as exc_info:
                await client._request("GET", "test/endpoint")
            
            assert "Request timed out" in str(exc_info.value)

    async def test_request_request_error(self):
        """Test handling of RequestError in request."""
        original_httpx_async_client = httpx.AsyncClient # Store original
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncClientClass:
            # Setup mock client
            mock_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original
            MockAsyncClientClass.return_value = mock_client_instance
            mock_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_client_instance.aclose = AsyncMock()
            mock_client = mock_client_instance # maintain variable name for minimal diff
            
            # Setup request exception
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "GET"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test/endpoint")
            mock_client.request.side_effect = httpx.RequestError("Connection failed", request=mock_request)
            
            client = AsyncVeniceClient(api_key="test-key")
            
            # Test that RequestError is handled
            with pytest.raises(VeniceError) as exc_info:
                await client._request("GET", "test/endpoint")
            
            assert "Request failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_request_json_decode_error(self):
        """Test handling of JSONDecodeError in stream request."""
        original_httpx_async_client = httpx.AsyncClient # Store original
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncClientClass:
            # Setup mock client and response
            mock_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original
            MockAsyncClientClass.return_value = mock_client_instance
            mock_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_client_instance.aclose = AsyncMock()
            mock_client = mock_client_instance # maintain variable name for minimal diff
            mock_response = AsyncMock()
            mock_client.stream.return_value.__aenter__.return_value = mock_response
            
            # Setup line response with invalid JSON using proper async iterator
            iterator_mock = create_async_iterator_mock(["data: {invalid json}"])
            # Use MagicMock with side_effect to directly return the iterator (not a coroutine)
            mock_response.aiter_lines = MagicMock(side_effect=lambda: iterator_mock)
            mock_response.aread = AsyncMock()
            mock_response.aclose = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            
            client = AsyncVeniceClient(api_key="test-key")
            
            # Consume the stream iterator to trigger the JSONDecodeError
            chunks = []
            async for chunk in client._stream_request("POST", "chat/completions"):
                chunks.append(chunk)
            
            # Should not have added any chunks due to the JSONDecodeError
            assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_stream_request_http_status_error(self):
        """Test handling of HTTPStatusError in stream request."""
        original_httpx_async_client = httpx.AsyncClient # Store original
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncClientClass:
            # Setup mock client
            mock_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original
            MockAsyncClientClass.return_value = mock_client_instance
            mock_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_client_instance.aclose = AsyncMock()
            mock_client = mock_client_instance # maintain variable name for minimal diff
            
            # Setup response to raise HTTPStatusError
            mock_response = MagicMock()  # Use MagicMock for sync methods
            # Add proper async iterator (empty because it will raise error before iteration)
            iterator_mock = create_async_iterator_mock([])
            mock_response.aiter_lines = MagicMock(side_effect=lambda: iterator_mock)
            mock_response.aread = AsyncMock()
            mock_response.aclose = AsyncMock()
            # Make sure status_code is an actual integer
            mock_response.status_code = 404
            
            # Create a request mock with proper attributes
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
            
            # Use MagicMock instead of AsyncMock for raise_for_status
            mock_error = httpx.HTTPStatusError(
                "404 Not Found", request=mock_request, response=mock_response
            )
            mock_response.raise_for_status = MagicMock(side_effect=mock_error)
            mock_client.stream.return_value.__aenter__.return_value = mock_response
            
            client = AsyncVeniceClient(api_key="test-key")
            
            # Test that HTTPStatusError is handled
            with pytest.raises(VeniceError):
                async for _ in client._stream_request("POST", "chat/completions"):
                    pass

    @pytest.mark.asyncio
    async def test_stream_request_timeout_exception(self):
        """Test handling of TimeoutException in stream request."""
        original_httpx_async_client = httpx.AsyncClient # Store original
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncClientClass:
            # Setup mock client
            mock_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original
            MockAsyncClientClass.return_value = mock_client_instance
            mock_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_client_instance.aclose = AsyncMock()
            mock_client = mock_client_instance # maintain variable name for minimal diff
            
            # Setup timeout exception with proper request attribute
            # 1. Create a mock httpx.Request object
            mock_httpx_request = MagicMock(spec=httpx.Request)
            mock_httpx_request.method = "POST"
            mock_httpx_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
            
            # 2. Instantiate the TimeoutException
            timeout_exception_instance = httpx.TimeoutException("Stream timed out")
            
            # 3. Manually set the internal _request attribute
            timeout_exception_instance._request = mock_httpx_request
            
            # 4. Assign the configured exception as the side_effect
            mock_client.stream.side_effect = timeout_exception_instance
            
            client = AsyncVeniceClient(api_key="test-key")
            
            # Test that TimeoutException is handled
            with pytest.raises(VeniceError) as exc_info:
                async for _ in client._stream_request("POST", "chat/completions"):
                    pass
            
            assert "Stream request timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_request_request_error(self):
        """Test handling of RequestError in stream request."""
        original_httpx_async_client = httpx.AsyncClient # Store original
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncClientClass:
            # Setup mock client
            mock_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original
            MockAsyncClientClass.return_value = mock_client_instance
            mock_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_client_instance.aclose = AsyncMock()
            mock_client = mock_client_instance # maintain variable name for minimal diff
            
            # Setup request exception
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
            mock_client.stream.side_effect = httpx.RequestError("Stream connection failed", request=mock_request)
            
            client = AsyncVeniceClient(api_key="test-key")
            
            # Test that RequestError is handled
            with pytest.raises(VeniceError) as exc_info:
                async for _ in client._stream_request("POST", "chat/completions"):
                    pass
            
            assert "Stream request failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_request_raw_errors(self):
        """Test error handling in _stream_request_raw."""
        original_httpx_async_client = httpx.AsyncClient # Store original
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncClientClass:
            # Setup mock client
            mock_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original
            MockAsyncClientClass.return_value = mock_client_instance
            mock_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_client_instance.aclose = AsyncMock()
            mock_client = mock_client_instance # maintain variable name for minimal diff
            
            # Test HTTP Status Error
            mock_response = MagicMock()  # Use MagicMock for sync methods
            # Add proper async iterator for bytes with empty iterator
            iterator_mock = create_async_iterator_mock([])
            mock_response.aiter_bytes = AsyncMock(side_effect=lambda: iterator_mock)
            mock_response.aread = AsyncMock()
            mock_response.aclose = AsyncMock()
            # Make sure status_code is an actual integer
            mock_response.status_code = 404
            
            # Create a request mock with proper attributes
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/audio/speech")
            
            # Use MagicMock instead of AsyncMock for raise_for_status
            mock_error = httpx.HTTPStatusError(
                "404 Not Found", request=mock_request, response=mock_response
            )
            mock_response.raise_for_status = MagicMock(side_effect=mock_error)
            mock_client.stream.return_value.__aenter__.return_value = mock_response
            
            client = AsyncVeniceClient(api_key="test-key")
            
            with pytest.raises(VeniceError):
                async for _ in client._stream_request_raw("POST", "audio/speech"):
                    pass

            # Test TimeoutException with proper request attribute
            # 1. Create a mock httpx.Request object
            mock_httpx_request = MagicMock(spec=httpx.Request)
            mock_httpx_request.method = "POST"
            mock_httpx_request.url = httpx.URL("https://api.venice.ai/api/v1/audio/speech")
            
            # 2. Instantiate the TimeoutException
            timeout_exception_instance = httpx.TimeoutException("Stream timed out")
            
            # 3. Manually set the internal _request attribute
            timeout_exception_instance._request = mock_httpx_request
            
            # 4. Assign the configured exception as the side_effect
            mock_client.stream.side_effect = timeout_exception_instance
            
            with pytest.raises(VeniceError) as exc_info:
                async for _ in client._stream_request_raw("POST", "audio/speech"):
                    pass
            
            assert "Stream request timed out" in str(exc_info.value)
            
            # Test RequestError
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/audio/speech")
            mock_client.stream.side_effect = httpx.RequestError("Stream connection failed", request=mock_request)
            
            with pytest.raises(VeniceError) as exc_info:
                async for _ in client._stream_request_raw("POST", "audio/speech"):
                    pass
            
            assert "Stream request failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_request_multipart(self, caplog):
        """Test that _request_multipart properly handles headers and logging."""
        # Define original_httpx_async_client at the function level
        original_httpx_async_client = httpx.AsyncClient
        
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncClientClass, \
             caplog.at_level(logging.DEBUG):
            
            # Setup mock client
            mock_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncClientClass.return_value = mock_client_instance
            # mock_client_instance.headers and .aclose are already set by the previous diff application for this test
            mock_client = mock_client_instance # maintain variable name for minimal diff
            
            # Setup mock response
            mock_response = MagicMock()  # Use MagicMock instead of AsyncMock for sync methods
            mock_response.json = AsyncMock(return_value={"status": "success"})
            mock_response.aread = AsyncMock()
            mock_response.aclose = AsyncMock()
            mock_response.text = '{"status": "success"}'
            mock_response.status_code = 200
            mock_response.headers = {"Content-Type": "application/json"}
            mock_client.request.return_value = mock_response
            
            # Initialize client with Auth and User-Agent
            client = AsyncVeniceClient(api_key="test-key")
            client._client.headers = {
                "Authorization": "Bearer test-key",
                "User-Agent": "TestAgent/1.0"
            }
            
            # Make a multipart request
            files = {"file": ("test.jpg", b"image data", "image/jpeg")}
            data = {"parameter": "value"}
            
            result = await client._request_multipart(
                "POST", 
                "test/upload", 
                files=files,
                data=data,
                headers={"Custom-Header": "Value"},
                params={"query": "param"}
            )
            
            # Verify result
            assert result == {"status": "success"}
            
            # Verify request was made with correct parameters
            mock_client.request.assert_called_once()
            call_args = mock_client.request.call_args[1]
            
            assert call_args["method"] == "POST"
            assert "test/upload" in str(call_args["url"])
            assert call_args["files"] == files
            assert call_args["data"] == data
            assert call_args["params"] == {"query": "param"}
            assert call_args["headers"]["Authorization"] == "Bearer test-key"
            assert call_args["headers"]["User-Agent"] == "TestAgent/1.0"
            assert call_args["headers"]["Custom-Header"] == "Value"

    @pytest.mark.asyncio
    async def test_request_multipart_errors(self):
        """Test error handling in _request_multipart."""
        original_httpx_async_client = httpx.AsyncClient # Store original
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncClientClass:
            # Setup mock client
            mock_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original
            MockAsyncClientClass.return_value = mock_client_instance
            mock_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_client_instance.aclose = AsyncMock()
            mock_client = mock_client_instance # maintain variable name for minimal diff
            
            # Test HTTP Status Error with JSON response
            mock_response = MagicMock()  # Use MagicMock for sync methods
            mock_error = httpx.HTTPStatusError(
                "400 Bad Request", request=MagicMock(method="POST", url=httpx.URL("https://api.venice.ai/api/v1/test/upload")), response=mock_response
            )
            mock_response.raise_for_status.side_effect = mock_error
            mock_response.json = AsyncMock(return_value={"error": "Invalid request"})
            mock_response.aread = AsyncMock()
            mock_response.aclose = AsyncMock()
            mock_response.text = '{"error": "Invalid request"}'
            mock_response.status_code = 400
            mock_client.request.return_value = mock_response
            
            # Mock _make_status_error to return VeniceError
            with patch('venice_ai._async_client._make_status_error') as mock_make_error:
                mock_make_error.return_value = VeniceError("Test error")
                
                client = AsyncVeniceClient(api_key="test-key")
                
                with pytest.raises(VeniceError):
                    await client._request_multipart(
                        "POST", "test/upload", files={"file": ("test.jpg", b"data", "image/jpeg")}
                    )
            
            
            # Test HTTP Status Error with non-JSON response
            mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
            mock_response.aread = AsyncMock()
            mock_response.aclose = AsyncMock()
            mock_response.text = "Not JSON"
            
            with pytest.raises(VeniceError):
                await client._request_multipart(
                    "POST", "test/upload", files={"file": ("test.jpg", b"data", "image/jpeg")}
                )
            
            # Test TimeoutException
            # 1. Create a mock httpx.Request object
            mock_httpx_request_for_timeout = MagicMock(spec=httpx.Request)
            mock_httpx_request_for_timeout.method = "POST" # Matches the _request_multipart call
            
            # Safely get base_url string for constructing the mock request URL
            # The `client` instance is expected to be AsyncVeniceClient(api_key="test-key")
            # Its base_url should typically be "https://api.venice.ai/api/v1/" (which includes a trailing slash)
            client_base_url_obj = getattr(client, "base_url", None)
            
            base_url_for_mock_str: str
            if client_base_url_obj is None:
                # This indicates an unexpected issue with the client instance's state.
                # For test stability, use the known default base URL.
                # Consider adding a log warning here in a real debugging scenario, e.g.:
                # import logging
                # logging.warning(
                #     "test_request_multipart_errors: client.base_url was missing. "
                #     "Falling back to default base URL for mock request setup."
                # )
                base_url_for_mock_str = "https://api.venice.ai/api/v1/" # Default, ends with slash
            else:
                base_url_for_mock_str = str(client_base_url_obj) # Should also end with slash

            path_segment = "test/upload" # Path segment for the multipart request
            
            # Construct the full URL. Since base_url_for_mock_str ends with a slash,
            # directly append the path_segment.
            full_mock_url_str = f"{base_url_for_mock_str}{path_segment}"
            mock_httpx_request_for_timeout.url = httpx.URL(full_mock_url_str)

            # 2. Instantiate the TimeoutException
            timeout_exception_instance = httpx.TimeoutException("Request timed out")

            # 3. Manually set the internal _request attribute
            timeout_exception_instance._request = mock_httpx_request_for_timeout

            # 4. Assign the configured exception instance as the side_effect
            mock_client.request.side_effect = timeout_exception_instance

            with pytest.raises(VeniceError) as exc_info:
                await client._request_multipart(
                    "POST", "test/upload", files={"file": ("test.jpg", b"data", "image/jpeg")}
                )
            
            assert "Request timed out" in str(exc_info.value)
            
            # Test RequestError
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test/upload")
            mock_client.request.side_effect = httpx.RequestError("Connection failed", request=mock_request)
            
            with pytest.raises(VeniceError) as exc_info:
                await client._request_multipart(
                    "POST", "test/upload", files={"file": ("test.jpg", b"data", "image/jpeg")}
                )
            
            assert "Request failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_client_context_manager(self):
        """Test async context manager functionality."""
        original_httpx_async_client = httpx.AsyncClient # Store original
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncClientClass:
            # Setup mock client
            mock_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original
            MockAsyncClientClass.return_value = mock_client_instance
            mock_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_client_instance.aclose = AsyncMock()
            mock_client = mock_client_instance # maintain variable name for minimal diff
            
            # Test __aenter__ and __aexit__
            async with AsyncVeniceClient(api_key="test-key") as client:
                # __aenter__ should return self
                assert isinstance(client, AsyncVeniceClient)
            
            # __aexit__ should call close which calls aclose on the internal client
            mock_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_request_header_handling(self):
        """Test that GET requests properly handle Content-Type and Accept headers."""
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncClientClass:
            # Setup mock client with no spec
            mock_client_instance = AsyncMock()
            mock_client_instance.headers = { # Configure the instance mock
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "TestAgent/1.0"
            }
            MockAsyncClientClass.return_value = mock_client_instance
            mock_client = mock_client_instance # maintain variable name for minimal diff
            
            # Setup mock response
            mock_response = MagicMock()  # Use MagicMock for sync methods
            mock_response.json = AsyncMock(return_value={"result": "success"})
            mock_response.aread = AsyncMock()
            mock_response.aclose = AsyncMock()
            mock_client.request.return_value = mock_response
            
            client = AsyncVeniceClient(api_key="test-key")
            
            # Make GET request
            await client._request("GET", "test/endpoint")
            
            # Verify headers were properly handled
            call_args = mock_client.request.call_args[1]
            headers = call_args["headers"]
            
            # For GET requests, Content-Type and Accept should be removed
            assert "Content-Type" not in headers
            assert "Accept" not in headers
            assert headers["Authorization"] == "Bearer test-key"
            assert headers["User-Agent"] == "TestAgent/1.0"