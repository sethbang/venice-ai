"""
Comprehensive tests for the _request_multipart method in AsyncVeniceClient.

This file focuses exclusively on testing the async multipart request functionality
to ensure complete code coverage for this method.
"""

import pytest
import httpx
import json
import io
import logging
from unittest.mock import patch, MagicMock, AsyncMock, call

# Add logger patch for the _async_client module
from venice_ai._async_client import AsyncVeniceClient
from venice_ai.exceptions import VeniceError, APIError, InvalidRequestError


@pytest.mark.asyncio
class TestAsyncClientMultipart:
    """Test suite dedicated to the _request_multipart method in AsyncVeniceClient."""

    @pytest.fixture
    async def mock_client(self):
        """Create an AsyncVeniceClient with a mocked httpx AsyncClient."""
        with patch('httpx.AsyncClient') as mock_httpx:
            client = AsyncVeniceClient(api_key="test-api-key")
            # Setup default headers as they would be in a real client
            client._client.headers = {
                "Authorization": "Bearer test-api-key",
                "User-Agent": "VeniceAsyncClient/1.0",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            yield client

    async def test_request_multipart_basic(self, mock_client, caplog):
        """Test basic functionality of _request_multipart with minimal parameters."""
        with caplog.at_level(logging.DEBUG):
            # Set up mock response
            mock_response = MagicMock()
            mock_response.json.return_value = {"status": "success"}
            mock_response.status_code = 200
            mock_response.headers = {}  # Add headers attribute to prevent AttributeError
            mock_client._client.request = AsyncMock(return_value=mock_response)
            
            # Make a minimal multipart request
            files = {"file": ("test.txt", b"content", "text/plain")}
            result = await mock_client._request_multipart("POST", "upload", files=files)
            
            # Verify result
            assert result == {"status": "success"}
            
            # Verify request was made correctly
            mock_client._client.request.assert_awaited_once()
            call_args = mock_client._client.request.call_args[1]
            
            # Check request parameters
            assert call_args["method"] == "POST"
            assert "upload" in str(call_args["url"])
            assert call_args["files"] == files
            
            # Check headers handling
            assert "Authorization" in call_args["headers"]
            assert "User-Agent" in call_args["headers"]
            assert call_args["headers"]["Accept"] == "*/*"
            assert "Content-Type" not in call_args["headers"]  # Should be omitted for multipart
            
            # Verify debug logging occurred
            assert any("Sending async multipart request" in message for message in caplog.messages)

    async def test_request_multipart_all_parameters(self, mock_client):
        """Test _request_multipart with all possible parameters."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.status_code = 200
        mock_response.headers = {}  # Add headers attribute to prevent AttributeError
        mock_client._client.request = AsyncMock(return_value=mock_response)
        
        # Create request parameters
        files = {"file": ("test.jpg", b"image data", "image/jpeg")}
        data = {"model": "test-model", "parameter": "value"}
        headers = {"X-Custom-Header": "custom-value"}
        params = {"query": "param"}
        
        # Make request with all parameters
        result = await mock_client._request_multipart(
            "POST",
            "images/upload",
            files=files, 
            data=data,
            headers=headers,
            params=params
        )
        
        # Verify result
        assert result == {"status": "success"}
        
        # Verify request was made correctly
        mock_client._client.request.assert_awaited_once()
        call_args = mock_client._client.request.call_args[1]
        
        # Check all parameters were passed correctly
        assert call_args["method"] == "POST"
        assert "images/upload" in str(call_args["url"])
        assert call_args["files"] == files
        assert call_args["data"] == data
        assert call_args["params"] == params
        
        # Check headers
        assert "Authorization" in call_args["headers"]
        assert "User-Agent" in call_args["headers"]
        assert call_args["headers"]["X-Custom-Header"] == "custom-value"
        assert call_args["headers"]["Accept"] == "*/*"

    async def test_request_multipart_header_precedence(self, mock_client):
        """Test that request-specific headers override client defaults."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.status_code = 200  # Add status_code
        mock_response.headers = {}  # Add headers attribute to prevent AttributeError
        mock_client._client.request = AsyncMock(return_value=mock_response)
        
        # Set custom Accept header that should override the default
        headers = {
            "Accept": "application/json",  # Should override default */*
            "User-Agent": "CustomAsyncAgent/2.0"  # Should override client User-Agent
        }
        
        # Make request with custom headers
        files = {"file": ("test.txt", b"content", "text/plain")}
        result = await mock_client._request_multipart("POST", "upload", files=files, headers=headers)
        
        # Verify headers
        call_args = mock_client._client.request.call_args[1]
        assert call_args["headers"]["Accept"] == "application/json"
        assert call_args["headers"]["User-Agent"] == "CustomAsyncAgent/2.0"
        assert call_args["headers"]["Authorization"] == "Bearer test-api-key"

    async def test_request_multipart_various_file_formats(self, mock_client):
        """Test _request_multipart with different file formats."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.status_code = 200  # Add status_code
        mock_response.headers = {}  # Add headers attribute to prevent AttributeError
        mock_client._client.request = AsyncMock(return_value=mock_response)
        
        # Create various file formats
        image_file = ("image.jpg", b"image data", "image/jpeg")
        text_file = ("document.txt", b"text content", "text/plain")
        binary_file = ("data.bin", b"\x00\x01\x02", "application/octet-stream")
        
        # Setup files dict with multiple files
        files = {
            "image": image_file,
            "text": text_file,
            "binary": binary_file
        }
        
        # Make request
        result = await mock_client._request_multipart("POST", "upload/multiple", files=files)
        
        # Verify result
        assert result == {"status": "success"}
        
        # Verify files were passed correctly
        call_args = mock_client._client.request.call_args[1]
        assert call_args["files"] == files

    async def test_request_multipart_with_file_io(self, mock_client):
        """Test _request_multipart with file-like objects."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.status_code = 200  # Add status_code
        mock_response.headers = {}  # Add headers attribute to prevent AttributeError
        mock_client._client.request = AsyncMock(return_value=mock_response)
        
        # Create file-like objects
        file_obj = io.BytesIO(b"file content")
        
        # Make request with a file-like object
        files = {"file": ("filename.txt", file_obj, "text/plain")}
        result = await mock_client._request_multipart("POST", "upload", files=files)
        
        # Verify result
        assert result == {"status": "success"}
        
        # Verify file was passed correctly
        call_args = mock_client._client.request.call_args[1]
        assert call_args["files"] == files

    async def test_request_multipart_with_file_logging(self, mock_client, caplog):
        """Test _request_multipart with detailed file logging."""
        with caplog.at_level(logging.DEBUG):
            # Set up mock response
            mock_response = MagicMock()
            mock_response.json.return_value = {"status": "success"}
            mock_response.status_code = 200  # Add status_code
            mock_response.headers = {}  # Add headers attribute to prevent AttributeError
            mock_client._client.request = AsyncMock(return_value=mock_response)
            
            # Create a file tuple mimicking httpx format
            file_tuple = ("test.jpg", b"binary data", "image/jpeg")
            
            # Make request with files
            files = {"image": file_tuple}
            await mock_client._request_multipart("POST", "upload", files=files)
            
            # Verify file details were logged
            assert any("File 'image' details:" in message for message in caplog.messages)
            assert any("Files content type:" in message for message in caplog.messages)

    async def test_request_multipart_http_error_json(self, mock_client, caplog):
        """Test _request_multipart with HTTP error with JSON response."""
        # Set up HTTP error with JSON response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "POST"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/upload")
        mock_response.request = mock_request
        mock_response.json.return_value = {"error": "Bad request", "code": "invalid_file"}
        mock_response.text = '{"error": "Bad request", "code": "invalid_file"}'
        mock_response.headers = {}  # Add headers attribute to prevent AttributeError
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "400 Bad Request",
            request=mock_response.request, 
            response=mock_response
        )
        mock_client._client.request = AsyncMock(return_value=mock_response)
        
        with caplog.at_level(logging.DEBUG):
            # Make request that will fail
            with pytest.raises(InvalidRequestError) as excinfo:
                await mock_client._request_multipart("POST", "upload", files={"file": ("test.jpg", b"data", "image/jpeg")})
            
            # Verify exception has the JSON body
            assert excinfo.value.body == {"error": "Bad request", "code": "invalid_file"}
            assert excinfo.value.status_code == 400
            
            # Verify error was logged
            assert any("Error response body (full details):" in message for message in caplog.messages)

    async def test_request_multipart_http_error_non_json(self, mock_client, caplog):
        """Test _request_multipart with HTTP error with non-JSON response."""
        # Set up HTTP error with non-JSON response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "POST"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/upload")
        mock_response.request = mock_request
        mock_response.text = "Internal Server Error"
        mock_response.headers = {}  # Add headers attribute to prevent AttributeError
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=mock_response.request,
            response=mock_response
        )
        mock_client._client.request = AsyncMock(return_value=mock_response)
        
        with caplog.at_level(logging.DEBUG):
            # Make request that will fail
            with pytest.raises(APIError) as excinfo:
                await mock_client._request_multipart("POST", "upload", files={"file": ("test.jpg", b"data", "image/jpeg")})
            
            # Verify exception has structured error message
            expected_body = {"error": f"Non-JSON response from API (status {mock_response.status_code}): {mock_response.text[:500]}"}
            assert excinfo.value.body == expected_body
            assert excinfo.value.status_code == 500
            assert mock_response.text in str(excinfo.value)
            
            # Verify non-JSON error was logged
            assert any(
                record.levelname == "ERROR"
                and "Error response body (full details):" in record.message
                and "Internal Server Error" in record.message
                for record in caplog.records
            )

    async def test_request_multipart_timeout(self, mock_client):
        """Test _request_multipart with timeout exception."""
        # Set up timeout exception
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "POST"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/upload")
        timeout_error = httpx.TimeoutException("Request timed out", request=mock_request)
        mock_client._client.request = AsyncMock(side_effect=timeout_error)
        
        # Make request that will time out
        with pytest.raises(VeniceError) as excinfo:
            await mock_client._request_multipart("POST", "upload", files={"file": ("test.jpg", b"data", "image/jpeg")})
        
        # Verify exception message
        assert "Request timed out" in str(excinfo.value)
        assert excinfo.value.response is None

    async def test_request_multipart_request_error(self, mock_client):
        """Test _request_multipart with network/connection error."""
        # Set up request error
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "POST"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/upload")
        request_error = httpx.RequestError("Network connection failed", request=mock_request)
        mock_client._client.request = AsyncMock(side_effect=request_error)
        
        # Make request that will fail with network error
        with pytest.raises(VeniceError) as excinfo:
            await mock_client._request_multipart("POST", "upload", files={"file": ("test.jpg", b"data", "image/jpeg")})
        
        # Verify exception message
        assert "Request failed" in str(excinfo.value)
        assert excinfo.value.response is None

    async def test_request_multipart_empty_files(self, mock_client):
        """Test _request_multipart with empty files dict."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.status_code = 200  # Add status_code
        mock_response.headers = {}  # Add headers attribute to prevent AttributeError
        mock_client._client.request = AsyncMock(return_value=mock_response)
        
        # Make request with empty files dict
        result = await mock_client._request_multipart("POST", "upload", files={})
        
        # Verify result
        assert result == {"status": "success"}
        
        # Verify request was made correctly
        call_args = mock_client._client.request.call_args[1]
        assert call_args["files"] == {}

    async def test_request_multipart_no_data(self, mock_client):
        """Test _request_multipart with only files, no data."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.status_code = 200  # Add status_code
        mock_response.headers = {}  # Add headers attribute to prevent AttributeError
        mock_client._client.request = AsyncMock(return_value=mock_response)
        
        # Make request with files but no data
        files = {"file": ("test.txt", b"content", "text/plain")}
        result = await mock_client._request_multipart("POST", "upload", files=files)
        
        # Verify result
        assert result == {"status": "success"}
        
        # Verify request data parameter is None
        call_args = mock_client._client.request.call_args[1]
        assert "data" in call_args
        assert call_args["data"] is None

    async def test_request_multipart_no_params(self, mock_client):
        """Test _request_multipart with no URL parameters."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.status_code = 200  # Add status_code
        mock_response.headers = {}  # Add headers attribute to prevent AttributeError
        mock_client._client.request = AsyncMock(return_value=mock_response)
        
        # Make request without params
        files = {"file": ("test.txt", b"content", "text/plain")}
        result = await mock_client._request_multipart("POST", "upload", files=files)
        
        # Verify result
        assert result == {"status": "success"}
        
        # Verify request params parameter is None
        call_args = mock_client._client.request.call_args[1]
        assert "params" in call_args
        assert call_args["params"] is None

    async def test_request_multipart_response_logging(self, mock_client, caplog):
        """Test response logging in _request_multipart."""
        with caplog.at_level(logging.DEBUG):
            # Set up mock response with specific attributes for logging
            mock_response = MagicMock()
            mock_response.json.return_value = {"key": "value"}
            mock_response.text = '{"key": "value"}'
            mock_response.status_code = 200
            mock_response.headers = {"Content-Type": "application/json"}
            mock_client._client.request = AsyncMock(return_value=mock_response)
            
            # Make request
            files = {"file": ("test.txt", b"content", "text/plain")}
            await mock_client._request_multipart("POST", "upload", files=files)
            
            # Verify response logging
            assert any("Received async response with status code:" in message for message in caplog.messages)
            assert any("Response headers:" in message for message in caplog.messages)
            assert any("Response content (first 500 chars for JSON):" in message for message in caplog.messages)