"""
Comprehensive tests for the _request_multipart method in VeniceClient.

This file focuses exclusively on testing the multipart request functionality
to ensure complete code coverage for this method.
"""

import pytest
import httpx
import json
import io
import logging
from unittest.mock import patch, MagicMock, call

# Add logger patch for the _client module
from venice_ai._client import VeniceClient
from venice_ai.exceptions import VeniceError, APIError, InvalidRequestError


class TestClientMultipart:
    """Test suite dedicated to the _request_multipart method."""
    
    # Add a class-level mock for the logger
    @pytest.fixture(autouse=True)
    def setup_logger_mock(self):
        """Set up a mock for the logger in the client module."""
        with patch('venice_ai._client.logger', autospec=True) as mock_logger:
            yield mock_logger

    @pytest.fixture
    def mock_client(self):
        """Create a client with a mocked httpx client."""
        with patch('httpx.Client') as mock_httpx:
            client = VeniceClient(api_key="test-api-key")
            # Setup default headers as they would be in a real client
            client._client.headers = {
                "Authorization": "Bearer test-api-key",
                "User-Agent": "VeniceClient/1.0",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            yield client

    def test_request_multipart_basic(self, mock_client, setup_logger_mock):
        """Test basic functionality of _request_multipart with minimal parameters."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.status_code = 200
        mock_client._client.request.return_value = mock_response
        
        # Make a minimal multipart request
        files = {"file": ("test.txt", b"content", "text/plain")}
        result = mock_client._request_multipart("POST", "upload", files=files)
        
        # Verify result
        assert result == {"status": "success"}
        
        # Verify request was made correctly
        mock_client._client.request.assert_called_once()
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
        
        # Verify debug logging occurred using the mock logger instead of caplog
        assert any("Sending multipart request" in str(args[0]) for args, _ in setup_logger_mock.debug.call_args_list)

    def test_request_multipart_all_parameters(self, mock_client):
        """Test _request_multipart with all possible parameters."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.status_code = 200
        mock_client._client.request.return_value = mock_response
        
        # Create request parameters
        files = {"file": ("test.jpg", b"image data", "image/jpeg")}
        data = {"model": "test-model", "parameter": "value"}
        headers = {"X-Custom-Header": "custom-value"}
        params = {"query": "param"}
        
        # Make request with all parameters
        result = mock_client._request_multipart(
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
        mock_client._client.request.assert_called_once()
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

    def test_request_multipart_header_precedence(self, mock_client):
        """Test that request-specific headers override client defaults."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_client._client.request.return_value = mock_response
        
        # Set custom Accept header that should override the default
        headers = {
            "Accept": "application/json",  # Should override default */*
            "User-Agent": "CustomAgent/2.0"  # Should override client User-Agent
        }
        
        # Make request with custom headers
        files = {"file": ("test.txt", b"content", "text/plain")}
        result = mock_client._request_multipart("POST", "upload", files=files, headers=headers)
        
        # Verify headers
        call_args = mock_client._client.request.call_args[1]
        assert call_args["headers"]["Accept"] == "application/json"
        assert call_args["headers"]["User-Agent"] == "CustomAgent/2.0"
        assert call_args["headers"]["Authorization"] == "Bearer test-api-key"

    def test_request_multipart_various_file_formats(self, mock_client):
        """Test _request_multipart with different file formats."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_client._client.request.return_value = mock_response
        
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
        result = mock_client._request_multipart("POST", "upload/multiple", files=files)
        
        # Verify result
        assert result == {"status": "success"}
        
        # Verify files were passed correctly
        call_args = mock_client._client.request.call_args[1]
        assert call_args["files"] == files

    def test_request_multipart_with_file_io(self, mock_client):
        """Test _request_multipart with file-like objects."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_client._client.request.return_value = mock_response
        
        # Create file-like objects
        file_obj = io.BytesIO(b"file content")
        
        # Make request with a file-like object
        files = {"file": ("filename.txt", file_obj, "text/plain")}
        result = mock_client._request_multipart("POST", "upload", files=files)
        
        # Verify result
        assert result == {"status": "success"}
        
        # Verify file was passed correctly
        call_args = mock_client._client.request.call_args[1]
        assert call_args["files"] == files

    def test_request_multipart_http_error_json(self, mock_client, setup_logger_mock):
        """Test _request_multipart with HTTP error with JSON response."""
        # Set up HTTP error with JSON response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "POST"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/upload")
        mock_response.request = mock_request
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response_json = {"error": "Bad request", "code": "invalid_file"}
        mock_response.json.return_value = mock_response_json
        mock_response.text = '{"error": "Bad request", "code": "invalid_file"}'
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "400 Bad Request",
            request=mock_response.request,
            response=mock_response
        )
        mock_client._client.request.return_value = mock_response
        
        # Make request that will fail
        with pytest.raises(APIError) as excinfo:
            mock_client._request_multipart("POST", "upload", files={"file": ("test.jpg", b"data", "image/jpeg")})
        
        # Verify exception has the JSON body
        assert excinfo.value.body == mock_response_json
        assert excinfo.value.status_code == 400
        assert mock_response_json["error"] in str(excinfo.value)
        
        # If error is a dictionary with more details, check those too
        if "error" in mock_response_json and isinstance(mock_response_json["error"], dict):
            error_details = mock_response_json["error"]
            if "message" in error_details:
                message = error_details.get("message")
                if hasattr(excinfo.value, 'message'):
                    assert getattr(excinfo.value, 'message') == message
            if "type" in error_details:
                error_type = error_details.get("type")
                # Access error type through the body, not as a direct attribute
                if excinfo.value.body and isinstance(excinfo.value.body, dict):
                    body_error = excinfo.value.body.get("error", {})
                    if isinstance(body_error, dict) and "type" in body_error:
                        assert body_error.get("type") == error_type
        
        # Verify error was logged using the mock logger
        assert any("Error response body (full details):" in str(args[0]) for args, _ in setup_logger_mock.error.call_args_list)

    def test_request_multipart_http_error_non_json(self, mock_client, setup_logger_mock):
        """Test _request_multipart with HTTP error with non-JSON response."""
        # Set up HTTP error with non-JSON response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "POST"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/upload")
        mock_response.request = mock_request
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_response.text = "Internal Server Error"
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=mock_response.request,
            response=mock_response
        )
        mock_client._client.request.return_value = mock_response
        
        # Make request that will fail
        with pytest.raises(APIError) as excinfo:
            mock_client._request_multipart("POST", "upload", files={"file": ("test.jpg", b"data", "image/jpeg")})
        
        # Verify exception has structured error message
        expected_body = {"error": f"Non-JSON response from API (status {mock_response.status_code}): {mock_response.text[:500]}"}
        assert excinfo.value.body == expected_body
        assert excinfo.value.status_code == 500
        assert mock_response.text in str(excinfo.value)
        
        # Verify non-JSON error was logged using the mock logger
        # Verify non-JSON error was logged using the mock logger
        # The log message now includes the structured body.
        log_found = False
        for args, _ in setup_logger_mock.error.call_args_list:
            log_message = str(args[0])
            if "Error response body (full details):" in log_message and \
               f"'error': 'Non-JSON response from API (status {mock_response.status_code}): {mock_response.text[:500]}'" in log_message:
                log_found = True
                break
        assert log_found, f"Expected log message with non-JSON details not found. Actual logs: {setup_logger_mock.error.call_args_list}"

    def test_request_multipart_timeout(self, mock_client):
        """Test _request_multipart with timeout exception."""
        # Set up timeout exception
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "POST"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/upload")
        timeout_error = httpx.TimeoutException("Request timed out", request=mock_request)
        mock_client._client.request.side_effect = timeout_error
        
        # Make request that will time out
        with pytest.raises(VeniceError) as excinfo:
            mock_client._request_multipart("POST", "upload", files={"file": ("test.jpg", b"data", "image/jpeg")})
        
        # Verify exception message
        assert "Request timed out" in str(excinfo.value)
        assert excinfo.value.response is None

    def test_request_multipart_request_error(self, mock_client):
        """Test _request_multipart with network/connection error."""
        # Set up request error
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "POST"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/upload")
        request_error = httpx.RequestError("Network connection failed", request=mock_request)
        mock_client._client.request.side_effect = request_error
        
        # Make request that will fail with network error
        with pytest.raises(VeniceError) as excinfo:
            mock_client._request_multipart("POST", "upload", files={"file": ("test.jpg", b"data", "image/jpeg")})
        
        # Verify exception message
        assert "Request failed" in str(excinfo.value)
        assert excinfo.value.response is None

    def test_request_multipart_empty_files(self, mock_client):
        """Test _request_multipart with empty files dict."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_client._client.request.return_value = mock_response
        
        # Make request with empty files dict
        result = mock_client._request_multipart("POST", "upload", files={})
        
        # Verify result
        assert result == {"status": "success"}
        
        # Verify request was made correctly
        call_args = mock_client._client.request.call_args[1]
        assert call_args["files"] == {}

    def test_request_multipart_no_data(self, mock_client):
        """Test _request_multipart with only files, no data."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_client._client.request.return_value = mock_response
        
        # Make request with files but no data
        files = {"file": ("test.txt", b"content", "text/plain")}
        result = mock_client._request_multipart("POST", "upload", files=files)
        
        # Verify result
        assert result == {"status": "success"}
        
        # Verify request data parameter is None
        call_args = mock_client._client.request.call_args[1]
        assert "data" in call_args
        assert call_args["data"] is None

    def test_request_multipart_no_params(self, mock_client):
        """Test _request_multipart with no URL parameters."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_client._client.request.return_value = mock_response
        
        # Make request without params
        files = {"file": ("test.txt", b"content", "text/plain")}
        result = mock_client._request_multipart("POST", "upload", files=files)
        
        # Verify result
        assert result == {"status": "success"}
        
        # Verify request params parameter is None
        call_args = mock_client._client.request.call_args[1]
        assert "params" in call_args
        assert call_args["params"] is None