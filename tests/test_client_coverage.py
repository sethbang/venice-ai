"""
Targeted tests for VeniceClient to improve code coverage.

This module focuses on specific areas that need better coverage based on
the coverage report, particularly error handling paths and multipart requests.
"""

import pytest
import httpx
import json
import io
import logging
from typing import cast
from unittest.mock import patch, MagicMock, call

from venice_ai._client import VeniceClient, ChatResource
from venice_ai.exceptions import VeniceError, APIError, InvalidRequestError


class TestClientCoverageImprovement:
    """Tests focused on improving code coverage for VeniceClient."""

    @pytest.fixture
    def client(self):
        """Create a client with a mocked httpx client."""
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-api-key")
            # Add logger to ensure it's defined for _request_multipart
            logger = MagicMock()
            yield client

    def test_request_get_header_removal(self, client):
        """Test that GET requests properly remove Content-Type and Accept headers."""
        # Set up client headers
        client._client.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer test-key"
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "success"}
        mock_response.status_code = 200  # Add status_code
        mock_response.headers = {}  # Add headers attribute to prevent AttributeError
        client._client.request.return_value = mock_response

        # Make GET request
        client._request("GET", "test/endpoint")

        # Verify Content-Type and Accept headers were removed
        _, kwargs = client._client.request.call_args
        assert "Content-Type" not in kwargs["headers"]
        assert "Accept" not in kwargs["headers"]

    def test_request_json_decode_error_in_error_response(self, client):
        """Test handling of JSON decode errors in error responses."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.request = MagicMock()
        mock_response.text = "Not a JSON error response"
        mock_response.headers = {}  # Add headers attribute to prevent AttributeError

        # Set up HTTPStatusError
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found",
            request=mock_response.request,
            response=mock_response
        )

        # Make response.json() raise JSONDecodeError
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        client._client.request.return_value = mock_response

        # Call should handle the JSON decode error properly
        with pytest.raises(APIError) as excinfo:
            client._request("GET", "test/endpoint")

        # Check the exception body contains structured error message
        expected_body = {"error": f"Non-JSON response from API (status 404): Not a JSON error response"}
        assert excinfo.value.body == expected_body
        assert "404" in str(excinfo.value)

    def test_request_error_handling(self, client):
        """Test handling of RequestError exceptions."""
        request_error = httpx.RequestError("Connection failed", request=MagicMock())
        client._client.request.side_effect = request_error

        with pytest.raises(VeniceError) as excinfo:
            client._request("GET", "test/endpoint")

        assert "Request failed: Connection failed" in str(excinfo.value)
        assert excinfo.value.response is None

    def test_stream_request_json_decode_error(self, client):
        """Test handling of JSON decode errors in streaming responses."""
        # Create a mock stream context manager
        mock_response = MagicMock()
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_response
        mock_stream.__exit__.return_value = None
        client._client.stream.return_value = mock_stream

        # Return lines including invalid JSON
        mock_response.iter_lines.return_value = [
            "data: {\"valid\": true}",
            "data: {invalid json}",  # This should cause JSONDecodeError
            "data: {\"also_valid\": true}",
            "data: [DONE]"
        ]

        # Patch the logger instead of print
        with patch('venice_ai._client.logger') as mock_logger:
            # Collect all chunks
            chunks = list(client._stream_request("POST", "chat/completions"))

            # Should have 2 valid chunks
            assert len(chunks) == 2
            assert chunks[0] == {"valid": True}
            assert chunks[1] == {"also_valid": True}

            # Check that error messages were logged for malformed chunks
            # Retrieve all error log calls
            error_calls = [call for call in mock_logger.error.call_args_list]

            # Check that the expected number of error calls occurred (1 malformed chunk * 2 log messages per chunk)
            assert len(error_calls) == 2, "Expected 2 error log calls for malformed chunks"

            # Check the content of the log messages
            # The exact error message from json.JSONDecodeError can vary, so check that it starts with the expected prefix
            assert error_calls[0][0][0].startswith("Failed to parse JSON in streaming response:"), "First error message should start with expected prefix"
            assert error_calls[1][0][0] == "Problematic JSON string: '{invalid json}'", "Second error message should match problematic string"

    def test_stream_request_http_status_error(self, client):
        """Test handling of HTTP status errors in stream requests."""
        # Create a mock response that raises an HTTP error
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.request = MagicMock()
        mock_response.headers = {}  # Add headers attribute to prevent AttributeError
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized",
            request=mock_response.request,
            response=mock_response
        )

        # Set up stream context manager
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_response
        mock_stream.__exit__.return_value = None
        client._client.stream.return_value = mock_stream

        with pytest.raises(APIError) as excinfo:
            list(client._stream_request("POST", "chat/completions"))

        assert "401" in str(excinfo.value)

    def test_stream_request_timeout(self, client):
        """Test handling of timeout exceptions in stream requests."""
        # Create a mock httpx.Request object
        mock_httpx_request = MagicMock(spec=httpx.Request)
        mock_httpx_request.method = "POST"
        mock_httpx_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
        
        # Make client.stream raise a TimeoutException with proper request attribute
        timeout_error = httpx.TimeoutException("Stream timed out")
        timeout_error._request = mock_httpx_request
        client._client.stream.side_effect = timeout_error

        with pytest.raises(VeniceError) as excinfo:
            list(client._stream_request("POST", "chat/completions"))

        assert "Stream request timed out" in str(excinfo.value)

    def test_stream_request_request_error(self, client):
        """Test handling of request errors in stream requests."""
        # Make client.stream raise a RequestError
        request_error = httpx.RequestError("Stream failed", request=MagicMock())
        client._client.stream.side_effect = request_error

        with pytest.raises(VeniceError) as excinfo:
            list(client._stream_request("POST", "chat/completions"))

        assert "Stream request failed" in str(excinfo.value)

    def test_stream_request_raw_http_error(self, client):
        """Test handling of HTTP errors in raw stream requests."""
        # Create a mock response that raises an HTTP error
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.request = MagicMock()
        mock_response.headers = {}  # Add headers attribute to prevent AttributeError
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "403 Forbidden",
            request=mock_response.request,
            response=mock_response
        )

        # Set up stream context manager
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_response
        mock_stream.__exit__.return_value = None
        client._client.stream.return_value = mock_stream

        with pytest.raises(APIError) as excinfo:
            list(client._stream_request_raw("POST", "audio/speech"))

        assert "403" in str(excinfo.value)

    def test_stream_request_raw_timeout(self, client):
        """Test handling of timeout exceptions in raw stream requests."""
        # Create a mock httpx.Request object
        mock_httpx_request = MagicMock(spec=httpx.Request)
        mock_httpx_request.method = "POST"
        mock_httpx_request.url = httpx.URL("https://api.venice.ai/api/v1/audio/speech")
        
        # Make client.stream raise a TimeoutException with proper request attribute
        timeout_error = httpx.TimeoutException("Raw stream timed out")
        timeout_error._request = mock_httpx_request
        client._client.stream.side_effect = timeout_error

        with pytest.raises(VeniceError) as excinfo:
            list(client._stream_request_raw("POST", "audio/speech"))

        assert "Stream request timed out" in str(excinfo.value)

    def test_stream_request_raw_request_error(self, client):
        """Test handling of request errors in raw stream requests."""
        # Make client.stream raise a RequestError
        request_error = httpx.RequestError("Raw stream failed", request=MagicMock())
        client._client.stream.side_effect = request_error

        with pytest.raises(VeniceError) as excinfo:
            list(client._stream_request_raw("POST", "audio/speech"))

        assert "Stream request failed" in str(excinfo.value)

    def test_request_multipart_comprehensive(self):
        """Comprehensive test for _request_multipart covering all code paths."""
        with patch('httpx.Client') as mock_httpx_client:
            # Set up mock response
            mock_response = MagicMock()
            mock_response.json.return_value = {"status": "success"}
            mock_response.status_code = 200
            mock_response.text = '{"status": "success"}'
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.raise_for_status = MagicMock()  # No error

            # Set up mock client
            mock_client = mock_httpx_client.return_value
            mock_client.request.return_value = mock_response
            mock_client.headers = {
                "Authorization": "Bearer test-key",
                "User-Agent": "VeniceClient/1.0",
                "Content-Type": "application/json",  # Should be removed for multipart
                "Accept": "application/json"  # Should be replaced with */*
            }

            # Create client and patch logger
            with patch('venice_ai._client.logger') as mock_logger:
                client = VeniceClient(api_key="test-key")
                client._client = mock_client

                # Test with typical multipart request
                files = {"file": ("test.jpg", b"image data", "image/jpeg")}
                data = {"model": "test-model"}
                headers = {"X-Custom": "value"}
                params = {"param1": "value1"}

                result = client._request_multipart(
                    "POST",
                    "images/upload",
                    files=files,
                    data=data,
                    headers=headers,
                    params=params
                )

                # Verify success
                assert result == {"status": "success"}

                # Verify request was made correctly
                mock_client.request.assert_called_once()
                args, kwargs = mock_client.request.call_args

                # Check basic request properties
                assert kwargs["method"] == "POST"
                assert kwargs["url"] == client._base_url.join("images/upload")
                assert kwargs["files"] == files
                assert kwargs["data"] == data
                assert kwargs["params"] == params

                # Check headers
                assert "Authorization" in kwargs["headers"]
                assert "User-Agent" in kwargs["headers"]
                assert kwargs["headers"]["Authorization"] == "Bearer test-key"
                assert kwargs["headers"]["User-Agent"] == "VeniceClient/1.0"
                assert kwargs["headers"]["X-Custom"] == "value"
                assert kwargs["headers"]["Accept"] == "*/*"
                assert "Content-Type" not in kwargs["headers"]  # Should be set automatically by httpx

                # Verify logging happened
                assert mock_logger.debug.call_count >= 5

    def test_request_multipart_error_handling(self):
        """Test comprehensive error handling in _request_multipart."""
        with patch('httpx.Client') as mock_httpx_client:
            # Create client
            client = VeniceClient(api_key="test-key")

            # Set up various error scenarios

            # 1. HTTP status error with JSON response
            mock_response1 = MagicMock(spec=httpx.Response)
            mock_response1.status_code = 400
            mock_response1.request = MagicMock()
            mock_response1.json.return_value = {"error": "Bad request", "code": "invalid_file"}
            mock_response1.headers = {}  # Add headers attribute to prevent AttributeError
            mock_response1.raise_for_status.side_effect = httpx.HTTPStatusError(
                "400 Bad Request",
                request=mock_response1.request,
                response=mock_response1
            )

            # 2. HTTP status error with non-JSON response
            mock_response2 = MagicMock(spec=httpx.Response)
            mock_response2.status_code = 500
            mock_response2.request = MagicMock()
            mock_response2.text = "Internal Server Error"
            mock_response2.headers = {}  # Add headers attribute to prevent AttributeError
            mock_response2.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_response2.raise_for_status.side_effect = httpx.HTTPStatusError(
                "500 Internal Server Error",
                request=mock_response2.request,
                response=mock_response2
            )

            # 3. Timeout exception with proper request attribute
            mock_httpx_request = MagicMock(spec=httpx.Request)
            mock_httpx_request.method = "POST"
            mock_httpx_request.url = httpx.URL("https://api.venice.ai/api/v1/upload")
            
            timeout_error = httpx.TimeoutException("Request timed out")
            timeout_error._request = mock_httpx_request

            # 4. Request error
            request_error = httpx.RequestError("Network error", request=MagicMock())

            # Run tests with all error scenarios
            with patch('venice_ai._client.logger'):
                # Test 1: HTTP status error with JSON body
                mock_request = cast(MagicMock, client._client.request)
                mock_request.return_value = mock_response1
                with pytest.raises(InvalidRequestError) as excinfo:
                    client._request_multipart("POST", "upload", files={"file": ("test.jpg", b"data", "image/jpeg")})
                assert excinfo.value.body == {"error": "Bad request", "code": "invalid_file"}

                # Test 2: HTTP status error with non-JSON body
                mock_request.return_value = mock_response2
                with pytest.raises(APIError) as excinfo:
                    client._request_multipart("POST", "upload", files={"file": ("test.jpg", b"data", "image/jpeg")})
                expected_body = {
                    "error": f"Non-JSON response from API (status {mock_response2.status_code}): {mock_response2.text[:500]}"
                }
                assert excinfo.value.body == expected_body

                # Test 3: Timeout exception
                mock_request.side_effect = timeout_error
                with pytest.raises(VeniceError) as excinfo:
                    client._request_multipart("POST", "upload", files={"file": ("test.jpg", b"data", "image/jpeg")})
                assert "Request timed out" in str(excinfo.value)

                # Test 4: Request error
                mock_request.side_effect = request_error
                with pytest.raises(VeniceError) as excinfo:
                    client._request_multipart("POST", "upload", files={"file": ("test.jpg", b"data", "image/jpeg")})
                assert "Request failed" in str(excinfo.value)

    def test_context_manager_methods(self):
        """Test the context manager methods (__enter__ and __exit__)."""
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-key")

            # Test __enter__ returns self
            context_client = client.__enter__()
            assert context_client is client

            # Test __exit__ calls close
            with patch.object(client, 'close') as mock_close:
                client.__exit__(None, None, None)
                mock_close.assert_called_once()

            # Test with actual context manager
            mock_client = MagicMock()
            with patch('httpx.Client', return_value=mock_client):
                with VeniceClient(api_key="test-key") as cm_client:
                    assert isinstance(cm_client, VeniceClient)

                # Should call close on exit
                mock_client.close.assert_called_once()