import pytest
import httpx
import json
from typing import cast, Dict, Any
from unittest.mock import patch, MagicMock, Mock
from venice_ai.exceptions import APIError # Import APIError

from venice_ai._client import VeniceClient
from venice_ai.exceptions import APITimeoutError, APIConnectionError

class TestClientErrorHandlingFallbacks:
    """Tests targeting specific error handling and fallback scenarios in the VeniceClient."""
    
    def test_request_timeout_with_none_request(self):
        """Test _request error handling when TimeoutException has e.request=None."""
        with patch('httpx.Client') as mock_httpx_client:
            # Create a TimeoutException with request=None
            timeout_exception = httpx.TimeoutException("Connection timed out")
            # Explicitly set request to None to simulate this edge case
            timeout_exception.request = None  # type: ignore[assignment]
            
            mock_httpx_client.return_value.request.side_effect = timeout_exception
            client = VeniceClient(api_key="test-api-key")
            
            with pytest.raises(APITimeoutError) as excinfo:
                client._request("GET", "test_endpoint")
            
            # Verify the error handling created a fallback Request object
            assert "Request timed out" in str(excinfo.value)
            # Verify the request in the exception is not None
            assert excinfo.value.request is not None
            request = excinfo.value.request
            assert request.method == "GET"
            assert "test_endpoint" in str(request.url)

    def test_request_error_with_none_request(self):
        """Test _request error handling when RequestError has e.request=None."""
        with patch('httpx.Client') as mock_httpx_client:
            # Create a RequestError with request=None
            request_error = httpx.RequestError("Connection failed")
            # Explicitly set request to None to simulate this edge case
            request_error.request = None  # type: ignore[assignment]
            
            mock_httpx_client.return_value.request.side_effect = request_error
            client = VeniceClient(api_key="test-api-key")
            
            with pytest.raises(APIConnectionError) as excinfo:
                client._request("GET", "test_endpoint")
            
            # Verify the error handling created a fallback Request object
            assert "Request failed" in str(excinfo.value)
            # Verify the request in the exception is not None
            assert excinfo.value.request is not None
            request = excinfo.value.request
            assert request.method == "GET"
            assert "test_endpoint" in str(request.url)
            
    def test_stream_request_timeout_with_none_request(self):
        """Test _stream_request error handling when TimeoutException has e.request=None."""
        with patch('httpx.Client') as mock_httpx_client:
            # Create a TimeoutException with request=None
            timeout_exception = httpx.TimeoutException("Stream connection timed out")
            # Explicitly set request to None to simulate this edge case
            timeout_exception.request = None  # type: ignore[assignment]
            
            mock_httpx_client.return_value.stream.side_effect = timeout_exception
            client = VeniceClient(api_key="test-api-key")
            
            with pytest.raises(APITimeoutError) as excinfo:
                for _ in client._stream_request("POST", "chat/completions", json_data={"model": "venice-1"}):
                    pass  # Should raise before yielding anything
            
            # Verify the error handling created a fallback Request object
            assert "Stream request timed out" in str(excinfo.value)
            # Verify the request in the exception is not None
            assert excinfo.value.request is not None
            request = excinfo.value.request
            assert request.method == "POST"
            assert "chat/completions" in str(request.url)
            
    def test_stream_request_error_with_none_request(self):
        """Test _stream_request error handling when RequestError has e.request=None."""
        with patch('httpx.Client') as mock_httpx_client:
            # Create a RequestError with request=None
            request_error = httpx.RequestError("Stream connection failed")
            # Explicitly set request to None to simulate this edge case
            request_error.request = None  # type: ignore[assignment]
            
            mock_httpx_client.return_value.stream.side_effect = request_error
            client = VeniceClient(api_key="test-api-key")
            
            with pytest.raises(APIConnectionError) as excinfo:
                for _ in client._stream_request("POST", "chat/completions", json_data={"model": "venice-1"}):
                    pass  # Should raise before yielding anything
            
            # Verify the error handling created a fallback Request object
            assert "Stream request failed" in str(excinfo.value)
            # Verify the request in the exception is not None
            assert excinfo.value.request is not None
            request = excinfo.value.request
            assert request.method == "POST"
            assert "chat/completions" in str(request.url)
            
    def test_request_multipart_timeout_with_none_request(self):
        """Test _request_multipart error handling when TimeoutException has e.request=None."""
        with patch('httpx.Client') as mock_httpx_client:
            # Create a TimeoutException with request=None
            timeout_exception = httpx.TimeoutException("Multipart request timed out")
            # Explicitly set request to None to simulate this edge case
            timeout_exception.request = None  # type: ignore[assignment]
            
            mock_httpx_client.return_value.request.side_effect = timeout_exception
            client = VeniceClient(api_key="test-api-key")
            files = {"file": ("test.txt", b"content", "text/plain")}
            
            with pytest.raises(APITimeoutError) as excinfo:
                client._request_multipart("POST", "upload", files=files)
            
            # Verify the error handling created a fallback Request object
            assert "Request timed out" in str(excinfo.value)
            # Verify the request in the exception is not None
            assert excinfo.value.request is not None
            request = excinfo.value.request
            assert request.method == "POST"
            assert "upload" in str(request.url)
            
    def test_request_multipart_error_with_none_request(self):
        """Test _request_multipart error handling when RequestError has e.request=None."""
        with patch('httpx.Client') as mock_httpx_client:
            # Create a RequestError with request=None
            request_error = httpx.RequestError("Multipart request failed")
            # Explicitly set request to None to simulate this edge case
            request_error.request = None  # type: ignore[assignment]
            
            mock_httpx_client.return_value.request.side_effect = request_error
            client = VeniceClient(api_key="test-api-key")
            files = {"file": ("test.txt", b"content", "text/plain")}
            
            with pytest.raises(APIConnectionError) as excinfo:
                client._request_multipart("POST", "upload", files=files)
            
            # Verify the error handling created a fallback Request object
            assert "Request failed" in str(excinfo.value)
            # Verify the request in the exception is not None
            assert excinfo.value.request is not None
            request = excinfo.value.request
            assert request.method == "POST"
            assert "upload" in str(request.url)
            
    def test_stream_request_raw_timeout_with_none_request(self):
        """Test _stream_request_raw error handling when TimeoutException has e.request=None."""
        with patch('httpx.Client') as mock_httpx_client:
            # Create a TimeoutException with request=None
            timeout_exception = httpx.TimeoutException("Raw stream timed out")
            # Explicitly set request to None to simulate this edge case
            timeout_exception.request = None  # type: ignore[assignment]
            
            mock_httpx_client.return_value.stream.side_effect = timeout_exception
            client = VeniceClient(api_key="test-api-key")
            
            with pytest.raises(APITimeoutError) as excinfo:
                for _ in client._stream_request_raw("POST", "audio/speech", json_data={"text": "test"}):
                    pass  # Should raise before yielding anything
            
            # Verify the error handling created a fallback Request object
            assert "Stream request timed out" in str(excinfo.value)
            # Verify the request in the exception is not None
            assert excinfo.value.request is not None
            request = excinfo.value.request
            assert request.method == "POST"
            assert "audio/speech" in str(request.url)
            
    def test_stream_request_raw_error_with_none_request(self):
        """Test _stream_request_raw error handling when RequestError has e.request=None."""
        with patch('httpx.Client') as mock_httpx_client:
            # Create a RequestError with request=None
            request_error = httpx.RequestError("Raw stream failed")
            # Explicitly set request to None to simulate this edge case
            request_error.request = None  # type: ignore[assignment]
            
            mock_httpx_client.return_value.stream.side_effect = request_error
            client = VeniceClient(api_key="test-api-key")
            
            with pytest.raises(APIConnectionError) as excinfo:
                for _ in client._stream_request_raw("POST", "audio/speech", json_data={"text": "test"}):
                    pass  # Should raise before yielding anything
            
            # Verify the error handling created a fallback Request object
            assert "Stream request failed" in str(excinfo.value)
            # Verify the request in the exception is not None
            assert excinfo.value.request is not None
            request = excinfo.value.request
            assert request.method == "POST"
            assert "audio/speech" in str(request.url)


class TestTranslateHttpxErrorToApiError:
    """Tests for specific error translation paths in _translate_httpx_error_to_api_error."""
    
    def test_non_json_error_response(self):
        """Test handling of HTTPStatusError with non-JSON response body."""
        with patch('httpx.Client') as mock_httpx_client:
            # Setup response mock
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 400
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_response.text = "Plain text error message"
            
            # Setup request mock
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "GET"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
            
            # Create HTTPStatusError with mocked request and response
            http_status_error = httpx.HTTPStatusError(
                "400 Bad Request", 
                request=mock_request, 
                response=mock_response
            )
            
            client = VeniceClient(api_key="test-api-key")
            # Directly test the _translate_httpx_error_to_api_error method
            api_error = client._translate_httpx_error_to_api_error(http_status_error, mock_request)
            
            # Verify text fallback was used for body
            assert cast(APIError, api_error).body is not None # Assertion on casted object
            assert "Plain text error message" in str(cast(Dict[str, Any], cast(APIError, api_error).body)) # Nested cast
    
    def test_non_json_error_response_text_also_fails(self):
        """Test handling of HTTPStatusError when both json() and text access fail."""
        with patch('httpx.Client') as mock_httpx_client:
            # Setup response mock
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 400
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            
            # Make text access also fail
            type(mock_response).text = Mock(side_effect=Exception("Failed to get text"))
            
            # Setup request mock
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "GET"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
            
            # Create HTTPStatusError with mocked request and response
            http_status_error = httpx.HTTPStatusError(
                "400 Bad Request", 
                request=mock_request, 
                response=mock_response
            )
            
            client = VeniceClient(api_key="test-api-key")
            # Directly test the _translate_httpx_error_to_api_error method
            api_error = client._translate_httpx_error_to_api_error(http_status_error, mock_request)
            
            # Verify body is None when both json and text fail
            assert cast(APIError, api_error).status_code == 400
            assert "API error 400" in str(api_error)
    
    def test_timeout_exception_translation(self):
        """Test translation of TimeoutException to APITimeoutError."""
        with patch('httpx.Client') as mock_httpx_client:
            # Create TimeoutException
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "GET"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
            
            timeout_exception = httpx.TimeoutException("Connection timed out", request=mock_request)
            
            client = VeniceClient(api_key="test-api-key")
            api_error = client._translate_httpx_error_to_api_error(timeout_exception, mock_request)
            
            # Verify correct type of error is returned
            assert isinstance(api_error, APITimeoutError)
            assert "timed out" in str(api_error)
            assert api_error.request is mock_request
    
    def test_connect_error_translation(self):
        """Test translation of ConnectError to APIConnectionError."""
        with patch('httpx.Client') as mock_httpx_client:
            # Create ConnectError
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "GET"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
            
            connect_error = httpx.ConnectError("Failed to connect", request=mock_request)
            
            client = VeniceClient(api_key="test-api-key")
            api_error = client._translate_httpx_error_to_api_error(connect_error, mock_request)
            
            # Verify correct type of error is returned
            assert isinstance(api_error, APIConnectionError)
            assert "Failed to connect" in str(api_error)
            assert api_error.request is mock_request
    
    def test_generic_request_error_translation(self):
        """Test translation of generic RequestError to APIConnectionError."""
        with patch('httpx.Client') as mock_httpx_client:
            # Create a generic RequestError that is neither a TimeoutException nor a ConnectError
            class CustomRequestError(httpx.RequestError):
                pass
            
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "GET"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
            
            request_error = CustomRequestError("Generic request error", request=mock_request)
            
            client = VeniceClient(api_key="test-api-key")
            api_error = client._translate_httpx_error_to_api_error(request_error, mock_request)
            
            # Verify correct type of error is returned for fallback case
            assert isinstance(api_error, APIConnectionError)
            assert "Generic request error" in str(api_error)
            assert api_error.request is mock_request


class TestClientContextManager:
    """Test VeniceClient context manager functionality."""
    
    def test_exit_calls_close(self):
        """Test __exit__ method properly closes the client."""
        with patch('httpx.Client') as mock_httpx_client:
            client = VeniceClient(api_key="test-api-key")
            # Validate client has _client attribute
            assert hasattr(client, "_client")
            assert client._client is not None
            
            # Call __exit__ directly
            client.__exit__(None, None, None)
            
            # Verify close was called
            mock_httpx_client.return_value.close.assert_called_once()
    
    def test_exit_with_no_client(self):
        """Test __exit__ method safely handles case when _client is None."""
        with patch('httpx.Client') as mock_httpx_client:
            client = VeniceClient(api_key="test-api-key")
            # Force client._client to be None to test the hasattr/truthiness check
            client._client = cast(httpx.Client, None) # Explicit cast to satisfy type checker
            
            # Call __exit__ should not raise an error
            client.__exit__(None, None, None)
            
            # Verify close was not called
            mock_httpx_client.return_value.close.assert_not_called()
            
    def test_context_manager_with_custom_http_client(self):
        """Test context manager with a custom http_client."""
        # Create a custom HTTP client
        custom_client = MagicMock(spec=httpx.Client)
        
        # Use the client in a context manager
        with VeniceClient(api_key="test-api-key", http_client=custom_client) as client:
            assert client._client is custom_client
        
        # Verify the user-provided client was NOT closed on exit
        custom_client.close.assert_not_called()