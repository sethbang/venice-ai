import pytest
import httpx
import json
from unittest.mock import MagicMock, patch, Mock
from typing import Dict, Any, Iterator, cast

from venice_ai._client import VeniceClient
from venice_ai.exceptions import (
    VeniceError, APIError, AuthenticationError, PermissionDeniedError,
    NotFoundError, InvalidRequestError, RateLimitError, InternalServerError
)

class TestClientExtended:
    """Additional tests for VeniceClient to improve coverage."""
    
    @pytest.fixture
    def client(self):
        """Create a client with a mocked httpx client."""
        mock_client = MagicMock(spec=httpx.Client)
        client = VeniceClient(api_key="test-key")
        client._client = mock_client
        return client
    
    def test_request_get_removes_content_type_and_accept(self, client):
        """Test that GET requests remove Content-Type and Accept headers if not explicitly provided."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"result": "success"}
        mock_response.raise_for_status = MagicMock()
        mock_request = cast(MagicMock, client._client.request)
        mock_request.return_value = mock_response
        
        # Set initial headers to verify they get removed
        client._client.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer test-key"
        }
        
        result = client._request("GET", "test/endpoint")
        
        assert result == {"result": "success"}
        
        # Verify headers were removed for the GET request
        _, kwargs = client._client.request.call_args
        assert "Content-Type" not in kwargs["headers"]
        assert "Accept" not in kwargs["headers"]
        assert "Authorization" in kwargs["headers"]
    
    def test_request_get_keeps_headers_if_explicitly_provided(self, client):
        """Test that GET requests keep Content-Type and Accept if explicitly provided in headers."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"result": "success"}
        mock_response.raise_for_status = MagicMock()
        mock_request = cast(MagicMock, client._client.request)
        mock_request.return_value = mock_response
        
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
        
        result = client._request("GET", "test/endpoint", headers=explicit_headers)
        
        assert result == {"result": "success"}
        
        # Verify explicit headers were kept
        _, kwargs = client._client.request.call_args
        assert kwargs["headers"]["Content-Type"] == "text/plain"
        assert kwargs["headers"]["Accept"] == "text/plain"
    
    def test_request_raw_response(self, client):
        """Test requesting raw binary response instead of parsed JSON."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.content = b"binary content"
        mock_response.raise_for_status = MagicMock()
        mock_request = cast(MagicMock, client._client.request)
        mock_request.return_value = mock_response
        
        result = client._request("GET", "test/endpoint", raw_response=True)
        
        assert result == b"binary content"
        assert not mock_response.json.called
    
    def test_request_json_decode_error(self, client):
        """Test HTTP error with invalid JSON response."""
        mock_response = MagicMock(spec=httpx.Response)
        http_error = httpx.HTTPStatusError(
            "Error", request=MagicMock(method="POST", url=httpx.URL("https://api.venice.ai/api/v1/test/endpoint")), response=mock_response
        )
        mock_response.raise_for_status.side_effect = http_error
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.status_code = 400
        mock_request = cast(MagicMock, client._client.request)
        mock_request.return_value = mock_response
        
        with pytest.raises(InvalidRequestError):
            client._request("POST", "test/endpoint")
    
    def test_request_timeout(self, client):
        """Test handling of timeout exceptions."""
        mock_httpx_request = MagicMock(spec=httpx.Request)
        mock_httpx_request.method = "POST"
        mock_httpx_request.url = httpx.URL("https://api.venice.ai/api/v1/test/endpoint")
        timeout_error = httpx.TimeoutException("Timeout", request=mock_httpx_request)
        mock_request = cast(MagicMock, client._client.request)
        mock_request.side_effect = timeout_error
        
        with pytest.raises(VeniceError) as excinfo:
            client._request("POST", "test/endpoint")
        
        assert "Request timed out" in str(excinfo.value)
    
    def test_request_network_error(self, client):
        """Test handling of network errors."""
        mock_httpx_request = MagicMock(spec=httpx.Request)
        mock_httpx_request.method = "POST"
        mock_httpx_request.url = httpx.URL("https://api.venice.ai/api/v1/test/endpoint")
        network_error = httpx.NetworkError("Network error", request=mock_httpx_request)
        mock_request = cast(MagicMock, client._client.request)
        mock_request.side_effect = network_error
        
        with pytest.raises(VeniceError) as excinfo:
            client._request("POST", "test/endpoint")
        
        assert "Request failed" in str(excinfo.value)
    
    def test_request_multipart(self, client):
        """Test making a multipart request for file uploads."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"result": "success"}
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.raise_for_status = MagicMock()
        mock_request = cast(MagicMock, client._client.request)
        mock_request.return_value = mock_response
        
        # Set initial client headers
        client._client.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer test-key",
            "User-Agent": "Test Agent"
        }
        
        files = {"file": ("test.jpg", b"file content", "image/jpeg")}
        data = {"model": "test-model"}
        
        result = client._request_multipart("POST", "test/endpoint", files=files, data=data)
        
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
    
    def test_stream_request_empty_lines(self, client):
        """Test handling of empty lines in SSE stream."""
        # Mock the streaming response
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_stream
        mock_stream.__exit__.return_value = None
        
        # Define the lines to be returned by iter_lines
        lines = ["", "data: {\"chunk\": 1}", "", "data: {\"chunk\": 2}", "data: [DONE]"]
        mock_stream.iter_lines.return_value = lines
        mock_stream.raise_for_status = MagicMock()
        
        client._client.stream.return_value = mock_stream
        
        chunks = list(client._stream_request("POST", "test/endpoint"))
        
        assert len(chunks) == 2
        assert chunks[0] == {"chunk": 1}
        assert chunks[1] == {"chunk": 2}
    
    def test_stream_request_invalid_json(self, client):
        """Test handling of invalid JSON in stream."""
        # Mock the streaming response
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_stream
        mock_stream.__exit__.return_value = None
        
        # Define the lines with invalid JSON
        lines = [
            "data: {\"valid\": true}",
            "data: {invalid json}",  # This one should be skipped
            "data: {\"also_valid\": true}"
        ]
        mock_stream.iter_lines.return_value = lines
        mock_stream.raise_for_status = MagicMock()
        
        client._client.stream.return_value = mock_stream
        
        chunks = list(client._stream_request("POST", "test/endpoint"))
        
        # We should only get the valid JSON chunks
        assert len(chunks) == 2
        assert chunks[0] == {"valid": True}
        assert chunks[1] == {"also_valid": True}
    
    def test_stream_request_http_error(self, client):
        """Test handling of HTTP errors in stream requests."""
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_stream
        
        # Mock an HTTP error
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 401
        http_error = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(method="POST", url=httpx.URL("https://api.venice.ai/api/v1/test/endpoint")), response=mock_response
        )
        mock_stream.raise_for_status.side_effect = http_error
        
        client._client.stream.return_value = mock_stream
        
        with pytest.raises(AuthenticationError):
            list(client._stream_request("POST", "test/endpoint"))
    
    def test_stream_request_raw(self, client):
        """Test streaming raw binary data."""
        # Mock the streaming response
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_stream
        mock_stream.__exit__.return_value = None
        
        # Define the binary chunks to be yielded
        chunks = [b"chunk1", b"", b"chunk2"]  # Empty chunk should be skipped
        mock_stream.iter_bytes.return_value = chunks
        mock_stream.raise_for_status = MagicMock()
        
        client._client.stream.return_value = mock_stream
        
        received_chunks = list(client._stream_request_raw("POST", "test/endpoint"))
        
        # Empty chunk should be skipped
        assert len(received_chunks) == 2
        assert received_chunks[0] == b"chunk1"
        assert received_chunks[1] == b"chunk2"
    
    def test_delete_method(self, client):
        """Test the delete convenience method."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"result": "deleted"}
        mock_response.raise_for_status = MagicMock()
        mock_request = cast(MagicMock, client._client.request)
        mock_request.return_value = mock_response
        
        result = client.delete("test/endpoint")
        
        assert result == {"result": "deleted"}
        # Delete calls _request, which then calls _client.request
        # So we should check that _client.request was called with the correct method and URL
        client._client.request.assert_called_once()
        args, kwargs = client._client.request.call_args
        assert kwargs["method"] == "DELETE"
        assert kwargs["url"] == client._base_url.join("test/endpoint")
    
    def test_close_and_context_manager(self):
        """Test client close method and context manager functionality."""
        client = VeniceClient(api_key="test-key")
        mock_client = MagicMock(spec=httpx.Client)
        client._client = mock_client
        
        # Test close method
        client.close()
        mock_client.close.assert_called_once()
        
        # Test context manager
        mock_client = MagicMock(spec=httpx.Client)
        with patch("httpx.Client", return_value=mock_client):
            with VeniceClient(api_key="test-key") as client:
                assert isinstance(client, VeniceClient)
                
            # Verify client was closed on exit
            mock_client.close.assert_called_once()

    def test_convenience_methods(self, client):
        """Test the get and post convenience methods."""
        # Mock response for both methods
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"result": "success"}
        mock_response.raise_for_status = MagicMock()
        mock_request = cast(MagicMock, client._client.request)
        mock_request.return_value = mock_response
        
        # Test GET method
        get_result = client.get("test/endpoint", params={"param": "value"})
        assert get_result == {"result": "success"}
        
        # Test POST method
        post_data = {"key": "value"}
        post_result = client.post("test/endpoint", json_data=post_data)
        assert post_result == {"result": "success"}
        
        # Verify both calls were made correctly
        assert client._client.request.call_count == 2
        
        calls = client._client.request.call_args_list
        
        # Check GET call
        get_call = calls[0]
        assert get_call[1]["method"] == "GET"
        assert get_call[1]["params"] == {"param": "value"}
        
        # Check POST call
        post_call = calls[1]
        assert post_call[1]["method"] == "POST"
        assert post_call[1]["json"] == post_data