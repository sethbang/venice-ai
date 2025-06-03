import pytest
import httpx
from unittest.mock import patch, MagicMock
from venice_ai._client import VeniceClient, ChatResource
from venice_ai.exceptions import VeniceError
from venice_ai import _constants
import json
import os


class TestVeniceClient:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=VeniceClient)
        client._api_key = "test-api-key"
        client._base_url = httpx.URL("https://api.venice.ai/api/v1/")
        client._timeout = 60.0
        client._max_retries = 2
        client._client = MagicMock(spec=httpx.Client)
        return client

    def test_initialization_with_api_key(self):
        with patch('httpx.Client') as mock_httpx_client:
            client = VeniceClient(api_key="test-api-key")
            assert client._api_key == "test-api-key"
            assert str(client._base_url).startswith("https://api.venice.ai/api/v1/")
            mock_httpx_client.assert_called_once()
            assert hasattr(client, 'chat')
            assert hasattr(client, 'models')
            assert hasattr(client, 'image')
            assert hasattr(client, 'audio')
            assert hasattr(client, 'billing')
            assert hasattr(client, 'embeddings')
            assert hasattr(client, 'api_keys')

    def test_initialization_without_api_key(self):
        # Ensure no environment variable is set for this test
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="The api_key client option must be set."):
                VeniceClient(api_key="")

    def test_initialization_with_custom_base_url(self):
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-api-key", base_url="https://custom.api.com")
            assert str(client._base_url).startswith("https://custom.api.com/")

    def test_initialization_with_custom_timeout_and_retries(self):
        with patch('httpx.Client') as mock_httpx_client:
            client = VeniceClient(api_key="test-api-key", timeout=30.0, max_retries=5)
            assert client._timeout.read == 30.0
            assert client._max_retries == 5
            mock_httpx_client.assert_called_once()
            
    def test_initialization_default_timeout(self):
        with patch('httpx.Client') as mock_httpx_client:
            client = VeniceClient(api_key="test-api-key", timeout=None)
            assert client._timeout.read == _constants.DEFAULT_TIMEOUT.read
            mock_httpx_client.assert_called_once()

    def test_request_basic_get(self):
        with patch('httpx.Client') as mock_httpx_client:
            mock_httpx_client.return_value.request.return_value = MagicMock(json=MagicMock(return_value={"status": "success"}))
            client = VeniceClient(api_key="test-api-key")
            result = client._request("GET", "test_endpoint")
            assert result == {"status": "success"}

    def test_request_basic_post(self):
        with patch('httpx.Client') as mock_httpx_client:
            mock_httpx_client.return_value.request.return_value = MagicMock(json=MagicMock(return_value={"status": "success"}))
            client = VeniceClient(api_key="test-api-key")
            json_data = {"key": "value"}
            result = client._request("POST", "test_endpoint", json_data=json_data)
            assert result == {"status": "success"}

    def test_request_with_params(self):
        with patch('httpx.Client') as mock_httpx_client:
            mock_httpx_client.return_value.request.return_value = MagicMock(json=MagicMock(return_value={"status": "success"}))
            client = VeniceClient(api_key="test-api-key")
            params = {"param1": "value1"}
            result = client.get("test_endpoint", params=params)
            assert result == {"status": "success"}

    def test_request_error_handling_http_status(self):
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response) # Use spec for better mocking
            mock_response.status_code = 400 # Set the status code
            mock_request_for_status_error = MagicMock(spec=httpx.Request)
            mock_request_for_status_error.method = "GET" # Or the appropriate method
            mock_request_for_status_error.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint") # Or the appropriate URL
            mock_response.request = mock_request_for_status_error
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                message="Error", request=mock_response.request, response=mock_response
            )
            mock_httpx_client.return_value.request.return_value = mock_response
            client = VeniceClient(api_key="test-api-key")
            from venice_ai.exceptions import InvalidRequestError
            with pytest.raises(InvalidRequestError):
                client._request("GET", "test_endpoint")

    def test_request_error_handling_timeout(self):
        with patch('httpx.Client') as mock_httpx_client:
            mock_httpx_client.return_value.request.side_effect = httpx.TimeoutException("Timeout", request=MagicMock())
            client = VeniceClient(api_key="test-api-key")
            with pytest.raises(VeniceError, match="Request timed out"):
                client._request("GET", "test_endpoint")

    def test_request_error_handling_request_error(self):
        with patch('httpx.Client') as mock_httpx_client:
            mock_httpx_client.return_value.request.side_effect = httpx.RequestError("Request failed", request=MagicMock())
            client = VeniceClient(api_key="test-api-key")
            with pytest.raises(VeniceError, match="Request failed"):
                client._request("GET", "test_endpoint")

    def test_request_error_handling_non_json_body(self):
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 400
            mock_request_for_status_error = MagicMock(spec=httpx.Request)
            mock_request_for_status_error.method = "GET" # Or the appropriate method
            mock_request_for_status_error.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint") # Or the appropriate URL
            mock_response.request = mock_request_for_status_error
            mock_response.text = "Not JSON content" # Set text content
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                message="Error", request=mock_response.request, response=mock_response
            )
            mock_httpx_client.return_value.request.return_value = mock_response

            client = VeniceClient(api_key="test-api-key")
            from venice_ai.exceptions import InvalidRequestError
            with pytest.raises(InvalidRequestError) as excinfo:
                client._request("GET", "test_endpoint")

            # Check that the exception contains the non-JSON body in its message
            assert "API error 400" in str(excinfo.value)

    def test_request_raw_response(self):
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock()
            mock_response.content = b"raw bytes content"
            mock_response.raise_for_status = MagicMock()
            mock_httpx_client.return_value.request.return_value = mock_response

            client = VeniceClient(api_key="test-api-key")
            result = client._request("GET", "test_endpoint", raw_response=True)
            assert isinstance(result, bytes)
            assert result == b"raw bytes content"
            mock_httpx_client.return_value.request.assert_called_once()

    def test_request_raw_response_no_cast_to(self):
        """Test _request with raw_response=True and no cast_to returns raw content."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.content = b"raw binary data for no_cast_to"
            mock_response.raise_for_status = MagicMock()
            # Make .json() raise an error to ensure .content is used
            mock_response.json.side_effect = json.JSONDecodeError("Cannot decode", "doc", 0)
            mock_httpx_client.return_value.request.return_value = mock_response

            client = VeniceClient(api_key="test-api-key")
            result = client._request("GET", "some_endpoint", raw_response=True) # cast_to is implicitly None
            
            assert result == b"raw binary data for no_cast_to"
            mock_httpx_client.return_value.request.assert_called_once()

    def test_stream_request_basic(self):
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock()
            mock_response.iter_lines.return_value = [
                "data: {\"choices\": [{\"delta\": {\"content\": \"chunk1\"}}]}",
                "data: {\"choices\": [{\"delta\": {\"content\": \"chunk2\"}}]}",
                "data: [DONE]"
            ]
            mock_httpx_client.return_value.stream.return_value.__enter__.return_value = mock_response
            client = VeniceClient(api_key="test-api-key")
            chunks = list(client._stream_request("POST", "chat/completions", json_data={"model": "venice-1"}))
            assert len(chunks) == 2
            assert chunks[0]["choices"][0]["delta"].get("content", "") == "chunk1"
            assert chunks[1]["choices"][0]["delta"].get("content", "") == "chunk2"

    def test_stream_request_empty_lines(self):
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock()
            mock_response.iter_lines.return_value = [
                "",
                "data: {\"choices\": [{\"delta\": {\"content\": \"chunk1\"}}]}",
                "data: [DONE]"
            ]
            mock_httpx_client.return_value.stream.return_value.__enter__.return_value = mock_response
            client = VeniceClient(api_key="test-api-key")
            chunks = list(client._stream_request("POST", "chat/completions", json_data={"model": "venice-1"}))
            assert len(chunks) == 1
            assert chunks[0]["choices"][0]["delta"].get("content", "") == "chunk1"

    def test_stream_request_json_decode_error(self):
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock()
            mock_response.iter_lines.return_value = [
                "data: invalid_json",
                "data: {\"choices\": [{\"delta\": {\"content\": \"chunk1\"}}]}",
                "data: [DONE]"
            ]
            mock_httpx_client.return_value.stream.return_value.__enter__.return_value = mock_response
            # Patch the logger instead of print
            with patch('venice_ai._client.logger') as mock_logger:
                client = VeniceClient(api_key="test-api-key")
                chunks = list(client._stream_request("POST", "chat/completions", json_data={"model": "venice-1"}))
                assert len(chunks) == 1
                assert chunks[0]["choices"][0]["delta"].get("content", "") == "chunk1"
                # Check if the expected error message was logged
                mock_logger.error.assert_any_call(
                    "Failed to parse JSON in streaming response: Expecting value: line 1 column 1 (char 0)"
                )
                mock_logger.error.assert_any_call(
                    "Problematic JSON string: 'invalid_json'"
                )

    def test_stream_request_raw_basic(self):
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock()
            mock_response.iter_bytes.return_value = [b"chunk1", b"chunk2"]
            mock_httpx_client.return_value.stream.return_value.__enter__.return_value = mock_response
            client = VeniceClient(api_key="test-api-key")
            chunks = list(client._stream_request_raw("POST", "audio/speech", json_data={"text": "test"}))
            assert len(chunks) == 2
            assert chunks[0] == b"chunk1"
            assert chunks[1] == b"chunk2"

    def test_stream_request_error_handling_http_status(self):
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response) # Use spec for better mocking
            mock_response.status_code = 400 # Set the status code
            mock_response.request = MagicMock(spec=httpx.Request, method="POST", url=httpx.URL("https://api.venice.ai/api/v1/chat/completions")) # Add method and url
            mock_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
                message="Error", request=mock_response.request, response=mock_response
            ))
            
            mock_stream_context = MagicMock()
            mock_stream_context.__enter__.return_value = mock_response
            mock_stream_context.__exit__.return_value = None
            
            mock_httpx_client.return_value.stream.return_value = mock_stream_context
            
            client = VeniceClient(api_key="test-api-key")
            
            from venice_ai.exceptions import InvalidRequestError
            with pytest.raises(InvalidRequestError):
                for _ in client._stream_request("POST", "chat/completions", json_data={"model": "venice-1"}):
                    pass  # Should raise before yielding anything
    
    def test_stream_request_error_handling_timeout(self):
        with patch('httpx.Client') as mock_httpx_client:
            mock_httpx_client.return_value.stream = MagicMock(side_effect=httpx.TimeoutException("Stream timeout", request=MagicMock()))
            
            client = VeniceClient(api_key="test-api-key")
            
            with pytest.raises(VeniceError, match="Stream request timed out"):
                for _ in client._stream_request("POST", "chat/completions", json_data={"model": "venice-1"}):
                    pass  # Should raise before yielding anything
    
    def test_stream_request_error_handling_request_error(self):
        with patch('httpx.Client') as mock_httpx_client:
            mock_httpx_client.return_value.stream = MagicMock(side_effect=httpx.RequestError("Stream request failed", request=MagicMock()))
            
            client = VeniceClient(api_key="test-api-key")
            
            with pytest.raises(VeniceError, match="Stream request failed"):
                for _ in client._stream_request("POST", "chat/completions", json_data={"model": "venice-1"}):
                    pass  # Should raise before yielding anything
                    
    def test_stream_request_raw_error_handling(self):
        with patch('httpx.Client') as mock_httpx_client:
            mock_httpx_client.return_value.stream = MagicMock(side_effect=httpx.RequestError("Stream request failed", request=MagicMock()))
            
            client = VeniceClient(api_key="test-api-key")
            
            with pytest.raises(VeniceError, match="Stream request failed"):
                for _ in client._stream_request_raw("POST", "audio/speech", json_data={"text": "test"}):
                    pass  # Should raise before yielding anything
                    
    def test_stream_request_with_custom_headers(self):
        with patch('httpx.Client') as mock_httpx_client:
            mock_httpx_client.return_value.stream.return_value.__enter__.return_value = MagicMock(spec=httpx.Response)
            
            # Create a mock headers object that behaves like a dictionary
            # Initialize with the headers that VeniceClient sets during construction
            mock_headers = {
                "Accept": "application/json",
                "Authorization": "Bearer test-api-key"
            }
            
            class MockHeaders:
                def __init__(self, initial_headers):
                    self._headers = initial_headers.copy()
                
                def update(self, d):
                    self._headers.update(d)
                
                def __iter__(self):
                    return iter(self._headers)
                
                def items(self):
                    return self._headers.items()
                
                def __getitem__(self, key):
                    return self._headers[key]
                
                def __contains__(self, key):
                    return key in self._headers
                
                def keys(self):
                    return self._headers.keys()
                
                def values(self):
                    return self._headers.values()
            
            mock_headers_obj = MockHeaders(mock_headers)
            mock_httpx_client.return_value.headers = mock_headers_obj
            client = VeniceClient(api_key="test-api-key")
            client._client.headers.update({"User-Agent": "test-agent"}) # Simulate existing headers
            
            custom_headers = {"X-Custom-Header": "custom-value", "Accept": "application/json"}
            
            # Call _stream_request with a mocked async iterator
            mock_response_content = ["data: [DONE]"]
            mock_httpx_client.return_value.stream.return_value.__enter__.return_value.iter_lines.return_value = iter(mock_response_content)
            
            list(client._stream_request("POST", "chat/completions", headers=custom_headers))
            
            # Assert that httpx.Client's stream method was called with the correct headers
            call_args = mock_httpx_client.return_value.stream.call_args
            assert "Accept" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["Accept"] == "text/event-stream" # Should be overridden
            assert "X-Custom-Header" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["X-Custom-Header"] == "custom-value"
            assert "User-Agent" in call_args.kwargs["headers"] # Existing headers should be preserved
            assert call_args.kwargs["headers"]["User-Agent"] == "test-agent"
            
    def test_stream_request_raw_with_custom_headers(self):
        with patch('httpx.Client') as mock_httpx_client:
            mock_httpx_client.return_value.stream.return_value.__enter__.return_value = MagicMock(spec=httpx.Response)
            
            # Create a mock headers object that behaves like a dictionary
            # Initialize with the headers that VeniceClient sets during construction
            mock_headers = {
                "Accept": "application/json",
                "Authorization": "Bearer test-api-key"
            }
            
            class MockHeaders:
                def __init__(self, initial_headers):
                    self._headers = initial_headers.copy()
                
                def update(self, d):
                    self._headers.update(d)
                
                def __iter__(self):
                    return iter(self._headers)
                
                def items(self):
                    return self._headers.items()
                
                def __getitem__(self, key):
                    return self._headers[key]
                
                def __contains__(self, key):
                    return key in self._headers
                
                def keys(self):
                    return self._headers.keys()
                
                def values(self):
                    return self._headers.values()
            
            mock_headers_obj = MockHeaders(mock_headers)
            mock_httpx_client.return_value.headers = mock_headers_obj
            client = VeniceClient(api_key="test-api-key")
            client._client.headers.update({"User-Agent": "test-agent-raw"}) # Simulate existing headers
            
            custom_headers = {"X-Custom-Header": "custom-raw-value", "Content-Type": "audio/mpeg", "Accept": "audio/wav"}
            
            # Call _stream_request_raw with a mocked async iterator
            mock_response_content = [b"done"]
            mock_httpx_client.return_value.stream.return_value.__enter__.return_value.iter_bytes.return_value = iter(mock_response_content)
            
            list(client._stream_request_raw("POST", "audio/speech", headers=custom_headers))
            
            # Assert that httpx.Client's stream method was called with the correct headers
            call_args = mock_httpx_client.return_value.stream.call_args
            assert "Accept" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["Accept"] == "audio/wav" # Should be preserved
            assert "Content-Type" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["Content-Type"] == "audio/mpeg" # Should be preserved
            assert "X-Custom-Header" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["X-Custom-Header"] == "custom-raw-value"
            assert "User-Agent" in call_args.kwargs["headers"] # Existing headers should be preserved
            assert call_args.kwargs["headers"]["User-Agent"] == "test-agent-raw"

    def test_request_multipart_basic(self):
        with patch('httpx.Client') as mock_httpx_client:
            mock_httpx_client.return_value.request.return_value = MagicMock(json=MagicMock(return_value={"status": "success"}))
            client = VeniceClient(api_key="test-api-key")
            files = {"file": ("test.txt", b"content", "text/plain")}
            result = client._request_multipart("POST", "upload", files=files)
            assert result == {"status": "success"}

    def test_request_multipart_with_options(self):
        with patch('httpx.Client') as mock_httpx_client:
            mock_httpx_client.return_value.request.return_value = MagicMock(json=MagicMock(return_value={"status": "success"}))
            client = VeniceClient(api_key="test-api-key")
            files = {"file": ("test.txt", b"content", "text/plain")}
            data = {"field1": "value1", "field2": "value2"}
            headers = {"X-Custom-Header": "custom-value"}
            params = {"param1": "value1"}
            
            result = client._request_multipart(
                "POST",
                "upload",
                files=files,
                data=data,
                headers=headers,
                params=params
            )
            
            assert result == {"status": "success"}
            mock_httpx_client.return_value.request.assert_called_once()
            # Extract the call arguments
            call_args = mock_httpx_client.return_value.request.call_args
            assert call_args.kwargs["files"] == files
            assert call_args.kwargs["data"] == data
            assert "X-Custom-Header" in call_args.kwargs["headers"]
            assert call_args.kwargs["params"] == params

    def test_request_multipart_error_handling(self):
        with patch('httpx.Client') as mock_httpx_client:
            mock_httpx_client.return_value.request.side_effect = httpx.RequestError("Multipart request failed", request=MagicMock())
            
            client = VeniceClient(api_key="test-api-key")
            files = {"file": ("test.txt", b"content", "text/plain")}
            
            with pytest.raises(VeniceError, match="Request failed"):
                client._request_multipart("POST", "upload", files=files)

    def test_request_multipart_default_headers(self):
        with patch('httpx.Client') as mock_httpx_client:
            mock_httpx_client.return_value.request.return_value = MagicMock(json=MagicMock(return_value={"status": "success"}))
            
            client = VeniceClient(api_key="test-api-key")
            # Ensure the default client headers are set for this test
            client._client.headers = {"Authorization": f"Bearer {client._api_key}", "User-Agent": "test-agent-sync"}

            files = {"file": ("test.txt", b"content", "text/plain")}
            # Call without providing 'Authorization' or 'Accept' in headers argument
            result = client._request_multipart("POST", "upload", files=files, headers={"X-Other-Sync-Header": "sync-value"})
            
            assert result == {"status": "success"}
            mock_httpx_client.return_value.request.assert_called_once()
            call_args = mock_httpx_client.return_value.request.call_args
            
            # Check that Authorization is present (from client default)
            assert "Authorization" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["Authorization"] == f"Bearer {client._api_key}"
            
            # Check that Accept is set to */* if not provided
            assert "Accept" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["Accept"] == "*/*"

            # Check that User-Agent is preserved
            assert "User-Agent" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["User-Agent"] == "test-agent-sync"

            # Check that other headers are preserved
            assert "X-Other-Sync-Header" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["X-Other-Sync-Header"] == "sync-value"

    def test_request_multipart_default_headers_with_none_headers_arg(self):
        """Test _request_multipart default headers when headers=None is passed."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_httpx_client.return_value.request.return_value = MagicMock(json=MagicMock(return_value={"status": "success"}))
            
            client = VeniceClient(api_key="test-api-key")
            # Ensure the default client headers are set for this test
            client._client.headers = {"Authorization": f"Bearer {client._api_key}", "User-Agent": "test-agent-sync-none"}

            files = {"file": ("test.txt", b"content", "text/plain")}
            # Call with headers=None
            result = client._request_multipart("POST", "upload", files=files, headers=None)
            
            assert result == {"status": "success"}
            mock_httpx_client.return_value.request.assert_called_once()
            call_args = mock_httpx_client.return_value.request.call_args
            
            assert "Authorization" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["Authorization"] == f"Bearer {client._api_key}"
            assert "Accept" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["Accept"] == "*/*"
            assert "User-Agent" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["User-Agent"] == "test-agent-sync-none"

    def test_request_error_handling_connect_error(self):
        """Test _request error handling for httpx.ConnectError."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_request_connect_error = MagicMock(spec=httpx.Request)
            mock_request_connect_error.method = "GET"
            mock_request_connect_error.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
            mock_httpx_client.return_value.request.side_effect = httpx.ConnectError("Connection failed", request=mock_request_connect_error)
            client = VeniceClient(api_key="test-api-key")
            with pytest.raises(VeniceError, match="Request failed: Connection failed"):
                client._request("GET", "test_endpoint")

    def test_request_error_handling_read_timeout(self):
        """Test _request error handling for httpx.ReadTimeout."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_request_read_timeout = MagicMock(spec=httpx.Request)
            mock_request_read_timeout.method = "GET"
            mock_request_read_timeout.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
            mock_httpx_client.return_value.request.side_effect = httpx.ReadTimeout("Read timed out", request=mock_request_read_timeout)
            client = VeniceClient(api_key="test-api-key")
            with pytest.raises(VeniceError, match="Request timed out: Read timed out"):
                client._request("GET", "test_endpoint")

    def test_stream_request_error_handling_connect_error(self):
        """Test _stream_request error handling for httpx.ConnectError."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_request_stream_connect_error = MagicMock(spec=httpx.Request)
            mock_request_stream_connect_error.method = "POST"
            mock_request_stream_connect_error.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
            mock_httpx_client.return_value.stream = MagicMock(side_effect=httpx.ConnectError("Stream connection failed", request=mock_request_stream_connect_error))
            client = VeniceClient(api_key="test-api-key")
            with pytest.raises(VeniceError, match="Stream request failed: Stream connection failed"):
                for _ in client._stream_request("POST", "chat/completions", json_data={"model": "venice-1"}):
                    pass

    def test_stream_request_raw_error_handling_connect_error(self):
        """Test _stream_request_raw error handling for httpx.ConnectError."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_request_raw_stream_connect_error = MagicMock(spec=httpx.Request)
            mock_request_raw_stream_connect_error.method = "POST"
            mock_request_raw_stream_connect_error.url = httpx.URL("https://api.venice.ai/api/v1/audio/speech")
            mock_httpx_client.return_value.stream = MagicMock(side_effect=httpx.ConnectError("Raw stream connection failed", request=mock_request_raw_stream_connect_error))
            client = VeniceClient(api_key="test-api-key")
            with pytest.raises(VeniceError, match="Stream request failed: Raw stream connection failed"):
                for _ in client._stream_request_raw("POST", "audio/speech", json_data={"text": "test"}):
                    pass

    def test_enter_method(self):
        """Test __enter__ method."""
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-api-key")
            entered_client = client.__enter__()
            assert entered_client is client

    def test_exit_method(self):
        """Test __exit__ method."""
        with patch('httpx.Client') as mock_httpx_client:
            client = VeniceClient(api_key="test-api-key")
            client._client = mock_httpx_client.return_value # Ensure the mock client is used
            
            client.__exit__(None, None, None) # Call with no exception

            mock_httpx_client.return_value.close.assert_called_once()

    def test_exit_method_with_exception(self):
        """Test __exit__ method when an exception occurs."""
        with patch('httpx.Client') as mock_httpx_client:
            client = VeniceClient(api_key="test-api-key")
            client._client = mock_httpx_client.return_value # Ensure the mock client is used
            
            try:
                # Simulate an exception within the context
                client.__exit__(ValueError, ValueError("Test"), None)
            except ValueError:
                pass # Catch the re-raised exception

            mock_httpx_client.return_value.close.assert_called_once()

    def test_context_manager(self):
        with patch('httpx.Client') as mock_httpx_client:
            with VeniceClient(api_key="test-api-key") as client:
                assert isinstance(client, VeniceClient)
            mock_httpx_client.return_value.close.assert_called_once()

    def test_close(self):
        with patch('httpx.Client') as mock_httpx_client:
            client = VeniceClient(api_key="test-api-key")
            client.close()
            mock_httpx_client.return_value.close.assert_called_once()

    def test_client_closes_transport_on_close(self):
        """Test that VeniceClient.close() properly closes the underlying httpx.Client transport."""
        with patch('httpx.Client') as mock_httpx_client:
            client = VeniceClient(api_key="test-api-key")
            client.close()
            mock_httpx_client.return_value.close.assert_called_once()

    def test_stream_response_sync_iteration_error_handling(self):
        """Test that Stream.__next__ properly handles httpx.HTTPError during iteration."""
        with patch('httpx.Client') as mock_httpx_client:
            # Create a mock iterator that raises httpx.ReadError on the second call to next()
            mock_iterator = MagicMock()
            
            # First call returns a chunk, second call raises httpx.ReadError
            def side_effect():
                yield {"choices": [{"delta": {"content": "chunk1"}}]}
                # Simulate an httpx.ReadError during stream iteration
                mock_request = MagicMock(spec=httpx.Request)
                mock_request.method = "POST"
                mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
                raise httpx.ReadError("Connection broken during stream", request=mock_request)
            
            mock_iterator.__iter__ = MagicMock(return_value=iter(side_effect()))
            mock_iterator.__next__ = MagicMock(side_effect=iter(side_effect()).__next__)
            
            client = VeniceClient(api_key="test-api-key")
            
            # Import Stream class
            from venice_ai.streaming import Stream
            
            # Create a Stream instance with the mock iterator
            stream = Stream(mock_iterator, client=client)
            
            # First iteration should work
            first_chunk = next(stream)
            assert first_chunk["choices"][0]["delta"]["content"] == "chunk1"
            
            # Second iteration should raise an APIError (translated from httpx.ReadError)
            from venice_ai.exceptions import APIConnectionError
            with pytest.raises(APIConnectionError, match="Connection broken during stream"):
                next(stream)

    def test_stream_response_consumed_error(self):
        """Test that Stream raises StreamConsumedError when attempting to re-iterate after consumption."""
        with patch('httpx.Client') as mock_httpx_client:
            # Create a mock iterator that yields one chunk then stops
            mock_iterator = MagicMock()
            
            def side_effect():
                yield {"choices": [{"delta": {"content": "chunk1"}}]}
                # Iterator is exhausted after one item
            
            mock_iterator.__iter__ = MagicMock(return_value=iter(side_effect()))
            mock_iterator.__next__ = MagicMock(side_effect=iter(side_effect()).__next__)
            
            client = VeniceClient(api_key="test-api-key")
            
            # Import Stream class
            from venice_ai.streaming import Stream
            
            # Create a Stream instance with the mock iterator
            stream = Stream(mock_iterator, client=client)
            
            # First iteration should work and consume the stream
            first_chunk = next(stream)
            assert first_chunk["choices"][0]["delta"]["content"] == "chunk1"
            
            # Stream should be exhausted now, trying to get next should raise StopIteration
            with pytest.raises(StopIteration):
                next(stream)
            
            # Now attempting to iterate again should raise StreamConsumedError
            from venice_ai.exceptions import StreamConsumedError
            with pytest.raises(StreamConsumedError, match="Cannot iterate over a consumed stream"):
                for chunk in stream:
                    pass  # Should raise before yielding anything

    def test_stream_response_closed_error(self):
        """Test that Stream raises StreamClosedError when attempting to iterate over a closed stream."""
        with patch('httpx.Client') as mock_httpx_client:
            # Create a mock iterator that raises httpx.StreamClosed when accessed
            mock_iterator = MagicMock()
            
            def side_effect():
                # Simulate httpx.StreamClosed being raised during iteration
                mock_request = MagicMock(spec=httpx.Request)
                mock_request.method = "POST"
                mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
                raise httpx.StreamClosed()
            
            mock_iterator.__iter__ = MagicMock(return_value=iter([]))
            mock_iterator.__next__ = MagicMock(side_effect=side_effect)
            
            client = VeniceClient(api_key="test-api-key")
            
            # Import Stream class
            from venice_ai.streaming import Stream
            
            # Create a Stream instance with the mock iterator
            stream = Stream(mock_iterator, client=client)
            
            # Attempting to iterate should raise StreamClosedError (translated from httpx.StreamClosed)
            from venice_ai.exceptions import StreamClosedError
            with pytest.raises(StreamClosedError, match="Stream has been closed"):
                next(stream)

    def test_timeout_connect_error_raw_response(self):
        """Test _request error handling for httpx.ConnectTimeout with raw_response=True."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_request_connect_timeout = MagicMock(spec=httpx.Request)
            mock_request_connect_timeout.method = "GET"
            mock_request_connect_timeout.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
            mock_httpx_client.return_value.request.side_effect = httpx.ConnectTimeout("Connection timed out", request=mock_request_connect_timeout)
            client = VeniceClient(api_key="test-api-key")
            from venice_ai.exceptions import APITimeoutError
            with pytest.raises(APITimeoutError) as exc_info:
                client._request("GET", "test_endpoint", raw_response=True)
            
            # Verify the APITimeoutError has the correct request attribute
            error = exc_info.value
            assert error.request is not None
            assert error.request.method == "GET"
            assert str(error.request.url) == "https://api.venice.ai/api/v1/test_endpoint"
            # For connection timeouts, there should be no response
            assert error.response is None

    def test_build_request_auth_headers_default_token_retention(self):
        """Test that build_request correctly handles authentication headers with default token retention."""
        with patch('httpx.Client'):
            # Test 1: Client initialized with API key
            client = VeniceClient(api_key="test-api-key")
            
            # Build a request and check that auth headers are included
            request_info = client.build_request("POST", "chat/completions", json_data={"model": "venice-1"})
            
            assert "Authorization" in request_info["headers"]
            assert request_info["headers"]["Authorization"] == "Bearer test-api-key"
            assert request_info["method"] == "POST"
            assert request_info["url"].endswith("chat/completions")
            assert request_info["json"] == {"model": "venice-1"}
            
            # Test 2: Multiple calls should retain the same token (default token retention)
            request_info2 = client.build_request("GET", "models")
            assert "Authorization" in request_info2["headers"]
            assert request_info2["headers"]["Authorization"] == "Bearer test-api-key"
            
            # Test 3: Client initialized without API key but with environment variable
            with patch.dict('os.environ', {'VENICE_API_KEY': 'env-api-key'}):
                client_env = VeniceClient()
                request_info3 = client_env.build_request("POST", "chat/completions")
                
                assert "Authorization" in request_info3["headers"]
                assert request_info3["headers"]["Authorization"] == "Bearer env-api-key"
                
                # Test token retention for env-based client
                request_info4 = client_env.build_request("GET", "models")
                assert "Authorization" in request_info4["headers"]
                assert request_info4["headers"]["Authorization"] == "Bearer env-api-key"
            
            # Test 4: Custom headers should be merged with auth headers
            custom_headers = {"X-Custom-Header": "custom-value"}
            request_info5 = client.build_request("POST", "chat/completions", headers=custom_headers)
            
            assert "Authorization" in request_info5["headers"]
            assert request_info5["headers"]["Authorization"] == "Bearer test-api-key"
            assert "X-Custom-Header" in request_info5["headers"]
            assert request_info5["headers"]["X-Custom-Header"] == "custom-value"

class TestChatResource:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=VeniceClient)
        client._api_key = "test-api-key"
        client._base_url = httpx.URL("https://api.venice.ai/api/v1/")
        client._client = MagicMock(spec=httpx.Client)
        return client

    def test_initialization(self, mock_client):
        resource = ChatResource(mock_client)
        assert resource._client == mock_client
        assert hasattr(resource, 'completions')