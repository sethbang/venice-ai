import pytest
import httpx
from unittest.mock import patch, MagicMock
from venice_ai._client import VeniceClient, ChatResource
from venice_ai._client_with_retries import VeniceClientWithRetries
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
            client = VeniceClientWithRetries(api_key="test-api-key", timeout=30.0, max_retries=5)
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

    # A1: Comprehensive tests for _request() method
    def test_request_get_method_success(self):
        """Test _request with GET method and successful JSON response."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.json.return_value = {"data": "success", "status": "ok"}
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            result = client._request("GET", "models")
            
            assert result == {"data": "success", "status": "ok"}
            mock_httpx_client.return_value.request.assert_called_once()
            call_args = mock_httpx_client.return_value.request.call_args
            assert call_args.kwargs["method"] == "GET"
            assert str(call_args.kwargs["url"]).endswith("models")

    def test_request_post_method_with_json_data(self):
        """Test _request with POST method and JSON data."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.json.return_value = {"id": "chat-123", "object": "chat.completion"}
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            json_data = {"model": "venice-1", "messages": [{"role": "user", "content": "Hello"}]}
            result = client._request("POST", "chat/completions", json_data=json_data)
            
            assert result == {"id": "chat-123", "object": "chat.completion"}
            call_args = mock_httpx_client.return_value.request.call_args
            assert call_args.kwargs["method"] == "POST"
            assert call_args.kwargs["json"] == json_data

    def test_request_put_method_success(self):
        """Test _request with PUT method."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.json.return_value = {"updated": True}
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            result = client._request("PUT", "resource/123", json_data={"name": "updated"})
            
            assert result == {"updated": True}
            call_args = mock_httpx_client.return_value.request.call_args
            assert call_args.kwargs["method"] == "PUT"

    def test_request_delete_method_success(self):
        """Test _request with DELETE method."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.json.return_value = {"deleted": True}
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            result = client._request("DELETE", "resource/123")
            
            assert result == {"deleted": True}
            call_args = mock_httpx_client.return_value.request.call_args
            assert call_args.kwargs["method"] == "DELETE"

    def test_request_patch_method_success(self):
        """Test _request with PATCH method."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.json.return_value = {"patched": True}
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            result = client._request("PATCH", "resource/123", json_data={"field": "value"})
            
            assert result == {"patched": True}
            call_args = mock_httpx_client.return_value.request.call_args
            assert call_args.kwargs["method"] == "PATCH"

    def test_request_with_params_success(self):
        """Test _request with URL parameters."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.json.return_value = {"results": []}
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            params = {"limit": 10, "offset": 0, "filter": "active"}
            result = client._request("GET", "models", params=params)
            
            assert result == {"results": []}
            call_args = mock_httpx_client.return_value.request.call_args
            assert call_args.kwargs["params"] == params

    def test_request_with_custom_headers(self):
        """Test _request with custom headers."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.json.return_value = {"success": True}
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            headers = {"X-Custom-Header": "custom-value", "X-Request-ID": "req-123"}
            result = client._request("GET", "models", headers=headers)
            
            assert result == {"success": True}
            call_args = mock_httpx_client.return_value.request.call_args
            request_headers = call_args.kwargs["headers"]
            assert "X-Custom-Header" in request_headers
            assert request_headers["X-Custom-Header"] == "custom-value"
            assert "X-Request-ID" in request_headers
            assert request_headers["X-Request-ID"] == "req-123"

    def test_request_with_custom_timeout(self):
        """Test _request with custom timeout."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.json.return_value = {"success": True}
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            custom_timeout = httpx.Timeout(30.0)
            result = client._request("GET", "models", timeout=custom_timeout)
            
            assert result == {"success": True}
            call_args = mock_httpx_client.return_value.request.call_args
            assert call_args.kwargs["timeout"] == custom_timeout

    def test_request_url_construction(self):
        """Test _request correctly constructs URLs from base_url and path."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.json.return_value = {"success": True}
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            result = client._request("GET", "models/list")
            
            assert result == {"success": True}
            call_args = mock_httpx_client.return_value.request.call_args
            url = str(call_args.kwargs["url"])
            assert "api.venice.ai/api/v1/models/list" in url

    def test_request_raw_response_success(self):
        """Test _request with raw_response=True returns raw response content."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.content = b"raw binary content"
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            result = client._request("GET", "models", raw_response=True)
            
            assert result == b"raw binary content"
            mock_response.raise_for_status.assert_called_once()

    def test_request_api_error_400_invalid_request(self):
        """Test _request handles 400 Bad Request errors correctly."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 400
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
            mock_response.request = mock_request
            mock_response.json.return_value = {"error": {"message": "Invalid model specified"}}
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                message="Bad Request", request=mock_request, response=mock_response
            )
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            from venice_ai.exceptions import InvalidRequestError
            with pytest.raises(InvalidRequestError) as exc_info:
                client._request("POST", "chat/completions", json_data={"model": "invalid"})
            
            error = exc_info.value
            assert error.status_code == 400
            assert error.request is not None
            assert error.response is not None

    def test_request_api_error_401_authentication(self):
        """Test _request handles 401 Unauthorized errors correctly."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 401
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "GET"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/models")
            mock_response.request = mock_request
            mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                message="Unauthorized", request=mock_request, response=mock_response
            )
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="invalid-key")
            from venice_ai.exceptions import AuthenticationError
            with pytest.raises(AuthenticationError) as exc_info:
                client._request("GET", "models")
            
            error = exc_info.value
            assert error.status_code == 401

    def test_request_api_error_403_permission_denied(self):
        """Test _request handles 403 Forbidden errors correctly."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 403
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/admin/users")
            mock_response.request = mock_request
            mock_response.json.return_value = {"error": {"message": "Insufficient permissions"}}
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                message="Forbidden", request=mock_request, response=mock_response
            )
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            from venice_ai.exceptions import PermissionDeniedError
            with pytest.raises(PermissionDeniedError) as exc_info:
                client._request("POST", "admin/users")
            
            error = exc_info.value
            assert error.status_code == 403

    def test_request_api_error_404_not_found(self):
        """Test _request handles 404 Not Found errors correctly."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 404
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "GET"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/models/nonexistent")
            mock_response.request = mock_request
            mock_response.json.return_value = {"error": {"message": "Model not found"}}
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                message="Not Found", request=mock_request, response=mock_response
            )
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            from venice_ai.exceptions import NotFoundError
            with pytest.raises(NotFoundError) as exc_info:
                client._request("GET", "models/nonexistent")
            
            error = exc_info.value
            assert error.status_code == 404

    def test_request_api_error_422_unprocessable_entity(self):
        """Test _request handles 422 Unprocessable Entity errors correctly."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 422
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
            mock_response.request = mock_request
            mock_response.json.return_value = {"error": {"message": "Validation failed"}}
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                message="Unprocessable Entity", request=mock_request, response=mock_response
            )
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            from venice_ai.exceptions import UnprocessableEntityError
            with pytest.raises(UnprocessableEntityError) as exc_info:
                client._request("POST", "chat/completions", json_data={"invalid": "data"})
            
            error = exc_info.value
            assert error.status_code == 422

    def test_request_api_error_429_rate_limit(self):
        """Test _request handles 429 Rate Limit errors correctly."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 429
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
            mock_response.request = mock_request
            mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
            mock_response.headers = {"Retry-After": "60"}
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                message="Too Many Requests", request=mock_request, response=mock_response
            )
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            from venice_ai.exceptions import RateLimitError
            with pytest.raises(RateLimitError) as exc_info:
                client._request("POST", "chat/completions")
            
            error = exc_info.value
            assert error.status_code == 429
            assert error.retry_after_seconds == 60

    def test_request_api_error_500_internal_server_error(self):
        """Test _request handles 500 Internal Server Error correctly."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 500
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
            mock_response.request = mock_request
            mock_response.json.return_value = {"error": {"message": "Internal server error"}}
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                message="Internal Server Error", request=mock_request, response=mock_response
            )
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            from venice_ai.exceptions import InternalServerError
            with pytest.raises(InternalServerError) as exc_info:
                client._request("POST", "chat/completions")
            
            error = exc_info.value
            assert error.status_code == 500

    def test_request_api_error_503_service_unavailable(self):
        """Test _request handles 503 Service Unavailable errors correctly."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 503
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "GET"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/models")
            mock_response.request = mock_request
            mock_response.json.return_value = {"error": {"message": "Service temporarily unavailable"}}
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                message="Service Unavailable", request=mock_request, response=mock_response
            )
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            from venice_ai.exceptions import ServiceUnavailableError
            with pytest.raises(ServiceUnavailableError) as exc_info:
                client._request("GET", "models")
            
            error = exc_info.value
            assert error.status_code == 503

    def test_request_timeout_exception(self):
        """Test _request handles httpx.TimeoutException correctly."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
            timeout_error = httpx.TimeoutException("Request timed out", request=mock_request)
            mock_httpx_client.return_value.request.side_effect = timeout_error
            
            client = VeniceClient(api_key="test-api-key")
            from venice_ai.exceptions import APITimeoutError
            with pytest.raises(APITimeoutError) as exc_info:
                client._request("POST", "chat/completions")
            
            error = exc_info.value
            assert "Request timed out" in str(error)
            assert error.request is not None
            assert error.original_error is timeout_error

    def test_request_connect_error(self):
        """Test _request handles httpx.ConnectError correctly."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "GET"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/models")
            connect_error = httpx.ConnectError("Connection failed", request=mock_request)
            mock_httpx_client.return_value.request.side_effect = connect_error
            
            client = VeniceClient(api_key="test-api-key")
            from venice_ai.exceptions import APIConnectionError
            with pytest.raises(APIConnectionError) as exc_info:
                client._request("GET", "models")
            
            error = exc_info.value
            assert "Request failed: Connection failed" in str(error)
            assert error.request is not None
            assert error.original_error is connect_error

    def test_request_generic_request_error(self):
        """Test _request handles generic httpx.RequestError correctly."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "GET"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/models")
            request_error = httpx.RequestError("Network error", request=mock_request)
            mock_httpx_client.return_value.request.side_effect = request_error
            
            client = VeniceClient(api_key="test-api-key")
            from venice_ai.exceptions import APIConnectionError
            with pytest.raises(APIConnectionError) as exc_info:
                client._request("GET", "models")
            
            error = exc_info.value
            assert "Request failed: Network error" in str(error)
            assert error.request is not None
            assert error.original_error is request_error

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
            mock_response.status_code = 200
            mock_response.headers = {}
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
            assert chunks[0].choices[0].delta.content == "chunk1"
            assert chunks[1].choices[0].delta.content == "chunk2"

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
            assert chunks[0].choices[0].delta.content == "chunk1"

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
                assert chunks[0].choices[0].delta.content or "" == "chunk1"
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

    # A2: Comprehensive tests for _stream_request() method
    def test_stream_request_successful_sse_parsing(self):
        """Test _stream_request successfully parses SSE data chunks."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.iter_lines.return_value = [
                "data: {\"id\": \"chunk-1\", \"choices\": [{\"delta\": {\"content\": \"Hello\"}}]}",
                "data: {\"id\": \"chunk-2\", \"choices\": [{\"delta\": {\"content\": \" world\"}}]}",
                "data: {\"id\": \"chunk-3\", \"choices\": [{\"delta\": {\"content\": \"!\"}}]}",
                "data: [DONE]"
            ]
            mock_response.raise_for_status = MagicMock()
            mock_stream_context = MagicMock()
            mock_stream_context.__enter__.return_value = mock_response
            mock_stream_context.__exit__.return_value = None
            mock_httpx_client.return_value.stream.return_value = mock_stream_context
            
            client = VeniceClient(api_key="test-api-key")
            chunks = list(client._stream_request("POST", "chat/completions", json_data={"model": "venice-1"}))
            
            assert len(chunks) == 3
            assert chunks[0].id == "chunk-1"
            assert chunks[0].choices[0].delta.content == "Hello"
            assert chunks[1].choices[0].delta.content == " world"
            assert chunks[2].choices[0].delta.content == "!"

    def test_stream_request_with_custom_headers_and_params(self):
        """Test _stream_request with custom headers and parameters."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.iter_lines.return_value = [
                "data: {\"choices\": [{\"delta\": {\"content\": \"test\"}}]}",
                "data: [DONE]"
            ]
            mock_response.raise_for_status = MagicMock()
            mock_stream_context = MagicMock()
            mock_stream_context.__enter__.return_value = mock_response
            mock_stream_context.__exit__.return_value = None
            mock_httpx_client.return_value.stream.return_value = mock_stream_context
            
            client = VeniceClient(api_key="test-api-key")
            headers = {"X-Stream-ID": "stream-123"}
            params = {"stream": "true", "temperature": "0.7"}
            
            chunks = list(client._stream_request(
                "POST",
                "chat/completions",
                json_data={"model": "venice-1"},
                headers=headers,
                params=params
            ))
            
            assert len(chunks) == 1
            call_args = mock_httpx_client.return_value.stream.call_args
            assert call_args.kwargs["params"] == params
            assert "X-Stream-ID" in call_args.kwargs["headers"]

    def test_stream_request_handles_bytes_lines(self):
        """Test _stream_request correctly handles byte string lines."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.iter_lines.return_value = [
                b"data: {\"choices\": [{\"delta\": {\"content\": \"byte test\"}}]}",
                b"data: [DONE]"
            ]
            mock_response.raise_for_status = MagicMock()
            mock_stream_context = MagicMock()
            mock_stream_context.__enter__.return_value = mock_response
            mock_stream_context.__exit__.return_value = None
            mock_httpx_client.return_value.stream.return_value = mock_stream_context
            
            client = VeniceClient(api_key="test-api-key")
            chunks = list(client._stream_request("POST", "chat/completions"))
            
            assert len(chunks) == 1
            assert chunks[0].choices[0].delta.content == "byte test"

    def test_stream_request_skips_non_data_lines(self):
        """Test _stream_request skips lines that don't start with 'data: '."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.iter_lines.return_value = [
                "event: start",
                "id: 123",
                "data: {\"choices\": [{\"delta\": {\"content\": \"valid\"}}]}",
                "retry: 3000",
                "data: [DONE]"
            ]
            mock_response.raise_for_status = MagicMock()
            mock_stream_context = MagicMock()
            mock_stream_context.__enter__.return_value = mock_response
            mock_stream_context.__exit__.return_value = None
            mock_httpx_client.return_value.stream.return_value = mock_stream_context
            
            client = VeniceClient(api_key="test-api-key")
            chunks = list(client._stream_request("POST", "chat/completions"))
            
            assert len(chunks) == 1
            assert chunks[0].choices[0].delta.content == "valid"

    def test_stream_request_handles_malformed_json_gracefully(self):
        """Test _stream_request continues processing after encountering malformed JSON."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.iter_lines.return_value = [
                "data: {\"choices\": [{\"delta\": {\"content\": \"first\"}}]}",
                "data: {invalid json}",
                "data: {\"choices\": [{\"delta\": {\"content\": \"second\"}}]}",
                "data: [DONE]"
            ]
            mock_response.raise_for_status = MagicMock()
            mock_stream_context = MagicMock()
            mock_stream_context.__enter__.return_value = mock_response
            mock_stream_context.__exit__.return_value = None
            mock_httpx_client.return_value.stream.return_value = mock_stream_context
            
            with patch('venice_ai._client.logger') as mock_logger:
                client = VeniceClient(api_key="test-api-key")
                chunks = list(client._stream_request("POST", "chat/completions"))
                
                assert len(chunks) == 2
                assert chunks[0].choices[0].delta.content == "first"
                assert chunks[1].choices[0].delta.content == "second"
                # Verify error was logged
                mock_logger.error.assert_called()

    def test_stream_request_api_error_before_streaming(self):
        """Test _stream_request handles API errors that occur before streaming starts."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 400
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
            mock_response.request = mock_request
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                message="Bad Request", request=mock_request, response=mock_response
            )
            mock_stream_context = MagicMock()
            mock_stream_context.__enter__.return_value = mock_response
            mock_stream_context.__exit__.return_value = None
            mock_httpx_client.return_value.stream.return_value = mock_stream_context
            
            client = VeniceClient(api_key="test-api-key")
            from venice_ai.exceptions import InvalidRequestError
            with pytest.raises(InvalidRequestError):
                for _ in client._stream_request("POST", "chat/completions", json_data={"invalid": "model"}):
                    pass

    def test_stream_request_timeout_during_setup(self):
        """Test _stream_request handles timeout during stream setup."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
            timeout_error = httpx.TimeoutException("Stream setup timeout", request=mock_request)
            mock_httpx_client.return_value.stream.side_effect = timeout_error
            
            client = VeniceClient(api_key="test-api-key")
            from venice_ai.exceptions import APITimeoutError
            with pytest.raises(APITimeoutError) as exc_info:
                for _ in client._stream_request("POST", "chat/completions"):
                    pass
            
            error = exc_info.value
            assert "Stream request timed out" in str(error)
            assert error.original_error is timeout_error

    def test_stream_request_connection_error_during_setup(self):
        """Test _stream_request handles connection errors during stream setup."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
            connect_error = httpx.ConnectError("Stream connection failed", request=mock_request)
            mock_httpx_client.return_value.stream.side_effect = connect_error
            
            client = VeniceClient(api_key="test-api-key")
            from venice_ai.exceptions import APIConnectionError
            with pytest.raises(APIConnectionError) as exc_info:
                for _ in client._stream_request("POST", "chat/completions"):
                    pass
            
            error = exc_info.value
            assert "Stream request failed" in str(error)
            assert error.original_error is connect_error

    def test_stream_request_read_error_during_iteration(self):
        """Test _stream_request handles ReadError during stream iteration."""
        with patch('httpx.Client') as mock_httpx_client:
            def iter_lines_side_effect():
                yield "data: {\"choices\": [{\"delta\": {\"content\": \"first\"}}]}"
                # Simulate ReadError during iteration
                mock_request = MagicMock(spec=httpx.Request)
                mock_request.method = "POST"
                mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
                raise httpx.ReadError("Connection broken during read", request=mock_request)
            
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.iter_lines.side_effect = iter_lines_side_effect
            mock_response.raise_for_status = MagicMock()
            mock_stream_context = MagicMock()
            mock_stream_context.__enter__.return_value = mock_response
            mock_stream_context.__exit__.return_value = None
            mock_httpx_client.return_value.stream.return_value = mock_stream_context
            
            client = VeniceClient(api_key="test-api-key")
            from venice_ai.exceptions import APIConnectionError
            
            iterator = client._stream_request("POST", "chat/completions")
            # First chunk should work
            first_chunk = next(iterator)
            assert first_chunk.choices[0].delta.content == "first"
            
            # Second iteration should raise APIConnectionError
            with pytest.raises(APIConnectionError) as exc_info:
                next(iterator)
            
            error = exc_info.value
            assert "Connection broken during read" in str(error)

    def test_stream_request_stream_consumed_error_during_iteration(self):
        """Test _stream_request handles StreamConsumed error during iteration."""
        with patch('httpx.Client') as mock_httpx_client:
            def iter_lines_side_effect():
                yield "data: {\"choices\": [{\"delta\": {\"content\": \"first\"}}]}"
                # Simulate StreamConsumed error during iteration
                mock_request = MagicMock(spec=httpx.Request)
                mock_request.method = "POST"
                mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
                raise httpx.StreamConsumed()
            
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.iter_lines.side_effect = iter_lines_side_effect
            mock_response.raise_for_status = MagicMock()
            mock_stream_context = MagicMock()
            mock_stream_context.__enter__.return_value = mock_response
            mock_stream_context.__exit__.return_value = None
            mock_httpx_client.return_value.stream.return_value = mock_stream_context
            
            client = VeniceClient(api_key="test-api-key")
            from venice_ai.exceptions import APIError
            
            iterator = client._stream_request("POST", "chat/completions")
            # First chunk should work
            first_chunk = next(iterator)
            assert first_chunk.choices[0].delta.content == "first"
            
            # Second iteration should raise APIError
            with pytest.raises(APIError) as exc_info:
                next(iterator)
            
            error = exc_info.value
            assert "Stream already consumed" in str(error)

    def test_stream_request_stream_closed_error_during_iteration(self):
        """Test _stream_request handles StreamClosed error during iteration."""
        with patch('httpx.Client') as mock_httpx_client:
            def iter_lines_side_effect():
                yield "data: {\"choices\": [{\"delta\": {\"content\": \"first\"}}]}"
                # Simulate StreamClosed error during iteration
                raise httpx.StreamClosed()
            
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.iter_lines.side_effect = iter_lines_side_effect
            mock_response.raise_for_status = MagicMock()
            mock_stream_context = MagicMock()
            mock_stream_context.__enter__.return_value = mock_response
            mock_stream_context.__exit__.return_value = None
            mock_httpx_client.return_value.stream.return_value = mock_stream_context
            
            client = VeniceClient(api_key="test-api-key")
            from venice_ai.exceptions import APIError
            
            iterator = client._stream_request("POST", "chat/completions")
            # First chunk should work
            first_chunk = next(iterator)
            assert first_chunk.choices[0].delta.content == "first"
            
            # Second iteration should raise APIError
            with pytest.raises(APIError) as exc_info:
                next(iterator)
            
            error = exc_info.value
            assert "Stream already closed" in str(error)

    def test_stream_request_header_handling_for_get_requests(self):
        """Test _stream_request properly handles headers for GET requests."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.iter_lines.return_value = ["data: [DONE]"]
            mock_response.raise_for_status = MagicMock()
            mock_stream_context = MagicMock()
            mock_stream_context.__enter__.return_value = mock_response
            mock_stream_context.__exit__.return_value = None
            mock_httpx_client.return_value.stream.return_value = mock_stream_context
            
            # Mock client headers
            client = VeniceClient(api_key="test-api-key")
            client._client.headers = {"Authorization": "Bearer test-api-key", "Content-Type": "application/json"}
            
            list(client._stream_request("GET", "events"))
            
            call_args = mock_httpx_client.return_value.stream.call_args
            headers = call_args.kwargs["headers"]
            # For GET requests, Content-Type should be removed
            assert "Content-Type" not in headers or headers.get("Content-Type") != "application/json"
            assert "Authorization" in headers

    def test_stream_request_header_handling_for_post_requests(self):
        """Test _stream_request properly handles headers for POST requests with JSON data."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.iter_lines.return_value = ["data: [DONE]"]
            mock_response.raise_for_status = MagicMock()
            mock_stream_context = MagicMock()
            mock_stream_context.__enter__.return_value = mock_response
            mock_stream_context.__exit__.return_value = None
            mock_httpx_client.return_value.stream.return_value = mock_stream_context
            
            client = VeniceClient(api_key="test-api-key")
            client._client.headers = {"Authorization": "Bearer test-api-key"}
            
            list(client._stream_request("POST", "chat/completions", json_data={"model": "venice-1"}))
            
            call_args = mock_httpx_client.return_value.stream.call_args
            headers = call_args.kwargs["headers"]
            # For POST requests with JSON data, Content-Type should be set
            assert headers["Content-Type"] == "application/json"
            assert headers["Accept"] == "text/event-stream"
            assert "Authorization" in headers

    # A3: Comprehensive tests for _request_multipart() method
    def test_request_multipart_successful_upload(self):
        """Test _request_multipart with successful file upload."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.json.return_value = {"id": "upload-123", "status": "success"}
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            files = {"image": ("test.jpg", b"fake_image_data", "image/jpeg")}
            result = client._request_multipart("POST", "image/upscale", files=files)
            
            assert result == {"id": "upload-123", "status": "success"}
            call_args = mock_httpx_client.return_value.request.call_args
            assert call_args.kwargs["method"] == "POST"
            assert call_args.kwargs["files"] == files

    def test_request_multipart_with_form_data(self):
        """Test _request_multipart with both files and form data."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.json.return_value = {"processed": True}
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            files = {"document": ("doc.pdf", b"pdf_content", "application/pdf")}
            data = {"scale_factor": "2", "format": "png", "quality": "high"}
            result = client._request_multipart("POST", "process", files=files, data=data)
            
            assert result == {"processed": True}
            call_args = mock_httpx_client.return_value.request.call_args
            assert call_args.kwargs["files"] == files
            assert call_args.kwargs["data"] == data

    def test_request_multipart_with_custom_headers_and_params(self):
        """Test _request_multipart with custom headers and URL parameters."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.json.return_value = {"uploaded": True}
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            files = {"file": ("data.txt", b"text_data", "text/plain")}
            headers = {"X-Upload-ID": "upload-456", "X-Client-Version": "1.0"}
            params = {"async": "true", "callback_url": "https://example.com/callback"}
            
            result = client._request_multipart(
                "POST", "upload",
                files=files,
                headers=headers,
                params=params
            )
            
            assert result == {"uploaded": True}
            call_args = mock_httpx_client.return_value.request.call_args
            assert call_args.kwargs["params"] == params
            request_headers = call_args.kwargs["headers"]
            assert "X-Upload-ID" in request_headers
            assert request_headers["X-Upload-ID"] == "upload-456"
            assert "X-Client-Version" in request_headers

    def test_request_multipart_with_custom_timeout(self):
        """Test _request_multipart with custom timeout."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.json.return_value = {"success": True}
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            files = {"file": ("large.zip", b"large_file_data", "application/zip")}
            custom_timeout = httpx.Timeout(120.0)
            
            result = client._request_multipart("POST", "upload", files=files, timeout=custom_timeout)
            
            assert result == {"success": True}
            call_args = mock_httpx_client.return_value.request.call_args
            assert call_args.kwargs["timeout"] == custom_timeout

    def test_request_multipart_raw_response(self):
        """Test _request_multipart with raw_response=True."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.content = b"raw_binary_response"
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            files = {"image": ("photo.jpg", b"image_data", "image/jpeg")}
            
            result = client._request_multipart("POST", "process", files=files, raw_response=True)
            
            assert result == b"raw_binary_response"
            mock_response.raise_for_status.assert_called_once()

    def test_request_multipart_preserves_auth_headers(self):
        """Test _request_multipart preserves Authorization and User-Agent headers."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.json.return_value = {"authorized": True}
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            # Mock the client headers
            client._client.headers = {
                "Authorization": "Bearer test-api-key",
                "User-Agent": "venice-ai-python/1.0.0"
            }
            
            files = {"file": ("test.txt", b"content", "text/plain")}
            result = client._request_multipart("POST", "upload", files=files)
            
            assert result == {"authorized": True}
            call_args = mock_httpx_client.return_value.request.call_args
            headers = call_args.kwargs["headers"]
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer test-api-key"
            assert "User-Agent" in headers
            assert headers["User-Agent"] == "venice-ai-python/1.0.0"

    def test_request_multipart_sets_default_accept_header(self):
        """Test _request_multipart sets Accept: */* by default."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.json.return_value = {"success": True}
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            client._client.headers = {"Authorization": "Bearer test-api-key"}
            
            files = {"file": ("test.txt", b"content", "text/plain")}
            result = client._request_multipart("POST", "upload", files=files)
            
            assert result == {"success": True}
            call_args = mock_httpx_client.return_value.request.call_args
            headers = call_args.kwargs["headers"]
            assert headers["Accept"] == "*/*"

    def test_request_multipart_api_error_400(self):
        """Test _request_multipart handles 400 Bad Request errors."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 400
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/upload")
            mock_response.request = mock_request
            mock_response.json.return_value = {"error": {"message": "Invalid file format"}}
            mock_response.headers = {}
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                message="Bad Request", request=mock_request, response=mock_response
            )
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            files = {"file": ("invalid.txt", b"content", "text/plain")}
            
            from venice_ai.exceptions import InvalidRequestError
            with pytest.raises(InvalidRequestError) as exc_info:
                client._request_multipart("POST", "upload", files=files)
            
            error = exc_info.value
            assert error.status_code == 400

    def test_request_multipart_api_error_413_payload_too_large(self):
        """Test _request_multipart handles 413 Payload Too Large errors."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 413
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/upload")
            mock_response.request = mock_request
            mock_response.json.return_value = {"error": {"message": "File too large"}}
            mock_response.headers = {}
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                message="Payload Too Large", request=mock_request, response=mock_response
            )
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            files = {"file": ("huge.zip", b"x" * 1000000, "application/zip")}
            
            from venice_ai.exceptions import InvalidRequestError
            with pytest.raises(InvalidRequestError) as exc_info:
                client._request_multipart("POST", "upload", files=files)
            
            error = exc_info.value
            assert error.status_code == 413

    def test_request_multipart_timeout_error(self):
        """Test _request_multipart handles timeout errors."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/upload")
            timeout_error = httpx.TimeoutException("Upload timed out", request=mock_request)
            mock_httpx_client.return_value.request.side_effect = timeout_error
            
            client = VeniceClient(api_key="test-api-key")
            files = {"file": ("slow.zip", b"data", "application/zip")}
            
            from venice_ai.exceptions import APITimeoutError
            with pytest.raises(APITimeoutError) as exc_info:
                client._request_multipart("POST", "upload", files=files)
            
            error = exc_info.value
            assert "Request timed out" in str(error)
            assert error.original_error is timeout_error

    def test_request_multipart_connection_error(self):
        """Test _request_multipart handles connection errors."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/upload")
            connect_error = httpx.ConnectError("Upload connection failed", request=mock_request)
            mock_httpx_client.return_value.request.side_effect = connect_error
            
            client = VeniceClient(api_key="test-api-key")
            files = {"file": ("test.txt", b"data", "text/plain")}
            
            from venice_ai.exceptions import APIConnectionError
            with pytest.raises(APIConnectionError) as exc_info:
                client._request_multipart("POST", "upload", files=files)
            
            error = exc_info.value
            assert "Request failed" in str(error)
            assert error.original_error is connect_error

    def test_request_multipart_generic_request_error(self):
        """Test _request_multipart handles generic request errors."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/upload")
            request_error = httpx.RequestError("Upload failed", request=mock_request)
            mock_httpx_client.return_value.request.side_effect = request_error
            
            client = VeniceClient(api_key="test-api-key")
            files = {"file": ("test.txt", b"data", "text/plain")}
            
            from venice_ai.exceptions import APIConnectionError
            with pytest.raises(APIConnectionError) as exc_info:
                client._request_multipart("POST", "upload", files=files)
            
            error = exc_info.value
            assert "Request failed" in str(error)
            assert error.original_error is request_error

    def test_request_multipart_multiple_files(self):
        """Test _request_multipart with multiple files."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.json.return_value = {"files_processed": 2}
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_httpx_client.return_value.request.return_value = mock_response
            
            client = VeniceClient(api_key="test-api-key")
            files = {
                "image1": ("photo1.jpg", b"image1_data", "image/jpeg"),
                "image2": ("photo2.png", b"image2_data", "image/png")
            }
            
            result = client._request_multipart("POST", "batch_process", files=files)
            
            assert result == {"files_processed": 2}
            call_args = mock_httpx_client.return_value.request.call_args
            assert call_args.kwargs["files"] == files

    # A4: Comprehensive tests for _translate_httpx_error_to_api_error() method
    def test_translate_httpx_status_error_400(self):
        """Test _translate_httpx_error_to_api_error with 400 status error."""
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-api-key")
            
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 400
            mock_response.json.return_value = {"error": {"message": "Bad request"}}
            mock_response.text = '{"error": {"message": "Bad request"}}'
            
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test")
            
            http_error = httpx.HTTPStatusError("Bad Request", request=mock_request, response=mock_response)
            
            result = client._translate_httpx_error_to_api_error(http_error, mock_request)
            
            from venice_ai.exceptions import InvalidRequestError
            assert isinstance(result, InvalidRequestError)
            assert result.status_code == 400
            assert result.request is mock_request
            assert result.response is mock_response

    def test_translate_httpx_status_error_401(self):
        """Test _translate_httpx_error_to_api_error with 401 status error."""
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-api-key")
            
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 401
            mock_response.json.return_value = {"error": {"message": "Unauthorized"}}
            mock_response.text = '{"error": {"message": "Unauthorized"}}'
            
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "GET"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/models")
            
            http_error = httpx.HTTPStatusError("Unauthorized", request=mock_request, response=mock_response)
            
            result = client._translate_httpx_error_to_api_error(http_error, mock_request)
            
            from venice_ai.exceptions import AuthenticationError
            assert isinstance(result, AuthenticationError)
            assert result.status_code == 401

    def test_translate_httpx_status_error_403(self):
        """Test _translate_httpx_error_to_api_error with 403 status error."""
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-api-key")
            
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 403
            mock_response.json.return_value = {"error": {"message": "Forbidden"}}
            mock_response.text = '{"error": {"message": "Forbidden"}}'
            
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/admin")
            
            http_error = httpx.HTTPStatusError("Forbidden", request=mock_request, response=mock_response)
            
            result = client._translate_httpx_error_to_api_error(http_error, mock_request)
            
            from venice_ai.exceptions import PermissionDeniedError
            assert isinstance(result, PermissionDeniedError)
            assert result.status_code == 403

    def test_translate_httpx_status_error_404(self):
        """Test _translate_httpx_error_to_api_error with 404 status error."""
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-api-key")
            
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 404
            mock_response.json.return_value = {"error": {"message": "Not found"}}
            mock_response.text = '{"error": {"message": "Not found"}}'
            
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "GET"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/models/nonexistent")
            
            http_error = httpx.HTTPStatusError("Not Found", request=mock_request, response=mock_response)
            
            result = client._translate_httpx_error_to_api_error(http_error, mock_request)
            
            from venice_ai.exceptions import NotFoundError
            assert isinstance(result, NotFoundError)
            assert result.status_code == 404

    def test_translate_httpx_status_error_422(self):
        """Test _translate_httpx_error_to_api_error with 422 status error."""
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-api-key")
            
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 422
            mock_response.json.return_value = {"error": {"message": "Validation failed"}}
            mock_response.text = '{"error": {"message": "Validation failed"}}'
            
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
            
            http_error = httpx.HTTPStatusError("Unprocessable Entity", request=mock_request, response=mock_response)
            
            result = client._translate_httpx_error_to_api_error(http_error, mock_request)
            
            from venice_ai.exceptions import UnprocessableEntityError
            assert isinstance(result, UnprocessableEntityError)
            assert result.status_code == 422

    def test_translate_httpx_status_error_429_with_retry_after(self):
        """Test _translate_httpx_error_to_api_error with 429 status error and Retry-After header."""
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-api-key")
            
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 429
            mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
            mock_response.text = '{"error": {"message": "Rate limit exceeded"}}'
            mock_response.headers = {"Retry-After": "120"}
            
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
            
            http_error = httpx.HTTPStatusError("Too Many Requests", request=mock_request, response=mock_response)
            
            result = client._translate_httpx_error_to_api_error(http_error, mock_request)
            
            from venice_ai.exceptions import RateLimitError
            assert isinstance(result, RateLimitError)
            assert result.status_code == 429
            assert result.retry_after_seconds == 120

    def test_translate_httpx_status_error_500(self):
        """Test _translate_httpx_error_to_api_error with 500 status error."""
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-api-key")
            
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 500
            mock_response.json.return_value = {"error": {"message": "Internal server error"}}
            mock_response.text = '{"error": {"message": "Internal server error"}}'
            
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
            
            http_error = httpx.HTTPStatusError("Internal Server Error", request=mock_request, response=mock_response)
            
            result = client._translate_httpx_error_to_api_error(http_error, mock_request)
            
            from venice_ai.exceptions import InternalServerError
            assert isinstance(result, InternalServerError)
            assert result.status_code == 500

    def test_translate_httpx_status_error_503(self):
        """Test _translate_httpx_error_to_api_error with 503 status error."""
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-api-key")
            
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 503
            mock_response.json.return_value = {"error": {"message": "Service unavailable"}}
            mock_response.text = '{"error": {"message": "Service unavailable"}}'
            
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "GET"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/models")
            
            http_error = httpx.HTTPStatusError("Service Unavailable", request=mock_request, response=mock_response)
            
            result = client._translate_httpx_error_to_api_error(http_error, mock_request)
            
            from venice_ai.exceptions import ServiceUnavailableError
            assert isinstance(result, ServiceUnavailableError)
            assert result.status_code == 503

    def test_translate_httpx_timeout_exception(self):
        """Test _translate_httpx_error_to_api_error with TimeoutException."""
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-api-key")
            
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
            
            timeout_error = httpx.TimeoutException("Request timed out", request=mock_request)
            
            result = client._translate_httpx_error_to_api_error(timeout_error, mock_request)
            
            from venice_ai.exceptions import APITimeoutError
            assert isinstance(result, APITimeoutError)
            assert "Request timed out" in str(result)
            assert result.request is mock_request
            assert result.original_error is timeout_error

    def test_translate_httpx_timeout_exception_for_stream(self):
        """Test _translate_httpx_error_to_api_error with TimeoutException for stream request."""
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-api-key")
            
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
            
            timeout_error = httpx.TimeoutException("Stream timed out", request=mock_request)
            
            result = client._translate_httpx_error_to_api_error(timeout_error, mock_request, is_stream=True)
            
            from venice_ai.exceptions import APITimeoutError
            assert isinstance(result, APITimeoutError)
            assert "Stream request timed out" in str(result)

    def test_translate_httpx_connect_error(self):
        """Test _translate_httpx_error_to_api_error with ConnectError."""
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-api-key")
            
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "GET"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/models")
            
            connect_error = httpx.ConnectError("Connection failed", request=mock_request)
            
            result = client._translate_httpx_error_to_api_error(connect_error, mock_request)
            
            from venice_ai.exceptions import APIConnectionError
            assert isinstance(result, APIConnectionError)
            assert "Request failed" in str(result)
            assert result.request is mock_request
            assert result.original_error is connect_error

    def test_translate_httpx_connect_error_for_stream(self):
        """Test _translate_httpx_error_to_api_error with ConnectError for stream request."""
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-api-key")
            
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
            
            connect_error = httpx.ConnectError("Stream connection failed", request=mock_request)
            
            result = client._translate_httpx_error_to_api_error(connect_error, mock_request, is_stream=True)
            
            from venice_ai.exceptions import APIConnectionError
            assert isinstance(result, APIConnectionError)
            assert "Stream request failed" in str(result)

    def test_translate_httpx_generic_request_error(self):
        """Test _translate_httpx_error_to_api_error with generic RequestError."""
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-api-key")
            
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
            
            request_error = httpx.RequestError("Network error", request=mock_request)
            
            result = client._translate_httpx_error_to_api_error(request_error, mock_request)
            
            from venice_ai.exceptions import APIConnectionError
            assert isinstance(result, APIConnectionError)
            assert "Request failed" in str(result)
            assert result.request is mock_request
            assert result.original_error is request_error

    def test_translate_httpx_error_with_missing_request(self):
        """Test _translate_httpx_error_to_api_error when error.request is None."""
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-api-key")
            
            # Create a mock error where accessing .request raises RuntimeError
            mock_error = MagicMock(spec=httpx.TimeoutException)
            mock_error.request = None
            mock_error.args = ("Timeout occurred",)
            
            default_request = MagicMock(spec=httpx.Request)
            default_request.method = "GET"
            default_request.url = httpx.URL("https://api.venice.ai/api/v1/models")
            
            result = client._translate_httpx_error_to_api_error(mock_error, default_request)
            
            from venice_ai.exceptions import APITimeoutError
            assert isinstance(result, APITimeoutError)
            assert result.request is default_request

    def test_translate_httpx_error_non_json_response(self):
        """Test _translate_httpx_error_to_api_error with non-JSON response body."""
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-api-key")
            
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 400
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)
            mock_response.text = "Plain text error message"
            
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test")
            
            http_error = httpx.HTTPStatusError("Bad Request", request=mock_request, response=mock_response)
            
            result = client._translate_httpx_error_to_api_error(http_error, mock_request)
            
            from venice_ai.exceptions import InvalidRequestError
            assert isinstance(result, InvalidRequestError)
            assert result.status_code == 400
            # The error should contain information about the non-JSON response
            assert "Non-JSON response" in str(result) or "Plain text error message" in str(result)

    def test_translate_httpx_error_empty_response(self):
        """Test _translate_httpx_error_to_api_error with empty response body."""
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-api-key")
            
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 500
            mock_response.json.side_effect = json.JSONDecodeError("No JSON", "doc", 0)
            mock_response.text = ""
            
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "GET"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/models")
            
            http_error = httpx.HTTPStatusError("Internal Server Error", request=mock_request, response=mock_response)
            
            result = client._translate_httpx_error_to_api_error(http_error, mock_request)
            
            from venice_ai.exceptions import InternalServerError
            assert isinstance(result, InternalServerError)
            assert result.status_code == 500

    # A5: Comprehensive tests for context management and close() method
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

    def test_context_manager_with_request_operations(self):
        """Test context manager works correctly when performing request operations."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.json.return_value = {"models": ["model1", "model2"]}
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_httpx_client.return_value.request.return_value = mock_response
            
            with VeniceClient(api_key="test-api-key") as client:
                # Perform a request operation within the context
                result = client._request("GET", "models")
                assert result == {"models": ["model1", "model2"]}
                
            # Verify close was called when exiting context
            mock_httpx_client.return_value.close.assert_called_once()

    def test_context_manager_with_exception_during_request(self):
        """Test context manager properly closes client even when request raises exception."""
        with patch('httpx.Client') as mock_httpx_client:
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "GET"
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/models")
            timeout_error = httpx.TimeoutException("Request timed out", request=mock_request)
            mock_httpx_client.return_value.request.side_effect = timeout_error
            
            from venice_ai.exceptions import APITimeoutError
            with pytest.raises(APITimeoutError):
                with VeniceClient(api_key="test-api-key") as client:
                    client._request("GET", "models")
                    
            # Verify close was still called despite the exception
            mock_httpx_client.return_value.close.assert_called_once()

    def test_close_method_idempotent(self):
        """Test that calling close() multiple times is safe."""
        with patch('httpx.Client') as mock_httpx_client:
            client = VeniceClient(api_key="test-api-key")
            
            # Call close multiple times
            client.close()
            client.close()
            client.close()
            
            # Should only call the underlying close once due to idempotency
            assert mock_httpx_client.return_value.close.call_count == 1

    def test_close_method_after_context_manager(self):
        """Test that calling close() after using as context manager is safe."""
        with patch('httpx.Client') as mock_httpx_client:
            with VeniceClient(api_key="test-api-key") as client:
                pass  # Do nothing in context
            
            # Calling close again should be safe due to idempotency
            client.close()
            
            # Should have been called only once due to idempotency
            assert mock_httpx_client.return_value.close.call_count == 1

    def test_exit_method_returns_false(self):
        """Test that __exit__ method returns False to not suppress exceptions."""
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-api-key")
            
            # __exit__ should return False (or None) to not suppress exceptions
            result = client.__exit__(None, None, None)
            assert result is None or result is False
            
            # Test with exception info
            result = client.__exit__(ValueError, ValueError("test"), None)
            assert result is None or result is False

    def test_context_manager_nested_usage(self):
        """Test that nested context manager usage works correctly."""
        with patch('httpx.Client') as mock_httpx_client:
            with VeniceClient(api_key="test-api-key") as client1:
                with VeniceClient(api_key="test-api-key-2") as client2:
                    assert client1 is not client2
                    assert isinstance(client1, VeniceClient)
                    assert isinstance(client2, VeniceClient)
                    
            # Both clients should have been closed
            assert mock_httpx_client.return_value.close.call_count == 2

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


class TestBaseClient:
    """Test suite for BaseClient initialization and configuration."""
    
    @pytest.fixture
    def api_key(self):
        """Fixture for consistent API key across tests."""
        return "test-api-key"
    
    @pytest.fixture
    def base_url(self):
        """Fixture for consistent base URL across tests."""
        return "https://custom.api.com"
    
    def test_initialization_with_api_key(self, api_key):
        """Test BaseClient initialization with API key."""
        from venice_ai._client import BaseClient
        with patch('httpx.Client') as mock_httpx_client:
            client = BaseClient(api_key=api_key)
            
            # Verify basic properties
            assert client._api_key == api_key
            assert str(client._base_url) == "https://api.venice.ai/api/v1/"
            assert client._timeout == _constants.DEFAULT_TIMEOUT
            
            # Verify httpx.Client was not called during BaseClient init
            mock_httpx_client.assert_not_called()
    
    def test_initialization_with_environment_api_key(self):
        """Test BaseClient initialization with API key from environment."""
        from venice_ai._client import BaseClient
        with patch.dict('os.environ', {'VENICE_API_KEY': 'env-api-key'}):
            with patch('httpx.Client') as mock_httpx_client:
                client = BaseClient()
                
                assert client._api_key == "env-api-key"
                mock_httpx_client.assert_not_called()
    
    def test_initialization_without_api_key_raises_error(self):
        """Test BaseClient initialization fails without API key."""
        from venice_ai._client import BaseClient
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="The api_key client option must be set."):
                BaseClient(api_key="")
    
    def test_initialization_with_custom_base_url(self, api_key, base_url):
        """Test BaseClient initialization with custom base URL."""
        from venice_ai._client import BaseClient
        with patch('httpx.Client') as mock_httpx_client:
            client = BaseClient(api_key=api_key, base_url=base_url)
            
            assert str(client._base_url) == f"{base_url}/"
            mock_httpx_client.assert_not_called()
    
    def test_initialization_base_url_normalization(self, api_key):
        """Test BaseClient normalizes base URL by adding trailing slash."""
        from venice_ai._client import BaseClient
        with patch('httpx.Client') as mock_httpx_client:
            # Test URL without trailing slash
            client1 = BaseClient(api_key=api_key, base_url="https://api.example.com")
            assert str(client1._base_url) == "https://api.example.com/"
            
            # Test URL with trailing slash
            client2 = BaseClient(api_key=api_key, base_url="https://api.example.com/")
            assert str(client2._base_url) == "https://api.example.com/"
            
            mock_httpx_client.assert_not_called()
    
    def test_initialization_with_custom_timeout_float(self, api_key):
        """Test BaseClient initialization with custom timeout as float."""
        from venice_ai._client import BaseClient
        with patch('httpx.Client') as mock_httpx_client:
            client = BaseClient(api_key=api_key, timeout=30.0)
            
            assert isinstance(client._timeout, httpx.Timeout)
            assert client._timeout.read == 30.0
            mock_httpx_client.assert_not_called()
    
    def test_initialization_with_custom_timeout_httpx_timeout(self, api_key):
        """Test BaseClient initialization with custom httpx.Timeout object."""
        from venice_ai._client import BaseClient
        custom_timeout = httpx.Timeout(45.0, connect=10.0)
        with patch('httpx.Client') as mock_httpx_client:
            client = BaseClient(api_key=api_key, timeout=custom_timeout)
            
            assert client._timeout is custom_timeout
            assert client._timeout.read == 45.0
            assert client._timeout.connect == 10.0
            mock_httpx_client.assert_not_called()
    
    def test_initialization_with_none_timeout_uses_default(self, api_key):
        """Test BaseClient initialization with None timeout uses default."""
        from venice_ai._client import BaseClient
        with patch('httpx.Client') as mock_httpx_client:
            client = BaseClient(api_key=api_key, timeout=None)
            
            assert client._timeout == _constants.DEFAULT_TIMEOUT
            mock_httpx_client.assert_not_called()
    
    def test_initialization_with_default_timeout_parameter(self, api_key):
        """Test BaseClient initialization with default_timeout parameter takes precedence."""
        from venice_ai._client import BaseClient
        default_timeout = httpx.Timeout(25.0, connect=5.0)
        with patch('httpx.Client') as mock_httpx_client:
            client = BaseClient(api_key=api_key, timeout=30.0, default_timeout=default_timeout)
            
            # default_timeout should take precedence over timeout
            assert client._timeout is default_timeout
            assert client._timeout.read == 25.0
            assert client._timeout.connect == 5.0
            mock_httpx_client.assert_not_called()
    
    def test_initialization_with_http_transport_options(self, api_key):
        """Test BaseClient initialization with HTTP transport options."""
        from venice_ai._client import BaseClient
        transport_options = {"retries": 3, "backoff_factor": 0.5}
        
        with patch('httpx.Client') as mock_httpx_client:
            client = BaseClient(api_key=api_key, http_transport_options=transport_options)
            
            assert client._http_transport_options == transport_options
            mock_httpx_client.assert_not_called()
    
    def test_initialization_with_httpx_client_kwargs(self, api_key):
        """Test BaseClient initialization collects httpx client kwargs."""
        from venice_ai._client import BaseClient
        from venice_ai.utils import NOT_GIVEN
        
        with patch('httpx.Client') as mock_httpx_client:
            client = BaseClient(
                api_key=api_key,
                proxy="http://proxy.example.com:8080",
                limits=httpx.Limits(max_connections=100),
                verify=False,
                trust_env=True,
                http1=True,
                http2=False
            )
            
            # Verify httpx kwargs are stored
            assert client._proxy == "http://proxy.example.com:8080"
            assert isinstance(client._limits, httpx.Limits)
            assert client._verify is False
            assert client._trust_env is True
            assert client._http1 is True
            assert client._http2 is False
            
            # Verify NOT_GIVEN values are preserved
            assert client._transport is NOT_GIVEN
            assert client._cert is NOT_GIVEN
            
            mock_httpx_client.assert_not_called()
    
    def test_build_raw_client_with_default_transport(self, api_key):
        """Test _build_raw_client creates httpx.Client with default transport."""
        from venice_ai._client import BaseClient
        
        with patch('httpx.Client') as mock_httpx_client:
            mock_client_instance = MagicMock()
            mock_httpx_client.return_value = mock_client_instance
            
            client = BaseClient(api_key=api_key)
            raw_client = client._build_raw_client()
            
            # Verify httpx.Client was called with correct arguments
            mock_httpx_client.assert_called_once()
            call_kwargs = mock_httpx_client.call_args.kwargs
            
            assert "timeout" in call_kwargs
            assert call_kwargs["timeout"] == client._timeout
            assert "transport" in call_kwargs
            assert isinstance(call_kwargs["transport"], httpx.HTTPTransport)
            
            assert raw_client is mock_client_instance
    
    def test_build_raw_client_with_custom_transport(self, api_key):
        """Test _build_raw_client uses custom transport when provided."""
        from venice_ai._client import BaseClient
        custom_transport = MagicMock(spec=httpx.BaseTransport)
        
        with patch('httpx.Client') as mock_httpx_client:
            mock_client_instance = MagicMock()
            mock_httpx_client.return_value = mock_client_instance
            
            client = BaseClient(api_key=api_key, transport=custom_transport)
            raw_client = client._build_raw_client()
            
            # Verify custom transport is used
            call_kwargs = mock_httpx_client.call_args.kwargs
            assert call_kwargs["transport"] is custom_transport
            
            assert raw_client is mock_client_instance
    
    def test_build_raw_client_includes_httpx_kwargs(self, api_key):
        """Test _build_raw_client includes all httpx client kwargs."""
        from venice_ai._client import BaseClient
        
        with patch('httpx.Client') as mock_httpx_client:
            mock_client_instance = MagicMock()
            mock_httpx_client.return_value = mock_client_instance
            
            client = BaseClient(
                api_key=api_key,
                proxy="http://proxy.example.com:8080",
                limits=httpx.Limits(max_connections=50),
                verify=False,
                trust_env=True
            )
            raw_client = client._build_raw_client()
            
            # Verify all kwargs are passed
            call_kwargs = mock_httpx_client.call_args.kwargs
            assert call_kwargs["proxy"] == "http://proxy.example.com:8080"
            assert isinstance(call_kwargs["limits"], httpx.Limits)
            assert call_kwargs["verify"] is False
            assert call_kwargs["trust_env"] is True
            
            assert raw_client is mock_client_instance
    
    def test_build_async_raw_client_with_default_transport(self, api_key):
        """Test _build_async_raw_client creates httpx.AsyncClient with default transport."""
        from venice_ai._client import BaseClient
        
        with patch('httpx.AsyncClient') as mock_async_client:
            mock_client_instance = MagicMock()
            mock_async_client.return_value = mock_client_instance
            
            client = BaseClient(api_key=api_key)
            async_raw_client = client._build_async_raw_client()
            
            # Verify httpx.AsyncClient was called with correct arguments
            mock_async_client.assert_called_once()
            call_kwargs = mock_async_client.call_args.kwargs
            
            assert "timeout" in call_kwargs
            assert call_kwargs["timeout"] == client._timeout
            assert "transport" in call_kwargs
            assert isinstance(call_kwargs["transport"], httpx.AsyncHTTPTransport)
            
            assert async_raw_client is mock_client_instance
    
    def test_build_async_raw_client_with_custom_async_transport(self, api_key):
        """Test _build_async_raw_client uses custom async transport when provided."""
        from venice_ai._client import BaseClient
        custom_async_transport = MagicMock(spec=httpx.AsyncBaseTransport)
        
        with patch('httpx.AsyncClient') as mock_async_client:
            mock_client_instance = MagicMock()
            mock_async_client.return_value = mock_client_instance
            
            client = BaseClient(api_key=api_key, async_transport=custom_async_transport)
            async_raw_client = client._build_async_raw_client()
            
            # Verify custom async transport is used
            call_kwargs = mock_async_client.call_args.kwargs
            assert call_kwargs["transport"] is custom_async_transport
            
            assert async_raw_client is mock_client_instance
    
    def test_build_async_raw_client_includes_httpx_kwargs(self, api_key):
        """Test _build_async_raw_client includes all httpx client kwargs."""
        from venice_ai._client import BaseClient
        
        with patch('httpx.AsyncClient') as mock_async_client:
            mock_client_instance = MagicMock()
            mock_async_client.return_value = mock_client_instance
            
            client = BaseClient(
                api_key=api_key,
                proxy="http://proxy.example.com:8080",
                limits=httpx.Limits(max_connections=50),
                verify=False,
                trust_env=True
            )
            async_raw_client = client._build_async_raw_client()
            
            # Verify all kwargs are passed
            call_kwargs = mock_async_client.call_args.kwargs
            assert call_kwargs["proxy"] == "http://proxy.example.com:8080"
            assert isinstance(call_kwargs["limits"], httpx.Limits)
            assert call_kwargs["verify"] is False
            assert call_kwargs["trust_env"] is True
            
            assert async_raw_client is mock_client_instance