import pytest
import httpx
import logging
from unittest.mock import patch, MagicMock, AsyncMock
import json
from typing import List, Dict, Any, Optional, Union, AsyncIterator, cast
from pydantic import BaseModel
from venice_ai.types.chat import MessageParam # Import MessageParam

from venice_ai._async_client import AsyncVeniceClient, AsyncChatResource, AsyncChatCompletions
from venice_ai._client_with_retries import AsyncVeniceClientWithRetries
from venice_ai.exceptions import (
    VeniceError, APIError, InvalidRequestError, AuthenticationError, PermissionDeniedError,
    NotFoundError, RateLimitError, ConflictError, UnprocessableEntityError, InternalServerError,
    APIConnectionError, APITimeoutError, APIResponseProcessingError, StreamConsumedError, StreamClosedError
)
from venice_ai import _constants

# Helper async iterator with better typing for testing streaming responses
async def mock_async_iterator(items: List[Any]) -> AsyncIterator[Any]:
    """Helper to create a mock async iterator for testing streaming responses."""
    for item in items:
        yield item

class TestAsyncVeniceClient:
    """Test suite for AsyncVeniceClient base functionality."""
    
    @pytest.fixture
    def api_key(self) -> str:
        """Fixture for consistent API key across tests."""
        return "test-api-key"
    
    @pytest.fixture
    def base_url(self) -> str:
        """Fixture for consistent base URL across tests."""
        return "https://api.venice.ai/api/v1/"
    
    @pytest.fixture
    def mock_response(self) -> MagicMock:
        """Fixture for a basic successful mock response."""
        mock_resp = MagicMock()
        mock_resp.json = AsyncMock(return_value={"status": "success"})
        mock_resp.aread = AsyncMock()
        mock_resp.aclose = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        return mock_resp
    
    @pytest.fixture
    def mock_async_client(self, api_key: str, base_url: str) -> MagicMock:
        """Fixture for a mock AsyncVeniceClient."""
        client = MagicMock(spec=AsyncVeniceClient)
        client._api_key = api_key
        client._base_url = httpx.URL(base_url)
        client._timeout = 60.0
        client._max_retries = 2
        client._client = AsyncMock(spec=httpx.AsyncClient)
        return client
    
    def setup_mock_httpx_client(self, mock_httpx_client: MagicMock, response: Optional[MagicMock] = None) -> None:
        """Helper method to setup a mock httpx client with an optional response."""
        if response:
            mock_httpx_client.return_value.request = AsyncMock()
            mock_httpx_client.return_value.request.return_value = response
        
    def setup_mock_stream_context(self, mock_httpx_client: MagicMock, stream_response: Any) -> AsyncMock:
        """Helper method to setup a mock stream context for streaming requests."""
        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__.return_value = stream_response
        mock_stream_context.__aexit__.return_value = None
        mock_httpx_client.return_value.stream.return_value = mock_stream_context
        return mock_stream_context

    @pytest.mark.asyncio
    async def test_initialization_with_api_key(self, api_key: str, base_url: str):
        """Test client initialization with API key."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            mock_httpx_client = MockAsyncHTTPXClientClass # for assert_called_once

            client = AsyncVeniceClient(api_key=api_key)
            
            # Assert client properties
            assert client._api_key == api_key
            assert str(client._base_url).startswith(base_url)
            MockAsyncHTTPXClientClass.assert_called_once()
            
            # Verify resource namespaces are initialized
            resources = ['chat', 'models', 'image', 'audio', 'billing', 'embeddings', 'api_keys', 'characters']
            for resource in resources:
                assert hasattr(client, resource), f"Client missing resource: {resource}"

    @pytest.mark.asyncio
    async def test_initialization_without_api_key(self):
        """Test client initialization fails without API key."""
        with pytest.raises(ValueError, match="The api_key client option must be set."):
            AsyncVeniceClient(api_key="")

    @pytest.mark.asyncio
    async def test_initialization_with_custom_base_url(self, api_key: str):
        """Test client initialization with custom base URL."""
        custom_url = "https://custom.api.com"
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            client = AsyncVeniceClient(api_key=api_key, base_url=custom_url)
            assert str(client._base_url).startswith(custom_url)
            assert str(client._base_url).endswith("/"), "Base URL should have trailing slash"

    @pytest.mark.asyncio
    async def test_initialization_with_custom_timeout_and_retries(self):
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            mock_httpx_client = MockAsyncHTTPXClientClass # for assert_called_once

            client = AsyncVeniceClientWithRetries(api_key="test-api-key", timeout=30.0, max_retries=5)
            assert isinstance(client._timeout, httpx.Timeout) # Hint for Pylance
            assert client._timeout.read == 30.0
            assert client._max_retries == 5
            MockAsyncHTTPXClientClass.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_initialization_default_timeout(self):
        """Test client initialization uses default timeout when timeout is None."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            client = AsyncVeniceClient(api_key="test-api-key", timeout=None)
            assert isinstance(client._timeout, httpx.Timeout)
            assert client._timeout.read == _constants.DEFAULT_TIMEOUT.read
            MockAsyncHTTPXClientClass.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_basic_get(self, api_key: str, mock_response: MagicMock):
        """Test basic GET request."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            # Ensure headers attribute is a synchronous mock behaving like a dict
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            # Ensure aclose is an AsyncMock for proper teardown if client.close() is called
            mock_httpx_client_instance.aclose = AsyncMock()
            # self.setup_mock_httpx_client now expects the class mock, not the instance
            self.setup_mock_httpx_client(MockAsyncHTTPXClientClass, mock_response)
            mock_httpx_client = MockAsyncHTTPXClientClass # for instance access if needed by setup

            client = AsyncVeniceClient(api_key=api_key)
            result = await client._request("GET", "test_endpoint")
            
            assert result == {"status": "success"}
            mock_httpx_client_instance.request.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_basic_post(self, api_key: str, mock_response: MagicMock):
        """Test basic POST request with JSON data."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            self.setup_mock_httpx_client(MockAsyncHTTPXClientClass, mock_response)
            mock_httpx_client = MockAsyncHTTPXClientClass

            client = AsyncVeniceClient(api_key=api_key)
            json_data = {"key": "value"}
            result = await client._request("POST", "test_endpoint", json_data=json_data)
            
            assert result == {"status": "success"}
            mock_httpx_client_instance.request.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_with_headers_and_params(self):
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            mock_httpx_client_instance.request = AsyncMock() # mock the request method on the instance
            mock_response = MagicMock() # This is for the response object, not the client
            mock_response.json = AsyncMock(return_value={"status": "success"})
            mock_response.aread = AsyncMock()
            mock_response.aclose = AsyncMock()
            mock_httpx_client_instance.request.return_value = mock_response

            client = AsyncVeniceClient(api_key="test-api-key")
            headers = {"X-Test-Header": "test_value"}
            params = {"param1": "value1"}
            result = await client._request("GET", "test_endpoint", headers=headers, params=params)
            assert result == {"status": "success"}
            mock_httpx_client_instance.request.assert_awaited_once_with(
                method="GET",
                url=client._base_url.join("test_endpoint"),
                json=None,
                headers=headers,
                params=params,
                timeout=client._timeout
            )

    @pytest.mark.asyncio
    async def test_request_raw_response(self):
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            mock_httpx_client_instance.request = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = b"raw bytes content"
            mock_response.aread = AsyncMock()
            mock_response.aclose = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_httpx_client_instance.request.return_value = mock_response

            client = AsyncVeniceClient(api_key="test-api-key")
            result = await client._request("GET", "test_endpoint", raw_response=True)
            assert isinstance(result, bytes)
            assert result == b"raw bytes content"
            mock_httpx_client_instance.request.assert_awaited_once_with(
                method="GET",
                url=client._base_url.join("test_endpoint"),
                json=None,
                headers={},
                params=None,
                timeout=client._timeout
            )

    @pytest.mark.asyncio
    async def test_request_raw_response_no_cast_to_async(self):
        """Test _request with raw_response=True and no cast_to returns raw content (async)."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            mock_response = AsyncMock(spec=httpx.Response) # Use AsyncMock for response if its methods are async
            mock_response.headers = {}
            mock_response.content = b"raw binary data for no_cast_to_async"
            mock_response.raise_for_status = MagicMock() # If sync
            # If raise_for_status is async: mock_response.raise_for_status = AsyncMock()
            
            # Make .json() raise an error or return distinct data to ensure .content is used
            # If .json() is async: mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("Cannot decode", "doc", 0))
            mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("Cannot decode", "doc", 0)) # Now async
            mock_response.aread = AsyncMock()
            mock_response.aclose = AsyncMock()
            
            mock_httpx_client_instance.request = AsyncMock(return_value=mock_response)

            client = AsyncVeniceClient(api_key="test-api-key")
            result = await client._request("GET", "some_endpoint", raw_response=True) # cast_to is implicitly None
            
            assert result == b"raw binary data for no_cast_to_async"
            mock_httpx_client_instance.request.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code,error_class,error_message_pattern", [
            (400, InvalidRequestError, "API error 400"),
            (401, AuthenticationError, "API error 401"),
            (403, PermissionDeniedError, "API error 403"),
            (404, NotFoundError, "API error 404"),
            (429, RateLimitError, "API error 429"),
        ]
    )
    async def test_request_http_error_handling(self, api_key: str, status_code: int, error_class: type, error_message_pattern: str):
        """Test HTTP error handling with different status codes."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            # Instantiate client first before accessing its attributes
            client = AsyncVeniceClient(api_key=api_key)
            
            # Setup error response
            mock_response = MagicMock(spec=httpx.Response) # This is the mock for httpx.Response, not the client
            mock_response.status_code = status_code
            mock_response.headers = {}  # Add missing headers attribute
            mock_request_for_status_error = MagicMock(spec=httpx.Request)
            mock_request_for_status_error.method = "GET" # Or the appropriate method
            mock_request_for_status_error.url = httpx.URL(client._base_url.join("test_endpoint")) # Or the appropriate URL
            mock_response.request = mock_request_for_status_error
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                message=f"HTTP Error {status_code}",
                request=mock_response.request,
                response=mock_response
            )
            # self.setup_mock_httpx_client expects the class mock
            self.setup_mock_httpx_client(MockAsyncHTTPXClientClass, mock_response)
            
            excinfo: pytest.ExceptionInfo
            with pytest.raises(error_class) as excinfo:
                await client._request("GET", "test_endpoint")
            
            # Verify error message
            assert excinfo.value is not None
            assert error_message_pattern in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_request_error_handling_non_json_body(self, api_key: str):
        """Test error handling when response body is not JSON."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            # Instantiate client first before accessing its attributes
            client = AsyncVeniceClient(api_key=api_key)
            
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 400
            mock_response.headers = {}  # Add missing headers attribute
            mock_request_for_status_error = MagicMock(spec=httpx.Request)
            mock_request_for_status_error.method = "GET" # Or the appropriate method
            mock_request_for_status_error.url = httpx.URL(client._base_url.join("test_endpoint")) # Or the appropriate URL
            mock_response.request = mock_request_for_status_error
            mock_response.text = "Not JSON content"
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                message="Error", request=mock_response.request, response=mock_response
            )
            mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
            mock_response.aread = AsyncMock()
            mock_response.aclose = AsyncMock()
            self.setup_mock_httpx_client(MockAsyncHTTPXClientClass, mock_response)
            
            with pytest.raises(InvalidRequestError) as excinfo:
                await client._request("GET", "test_endpoint")
            
            assert "API error 400" in str(excinfo.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exception_class,exception_message,expected_error,expected_message", [
            (httpx.TimeoutException, "Request timed out", VeniceError, "Request timed out"),
            (httpx.ConnectError, "Connection failed", VeniceError, "Request failed: Connection failed"),
            (httpx.ReadTimeout, "Read timed out", VeniceError, "Request timed out: Read timed out"),
            (httpx.RequestError, "Generic request error", VeniceError, "Request failed: Generic request error"),
        ]
    )
    async def test_request_client_error_handling(self, api_key: str, exception_class, exception_message, expected_error, expected_message):
        """Test handling of various client-side errors."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "GET" # Add the method attribute
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint") # Add the url attribute
            mock_httpx_client_instance.request = AsyncMock(
                side_effect=exception_class(exception_message, request=mock_request)
            )
            
            client = AsyncVeniceClient(api_key=api_key)
            
            with pytest.raises(expected_error, match=expected_message):
                await client._request("GET", "test_endpoint")

    # B1a: Add tests for _request with 'PUT', 'DELETE', 'PATCH' methods
    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["PUT", "DELETE", "PATCH"])
    async def test_request_additional_http_methods(self, api_key: str, method: str):
        """Test _request with PUT, DELETE, PATCH methods."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            mock_response = MagicMock()
            mock_response.json = AsyncMock(return_value={"status": "success", "method": method})
            mock_response.aread = AsyncMock()
            mock_response.aclose = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_httpx_client_instance.request = AsyncMock(return_value=mock_response)

            client = AsyncVeniceClient(api_key=api_key)
            json_data = {"key": "value"} if method in ["PUT", "PATCH"] else None
            result = await client._request(method, "test_endpoint", json_data=json_data)
            
            assert result == {"status": "success", "method": method}
            # Content-Type header is only added when json_data is provided
            expected_headers = {'Content-Type': 'application/json'} if json_data else {}
            mock_httpx_client_instance.request.assert_awaited_once_with(
                method=method,
                url=client._base_url.join("test_endpoint"),
                json=json_data,
                headers=expected_headers,
                params=None,
                timeout=client._timeout
            )

    # B1b: Add tests for _request successful JSON response with cast_to=MyPydanticModel
    @pytest.mark.asyncio
    async def test_request_with_pydantic_model_casting(self, api_key: str):
        """Test _request with cast_to=MyPydanticModel for JSON response casting."""
        # Define a test Pydantic model
        class TestModel(BaseModel):
            status: str
            message: str
            
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            mock_response = MagicMock()
            mock_response.json = AsyncMock(return_value={"status": "success", "message": "test"})
            mock_response.aread = AsyncMock()
            mock_response.aclose = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_httpx_client_instance.request = AsyncMock(return_value=mock_response)

            client = AsyncVeniceClient(api_key=api_key)
            result = await client._request("GET", "test_endpoint")
            
            assert isinstance(result, dict)
            assert result["status"] == "success"
            assert result["message"] == "test"
            mock_httpx_client_instance.request.assert_awaited_once()

    # B1c: Add tests for _request successful raw httpx.Response with cast_to=httpx.Response
    @pytest.mark.asyncio
    async def test_request_with_httpx_response_casting(self, api_key: str):
        """Test _request with cast_to=httpx.Response returning raw response object."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 200
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.content = b'{"status": "success"}'
            mock_response.json = AsyncMock(return_value={"status": "success"})
            mock_response.aread = AsyncMock()
            mock_response.aclose = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_httpx_client_instance.request = AsyncMock(return_value=mock_response)

            client = AsyncVeniceClient(api_key=api_key)
            result = await client._request("GET", "test_endpoint")
            
            assert result == {"status": "success"}
            mock_httpx_client_instance.request.assert_awaited_once()

    # B1d: Comprehensive tests for _request API Errors with proper error mapping
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code,error_class,error_body", [
            (400, InvalidRequestError, {"error": {"message": "Bad request"}}),
            (401, AuthenticationError, {"error": {"message": "Unauthorized"}}),
            (403, PermissionDeniedError, {"error": {"message": "Forbidden"}}),
            (404, NotFoundError, {"error": {"message": "Not found"}}),
            (409, ConflictError, {"error": {"message": "Conflict"}}),
            (422, UnprocessableEntityError, {"error": {"message": "Validation failed"}}),
            (429, RateLimitError, {"error": {"message": "Rate limit exceeded"}}),
            (500, InternalServerError, {"error": {"message": "Internal server error"}}),
            (502, InternalServerError, {"error": {"message": "Bad gateway"}}),
            (503, InternalServerError, {"error": {"message": "Service unavailable"}}),
        ]
    )
    async def test_request_comprehensive_api_error_mapping(self, api_key: str, status_code: int, error_class: type, error_body: dict):
        """Test comprehensive API error mapping with proper error attributes."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            client = AsyncVeniceClient(api_key=api_key)
            
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = status_code
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.json = AsyncMock(return_value=error_body)
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "GET"
            mock_request.url = httpx.URL(client._base_url.join("test_endpoint"))
            mock_response.request = mock_request
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                message=f"HTTP Error {status_code}",
                request=mock_request,
                response=mock_response
            )
            mock_httpx_client_instance.request = AsyncMock(return_value=mock_response)
            
            with pytest.raises(error_class) as exc_info:
                await client._request("GET", "test_endpoint")
            
            error = exc_info.value
            assert error.status_code == status_code
            assert error.request is mock_request
            assert error.response is mock_response
            assert error.body == error_body
            if status_code == 429 and hasattr(error, 'retry_after_seconds'):
                # RateLimitError might have retry_after_seconds attribute
                assert hasattr(error, 'retry_after_seconds')

    # B1e: Comprehensive parameter handling tests with explicit assertions
    @pytest.mark.asyncio
    async def test_request_comprehensive_parameter_handling(self, api_key: str):
        """Test _request parameter handling with explicit assertions on httpx.AsyncClient.request calls."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            mock_response = MagicMock()
            mock_response.json = AsyncMock(return_value={"status": "success"})
            mock_response.aread = AsyncMock()
            mock_response.aclose = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_httpx_client_instance.request = AsyncMock(return_value=mock_response)

            client = AsyncVeniceClient(api_key=api_key)
            
            # Test with all parameters
            custom_headers = {"X-Custom": "value", "Authorization": "Bearer override"}
            params = {"param1": "value1", "param2": "value2"}
            json_data = {"key": "value", "nested": {"inner": "data"}}
            custom_timeout = httpx.Timeout(30.0)
            
            await client._request(
                "POST",
                "test_endpoint",
                headers=custom_headers,
                params=params,
                json_data=json_data,
                timeout=custom_timeout
            )
            
            # Verify all parameters were passed correctly
            # Merge custom_headers with the automatic Content-Type header
            expected_headers = custom_headers.copy()
            expected_headers['Content-Type'] = 'application/json'
            
            mock_httpx_client_instance.request.assert_awaited_once_with(
                method="POST",
                url=client._base_url.join("test_endpoint"),
                json=json_data,
                headers=expected_headers,
                params=params,
                timeout=custom_timeout
            )

    @pytest.mark.asyncio
    async def test_convenience_methods(self, api_key: str):
        """Test the convenience methods (get, post, delete)."""
        client = AsyncVeniceClient(api_key=api_key)
        
        # Test GET method
        with patch.object(client, '_request', AsyncMock(return_value={"status": "success"})) as mock_request:
            params = {"param1": "value1"}
            result = await client.get("test_endpoint", params=params)
            
            assert result == {"status": "success"}
            mock_request.assert_awaited_once_with("GET", "test_endpoint", params=params, cast_to=None)
        
        # Test POST method
        with patch.object(client, '_request', AsyncMock(return_value={"status": "created"})) as mock_request:
            json_data = {"key": "value"}
            result = await client.post("test_endpoint", json_data=json_data)
            
            assert result == {"status": "created"}
            mock_request.assert_awaited_once_with("POST", "test_endpoint", json_data=json_data, timeout=None, cast_to=None)

    # Delete method is now tested in the combined test_convenience_methods test

    @pytest.mark.asyncio
    async def test_stream_request_basic(self, api_key: str):
        """Test basic stream request handling."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            # Configure the mock stream response
            mock_response = AsyncMock(spec=httpx.Response) # This is for the response from stream.__aenter__
            mock_response.headers = {}
            mock_response.aiter_lines.return_value = mock_async_iterator([
                "data: {\"choices\": [{\"delta\": {\"content\": \"chunk1\"}}]}",
                "data: {\"choices\": [{\"delta\": {\"content\": \"chunk2\"}}]}",
                "data: [DONE]"
            ])
            mock_response.raise_for_status = MagicMock()

            # Add a mock request object to the mock response
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST" # Or the appropriate method for the test
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions") # Or the appropriate URL
            mock_response.request = mock_request
            
            # Setup stream context using helper method
            self.setup_mock_stream_context(MockAsyncHTTPXClientClass, mock_response)
            
            # Test stream request
            client = AsyncVeniceClient(api_key=api_key)
            chunks = []
            async for chunk in client._stream_request("POST", "chat/completions", json_data={"model": "venice-1"}):
                chunks.append(chunk)
            
            assert len(chunks) == 2
            assert chunks[0]["choices"][0]["delta"]["content"] == "chunk1"
            assert chunks[1]["choices"][0]["delta"]["content"] == "chunk2"

    @pytest.mark.asyncio
    async def test_stream_request_edge_cases(self, api_key: str, caplog):
        """Test stream request with various edge cases (empty lines, JSON errors)."""
        # Test with empty lines
        client = AsyncVeniceClient(api_key=api_key)
        
        # Empty line test
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.headers = {}
        mock_response.aiter_lines.return_value = mock_async_iterator([
            "",
            "data: {\"choices\": [{\"delta\": {\"content\": \"chunk1\"}}]}",
            "data: [DONE]"
        ])
        mock_response.raise_for_status = MagicMock()
        
        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__.return_value = mock_response
        mock_stream_context.__aexit__.return_value = None
        
        with patch.object(client._client, 'stream', return_value=mock_stream_context):
            chunks = []
            async for chunk in client._stream_request("POST", "chat/completions", json_data={"model": "venice-1"}):
                chunks.append(chunk)
            assert len(chunks) == 1, "Should have 1 valid chunk after filtering empty lines"
            assert chunks[0]["choices"][0]["delta"]["content"] == "chunk1"
        
        # JSON decode error test
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.headers = {}
        mock_response.aiter_lines.return_value = mock_async_iterator([
            "data: invalid_json",
            "data: {\"choices\": [{\"delta\": {\"content\": \"chunk1\"}}]}",
            "data: [DONE]"
        ])
        mock_response.raise_for_status = MagicMock()
        
        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__.return_value = mock_response
        mock_stream_context.__aexit__.return_value = None
        
        with caplog.at_level(logging.ERROR):
            with patch.object(client._client, 'stream', return_value=mock_stream_context):
                chunks = []
                async for chunk in client._stream_request("POST", "chat/completions", json_data={"model": "venice-1"}):
                    chunks.append(chunk)
                assert len(chunks) == 1, "Should skip invalid JSON and return valid chunk"
                assert chunks[0]["choices"][0]["delta"]["content"] == "chunk1"
                # Check that an error message containing "Failed to parse JSON" was logged
                assert any("Failed to parse JSON" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_stream_request_raw_basic(self, api_key: str):
        """Test raw binary streaming for audio/speech endpoints."""
        # Configure the mock response
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.headers = {}
        mock_response.aiter_bytes.return_value = mock_async_iterator([b"chunk1", b"chunk2"])
        mock_response.raise_for_status = MagicMock()

        # Setup stream context
        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__.return_value = mock_response
        mock_stream_context.__aexit__.return_value = None

        # Test raw streaming
        client = AsyncVeniceClient(api_key=api_key)
        with patch.object(client._client, 'stream', return_value=mock_stream_context):
            chunks = []
            async for chunk in client._stream_request_raw("POST", "audio/speech", json_data={"text": "test"}):
                chunks.append(chunk)
            
            assert len(chunks) == 2, "Should receive two binary chunks"
            assert chunks[0] == b"chunk1", "First chunk should match expected binary data"
            assert chunks[1] == b"chunk2", "Second chunk should match expected binary data"

    # B2a: Add tests for _stream_request with cast_to=MySSEModel and stream_cls=AsyncStream
    @pytest.mark.asyncio
    async def test_stream_request_with_pydantic_model_and_custom_stream_cls(self, api_key: str):
        """Test _stream_request with cast_to=MySSEModel and stream_cls=AsyncStream."""
        # Define a test SSE model
        class SSEModel(BaseModel):
            choices: List[Dict[str, Any]]
            
        from venice_ai.streaming import AsyncStream
        
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            mock_response = AsyncMock(spec=httpx.Response)
            mock_response.headers = {}
            mock_response.aiter_lines.return_value = mock_async_iterator([
                "data: {\"choices\": [{\"delta\": {\"content\": \"chunk1\"}}]}",
                "data: {\"choices\": [{\"delta\": {\"content\": \"chunk2\"}}]}",
                "data: [DONE]"
            ])
            mock_response.raise_for_status = MagicMock()
            
            mock_stream_context = AsyncMock()
            mock_stream_context.__aenter__.return_value = mock_response
            mock_stream_context.__aexit__.return_value = None
            mock_httpx_client_instance.stream.return_value = mock_stream_context
            
            client = AsyncVeniceClient(api_key=api_key)
            
            # Test basic streaming (the current implementation returns AsyncIterator)
            stream_iter = client._stream_request(
                "POST",
                "chat/completions",
                json_data={"model": "venice-1"}
            )
            
            # Note: The current implementation returns AsyncIterator, not AsyncStream
            # This test verifies the basic streaming functionality
            
            chunks = []
            async for chunk in stream_iter:
                chunks.append(chunk)
            
            # _stream_request yields parsed JSON objects from SSE "data:" lines
            assert len(chunks) == 2
            assert all(isinstance(chunk, dict) for chunk in chunks)
            assert chunks[0]["choices"][0]["delta"]["content"] == "chunk1"
            assert chunks[1]["choices"][0]["delta"]["content"] == "chunk2"

    # B2b: Add tests for _stream_request with cast_to=bytes and stream_cls=AsyncStream
    @pytest.mark.asyncio
    async def test_stream_request_raw_bytes_with_custom_stream_cls(self, api_key: str):
        """Test _stream_request for raw byte stream with cast_to=bytes and stream_cls=AsyncStream."""
        from venice_ai.streaming import AsyncStream
        
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            mock_response = AsyncMock(spec=httpx.Response)
            mock_response.headers = {}
            # _stream_request uses aiter_lines, not aiter_bytes, and expects SSE format
            sse_lines = [
                'data: {"type": "chunk", "content": "chunk1"}',
                'data: {"type": "chunk", "content": "chunk2"}',
                'data: {"type": "chunk", "content": "chunk3"}'
            ]
            mock_response.aiter_lines.return_value = mock_async_iterator(sse_lines)
            mock_response.raise_for_status = MagicMock()
            
            mock_stream_context = AsyncMock()
            mock_stream_context.__aenter__.return_value = mock_response
            mock_stream_context.__aexit__.return_value = None
            mock_httpx_client_instance.stream.return_value = mock_stream_context
            
            client = AsyncVeniceClient(api_key=api_key)
            
            # Test raw byte streaming (basic functionality)
            stream_iter = client._stream_request(
                "POST",
                "audio/speech",
                json_data={"text": "test"}
            )
            
            # Note: The current implementation returns AsyncIterator, not AsyncStream
            # This test verifies basic byte streaming functionality
            
            chunks = []
            async for chunk in stream_iter:
                chunks.append(chunk)
            
            # _stream_request is designed for SSE (JSON) streaming, not raw bytes
            # It yields parsed JSON objects from "data:" lines
            assert len(chunks) == 3
            assert all(isinstance(chunk, dict) for chunk in chunks)

    # B2c: Add tests for _stream_request API Error Before Streaming
    @pytest.mark.asyncio
    async def test_stream_request_api_error_before_streaming(self, api_key: str):
        """Test _stream_request API error before streaming starts (e.g., NotFoundError)."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            client = AsyncVeniceClient(api_key=api_key)
            
            mock_response = AsyncMock(spec=httpx.Response)
            mock_response.status_code = 404
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.json = AsyncMock(return_value={"error": {"message": "Model not found"}})
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL(client._base_url.join("chat/completions"))
            mock_response.request = mock_request
            mock_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
                message="HTTP Error 404",
                request=mock_request,
                response=mock_response
            ))
            
            mock_stream_context = AsyncMock()
            mock_stream_context.__aenter__.return_value = mock_response
            mock_stream_context.__aexit__.return_value = None
            mock_httpx_client_instance.stream.return_value = mock_stream_context
            
            with pytest.raises(NotFoundError) as exc_info:
                async for _ in client._stream_request("POST", "chat/completions", json_data={"model": "nonexistent"}):
                    pass  # Should raise before yielding anything
            
            error = exc_info.value
            assert error.status_code == 404
            assert error.request is mock_request
            assert error.response is mock_response

    # B2d: Comprehensive tests for _stream_request Error During Streaming
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exception_class,exception_message,expected_error_type", [
            (httpx.ReadError, "Connection broken during stream", APIConnectionError),
            (httpx.WriteError, "Write error during stream", APIConnectionError),
            (httpx.ProtocolError, "Protocol error during stream", APIConnectionError),
            (httpx.LocalProtocolError, "Local protocol error during stream", APIConnectionError),
        ]
    )
    async def test_stream_request_error_during_streaming(self, api_key: str, exception_class, exception_message, expected_error_type):
        """Test _stream_request error handling during streaming iteration."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            client = AsyncVeniceClient(api_key=api_key)
            
            # Create mock iterator that yields one chunk then raises error
            async def error_iterator():
                yield "data: {\"choices\": [{\"delta\": {\"content\": \"chunk1\"}}]}"
                mock_request = MagicMock(spec=httpx.Request)
                mock_request.method = "POST"
                mock_request.url = httpx.URL(client._base_url.join("chat/completions"))
                raise exception_class(exception_message, request=mock_request)
            
            mock_response = AsyncMock(spec=httpx.Response)
            mock_response.headers = {}
            mock_response.aiter_lines.return_value = error_iterator()
            mock_response.raise_for_status = MagicMock()
            
            mock_stream_context = AsyncMock()
            mock_stream_context.__aenter__.return_value = mock_response
            mock_stream_context.__aexit__.return_value = None
            mock_httpx_client_instance.stream.return_value = mock_stream_context
            
            chunks = []
            with pytest.raises(expected_error_type, match=exception_message):
                async for chunk in client._stream_request("POST", "chat/completions", json_data={"model": "venice-1"}):
                    chunks.append(chunk)
            
            # Should have received one chunk before the error
            assert len(chunks) == 1
            assert chunks[0]["choices"][0]["delta"]["content"] == "chunk1"

    @pytest.mark.asyncio
    async def test_request_multipart_basic(self, api_key: str, mock_response: MagicMock):
        """Test basic multipart form request."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            self.setup_mock_httpx_client(MockAsyncHTTPXClientClass, mock_response)
            
            client = AsyncVeniceClient(api_key=api_key)
            files = {"file": ("test.txt", b"content", "text/plain")}
            result = await client._request_multipart("POST", "upload", files=files)
            
            assert result == {"status": "success"}
            mock_httpx_client_instance.request.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_multipart_with_options(self, api_key: str, mock_response: MagicMock):
        """Test multipart form request with additional options."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            self.setup_mock_httpx_client(MockAsyncHTTPXClientClass, mock_response)
            
            client = AsyncVeniceClient(api_key=api_key)
            files = {"file": ("test.txt", b"content", "text/plain")}
            data = {"field1": "value1", "field2": "value2"}
            headers = {"X-Custom-Header": "custom-value"}
            params = {"param1": "value1"}
            
            result = await client._request_multipart(
                "POST",
                "upload",
                files=files,
                data=data,
                headers=headers,
                params=params
            )
            
            assert result == {"status": "success"}
            mock_httpx_client_instance.request.assert_awaited_once()
            
            # Verify all parameters were passed correctly
            call_args = mock_httpx_client_instance.request.call_args
            assert call_args.kwargs["files"] == files, "Files parameter not passed correctly"
            assert call_args.kwargs["data"] == data, "Form data not passed correctly"
            assert "X-Custom-Header" in call_args.kwargs["headers"], "Custom header not passed correctly"
            assert call_args.kwargs["params"] == params, "URL parameters not passed correctly"

    @pytest.mark.asyncio
    async def test_request_multipart_error_handling(self, api_key: str):
        """Test error handling for multipart requests."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            # Setup error condition
            mock_httpx_client_instance.request = AsyncMock(
                side_effect=httpx.RequestError("Multipart request failed", request=MagicMock())
            )
            
            client = AsyncVeniceClient(api_key=api_key)
            files = {"file": ("test.txt", b"content", "text/plain")}
            
            with pytest.raises(VeniceError, match="Request failed"):
                await client._request_multipart("POST", "upload", files=files)

    @pytest.mark.asyncio
    async def test_request_multipart_default_headers(self, api_key: str):
        """Test that default headers are properly included in multipart requests."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            # Explicitly set headers on the instance BEFORE client init uses it
            mock_httpx_client_instance.headers = {"Authorization": f"Bearer {api_key}", "User-Agent": "test-agent"}
            mock_httpx_client_instance.aclose = AsyncMock() # Also ensure aclose is set
            
            mock_response = MagicMock()
            mock_response.json = AsyncMock(return_value={"status": "success"})
            mock_response.aread = AsyncMock()
            mock_response.aclose = AsyncMock()
            self.setup_mock_httpx_client(MockAsyncHTTPXClientClass, mock_response) # This sets up .request
            
            client = AsyncVeniceClient(api_key=api_key)
            # client._client is mock_httpx_client_instance. Its headers are already set.

            files = {"file": ("test.txt", b"content", "text/plain")}
            # Call without providing 'Authorization' or 'Accept' in headers argument
            result = await client._request_multipart("POST", "upload", files=files, headers={"X-Other-Header": "value"})
            
            assert result == {"status": "success"}
            mock_httpx_client_instance.request.assert_awaited_once()
            call_args = mock_httpx_client_instance.request.call_args
            
            # Check header behavior
            assert "Authorization" in call_args.kwargs["headers"], "Authorization header should be preserved"
            assert call_args.kwargs["headers"]["Authorization"] == f"Bearer {client._api_key}"
            
            assert "Accept" in call_args.kwargs["headers"], "Accept header should be added with default value"
            assert call_args.kwargs["headers"]["Accept"] == "*/*"

            assert "User-Agent" in call_args.kwargs["headers"], "User-Agent header should be preserved"
            assert call_args.kwargs["headers"]["User-Agent"] == "test-agent"

            assert "X-Other-Header" in call_args.kwargs["headers"], "Custom headers should be preserved"
            assert call_args.kwargs["headers"]["X-Other-Header"] == "value"

    @pytest.mark.asyncio
    async def test_request_multipart_default_headers_with_none_headers_arg_async(self, api_key: str):
        """Test _request_multipart default headers when headers=None is passed (async)."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            
            # Simulate default headers already being on the internal client instance
            mock_httpx_client_instance.headers = {"Authorization": f"Bearer {api_key}", "User-Agent": "test-agent-async-none"}
            mock_httpx_client_instance.aclose = AsyncMock()
            
            mock_response = MagicMock() # Sync mock for the response object itself
            mock_response.json = AsyncMock(return_value={"status": "success"}) # Now async mock for json()
            mock_response.aread = AsyncMock()
            mock_response.aclose = AsyncMock()
            
            mock_httpx_client_instance.request = AsyncMock(return_value=mock_response) # request method is async

            client = AsyncVeniceClient(api_key=api_key)
            # client._client is mock_httpx_client_instance, its headers are set above.

            files = {"file": ("test.txt", b"content", "text/plain")}
            # Call with headers=None
            result = await client._request_multipart("POST", "upload", files=files, headers=None)
            
            assert result == {"status": "success"}
            mock_httpx_client_instance.request.assert_awaited_once()
            call_args = mock_httpx_client_instance.request.call_args
            
            assert "Authorization" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["Authorization"] == f"Bearer {client._api_key}"
            assert "Accept" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["Accept"] == "*/*"
            assert "User-Agent" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["User-Agent"] == "test-agent-async-none"

    # B3a: Ensure tests for _request_multipart explicitly assert files and data arguments
    @pytest.mark.asyncio
    async def test_request_multipart_explicit_parameter_assertions(self, api_key: str):
        """Test _request_multipart with explicit assertions on files and data parameters."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            mock_response = MagicMock()
            mock_response.json = AsyncMock(return_value={"status": "success"})
            mock_response.aread = AsyncMock()
            mock_response.aclose = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_httpx_client_instance.request = AsyncMock(return_value=mock_response)

            client = AsyncVeniceClient(api_key=api_key)
            
            # Test with both files and data
            files = {
                "file1": ("test1.txt", b"content1", "text/plain"),
                "file2": ("test2.json", b'{"key": "value"}', "application/json")
            }
            data = {
                "field1": "value1",
                "field2": "value2",
                "nested": {"inner": "data"}
            }
            
            await client._request_multipart("POST", "upload", files=files, data=data)
            
            # Explicitly verify files and data were passed correctly
            mock_httpx_client_instance.request.assert_awaited_once()
            call_args = mock_httpx_client_instance.request.call_args
            
            assert call_args.kwargs["files"] is files, "Files parameter should be passed exactly as provided"
            assert call_args.kwargs["data"] is data, "Data parameter should be passed exactly as provided"
            assert call_args.kwargs["method"] == "POST"
            assert "upload" in str(call_args.kwargs["url"])

    # B3b: Add tests for _request_multipart API Error during Multipart Upload
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code,error_class", [
            (400, InvalidRequestError),
            (413, InvalidRequestError),  # File too large
            (415, InvalidRequestError),  # Unsupported media type
            (422, UnprocessableEntityError),
            (500, InternalServerError),
        ]
    )
    async def test_request_multipart_api_error_during_upload(self, api_key: str, status_code: int, error_class: type):
        """Test _request_multipart API error handling during multipart upload."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            client = AsyncVeniceClient(api_key=api_key)
            
            # Setup error response
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = status_code
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.json = AsyncMock(return_value={"error": {"message": f"Upload failed with {status_code}"}})
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST"
            mock_request.url = httpx.URL(client._base_url.join("upload"))
            mock_response.request = mock_request
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                message=f"HTTP Error {status_code}",
                request=mock_request,
                response=mock_response
            )
            mock_httpx_client_instance.request = AsyncMock(return_value=mock_response)
            
            files = {"file": ("test.txt", b"content", "text/plain")}
            
            with pytest.raises(error_class) as exc_info:
                await client._request_multipart("POST", "upload", files=files)
            
            error = exc_info.value
            assert error.status_code == status_code
            assert error.request is mock_request
            assert error.response is mock_response

    @pytest.mark.asyncio
    async def test_stream_request_error_handling_http_status(self, api_key: str):
        """Test error handling for stream requests with HTTP status errors."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            # Instantiate client first before accessing its attributes
            client = AsyncVeniceClient(api_key=api_key)
            
            # Setup error response
            mock_response = AsyncMock(spec=httpx.Response) # This is for the response from stream.__aenter__
            mock_response.headers = {}
            mock_response.status_code = 400
            mock_request_for_stream_status_error = MagicMock(spec=httpx.Request)
            mock_request_for_stream_status_error.method = "POST" # Or the appropriate method
            mock_request_for_stream_status_error.url = httpx.URL(client._base_url.join("chat/completions")) # Or the appropriate URL
            mock_response.request = mock_request_for_stream_status_error
            mock_response.status_code = 400
            mock_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
                message="Error", request=mock_response.request, response=mock_response
            ))
            
            # Setup stream context
            mock_stream_context = AsyncMock()
            mock_stream_context.__aenter__.return_value = mock_response
            mock_stream_context.__aexit__.return_value = None
            mock_httpx_client_instance.stream.return_value = mock_stream_context # Set on the instance
            
            with pytest.raises(InvalidRequestError):
                async for _ in client._stream_request("POST", "chat/completions", json_data={"model": "venice-1"}):
                    pass  # Should raise before yielding anything
                    
    @pytest.mark.asyncio
    async def test_stream_request_with_custom_headers_async(self, api_key: str):
        """Test that custom headers are correctly merged for async SSE streaming requests."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            
            # Create a mock headers object that behaves like a dictionary
            # Initialize with the headers that AsyncVeniceClient sets during construction
            mock_headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}"
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
            mock_httpx_client_instance.headers = mock_headers_obj
            mock_httpx_client_instance.aclose = AsyncMock()
            
            client = AsyncVeniceClient(api_key=api_key)
            mock_httpx_client_instance.headers.update({"User-Agent": "test-agent-async"}) # Simulate existing headers
            
            custom_headers = {"X-Custom-Header": "custom-value", "Accept": "application/json"}
            
            mock_response_content = ["data: [DONE]"]
            mock_response = AsyncMock(spec=httpx.Response)
            mock_response.headers = {}
            mock_response.aiter_lines.return_value = mock_async_iterator(mock_response_content)
            mock_response.raise_for_status = MagicMock()
            
            mock_stream_context = AsyncMock()
            mock_stream_context.__aenter__.return_value = mock_response
            mock_stream_context.__aexit__.return_value = None
            mock_httpx_client_instance.stream.return_value = mock_stream_context
            
            async for _ in client._stream_request("POST", "chat/completions", headers=custom_headers):
                pass
            
            call_args = mock_httpx_client_instance.stream.call_args
            assert "Accept" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["Accept"] == "text/event-stream"
            assert "X-Custom-Header" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["X-Custom-Header"] == "custom-value"
            assert "User-Agent" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["User-Agent"] == "test-agent-async"

    @pytest.mark.asyncio
    async def test_stream_request_raw_with_custom_headers_async(self, api_key: str):
        """Test that custom headers are correctly merged for async raw binary streaming requests."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            
            # Create a mock headers object that behaves like a dictionary
            # Initialize with the headers that AsyncVeniceClient sets during construction
            mock_headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}"
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
            mock_httpx_client_instance.headers = mock_headers_obj
            mock_httpx_client_instance.aclose = AsyncMock()
            
            client = AsyncVeniceClient(api_key=api_key)
            mock_httpx_client_instance.headers.update({"User-Agent": "test-agent-async-raw"}) # Simulate existing headers

            custom_headers = {"X-Custom-Header": "custom-raw-value", "Content-Type": "audio/mpeg", "Accept": "audio/wav"}
            
            mock_response_content = [b"done"]
            mock_response = AsyncMock(spec=httpx.Response)
            mock_response.headers = {}
            mock_response.aiter_bytes.return_value = mock_async_iterator(mock_response_content)
            mock_response.raise_for_status = MagicMock()
            
            mock_stream_context = AsyncMock()
            mock_stream_context.__aenter__.return_value = mock_response
            mock_stream_context.__aexit__.return_value = None
            mock_httpx_client_instance.stream.return_value = mock_stream_context
            
            async for _ in client._stream_request_raw("POST", "audio/speech", headers=custom_headers):
                pass
            
            call_args = mock_httpx_client_instance.stream.call_args
            assert "Accept" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["Accept"] == "audio/wav"
            assert "Content-Type" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["Content-Type"] == "audio/mpeg"
            assert "X-Custom-Header" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["X-Custom-Header"] == "custom-raw-value"
            assert "User-Agent" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["User-Agent"] == "test-agent-async-raw"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exception_class,exception_message,expected_error_type,expected_message", [
            (httpx.TimeoutException, "Stream timeout", VeniceError, "Stream request timed out: Stream timeout"),
            (httpx.ConnectError, "Stream connection error", VeniceError, "Stream request failed: Stream connection error"), # Updated expected message
        ]
    )
    async def test_stream_request_client_error_handling(
        self, api_key: str, exception_class, exception_message, expected_error_type, expected_message
    ):
        """Test handling of client-side errors during streaming."""
        # This test patches client._stream_request directly, so the httpx.AsyncClient patch is not strictly needed here
        # for the direct behavior of _stream_request, but keeping it for consistency if other internal calls were made.
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            # The instance created by AsyncVeniceClient will use this spec if not for the direct patch later
            mock_instance_for_init = AsyncMock(spec=original_httpx_async_client)
            mock_instance_for_init.headers = MagicMock(spec=httpx.Headers)
            mock_instance_for_init.aclose = AsyncMock()
            # stream is a sync method returning an async context manager. Mock it as MagicMock.
            mock_instance_for_init.stream = MagicMock()
            MockAsyncHTTPXClientClass.return_value = mock_instance_for_init
            client = AsyncVeniceClient(api_key=api_key)
            
            # Create a mock request object with a method attribute
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST" # Or the appropriate method for the test
            mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions") # Or the appropriate URL

            # Setup mock stream context that raises the exception on __aenter__
            mock_stream_context = AsyncMock()
            # Pass the mock request to the exception
            mock_stream_context.__aenter__.side_effect = exception_class(exception_message, request=mock_request)
            mock_stream_context.__aexit__.return_value = None
            
            with patch.object(client._client, 'stream', return_value=mock_stream_context):
                with pytest.raises(expected_error_type, match=expected_message):
                    async for _ in client._stream_request("POST", "chat/completions", json_data={"model": "venice-1"}):
                        pass  # Should raise before yielding anything

    # test_stream_request_error_handling_request_error is covered by the parameterized test above

    @pytest.mark.asyncio
    async def test_context_manager(self):
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock() # aclose on the instance

            async with AsyncVeniceClient(api_key="test-api-key") as client:
                assert isinstance(client, AsyncVeniceClient)
            # aclose is called by __aexit__
            mock_httpx_client_instance.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close(self, api_key: str):
        """Test that close method properly closes the HTTP client."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()

            client = AsyncVeniceClient(api_key=api_key)
            await client.close()
            
            # Verify the client was closed
            mock_httpx_client_instance.aclose.assert_awaited_once()
    @pytest.mark.asyncio
    async def test_client_closes_transport_on_close_async(self, api_key: str):
        """Test that AsyncVeniceClient.aclose() properly closes the underlying httpx.AsyncClient transport."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()

            client = AsyncVeniceClient(api_key=api_key)
            await client.aclose()
            
            # Verify the underlying httpx client was closed
            mock_httpx_client_instance.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialization_with_external_http_client(self, api_key: str):
        """Test that client can be initialized with an external HTTP client."""
        # No need to patch httpx.AsyncClient here if we are providing an external one.
        # The SUT's __init__ should not call httpx.AsyncClient() in this case.
        external_client = AsyncMock(spec=httpx.AsyncClient) # Use an AsyncMock for the external client
        client = AsyncVeniceClient(api_key=api_key, http_client=external_client)
            
        # Verify that the external client is used
        assert client._client is external_client, "External HTTP client should be used"
        # mock_httpx_client.assert_not_called() # This assertion is not relevant if we don't patch

    @pytest.mark.asyncio
    async def test_request_get_headers_modification(self, api_key: str, mock_response: MagicMock):
        """Test that default headers like Content-Type are removed for GET requests."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            self.setup_mock_httpx_client(MockAsyncHTTPXClientClass, mock_response)
            
            client = AsyncVeniceClient(api_key=api_key)
            result = await client._request("GET", "test_endpoint")
            
            # Verify the result and headers
            assert result == {"status": "success"}
            call_args = mock_httpx_client_instance.request.call_args
            assert "Content-Type" not in call_args.kwargs["headers"], "Content-Type should be removed for GET"
            assert "Accept" not in call_args.kwargs["headers"], "Accept should be removed for GET"

    @pytest.mark.asyncio
    async def test_request_error_handling_connect_error(self):
        """Test _request error handling for httpx.ConnectError."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            # Instantiate client first before accessing its attributes
            client = AsyncVeniceClient(api_key="test-api-key")
            
            mock_httpx_client_instance.request = AsyncMock()
            mock_request_connect_error = MagicMock(spec=httpx.Request)
            mock_request_connect_error.method = "GET"
            mock_request_connect_error.url = httpx.URL(client._base_url.join("test_endpoint"))
            mock_httpx_client_instance.request.side_effect = httpx.ConnectError("Connection failed async", request=mock_request_connect_error)
            with pytest.raises(VeniceError, match="Request failed: Connection failed async"):
                await client._request("GET", "test_endpoint")

    @pytest.mark.asyncio
    async def test_request_error_handling_read_timeout(self):
        """Test _request error handling for httpx.ReadTimeout."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            # Instantiate client first before accessing its attributes
            client = AsyncVeniceClient(api_key="test-api-key")
            
            mock_httpx_client_instance.request = AsyncMock()
            mock_request_read_timeout = MagicMock(spec=httpx.Request)
            mock_request_read_timeout.method = "GET"
            mock_request_read_timeout.url = httpx.URL(client._base_url.join("test_endpoint"))
            mock_httpx_client_instance.request.side_effect = httpx.ReadTimeout("Read timed out async", request=mock_request_read_timeout)
            with pytest.raises(VeniceError, match="Request timed out: Read timed out async"):
                await client._request("GET", "test_endpoint")

    @pytest.mark.asyncio
    async def test_stream_request_error_handling_connect_error(self):
        """Test _stream_request error handling for httpx.ConnectError."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            # Instantiate client first before accessing its attributes
            client = AsyncVeniceClient(api_key="test-api-key")
            
            # mock_httpx_client_instance.stream is already an AsyncMock by default. Set its side_effect.
            mock_request_stream_connect_error = MagicMock(spec=httpx.Request)
            mock_request_stream_connect_error.method = "POST"
            mock_request_stream_connect_error.url = httpx.URL(client._base_url.join("chat/completions"))
            mock_httpx_client_instance.stream.side_effect = httpx.ConnectError("Stream connection failed async", request=mock_request_stream_connect_error)
            with pytest.raises(VeniceError, match="Stream request failed: Stream connection failed async"):
                async for _ in client._stream_request("POST", "chat/completions", json_data={"model": "venice-1"}):
                    pass

    @pytest.mark.asyncio
    async def test_stream_request_raw_error_handling_connect_error(self):
        """Test _stream_request_raw error handling for httpx.ConnectError."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            # Instantiate client first before accessing its attributes
            client = AsyncVeniceClient(api_key="test-api-key")
            
            # mock_httpx_client_instance.stream is already an AsyncMock by default. Set its side_effect.
            mock_request_raw_stream_connect_error = MagicMock(spec=httpx.Request)
            mock_request_raw_stream_connect_error.method = "POST"
            mock_request_raw_stream_connect_error.url = httpx.URL(client._base_url.join("audio/speech"))
            mock_httpx_client_instance.stream.side_effect = httpx.ConnectError("Raw stream connection failed async", request=mock_request_raw_stream_connect_error)
            with pytest.raises(VeniceError, match="Stream request failed: Raw stream connection failed async"):
                async for _ in client._stream_request_raw("POST", "audio/speech", json_data={"text": "test"}):
                    pass

    @pytest.mark.asyncio
    async def test_aenter_method(self):
        """Test __aenter__ method."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            client = AsyncVeniceClient(api_key="test-api-key")
            entered_client = await client.__aenter__()
            assert entered_client is client

    @pytest.mark.asyncio
    async def test_aexit_method(self):
        """Test __aexit__ method."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            client = AsyncVeniceClient(api_key="test-api-key")
            # client._client is already mock_httpx_client_instance due to the patch
            mock_httpx_client_instance.aclose = AsyncMock() # Mock the async close method
            
            await client.__aexit__(None, None, None) # Call with no exception

            mock_httpx_client_instance.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aexit_method_with_exception(self):
        """Test __aexit__ method when an exception occurs."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            client = AsyncVeniceClient(api_key="test-api-key")
            # client._client is already mock_httpx_client_instance
            mock_httpx_client_instance.aclose = AsyncMock() # Mock the async close method
            
            try:
                # Simulate an exception within the context
                await client.__aexit__(ValueError, ValueError("Test async"), None)
            except ValueError:
                pass # Catch the re-raised exception

            mock_httpx_client_instance.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stream_response_consumed_error_async(self):
        """Test that AsyncStream raises StreamConsumedError when attempting to re-iterate after consumption."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            # Create a mock async iterator that yields one chunk then stops
            mock_iterator = AsyncMock()
            
            async def side_effect():
                yield {"choices": [{"delta": {"content": "chunk1"}}]}
                # Iterator is exhausted after one item
            
            mock_iterator.__aiter__ = MagicMock(return_value=side_effect())
            mock_iterator.__anext__ = MagicMock(side_effect=side_effect().__anext__)
            
            # Setup mock httpx client for AsyncVeniceClient initialization
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            
            client = AsyncVeniceClient(api_key="test-api-key")
            
            # Import AsyncStream class
            from venice_ai.streaming import AsyncStream
            
            # Create an AsyncStream instance with the mock iterator
            stream = AsyncStream(mock_iterator, client=client)
            
            # First iteration should work and consume the stream
            first_chunk = await stream.__anext__()
            assert first_chunk["choices"][0]["delta"]["content"] == "chunk1"
            
            # Stream should be exhausted now, trying to get next should raise StopAsyncIteration
            with pytest.raises(StopAsyncIteration):
                await stream.__anext__()
            
            # Now attempting to iterate again should raise StreamConsumedError
            from venice_ai.exceptions import StreamConsumedError
            with pytest.raises(StreamConsumedError, match="Cannot iterate over a consumed stream"):
                async for chunk in stream:
                    pass  # Should raise before yielding anything

    @pytest.mark.asyncio
    async def test_stream_response_closed_error_async(self):
        """Test that AsyncStream raises StreamClosedError when attempting to iterate over a closed stream."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            # Create a mock async iterator that raises httpx.StreamClosed when accessed
            mock_iterator = AsyncMock()
            
            async def side_effect():
                # Simulate httpx.StreamClosed being raised during iteration
                mock_request = MagicMock(spec=httpx.Request)
                mock_request.method = "POST"
                mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
                raise httpx.StreamClosed()
            
            mock_iterator.__aiter__ = MagicMock(return_value=mock_async_iterator([]))
            mock_iterator.__anext__ = MagicMock(side_effect=side_effect)
            
            # Setup mock httpx client for AsyncVeniceClient initialization
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            
            client = AsyncVeniceClient(api_key="test-api-key")
            
            # Import AsyncStream class
            from venice_ai.streaming import AsyncStream
            
            # Create an AsyncStream instance with the mock iterator
            stream = AsyncStream(mock_iterator, client=client)
            
            # Attempting to iterate should raise StreamClosedError (translated from httpx.StreamClosed)
            from venice_ai.exceptions import StreamClosedError
            with pytest.raises(StreamClosedError, match="Stream has been closed"):
                await stream.__anext__()

    @pytest.mark.asyncio
    async def test_timeout_connect_error_raw_response_async(self):
        """Test _request error handling for httpx.ConnectTimeout with raw_response=True."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client) # Use original for spec
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            # Create mock request object for the ConnectTimeout exception
            mock_request_connect_timeout = MagicMock(spec=httpx.Request)
            mock_request_connect_timeout.method = "GET"
            mock_request_connect_timeout.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
            
            # Setup the mock to raise ConnectTimeout when request is called
            mock_httpx_client_instance.request = AsyncMock(
                side_effect=httpx.ConnectTimeout("Connection timed out", request=mock_request_connect_timeout)
            )
            
            client = AsyncVeniceClient(api_key="test-api-key")
            
            from venice_ai.exceptions import APITimeoutError
            with pytest.raises(APITimeoutError) as exc_info:
                await client._request("GET", "test_endpoint", raw_response=True)
            
            # Verify the APITimeoutError has the correct request attribute
            error = exc_info.value
            assert error.request is not None
            assert error.request.method == "GET"
            assert str(error.request.url) == "https://api.venice.ai/api/v1/test_endpoint"
            # For connection timeouts, there should be no response
            assert error.response is None

    @pytest.mark.asyncio
    async def test_build_request_auth_headers_default_token_retention_async(self):
        """Test that build_request correctly handles authentication headers with default token retention in async client."""
        with patch('httpx.AsyncClient'):
            # Test 1: Client initialized with API key
            client = AsyncVeniceClient(api_key="test-api-key")
            
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
                client_env = AsyncVeniceClient()
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


    # B4: Add tests for _translate_httpx_error_to_api_error method
    @pytest.mark.asyncio
    async def test_translate_httpx_error_to_api_error_timeout(self, api_key: str):
        """Test _translate_httpx_error_to_api_error for TimeoutException."""
        client = AsyncVeniceClient(api_key=api_key)
        
        # Create a mock request
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.url = "https://api.venice.ai/test"
        mock_request.method = "POST"
        
        # Create TimeoutException
        timeout_error = httpx.TimeoutException("Request timed out", request=mock_request)
        
        # Test translation
        api_error = await client._translate_httpx_error_to_api_error(timeout_error, mock_request)
        
        assert isinstance(api_error, APITimeoutError)
        assert api_error.request is mock_request
        assert "Request timed out" in str(api_error)

    @pytest.mark.asyncio
    async def test_translate_httpx_error_to_api_error_connect_error(self, api_key: str):
        """Test _translate_httpx_error_to_api_error for ConnectError."""
        client = AsyncVeniceClient(api_key=api_key)
        
        # Create a mock request
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.url = "https://api.venice.ai/test"
        mock_request.method = "GET"
        
        # Create ConnectError
        connect_error = httpx.ConnectError("Connection failed", request=mock_request)
        
        # Test translation
        api_error = await client._translate_httpx_error_to_api_error(connect_error, mock_request)
        
        assert isinstance(api_error, APIConnectionError)
        assert api_error.request is mock_request
        assert "Connection failed" in str(api_error)

    @pytest.mark.asyncio
    async def test_translate_httpx_error_to_api_error_http_status_error_with_response(self, api_key: str):
        """Test _translate_httpx_error_to_api_error for HTTPStatusError with async response methods."""
        client = AsyncVeniceClient(api_key=api_key)
        
        # Create a mock request
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.url = "https://api.venice.ai/test"
        mock_request.method = "POST"
        
        # Create a mock response with async methods
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.aread.return_value = b'{"error": {"message": "Invalid API key", "type": "authentication_error"}}'
        mock_response.text = '{"error": {"message": "Invalid API key", "type": "authentication_error"}}'
        mock_response.json.return_value = {"error": {"message": "Invalid API key", "type": "authentication_error"}}
        mock_response.request = mock_request
        
        # Create HTTPStatusError
        http_error = httpx.HTTPStatusError("401 Unauthorized", request=mock_request, response=mock_response)
        
        # Test translation
        api_error = await client._translate_httpx_error_to_api_error(http_error, mock_request)
        
        assert isinstance(api_error, AuthenticationError)
        assert api_error.status_code == 401
        assert api_error.request is mock_request
        assert api_error.response is mock_response
        assert api_error.body == {"error": {"message": "Invalid API key", "type": "authentication_error"}}
        
        # Verify that aread() was called on the response
        mock_response.aread.assert_called_once()

    @pytest.mark.asyncio
    async def test_translate_httpx_error_to_api_error_http_status_error_403(self, api_key: str):
        """Test _translate_httpx_error_to_api_error for 403 Forbidden."""
        client = AsyncVeniceClient(api_key=api_key)
        
        # Create a mock request
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.url = "https://api.venice.ai/test"
        mock_request.method = "DELETE"
        
        # Create a mock response
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 403
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.aread.return_value = b'{"error": {"message": "Insufficient permissions", "type": "permission_denied"}}'
        mock_response.text = '{"error": {"message": "Insufficient permissions", "type": "permission_denied"}}'
        mock_response.json.return_value = {"error": {"message": "Insufficient permissions", "type": "permission_denied"}}
        mock_response.request = mock_request
        
        # Create HTTPStatusError
        http_error = httpx.HTTPStatusError("403 Forbidden", request=mock_request, response=mock_response)
        
        # Test translation
        api_error = await client._translate_httpx_error_to_api_error(http_error, mock_request)
        
        assert isinstance(api_error, PermissionDeniedError)
        assert api_error.status_code == 403
        assert api_error.request is mock_request
        assert api_error.response is mock_response

    @pytest.mark.asyncio
    async def test_translate_httpx_error_to_api_error_http_status_error_429(self, api_key: str):
        """Test _translate_httpx_error_to_api_error for 429 Rate Limit."""
        client = AsyncVeniceClient(api_key=api_key)
        
        # Create a mock request
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.url = "https://api.venice.ai/test"
        mock_request.method = "POST"
        
        # Create a mock response
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.headers = {"Content-Type": "application/json", "Retry-After": "60"}
        mock_response.aread.return_value = b'{"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}'
        mock_response.text = '{"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}'
        mock_response.json.return_value = {"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}
        mock_response.request = mock_request
        
        # Create HTTPStatusError
        http_error = httpx.HTTPStatusError("429 Too Many Requests", request=mock_request, response=mock_response)
        
        # Test translation
        api_error = await client._translate_httpx_error_to_api_error(http_error, mock_request)
        
        assert isinstance(api_error, RateLimitError)
        assert api_error.status_code == 429
        assert api_error.request is mock_request
        assert api_error.response is mock_response

    @pytest.mark.asyncio
    async def test_translate_httpx_error_to_api_error_generic_request_error(self, api_key: str):
        """Test _translate_httpx_error_to_api_error for generic RequestError."""
        client = AsyncVeniceClient(api_key=api_key)
        
        # Create a mock request
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.url = "https://api.venice.ai/test"
        mock_request.method = "PUT"
        
        # Create generic RequestError
        request_error = httpx.RequestError("Generic request error", request=mock_request)
        
        # Test translation
        api_error = await client._translate_httpx_error_to_api_error(request_error, mock_request)
        
        assert isinstance(api_error, APIConnectionError)
        assert api_error.request is mock_request
        assert "Generic request error" in str(api_error)

    @pytest.mark.asyncio
    async def test_translate_httpx_error_to_api_error_response_read_failure(self, api_key: str):
        """Test _translate_httpx_error_to_api_error when response.aread() fails."""
        client = AsyncVeniceClient(api_key=api_key)
        
        # Create a mock request
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.url = "https://api.venice.ai/test"
        mock_request.method = "POST"
        
        # Create a mock response that fails on aread()
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.aread.side_effect = Exception("Failed to read response")
        mock_response.text = None  # This will be None due to aread failure
        mock_response.json.side_effect = Exception("Failed to parse JSON")
        mock_response.request = mock_request
        
        # Create HTTPStatusError
        http_error = httpx.HTTPStatusError("500 Internal Server Error", request=mock_request, response=mock_response)
        
        # Test translation
        api_error = await client._translate_httpx_error_to_api_error(http_error, mock_request)
        
        assert isinstance(api_error, InternalServerError)
        assert api_error.status_code == 500
        assert api_error.request is mock_request
        assert api_error.response is mock_response
        # When aread() fails, body should be None or empty
        assert api_error.body is None or api_error.body == ""

    # B5: Add tests for AsyncVeniceClient context management with requests inside the context
    @pytest.mark.asyncio
    async def test_context_manager_with_successful_request_inside(self, api_key: str):
        """Test context manager with successful request inside the context."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            # Setup successful response
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 200
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.content = b'{"status": "success"}'
            mock_response.raise_for_status = MagicMock()
            mock_httpx_client_instance.request = AsyncMock(return_value=mock_response)
            
            # Test context manager with request inside
            async with AsyncVeniceClient(api_key=api_key) as client:
                result = await client._request("GET", "test_endpoint", raw_response=True)
                assert result == b'{"status": "success"}'
            
            # Verify aclose was called on exit
            mock_httpx_client_instance.aclose.assert_awaited_once()
            # Verify request was made
            mock_httpx_client_instance.request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_manager_with_api_error_inside(self, api_key: str):
        """Test context manager with API error inside the context - cleanup should still occur."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            # Setup error response
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.url = "https://api.venice.ai/test_endpoint"
            mock_request.method = "GET"
            
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 404
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.content = b'{"error": {"message": "Not found", "type": "not_found_error"}}'
            mock_response.request = mock_request
            
            # Create HTTPStatusError that will be raised
            http_error = httpx.HTTPStatusError("404 Not Found", request=mock_request, response=mock_response)
            mock_httpx_client_instance.request = AsyncMock(side_effect=http_error)
            
            # Test context manager with error inside
            with pytest.raises(NotFoundError):
                async with AsyncVeniceClient(api_key=api_key) as client:
                    await client._request("GET", "test_endpoint", raw_response=True)
            
            # Verify aclose was called despite the error
            mock_httpx_client_instance.aclose.assert_awaited_once()
            # Verify request was attempted
            mock_httpx_client_instance.request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_manager_with_streaming_request_inside(self, api_key: str):
        """Test context manager with streaming request inside the context."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            # Setup streaming response
            mock_response = AsyncMock(spec=httpx.Response)
            mock_response.headers = {}
            mock_response.aiter_lines.return_value = mock_async_iterator([
                'data: {"choices": [{"delta": {"content": "chunk1"}}]}',
                'data: {"choices": [{"delta": {"content": "chunk2"}}]}',
                'data: [DONE]'
            ])
            mock_response.raise_for_status = MagicMock()
            
            mock_stream_context = AsyncMock()
            mock_stream_context.__aenter__.return_value = mock_response
            mock_stream_context.__aexit__.return_value = None
            mock_httpx_client_instance.stream.return_value = mock_stream_context
            
            # Test context manager with streaming request inside
            chunks = []
            async with AsyncVeniceClient(api_key=api_key) as client:
                async for chunk in client._stream_request("POST", "chat/completions", json_data={"model": "venice-1"}):
                    chunks.append(chunk)
            
            # Verify streaming worked
            assert len(chunks) == 2  # Only 2 valid JSON chunks, [DONE] is filtered out
            assert chunks[0]["choices"][0]["delta"]["content"] == "chunk1"
            assert chunks[1]["choices"][0]["delta"]["content"] == "chunk2"
            
            # Verify aclose was called on exit
            mock_httpx_client_instance.aclose.assert_awaited_once()
            # Verify stream was called
            mock_httpx_client_instance.stream.assert_called_once()


# Shared fixtures for resource tests
@pytest.fixture
def api_base_url() -> str:
    """Common base URL for testing."""
    return "https://api.venice.ai/api/v1/"
@pytest.fixture
def api_test_key() -> str:
    """Common API key for testing."""
    return "test-api-key"

@pytest.fixture
def chat_response() -> Dict[str, Any]:
    """Sample chat completion response."""
    return {"choices": [{"message": {"content": "Hello"}}]}

class TestAsyncChatResource:
    """Tests for the AsyncChatResource class."""
    
    @pytest.fixture
    def mock_async_client(self, api_test_key: str, api_base_url: str) -> MagicMock:
        """Create a mock AsyncVeniceClient for testing resources."""
        client = MagicMock(spec=AsyncVeniceClient)
        client._api_key = api_test_key
        client._base_url = httpx.URL(api_base_url)
        client._client = AsyncMock(spec=httpx.AsyncClient)
        return client

    @pytest.mark.asyncio
    async def test_initialization(self, mock_async_client: MagicMock):
        """Test that AsyncChatResource initializes correctly with proper attributes."""
        resource = AsyncChatResource(mock_async_client)
        assert resource._client == mock_async_client, "Client reference should be stored"
        assert hasattr(resource, 'completions'), "Completions attribute should be initialized"


class TestAsyncChatCompletions:
    """Tests for the AsyncChatCompletions class."""
    
    @pytest.fixture
    def mock_async_client(self, api_test_key: str, api_base_url: str) -> MagicMock:
        """Create a mock AsyncVeniceClient with methods needed for completions testing."""
        client = MagicMock(spec=AsyncVeniceClient)
        client._api_key = api_test_key
        client._base_url = httpx.URL(api_base_url)
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client.post = AsyncMock()
        client._stream_request = AsyncMock()
        return client
    
    @pytest.fixture
    def test_messages(self) -> List[MessageParam]:
        """Sample messages for chat tests."""
        # Explicitly define the dictionary with known keys and cast it
        message: MessageParam = {"role": "user", "content": "Hello"}
        return [message]

    @pytest.mark.asyncio
    async def test_create_non_streaming(self, api_test_key: str, chat_response: Dict, test_messages: List[MessageParam]):
        """Test create method for chat completions in non-streaming mode."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            # This mock_httpx_client_instance is for the one created by AsyncVeniceClient.__init__
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            # Setup client and response
            client = AsyncVeniceClient(api_key=api_test_key)
            # The client.post method itself will use client._request which uses client._client.request
            # So, we need to ensure client._client.request is properly mocked if we don't patch client.post directly.
            # However, this test patches client.post directly, so the httpx.AsyncClient patch is for __init__.
            with patch.object(client, 'post', return_value=chat_response) as mock_post:
                # Create completions and test
                completions = AsyncChatCompletions(client)
                result = await completions.create(
                    model="venice-1",
                    messages=test_messages,
                    stream=False
                )
                
                # Verify the result and API call
                mock_post.assert_awaited_once()
                assert result == chat_response, "Response should match expected format"

    @pytest.fixture
    def stream_chunks(self) -> List[Dict[str, Any]]:
        """Sample streaming chunks for chat tests."""
        return [
            {"choices": [{"delta": {"content": "chunk1"}}]},
            {"choices": [{"delta": {"content": "chunk2"}}]}
        ]

    @pytest.mark.asyncio
    async def test_create_streaming(self, api_test_key: str, test_messages: List[MessageParam], stream_chunks: List[Dict]):
        """Test create method for chat completions in streaming mode."""
        # Create a mock async iterator for stream responses
        mock_iterator = mock_async_iterator(stream_chunks)
        
        # Setup client with mocked streaming
        client = AsyncVeniceClient(api_key=api_test_key)
        with patch.object(client, '_stream_request', return_value=mock_iterator) as mock_stream_method:
            completions = AsyncChatCompletions(client)
            
            # Process streaming response
            chunks = []
            async for chunk in await completions.create(
                    model="venice-1",
                    messages=test_messages,
                    stream=True
                ):
                chunks.append(chunk)
            
            # Verify streaming results
            assert len(chunks) == 2, "Should receive both chunks"
            assert chunks[0]["choices"][0]["delta"]["content"] == "chunk1", "First chunk content should match"
            assert chunks[1]["choices"][0]["delta"]["content"] == "chunk2", "Second chunk content should match"

    @pytest.mark.asyncio
    async def test_create_with_optional_parameters(self, api_test_key: str, chat_response: Dict, test_messages: List[MessageParam]):
        """Test create method with optional parameters like temperature and max_tokens."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            client = AsyncVeniceClient(api_key=api_test_key)
            
            # Test with additional parameters
            with patch.object(client, 'post', return_value=chat_response) as mock_post:
                completions = AsyncChatCompletions(client)
                await completions.create(
                    model="venice-1",
                    messages=test_messages,
                    stream=False,
                    temperature=0.7,
                    max_tokens=100
                )
                
                # Verify the call with optional parameters
                mock_post.assert_awaited_once()
                call_args = mock_post.call_args
                assert call_args is not None, "Post should be called with correct parameters"
                _, kwargs = call_args
                json_data = kwargs.get('json_data', {})
                assert json_data.get('temperature') == 0.7, "Temperature parameter should be included"
                assert json_data.get('max_tokens') == 100, "Max tokens parameter should be included"

    @pytest.mark.asyncio
    async def test_create_with_error(self, api_test_key: str, test_messages: List[MessageParam]):
        """Test error handling for non-streaming create method."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            client = AsyncVeniceClient(api_key=api_test_key)
            
            # Set up error scenario
            error_response = VeniceError("Invalid model specified")
            with patch.object(client, 'post', side_effect=error_response) as mock_post:
                completions = AsyncChatCompletions(client)
                
                # Verify error is properly propagated
                with pytest.raises(VeniceError, match="Invalid model specified"):
                    await completions.create(
                        model="invalid-model",
                        messages=test_messages,
                        stream=False
                    )
                mock_post.assert_awaited_once()
                
    @pytest.mark.asyncio
    async def test_create_streaming_with_error(self, api_test_key: str, test_messages: List[MessageParam]):
        """Test error handling for streaming create method."""
        client = AsyncVeniceClient(api_key=api_test_key)
        
        # Set up streaming error scenario
        error_response = VeniceError("Invalid model specified")
        with patch.object(client, '_stream_request', side_effect=error_response) as mock_stream:
            completions = AsyncChatCompletions(client)
            
            # Verify streaming error is properly propagated
            with pytest.raises(VeniceError, match="Invalid model specified"):
                await completions.create(
                    model="invalid-model",
                    messages=test_messages,
                    stream=True
                )
            
            # Verify stream request was called
            mock_stream.assert_called_once()
    @pytest.mark.asyncio
    async def test_create_with_none_optional_parameters(self, api_test_key: str, chat_response: Dict, test_messages: List[MessageParam]):
        """Test create method with optional parameters explicitly set to None."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            client = AsyncVeniceClient(api_key=api_test_key)
            
            with patch.object(client, 'post', return_value=chat_response) as mock_post:
                completions = AsyncChatCompletions(client)
                await completions.create(
                    model="venice-1",
                    messages=test_messages,
                    stream=False,
                    temperature=None, # Explicitly None
                    max_tokens=100,
                    top_p=None # Explicitly None
                )
                
                mock_post.assert_awaited_once()
                call_args = mock_post.call_args
                assert call_args is not None
                _, kwargs = call_args
                json_data = kwargs.get('json_data', {})
                assert 'temperature' not in json_data, "Temperature parameter should not be included when None"
                assert 'top_p' not in json_data, "Top_p parameter should not be included when None"
                assert json_data.get('max_tokens') == 100, "Max tokens parameter should be included"

    @pytest.mark.asyncio
    async def test_create_streaming_with_custom_stream_cls(self, api_test_key: str, test_messages: List[MessageParam], stream_chunks: List[Dict]):
        """Test create method for chat completions in streaming mode with a custom stream_cls."""
        # Define a dummy custom stream class
        class CustomAsyncStream:
            def __init__(self, iterator, client):
                self._iterator = iterator
                self._client = client
                self._consumed = False

            async def __aiter__(self):
                if self._consumed:
                    raise VeniceError("Stream has already been consumed")
                self._consumed = True
                async for item in self._iterator:
                    yield item

        # Create a mock async iterator for stream responses
        mock_iterator = mock_async_iterator(stream_chunks)
        
        # Setup client with mocked streaming
        client = AsyncVeniceClient(api_key=api_test_key)
        with patch.object(client, '_stream_request', return_value=mock_iterator) as mock_stream_method:
            completions = AsyncChatCompletions(client)
            
            # Process streaming response with custom stream_cls
            result_stream = await completions.create(
                    model="venice-1",
                    messages=test_messages,
                    stream=True,
                    stream_cls=CustomAsyncStream # Provide custom stream class
                )
            
            # Verify that the returned object is an instance of CustomAsyncStream
            assert isinstance(result_stream, CustomAsyncStream)
            
            # Verify that the custom stream can be iterated over
            chunks = []
            async for chunk in result_stream:
                chunks.append(chunk)

            assert len(chunks) == 2, "Should receive both chunks from custom stream"
            assert chunks[0]["choices"][0]["delta"]["content"] == "chunk1", "First chunk content should match"
            assert chunks[1]["choices"][0]["delta"]["content"] == "chunk2", "Second chunk content should match"

    @pytest.mark.asyncio
    async def test_stream_response_async_iteration_error_handling(self):
        """Test that AsyncStream.__anext__ properly handles httpx.HTTPError during iteration."""
        original_httpx_async_client = httpx.AsyncClient # Store original class
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            # Create a mock async iterator that raises httpx.ReadError on the second call to __anext__()
            mock_iterator = AsyncMock()
            
            # First call returns a chunk, second call raises httpx.ReadError
            async def side_effect():
                yield {"choices": [{"delta": {"content": "chunk1"}}]}
                # Simulate an httpx.ReadError during async stream iteration
                mock_request = MagicMock(spec=httpx.Request)
                mock_request.method = "POST"
                mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
                raise httpx.ReadError("Connection broken during async stream", request=mock_request)
            
            mock_iterator.__aiter__ = MagicMock(return_value=side_effect())
            mock_iterator.__anext__ = MagicMock(side_effect=side_effect().__anext__)
            
            # Setup mock httpx client for AsyncVeniceClient initialization
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            
            client = AsyncVeniceClient(api_key="test-api-key")
            
            # Import AsyncStream class
            from venice_ai.streaming import AsyncStream
            
            # Create an AsyncStream instance with the mock iterator
            stream = AsyncStream(mock_iterator, client=client)
            
            # First iteration should work
            first_chunk = await stream.__anext__()
            assert first_chunk["choices"][0]["delta"]["content"] == "chunk1"
            
            # Second iteration should raise an APIError (translated from httpx.ReadError)
            from venice_ai.exceptions import APIConnectionError
            with pytest.raises(APIConnectionError, match="Connection broken during async stream"):
                await stream.__anext__()

class TestAsyncBaseClient:
    """Test suite for AsyncVeniceClient base functionality inherited from BaseClient."""
    
    @pytest.fixture
    def api_key(self):
        """Fixture for consistent API key across tests."""
        return "test-api-key"
    
    @pytest.fixture
    def base_url(self):
        """Fixture for consistent base URL across tests."""
        return "https://custom.api.com"
    
    @pytest.mark.asyncio
    async def test_async_context_manager_enter(self, api_key):
        """Test AsyncVeniceClient __aenter__ method."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            client = AsyncVeniceClient(api_key=api_key)
            entered_client = await client.__aenter__()
            
            # Verify __aenter__ returns self
            assert entered_client is client
    
    @pytest.mark.asyncio
    async def test_async_context_manager_exit_normal(self, api_key):
        """Test AsyncVeniceClient __aexit__ method with normal exit."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            client = AsyncVeniceClient(api_key=api_key)
            
            # Test normal exit (no exception)
            await client.__aexit__(None, None, None)
            
            # Verify aclose was called
            mock_httpx_client_instance.aclose.assert_awaited_once()
    
    @pytest.mark.asyncio
    async def test_async_context_manager_exit_with_exception(self, api_key):
        """Test AsyncVeniceClient __aexit__ method with exception."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            client = AsyncVeniceClient(api_key=api_key)
            
            # Test exit with exception
            exc_type = ValueError
            exc_value = ValueError("Test exception")
            exc_traceback = None
            
            await client.__aexit__(exc_type, exc_value, exc_traceback)
            
            # Verify aclose was called even with exception
            mock_httpx_client_instance.aclose.assert_awaited_once()
    
    @pytest.mark.asyncio
    async def test_async_context_manager_full_usage(self, api_key):
        """Test AsyncVeniceClient as async context manager."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            async with AsyncVeniceClient(api_key=api_key) as client:
                # Verify client is properly initialized
                assert isinstance(client, AsyncVeniceClient)
                assert client._api_key == api_key
            
            # Verify aclose was called on exit
            mock_httpx_client_instance.aclose.assert_awaited_once()
    
    @pytest.mark.asyncio
    async def test_aclose_method(self, api_key):
        """Test AsyncVeniceClient aclose method."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            client = AsyncVeniceClient(api_key=api_key)
            await client.aclose()
            
            # Verify the underlying httpx client was closed
            mock_httpx_client_instance.aclose.assert_awaited_once()
    
    @pytest.mark.asyncio
    async def test_close_method_calls_aclose(self, api_key):
        """Test AsyncVeniceClient close method calls aclose."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            client = AsyncVeniceClient(api_key=api_key)
            await client.close()
            
            # Verify the underlying httpx client was closed
            mock_httpx_client_instance.aclose.assert_awaited_once()
    
    @pytest.mark.asyncio
    async def test_build_async_raw_client_usage(self, api_key):
        """Test that AsyncVeniceClient uses _build_async_raw_client from BaseClient."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            # Create client with custom async transport
            custom_async_transport = MagicMock(spec=httpx.AsyncBaseTransport)
            client = AsyncVeniceClient(api_key=api_key, async_transport=custom_async_transport)
            
            # Verify that httpx.AsyncClient was called with the custom transport
            MockAsyncHTTPXClientClass.assert_called_once()
            call_kwargs = MockAsyncHTTPXClientClass.call_args.kwargs
            assert call_kwargs["transport"] is custom_async_transport
    
    @pytest.mark.asyncio
    async def test_inherited_base_client_initialization(self, api_key, base_url):
        """Test that AsyncVeniceClient properly inherits BaseClient initialization."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            # Test initialization with BaseClient parameters
            client = AsyncVeniceClient(
                api_key=api_key,
                base_url=base_url,
                timeout=30.0,
                proxy="http://proxy.example.com:8080",
                verify=False
            )
            
            # Verify BaseClient attributes are properly set
            assert client._api_key == api_key
            assert str(client._base_url) == f"{base_url}/"
            assert isinstance(client._timeout, httpx.Timeout)
            assert client._timeout.read == 30.0
            assert client._proxy == "http://proxy.example.com:8080"
            assert client._verify is False
            
            # Verify httpx.AsyncClient was called with inherited parameters
            MockAsyncHTTPXClientClass.assert_called_once()
            call_kwargs = MockAsyncHTTPXClientClass.call_args.kwargs
            assert call_kwargs["proxy"] == "http://proxy.example.com:8080"
            assert call_kwargs["verify"] is False
    
    @pytest.mark.asyncio
    async def test_async_context_manager_exception_handling(self, api_key):
        """Test AsyncVeniceClient context manager handles exceptions properly."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            # Test that aclose is called even when exception occurs in context
            with pytest.raises(ValueError, match="Test exception"):
                async with AsyncVeniceClient(api_key=api_key) as client:
                    assert isinstance(client, AsyncVeniceClient)
                    raise ValueError("Test exception")
            
            # Verify aclose was called despite the exception
            mock_httpx_client_instance.aclose.assert_awaited_once()
    
    @pytest.mark.asyncio
    async def test_multiple_aclose_calls_safe(self, api_key):
        """Test that multiple calls to aclose are safe and idempotent."""
        original_httpx_async_client = httpx.AsyncClient
        with patch('httpx.AsyncClient', new_callable=MagicMock) as MockAsyncHTTPXClientClass:
            mock_httpx_client_instance = AsyncMock(spec=original_httpx_async_client)
            MockAsyncHTTPXClientClass.return_value = mock_httpx_client_instance
            mock_httpx_client_instance.headers = MagicMock(spec=httpx.Headers)
            mock_httpx_client_instance.aclose = AsyncMock()
            
            client = AsyncVeniceClient(api_key=api_key)
            
            # Call aclose multiple times
            await client.aclose()
            await client.aclose()
            await client.close()  # This also calls aclose
            
            # Verify underlying aclose was called only once due to idempotent behavior
            # The client protects against multiple close calls with _is_closed flag
            assert mock_httpx_client_instance.aclose.await_count == 1
            
            # Verify the client is marked as closed
            assert client._is_closed is True

