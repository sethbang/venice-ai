import pytest
import httpx
import json
from unittest.mock import patch, MagicMock, AsyncMock
from venice_ai._resource import APIResource, AsyncAPIResource
from venice_ai._client import VeniceClient
from venice_ai._async_client import AsyncVeniceClient
from venice_ai.exceptions import (
    APIError, InvalidRequestError, AuthenticationError, 
    PermissionDeniedError, NotFoundError, RateLimitError,
    InternalServerError, APITimeoutError, APIConnectionError
)


class TestAPIResourceRawResponse:
    """Test coverage for _request_raw_response method in APIResource."""
    
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=VeniceClient)
        client._api_key = "test-api-key"
        client._base_url = httpx.URL("https://api.venice.ai/v1/")
        client._client = MagicMock(spec=httpx.Client)
        client._timeout = httpx.Timeout(60.0)
        # Mock the error translation method
        client._translate_httpx_error_to_api_error = MagicMock()
        return client
    
    def test_request_raw_response_basic(self, mock_client):
        """Test basic _request_raw_response functionality."""
        resource = APIResource(mock_client)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.content = b"raw response data"
        mock_client._client.request.return_value = mock_response
        
        options = {
            "body": {"key": "value"},
            "headers": {"Custom-Header": "value"},
            "timeout": 30.0
        }
        
        response = resource._request_raw_response("POST", "endpoint", options=options)
        
        assert response == mock_response
        mock_client._client.request.assert_called_once()
        args, kwargs = mock_client._client.request.call_args
        assert kwargs["method"] == "POST"
        assert str(kwargs["url"]).endswith("endpoint")
        assert kwargs["json"] == {"key": "value"}
        assert kwargs["timeout"] == 30.0
        assert "Custom-Header" in kwargs["headers"]
    
    def test_request_raw_response_no_body(self, mock_client):
        """Test _request_raw_response without body."""
        resource = APIResource(mock_client)
        mock_response = MagicMock(spec=httpx.Response)
        mock_client._client.request.return_value = mock_response
        
        options = {"headers": {"Accept": "application/octet-stream"}}
        
        response = resource._request_raw_response("GET", "download", options=options)
        
        assert response == mock_response
        args, kwargs = mock_client._client.request.call_args
        assert kwargs["json"] is None
        assert kwargs["headers"]["Accept"] == "application/octet-stream"
    
    def test_request_raw_response_default_timeout(self, mock_client):
        """Test _request_raw_response uses client default timeout when not specified."""
        resource = APIResource(mock_client)
        mock_response = MagicMock(spec=httpx.Response)
        mock_client._client.request.return_value = mock_response
        
        options = {"body": {"data": "test"}}
        
        response = resource._request_raw_response("POST", "endpoint", options=options)
        
        args, kwargs = mock_client._client.request.call_args
        assert kwargs["timeout"] == mock_client._timeout
    
    def test_request_raw_response_header_merging(self, mock_client):
        """Test header merging in _request_raw_response."""
        resource = APIResource(mock_client)
        mock_response = MagicMock(spec=httpx.Response)
        mock_client._client.request.return_value = mock_response
        
        # Set default headers on the client
        mock_client._client.headers = {"Authorization": "Bearer token", "User-Agent": "test-agent"}
        
        options = {
            "headers": {"Custom-Header": "value", "Authorization": "Bearer override-token"}
        }
        
        response = resource._request_raw_response("GET", "endpoint", options=options)
        
        args, kwargs = mock_client._client.request.call_args
        # Headers should be merged with provided headers taking precedence
        assert kwargs["headers"]["Authorization"] == "Bearer override-token"
        assert kwargs["headers"]["Custom-Header"] == "value"
        assert kwargs["headers"]["User-Agent"] == "test-agent"
    
    def test_request_raw_response_http_status_error(self, mock_client):
        """Test _request_raw_response with HTTPStatusError."""
        resource = APIResource(mock_client)
        
        # Create the error
        request = httpx.Request("POST", "https://api.venice.ai/v1/endpoint")
        response = httpx.Response(status_code=404, request=request)
        error = httpx.HTTPStatusError("Not Found", request=request, response=response)
        
        mock_client._client.request.side_effect = error
        
        # Mock the translation method to return a NotFoundError
        mock_client._translate_httpx_error_to_api_error.return_value = NotFoundError(
            message="Resource not found",
            request=request,
            response=response
        )
        
        options = {"body": {"key": "value"}}
        
        with pytest.raises(NotFoundError) as exc_info:
            resource._request_raw_response("POST", "endpoint", options=options)
        
        assert "Resource not found" in str(exc_info.value)
        # Check that the translation method was called
        mock_client._translate_httpx_error_to_api_error.assert_called_once()
        # Verify the arguments
        call_args = mock_client._translate_httpx_error_to_api_error.call_args[0]
        assert isinstance(call_args[0], httpx.HTTPStatusError)
        assert call_args[1].method == "POST"
        assert "endpoint" in str(call_args[1].url)
    
    def test_request_raw_response_request_error(self, mock_client):
        """Test _request_raw_response with RequestError."""
        resource = APIResource(mock_client)
        
        # Create the error
        request = httpx.Request("GET", "https://api.venice.ai/v1/endpoint")
        error = httpx.RequestError("Connection failed", request=request)
        
        mock_client._client.request.side_effect = error
        
        # Mock the translation method to return an APIConnectionError
        mock_client._translate_httpx_error_to_api_error.return_value = APIConnectionError(
            message="Connection failed",
            request=request,
            response=None,
            original_error=error
        )
        
        options = {}
        
        with pytest.raises(APIConnectionError) as exc_info:
            resource._request_raw_response("GET", "endpoint", options=options)
        
        assert "Connection failed" in str(exc_info.value)
        # Check that the translation method was called
        mock_client._translate_httpx_error_to_api_error.assert_called_once()
        # Verify the arguments
        call_args = mock_client._translate_httpx_error_to_api_error.call_args[0]
        assert isinstance(call_args[0], httpx.RequestError)
        assert call_args[1].method == "GET"
        assert "endpoint" in str(call_args[1].url)
    
    def test_request_raw_response_no_translate_method(self, mock_client):
        """Test _request_raw_response when client doesn't have _translate_httpx_error_to_api_error."""
        resource = APIResource(mock_client)
        
        # Remove the translation method
        delattr(mock_client, '_translate_httpx_error_to_api_error')
        
        # Create the error
        request = httpx.Request("POST", "https://api.venice.ai/v1/endpoint")
        error = httpx.RequestError("Connection failed", request=request)
        
        mock_client._client.request.side_effect = error
        
        options = {"body": {"key": "value"}}
        
        # Should raise the original httpx error
        with pytest.raises(httpx.RequestError) as exc_info:
            resource._request_raw_response("POST", "endpoint", options=options)
        
        assert "Connection failed" in str(exc_info.value)
    
    def test_request_raw_response_stream_mode(self, mock_client):
        """Test _request_raw_response with stream_mode parameter."""
        resource = APIResource(mock_client)
        mock_response = MagicMock(spec=httpx.Response)
        mock_client._client.request.return_value = mock_response
        
        options = {"body": {"stream": True}}
        
        # Note: stream_mode parameter is accepted but not used in current implementation
        response = resource._request_raw_response("POST", "stream", options=options, stream_mode=True)
        
        assert response == mock_response
        mock_client._client.request.assert_called_once()


@pytest.mark.asyncio
class TestAsyncAPIResourceRawResponse:
    """Test coverage for _arequest_raw_response method in AsyncAPIResource."""
    
    @pytest.fixture
    def mock_async_client(self):
        client = MagicMock(spec=AsyncVeniceClient)
        client._api_key = "test-api-key"
        client._base_url = httpx.URL("https://api.venice.ai/v1/")
        client._client = MagicMock(spec=httpx.AsyncClient)
        client._timeout = httpx.Timeout(60.0)
        # Mock the async error translation method
        client._translate_httpx_error_to_api_error = AsyncMock()
        return client
    
    async def test_arequest_raw_response_basic(self, mock_async_client):
        """Test basic _arequest_raw_response functionality."""
        resource = AsyncAPIResource(mock_async_client)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.content = b"raw async response data"
        mock_async_client._client.request = AsyncMock(return_value=mock_response)
        
        options = {
            "body": {"key": "async_value"},
            "headers": {"X-Async-Header": "value"},
            "timeout": 45.0
        }
        
        response = await resource._arequest_raw_response("POST", "async_endpoint", options=options)
        
        assert response == mock_response
        mock_async_client._client.request.assert_called_once()
        args, kwargs = mock_async_client._client.request.call_args
        assert kwargs["method"] == "POST"
        assert str(kwargs["url"]).endswith("async_endpoint")
        assert kwargs["json"] == {"key": "async_value"}
        assert kwargs["timeout"] == 45.0
        assert "X-Async-Header" in kwargs["headers"]
    
    async def test_arequest_raw_response_no_body(self, mock_async_client):
        """Test _arequest_raw_response without body."""
        resource = AsyncAPIResource(mock_async_client)
        mock_response = MagicMock(spec=httpx.Response)
        mock_async_client._client.request = AsyncMock(return_value=mock_response)
        
        options = {"headers": {"Accept": "text/plain"}}
        
        response = await resource._arequest_raw_response("GET", "text", options=options)
        
        assert response == mock_response
        args, kwargs = mock_async_client._client.request.call_args
        assert kwargs["json"] is None
        assert kwargs["headers"]["Accept"] == "text/plain"
    
    async def test_arequest_raw_response_default_timeout(self, mock_async_client):
        """Test _arequest_raw_response uses client default timeout when not specified."""
        resource = AsyncAPIResource(mock_async_client)
        mock_response = MagicMock(spec=httpx.Response)
        mock_async_client._client.request = AsyncMock(return_value=mock_response)
        
        options = {"body": {"async_data": "test"}}
        
        response = await resource._arequest_raw_response("POST", "endpoint", options=options)
        
        args, kwargs = mock_async_client._client.request.call_args
        assert kwargs["timeout"] == mock_async_client._timeout
    
    async def test_arequest_raw_response_header_merging(self, mock_async_client):
        """Test header merging in _arequest_raw_response."""
        resource = AsyncAPIResource(mock_async_client)
        mock_response = MagicMock(spec=httpx.Response)
        mock_async_client._client.request = AsyncMock(return_value=mock_response)
        
        # Set default headers on the async client
        mock_async_client._client.headers = {
            "Authorization": "Bearer async-token", 
            "User-Agent": "async-test-agent",
            "X-Default": "default-value"
        }
        
        options = {
            "headers": {
                "X-Custom": "custom-value", 
                "Authorization": "Bearer override-async-token",
                "X-Default": "override-value"
            }
        }
        
        response = await resource._arequest_raw_response("PUT", "update", options=options)
        
        args, kwargs = mock_async_client._client.request.call_args
        # Headers should be merged with provided headers taking precedence
        assert kwargs["headers"]["Authorization"] == "Bearer override-async-token"
        assert kwargs["headers"]["X-Custom"] == "custom-value"
        assert kwargs["headers"]["X-Default"] == "override-value"
        assert kwargs["headers"]["User-Agent"] == "async-test-agent"
    
    async def test_arequest_raw_response_http_status_error(self, mock_async_client):
        """Test _arequest_raw_response with HTTPStatusError."""
        resource = AsyncAPIResource(mock_async_client)
        
        # Create the error
        request = httpx.Request("POST", "https://api.venice.ai/v1/async_endpoint")
        response = httpx.Response(status_code=401, request=request)
        error = httpx.HTTPStatusError("Unauthorized", request=request, response=response)
        
        mock_async_client._client.request = AsyncMock(side_effect=error)
        
        # Mock the translation method to return an AuthenticationError
        mock_async_client._translate_httpx_error_to_api_error.return_value = AuthenticationError(
            message="Invalid API key",
            request=request,
            response=response
        )
        
        options = {"body": {"secure": "data"}}
        
        with pytest.raises(AuthenticationError) as exc_info:
            await resource._arequest_raw_response("POST", "async_endpoint", options=options)
        
        assert "Invalid API key" in str(exc_info.value)
        # Check that the translation method was called
        mock_async_client._translate_httpx_error_to_api_error.assert_called_once()
        # Verify the arguments
        call_args = mock_async_client._translate_httpx_error_to_api_error.call_args[0]
        assert isinstance(call_args[0], httpx.HTTPStatusError)
        assert call_args[1].method == "POST"
        assert "async_endpoint" in str(call_args[1].url)
    
    async def test_arequest_raw_response_request_error(self, mock_async_client):
        """Test _arequest_raw_response with RequestError."""
        resource = AsyncAPIResource(mock_async_client)
        
        # Create the error
        request = httpx.Request("GET", "https://api.venice.ai/v1/async_endpoint")
        error = httpx.RequestError("Async connection failed", request=request)
        
        mock_async_client._client.request = AsyncMock(side_effect=error)
        
        # Mock the translation method to return an APIConnectionError
        mock_async_client._translate_httpx_error_to_api_error.return_value = APIConnectionError(
            message="Async connection failed",
            request=request,
            response=None,
            original_error=error
        )
        
        options = {"headers": {"X-Async": "true"}}
        
        with pytest.raises(APIConnectionError) as exc_info:
            await resource._arequest_raw_response("GET", "async_endpoint", options=options)
        
        assert "Async connection failed" in str(exc_info.value)
        # Check that the translation method was called
        mock_async_client._translate_httpx_error_to_api_error.assert_called_once()
        # Verify the arguments
        call_args = mock_async_client._translate_httpx_error_to_api_error.call_args[0]
        assert isinstance(call_args[0], httpx.RequestError)
        assert call_args[1].method == "GET"
        assert "async_endpoint" in str(call_args[1].url)
    
    async def test_arequest_raw_response_no_translate_method(self, mock_async_client):
        """Test _arequest_raw_response when client doesn't have _translate_httpx_error_to_api_error."""
        resource = AsyncAPIResource(mock_async_client)
        
        # Remove the translation method
        delattr(mock_async_client, '_translate_httpx_error_to_api_error')
        
        # Create the error
        request = httpx.Request("POST", "https://api.venice.ai/v1/async_endpoint")
        error = httpx.RequestError("Async connection failed", request=request)
        
        mock_async_client._client.request = AsyncMock(side_effect=error)
        
        options = {"body": {"async_key": "async_value"}}
        
        # Should raise the original httpx error
        with pytest.raises(httpx.RequestError) as exc_info:
            await resource._arequest_raw_response("POST", "async_endpoint", options=options)
        
        assert "Async connection failed" in str(exc_info.value)
    
    async def test_arequest_raw_response_stream_mode(self, mock_async_client):
        """Test _arequest_raw_response with stream_mode parameter."""
        resource = AsyncAPIResource(mock_async_client)
        mock_response = MagicMock(spec=httpx.Response)
        mock_async_client._client.request = AsyncMock(return_value=mock_response)
        
        options = {"body": {"stream": True}}
        
        # Note: stream_mode parameter is accepted but not used in current implementation
        response = await resource._arequest_raw_response("POST", "stream", options=options, stream_mode=True)
        
        assert response == mock_response
        mock_async_client._client.request.assert_called_once()
    
    async def test_arequest_raw_response_timeout_error(self, mock_async_client):
        """Test _arequest_raw_response with timeout error."""
        resource = AsyncAPIResource(mock_async_client)
        
        # Create the timeout error
        request = httpx.Request("POST", "https://api.venice.ai/v1/slow_endpoint")
        error = httpx.TimeoutException("Request timed out", request=request)
        
        mock_async_client._client.request = AsyncMock(side_effect=error)
        
        # Mock the translation method to return an APITimeoutError
        mock_async_client._translate_httpx_error_to_api_error.return_value = APITimeoutError(
            message="Request timed out",
            request=request,
            response=None,
            original_error=error
        )
        
        options = {"body": {"data": "slow"}, "timeout": 1.0}
        
        with pytest.raises(APITimeoutError) as exc_info:
            await resource._arequest_raw_response("POST", "slow_endpoint", options=options)
        
        assert "Request timed out" in str(exc_info.value)
    
    async def test_arequest_raw_response_empty_options(self, mock_async_client):
        """Test _arequest_raw_response with empty options."""
        resource = AsyncAPIResource(mock_async_client)
        mock_response = MagicMock(spec=httpx.Response)
        mock_async_client._client.request = AsyncMock(return_value=mock_response)
        
        response = await resource._arequest_raw_response("GET", "simple", options={})
        
        assert response == mock_response
        args, kwargs = mock_async_client._client.request.call_args
        assert kwargs["json"] is None
        assert kwargs["timeout"] == mock_async_client._timeout