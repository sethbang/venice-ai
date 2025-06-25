import pytest
import httpx
import json
from typing import cast, Dict, Any
from unittest.mock import MagicMock, AsyncMock, patch

from venice_ai._async_client import AsyncVeniceClient
from venice_ai.exceptions import (
    APIError,
    AuthenticationError,
    PermissionDeniedError,
    InvalidRequestError,
    NotFoundError,
    ConflictError,
ServiceUnavailableError,
    UnprocessableEntityError,
    RateLimitError,
    InternalServerError,
    APITimeoutError,
    APIConnectionError,
)


class TestAsyncTranslateHttpxErrorToApiErrorComprehensive:
    """Comprehensive tests for _translate_httpx_error_to_api_error method in the asynchronous client."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code,expected_error_class",
        [
            (400, InvalidRequestError),
            (401, AuthenticationError),
            (403, PermissionDeniedError),
            (404, NotFoundError),
            (409, ConflictError),
            (413, InvalidRequestError),
            (415, InvalidRequestError),
            (422, UnprocessableEntityError),
            (429, RateLimitError),
            (500, InternalServerError),
            (502, InternalServerError),
            (503, ServiceUnavailableError),
            (504, InternalServerError),
            (418, APIError),  # Testing a non-standard status code
        ]
    )
    async def test_http_status_error_with_json_error_body(self, status_code, expected_error_class):
        """Test async _translate_httpx_error_to_api_error with HTTP status errors and JSON error body."""
        # Create client instance
        client = AsyncVeniceClient(api_key="test-api-key")
        
        # Setup request mock
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "GET"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
        
        # Setup response mock with JSON error body
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = status_code
        mock_response.headers = {}
        mock_response.json = AsyncMock(return_value={
            "error": {
                "message": f"Specific error for status {status_code}",
                "type": "invalid_request_error",
                "code": "some_code"
            }
        })
        mock_response.aread = AsyncMock()
        mock_response.aclose = AsyncMock()
        
        # Create HTTPStatusError
        http_error = httpx.HTTPStatusError(
            f"HTTP Error {status_code}",
            request=mock_request,
            response=mock_response
        )
        
        # Call async method and get error
        api_error = await client._translate_httpx_error_to_api_error(http_error, mock_request)
        
        # Verify correct error type and properties
        assert isinstance(api_error, expected_error_class)
        assert api_error.request is mock_request
        assert api_error.response is mock_response
        assert cast(APIError, api_error).status_code == status_code
        assert "Specific error for status" in str(api_error)
        assert "some_code" in str(api_error)
        assert cast(APIError, api_error).body is not None
        body = cast(Dict[str, Any], cast(APIError, api_error).body)
        assert "error" in body

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code,expected_error_class",
        [
            (400, InvalidRequestError),
            (401, AuthenticationError),
            (403, PermissionDeniedError),
            (404, NotFoundError),
            (429, RateLimitError),
            (500, InternalServerError),
        ]
    )
    async def test_http_status_error_with_json_nonstandard_error_body(self, status_code, expected_error_class):
        """Test async _translate_httpx_error_to_api_error with HTTP status errors and non-standard JSON error body."""
        # Create client instance
        client = AsyncVeniceClient(api_key="test-api-key")
        
        # Setup request mock
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "GET"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
        
        # Setup response mock with non-standard JSON error body
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = status_code
        mock_response.headers = {}
        mock_response.json = AsyncMock(return_value={
            "detail": f"Another error format for status {status_code}"
        })
        mock_response.aread = AsyncMock()
        mock_response.aclose = AsyncMock()
        
        # Create HTTPStatusError
        http_error = httpx.HTTPStatusError(
            f"HTTP Error {status_code}",
            request=mock_request,
            response=mock_response
        )
        
        # Call async method and get error
        api_error = await client._translate_httpx_error_to_api_error(http_error, mock_request)
        
        # Verify correct error type and properties
        assert isinstance(api_error, expected_error_class)
        assert api_error.request is mock_request
        assert api_error.response is mock_response
        assert cast(APIError, api_error).status_code == status_code
        assert cast(APIError, api_error).body is not None
        body = cast(Dict[str, Any], cast(APIError, api_error).body)
        assert "detail" in body

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code,expected_error_class",
        [
            (400, InvalidRequestError),
            (401, AuthenticationError),
            (404, NotFoundError),
            (500, InternalServerError),
        ]
    )
    async def test_http_status_error_with_plain_text_body(self, status_code, expected_error_class):
        """Test async _translate_httpx_error_to_api_error with HTTP status errors and plain text error body."""
        # Create client instance
        client = AsyncVeniceClient(api_key="test-api-key")
        
        # Setup request mock
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "GET"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
        
        # Setup response mock with plain text error body
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = status_code
        mock_response.headers = {}
        mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
        mock_response.aread = AsyncMock()
        mock_response.aclose = AsyncMock()
        mock_response.text = f"A plain text error occurred for status {status_code}"
        
        # Create HTTPStatusError
        http_error = httpx.HTTPStatusError(
            f"HTTP Error {status_code}",
            request=mock_request,
            response=mock_response
        )
        
        # Call async method and get error
        api_error = await client._translate_httpx_error_to_api_error(http_error, mock_request)
        
        # Verify correct error type and properties
        assert isinstance(api_error, expected_error_class)
        assert api_error.request is mock_request
        assert api_error.response is mock_response
        assert cast(APIError, api_error).status_code == status_code
        assert cast(APIError, api_error).body is not None
        assert "A plain text error occurred" in str(cast(APIError, api_error).body)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code,expected_error_class",
        [
            (400, InvalidRequestError),
            (429, RateLimitError),
            (500, InternalServerError),
        ]
    )
    async def test_http_status_error_with_unparseable_json_body(self, status_code, expected_error_class):
        """Test async _translate_httpx_error_to_api_error with HTTP status errors and unparseable JSON body."""
        # Create client instance
        client = AsyncVeniceClient(api_key="test-api-key")
        
        # Setup request mock
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "GET"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
        
        # Setup response mock with unparseable JSON body
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = status_code
        mock_response.headers = {}
        mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
        mock_response.aread = AsyncMock()
        mock_response.aclose = AsyncMock()
        mock_response.text = "not json{"
        
        # Create HTTPStatusError
        http_error = httpx.HTTPStatusError(
            f"HTTP Error {status_code}",
            request=mock_request,
            response=mock_response
        )
        
        # Call async method and get error
        api_error = await client._translate_httpx_error_to_api_error(http_error, mock_request)
        
        # Verify correct error type and properties
        assert isinstance(api_error, expected_error_class)
        assert api_error.request is mock_request
        assert api_error.response is mock_response
        assert cast(APIError, api_error).status_code == status_code
        assert cast(APIError, api_error).body is not None
        assert "not json{" in str(cast(APIError, api_error).body)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code,expected_error_class",
        [
            (400, InvalidRequestError),
            (500, InternalServerError),
        ]
    )
    async def test_http_status_error_with_empty_body(self, status_code, expected_error_class):
        """Test async _translate_httpx_error_to_api_error with HTTP status errors and empty body."""
        # Create client instance
        client = AsyncVeniceClient(api_key="test-api-key")
        
        # Setup request mock
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "GET"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
        
        # Setup response mock with empty body
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = status_code
        mock_response.headers = {}
        mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
        mock_response.aread = AsyncMock()
        mock_response.aclose = AsyncMock()
        mock_response.text = ""
        
        # Create HTTPStatusError
        http_error = httpx.HTTPStatusError(
            f"HTTP Error {status_code}",
            request=mock_request,
            response=mock_response
        )
        
        # Call async method and get error
        api_error = await client._translate_httpx_error_to_api_error(http_error, mock_request)
        
        # Verify correct error type and properties
        assert isinstance(api_error, expected_error_class)
        assert api_error.request is mock_request
        assert api_error.response is mock_response
        assert cast(APIError, api_error).status_code == status_code
        expected_message = f"API error {status_code} for {mock_request.method} {mock_request.url}"
        assert str(api_error) == expected_message

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code,expected_error_class",
        [
            (400, InvalidRequestError),
            (401, AuthenticationError),
            (429, RateLimitError),
            (500, InternalServerError),
        ]
    )
    async def test_http_status_error_streaming_response(self, status_code, expected_error_class):
        """Test async _translate_httpx_error_to_api_error with streaming HTTP status errors."""
        # Create client instance
        client = AsyncVeniceClient(api_key="test-api-key")
        
        # Setup request mock
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "POST"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
        
        # Setup response mock for streaming
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = status_code
        mock_response.headers = {}
        
        # For streaming case, we need to make json() an AsyncMock
        mock_response.json = AsyncMock(return_value={
            "error": {
                "message": f"Streaming error for status {status_code}",
                "type": "stream_error",
                "code": "stream_error_code"
            }
        })
        
        # Create HTTPStatusError
        http_error = httpx.HTTPStatusError(
            f"HTTP Error {status_code}",
            request=mock_request,
            response=mock_response
        )
        
        # Call async method with is_stream=True and get error
        api_error = await client._translate_httpx_error_to_api_error(http_error, mock_request, is_stream=True)
        
        # Verify correct error type and properties
        assert isinstance(api_error, expected_error_class)
        assert api_error.request is mock_request
        assert api_error.response is mock_response
        assert cast(APIError, api_error).status_code == status_code
        assert cast(APIError, api_error).body is not None
        body = cast(Dict[str, Any], cast(APIError, api_error).body)
        assert "error" in body
        assert "Streaming error for status" in str(api_error)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code,expected_error_class",
        [
            (400, InvalidRequestError),
            (500, InternalServerError),
        ]
    )
    async def test_http_status_error_streaming_response_json_fails(self, status_code, expected_error_class):
        """Test async _translate_httpx_error_to_api_error with streaming HTTP status errors when json() fails."""
        # Create client instance
        client = AsyncVeniceClient(api_key="test-api-key")
        
        # Setup request mock
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "POST"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
        
        # Setup response mock for streaming
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = status_code
        mock_response.headers = {}
        
        # For streaming case with JSON error
        mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
        mock_response.text = AsyncMock(return_value=f"Streaming plain text error for status {status_code}")
        
        # Create HTTPStatusError
        http_error = httpx.HTTPStatusError(
            f"HTTP Error {status_code}",
            request=mock_request,
            response=mock_response
        )
        
        # Call async method with is_stream=True and get error
        api_error = await client._translate_httpx_error_to_api_error(http_error, mock_request, is_stream=True)
        
        # Verify correct error type and properties
        assert isinstance(api_error, expected_error_class)
        assert api_error.request is mock_request
        assert api_error.response is mock_response
        assert cast(APIError, api_error).status_code == status_code
        assert cast(APIError, api_error).body is not None
        assert "Streaming plain text error for status" in str(cast(APIError, api_error).body)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code,expected_error_class",
        [
            (400, InvalidRequestError),
            (500, InternalServerError),
        ]
    )
    async def test_http_status_error_streaming_response_json_and_text_fail(self, status_code, expected_error_class):
        """Test async _translate_httpx_error_to_api_error with streaming HTTP status errors when both json() and text() fail."""
        # Create client instance
        client = AsyncVeniceClient(api_key="test-api-key")
        
        # Setup request mock
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "POST"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
        
        # Setup response mock for streaming
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = status_code
        mock_response.headers = {}
        
        # For streaming case with both JSON and text errors
        mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
        mock_response.text = AsyncMock(side_effect=Exception("Failed to get text"))
        
        # Create HTTPStatusError
        http_error = httpx.HTTPStatusError(
            f"HTTP Error {status_code}",
            request=mock_request,
            response=mock_response
        )
        
        # Call async method with is_stream=True and get error
        api_error = await client._translate_httpx_error_to_api_error(http_error, mock_request, is_stream=True)
        
        # Verify correct error type and properties
        assert isinstance(api_error, expected_error_class)
        assert api_error.request is mock_request
        assert api_error.response is mock_response
        assert cast(APIError, api_error).status_code == status_code
        assert f"API error {status_code}" in str(api_error)

    @pytest.mark.asyncio
    async def test_timeout_exception(self):
        """Test async _translate_httpx_error_to_api_error with TimeoutException."""
        # Create client instance
        client = AsyncVeniceClient(api_key="test-api-key")
        
        # Setup request mock
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "GET"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
        
        # Create TimeoutException
        timeout_error = httpx.TimeoutException("Timeout occurred", request=mock_request)
        
        # Call async method and get error
        api_error = await client._translate_httpx_error_to_api_error(timeout_error, mock_request)
        
        # Verify correct error type and properties
        assert isinstance(api_error, APITimeoutError)
        assert api_error.request is mock_request
        assert "Timeout occurred" in str(api_error)

    @pytest.mark.asyncio
    async def test_connect_error(self):
        """Test async _translate_httpx_error_to_api_error with ConnectError."""
        # Create client instance
        client = AsyncVeniceClient(api_key="test-api-key")
        
        # Setup request mock
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "GET"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
        
        # Create ConnectError
        connect_error = httpx.ConnectError("Connection failed", request=mock_request)
        
        # Call async method and get error
        api_error = await client._translate_httpx_error_to_api_error(connect_error, mock_request)
        
        # Verify correct error type and properties
        assert isinstance(api_error, APIConnectionError)
        assert api_error.request is mock_request
        assert "Connection failed" in str(api_error)

    @pytest.mark.asyncio
    async def test_generic_request_error(self):
        """Test async _translate_httpx_error_to_api_error with generic RequestError."""
        # Create client instance
        client = AsyncVeniceClient(api_key="test-api-key")
        
        # Setup request mock
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "GET"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
        
        # Create a generic RequestError that is neither a TimeoutException nor a ConnectError
        class CustomRequestError(httpx.RequestError):
            pass
        
        request_error = CustomRequestError("Generic request error", request=mock_request)
        
        # Call async method and get error
        api_error = await client._translate_httpx_error_to_api_error(request_error, mock_request)
        
        # Verify correct error type and properties
        assert isinstance(api_error, APIConnectionError)
        assert api_error.request is mock_request
        assert "Generic request error" in str(api_error)

    @pytest.mark.asyncio
    async def test_request_error_with_none_request_fallback(self):
        """Test async _translate_httpx_error_to_api_error with RequestError.request=None and fallback request."""
        # Create client instance
        client = AsyncVeniceClient(api_key="test-api-key")
        
        # Setup fallback request
        fallback_request = MagicMock(spec=httpx.Request)
        fallback_request.method = "POST"
        fallback_request.url = httpx.URL("https://api.venice.ai/api/v1/fallback_endpoint")
        
        # Create TimeoutException with request=None
        timeout_error = httpx.TimeoutException("Timeout occurred")
        timeout_error.request = None  # type: ignore[assignment]
        
        # Call async method and get error
        api_error = await client._translate_httpx_error_to_api_error(timeout_error, fallback_request)
        
        # Verify fallback request is used
        assert isinstance(api_error, APITimeoutError)
        assert api_error.request is fallback_request
        assert api_error.request is not None # Added assertion
        assert api_error.request.method == "POST"
        assert "fallback_endpoint" in str(api_error.request.url)

    @pytest.mark.asyncio
    async def test_timeout_exception_with_is_stream(self):
        """Test async _translate_httpx_error_to_api_error with TimeoutException and is_stream=True."""
        # Create client instance
        client = AsyncVeniceClient(api_key="test-api-key")
        
        # Setup request mock
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "POST"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
        
        # Create TimeoutException
        timeout_error = httpx.TimeoutException("Stream timeout occurred", request=mock_request)
        
        # Call async method with is_stream=True and get error
        api_error = await client._translate_httpx_error_to_api_error(timeout_error, mock_request, is_stream=True)
        
        # Verify correct error type and properties
        assert isinstance(api_error, APITimeoutError)
        assert api_error.request is mock_request
        assert "Stream request timed out" in str(api_error)
        assert "Stream timeout occurred" in str(api_error)

    @pytest.mark.asyncio
    async def test_connect_error_with_is_stream(self):
        """Test async _translate_httpx_error_to_api_error with ConnectError and is_stream=True."""
        # Create client instance
        client = AsyncVeniceClient(api_key="test-api-key")
        
        # Setup request mock
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "POST"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/chat/completions")
        
        # Create ConnectError
        connect_error = httpx.ConnectError("Stream connection failed", request=mock_request)
        
        # Call async method with is_stream=True and get error
        api_error = await client._translate_httpx_error_to_api_error(connect_error, mock_request, is_stream=True)
        
        # Verify correct error type and properties
        assert isinstance(api_error, APIConnectionError)
        assert api_error.request is mock_request
        assert "Stream request failed" in str(api_error)
        assert "Stream connection failed" in str(api_error)