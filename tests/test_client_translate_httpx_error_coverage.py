import pytest
import httpx
import json
from typing import cast, Dict, Any
from unittest.mock import MagicMock, patch

from venice_ai._client import VeniceClient
from venice_ai.exceptions import (
    APIError,
    AuthenticationError,
    PermissionDeniedError,
    InvalidRequestError,
    NotFoundError,
    ConflictError,
    UnprocessableEntityError,
    RateLimitError,
ServiceUnavailableError,
    InternalServerError,
    APITimeoutError,
    APIConnectionError,
)


class TestTranslateHttpxErrorToApiErrorComprehensive:
    """Comprehensive tests for _translate_httpx_error_to_api_error method in the synchronous client."""

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
    def test_http_status_error_with_json_error_body(self, status_code, expected_error_class):
        """Test _translate_httpx_error_to_api_error with HTTP status errors and JSON error body."""
        # Setup request mock
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "GET"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
        
        # Setup response mock with JSON error body
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = status_code
        mock_response.headers = {}
        mock_response.json.return_value = {
            "error": {
                "message": f"Specific error for status {status_code}",
                "type": "invalid_request_error",
                "code": "some_code"
            }
        }
        
        # Create HTTPStatusError
        http_error = httpx.HTTPStatusError(
            f"HTTP Error {status_code}",
            request=mock_request,
            response=mock_response
        )
        
        # Initialize client and translate error
        client = VeniceClient(api_key="test-api-key")
        api_error = client._translate_httpx_error_to_api_error(http_error, mock_request)
        
        # Verify correct error type and properties
        assert isinstance(api_error, expected_error_class)
        assert api_error.request is mock_request # Changed to check api_error.request
        assert api_error.response is mock_response
        assert cast(APIError, api_error).status_code == status_code
        assert "Specific error for status" in str(api_error)
        assert "some_code" in str(api_error)
        assert cast(APIError, api_error).body is not None
        body = cast(Dict[str, Any], cast(APIError, api_error).body)
        assert "error" in body

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
    def test_http_status_error_with_json_nonstandard_error_body(self, status_code, expected_error_class):
        """Test _translate_httpx_error_to_api_error with HTTP status errors and non-standard JSON error body."""
        # Setup request mock
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "GET"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
        
        # Setup response mock with non-standard JSON error body
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = status_code
        mock_response.headers = {}
        mock_response.json.return_value = {
            "detail": f"Another error format for status {status_code}"
        }
        
        # Create HTTPStatusError
        http_error = httpx.HTTPStatusError(
            f"HTTP Error {status_code}",
            request=mock_request,
            response=mock_response
        )
        
        # Initialize client and translate error
        client = VeniceClient(api_key="test-api-key")
        api_error = client._translate_httpx_error_to_api_error(http_error, mock_request)
        
        # Verify correct error type and properties
        assert isinstance(api_error, expected_error_class)
        assert api_error.request is mock_request # Changed to check api_error.request
        assert api_error.response is mock_response
        assert cast(APIError, api_error).status_code == status_code
        assert cast(APIError, api_error).body is not None
        body = cast(Dict[str, Any], cast(APIError, api_error).body)
        assert "detail" in body

    @pytest.mark.parametrize(
        "status_code,expected_error_class",
        [
            (400, InvalidRequestError),
            (401, AuthenticationError),
            (404, NotFoundError),
            (500, InternalServerError),
        ]
    )
    def test_http_status_error_with_plain_text_body(self, status_code, expected_error_class):
        """Test _translate_httpx_error_to_api_error with HTTP status errors and plain text error body."""
        # Setup request mock
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "GET"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
        
        # Setup response mock with plain text error body
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = status_code
        mock_response.headers = {}
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.text = f"A plain text error occurred for status {status_code}"
        
        # Create HTTPStatusError
        http_error = httpx.HTTPStatusError(
            f"HTTP Error {status_code}",
            request=mock_request,
            response=mock_response
        )
        
        # Initialize client and translate error
        client = VeniceClient(api_key="test-api-key")
        api_error = client._translate_httpx_error_to_api_error(http_error, mock_request)
        
        # Verify correct error type and properties
        assert isinstance(api_error, expected_error_class)
        assert api_error.request is mock_request # Changed to check api_error.request
        assert api_error.response is mock_response
        assert cast(APIError, api_error).status_code == status_code
        assert cast(APIError, api_error).body is not None
        assert "A plain text error occurred" in str(cast(APIError, api_error).body)

    @pytest.mark.parametrize(
        "status_code,expected_error_class",
        [
            (400, InvalidRequestError),
            (429, RateLimitError),
            (500, InternalServerError),
        ]
    )
    def test_http_status_error_with_unparseable_json_body(self, status_code, expected_error_class):
        """Test _translate_httpx_error_to_api_error with HTTP status errors and unparseable JSON body."""
        # Setup request mock
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "GET"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
        
        # Setup response mock with unparseable JSON body
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = status_code
        mock_response.headers = {}
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.text = "not json{"
        
        # Create HTTPStatusError
        http_error = httpx.HTTPStatusError(
            f"HTTP Error {status_code}",
            request=mock_request,
            response=mock_response
        )
        
        # Initialize client and translate error
        client = VeniceClient(api_key="test-api-key")
        api_error = client._translate_httpx_error_to_api_error(http_error, mock_request)
        
        # Verify correct error type and properties
        assert isinstance(api_error, expected_error_class)
        assert api_error.request is mock_request # Changed to check api_error.request
        assert api_error.response is mock_response
        assert cast(APIError, api_error).status_code == status_code
        assert cast(APIError, api_error).body is not None
        assert "not json{" in str(cast(APIError, api_error).body)

    @pytest.mark.parametrize(
        "status_code,expected_error_class",
        [
            (400, InvalidRequestError),
            (500, InternalServerError),
        ]
    )
    def test_http_status_error_with_empty_body(self, status_code, expected_error_class):
        """Test _translate_httpx_error_to_api_error with HTTP status errors and empty body."""
        # Setup request mock
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "GET"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
        
        # Setup response mock with empty body
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = status_code
        mock_response.headers = {}
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.text = ""
        
        # Create HTTPStatusError
        http_error = httpx.HTTPStatusError(
            f"HTTP Error {status_code}",
            request=mock_request,
            response=mock_response
        )
        
        # Initialize client and translate error
        client = VeniceClient(api_key="test-api-key")
        api_error = client._translate_httpx_error_to_api_error(http_error, mock_request)
        
        # Verify correct error type and properties
        assert isinstance(api_error, expected_error_class)
        assert api_error.request is mock_request # Changed to check api_error.request
        assert api_error.response is mock_response
        assert cast(APIError, api_error).status_code == status_code
        # Check that the error message contains the core information (status code and API error indication)
        # The exact format can be "API error {status_code} for {METHOD} {URL}"
        assert f"API error {status_code}" in str(api_error)
        assert f"for {mock_request.method} {mock_request.url}" in str(api_error)

    def test_timeout_exception(self):
        """Test _translate_httpx_error_to_api_error with TimeoutException."""
        # Setup request mock
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "GET"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
        
        # Create TimeoutException
        timeout_error = httpx.TimeoutException("Timeout occurred", request=mock_request)
        
        # Initialize client and translate error
        client = VeniceClient(api_key="test-api-key")
        api_error = client._translate_httpx_error_to_api_error(timeout_error, mock_request)
        
        # Verify correct error type and properties
        assert isinstance(api_error, APITimeoutError)
        assert api_error.request is mock_request
        assert api_error.request is not None # Assertion added
        assert "Timeout occurred" in str(api_error)

    def test_connect_error(self):
        """Test _translate_httpx_error_to_api_error with ConnectError."""
        # Setup request mock
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "GET"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
        
        # Create ConnectError
        connect_error = httpx.ConnectError("Connection failed", request=mock_request)
        
        # Initialize client and translate error
        client = VeniceClient(api_key="test-api-key")
        api_error = client._translate_httpx_error_to_api_error(connect_error, mock_request)
        
        # Verify correct error type and properties
        assert isinstance(api_error, APIConnectionError)
        assert api_error.request is mock_request
        assert api_error.request is not None # Assertion added
        assert "Connection failed" in str(api_error)

    def test_generic_request_error(self):
        """Test _translate_httpx_error_to_api_error with generic RequestError."""
        # Setup request mock
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "GET"
        mock_request.url = httpx.URL("https://api.venice.ai/api/v1/test_endpoint")
        
        # Create a generic RequestError that is neither a TimeoutException nor a ConnectError
        class CustomRequestError(httpx.RequestError):
            pass
        
        request_error = CustomRequestError("Generic request error", request=mock_request)
        
        # Initialize client and translate error
        client = VeniceClient(api_key="test-api-key")
        api_error = client._translate_httpx_error_to_api_error(request_error, mock_request)
        
        # Verify correct error type and properties
        assert isinstance(api_error, APIConnectionError)
        assert api_error.request is mock_request
        assert api_error.request is not None # Assertion added
        assert "Generic request error" in str(api_error)

    def test_request_error_with_none_request_fallback(self):
        """Test _translate_httpx_error_to_api_error with RequestError.request=None and fallback request."""
        # Setup fallback request
        fallback_request = MagicMock(spec=httpx.Request)
        fallback_request.method = "POST"
        fallback_request.url = httpx.URL("https://api.venice.ai/api/v1/fallback_endpoint")
        
        # Create TimeoutException with request=None
        timeout_error = httpx.TimeoutException("Timeout occurred")
        timeout_error.request = None  # type: ignore[assignment]
        
        # Initialize client and translate error
        client = VeniceClient(api_key="test-api-key")
        api_error = client._translate_httpx_error_to_api_error(timeout_error, fallback_request)
        
        # Verify fallback request is used
        assert isinstance(api_error, APITimeoutError)
        assert api_error.request is fallback_request
        assert api_error.request is not None # Assertion added
        assert api_error.request.method == "POST"
        assert "fallback_endpoint" in str(api_error.request.url)