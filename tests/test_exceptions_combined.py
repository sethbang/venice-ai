"""
Additional targeted tests to improve code coverage for venice_ai.exceptions.

This test suite specifically addresses coverage gaps indicated in the coverage
report, focusing on:
1. Direct instantiation of APIError (constructor coverage)
2. Various response body structures in _make_status_error
3. Complete coverage of all HTTP status code branches in _make_status_error
"""

import pytest
import httpx
from unittest.mock import MagicMock, patch

from venice_ai.exceptions import (
    VeniceError, APIError, AuthenticationError, PermissionDeniedError,
    InvalidRequestError, NotFoundError, ConflictError, UnprocessableEntityError,
    RateLimitError, InternalServerError, _make_status_error
)


class TestAPIErrorConstructor:
    """Tests specifically for APIError constructor coverage."""
    
    def test_api_error_init_with_response(self):
        """Test APIError constructor with complete initialization flow."""
        # Create mock response with status code
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        
        # Initialize APIError directly
        error = APIError("Rate limit exceeded", response=mock_response)
        
        # Verify that super().__init__ was called and attributes were set correctly
        assert error.message == "Rate limit exceeded"
        assert error.response == mock_response
        assert error.status_code == 429
        assert str(error) == "Rate limit exceeded"


class TestMakeStatusErrorBodyParsing:
    """Tests for _make_status_error body parsing logic."""
    
    def test_make_status_error_with_non_dict_body(self):
        """Test _make_status_error with a body that's not a dictionary."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        
        # Test with a string body
        error = _make_status_error(None, body="This is a string", response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "HTTP Status 400" in error.message
    
    def test_make_status_error_with_dict_without_error_key(self):
        """Test _make_status_error with a dict that doesn't have an 'error' key."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        
        body = {"message": "Invalid request", "code": "INVALID_REQUEST"}
        
        error = _make_status_error(None, body=body, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "HTTP Status 400" in error.message
    
    def test_make_status_error_with_non_dict_error_value(self):
        """Test _make_status_error where 'error' value is not a dictionary."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        
        body = {"error": "Something went wrong"}
        
        error = _make_status_error(None, body=body, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "HTTP Status 400" in error.message
    
    def test_make_status_error_with_error_missing_keys(self):
        """Test _make_status_error where 'error' dict doesn't have expected keys."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        
        # Error dict with neither 'message', 'detail', nor 'code'
        body = {"error": {"unexpected_key": "unexpected_value"}}
        
        error = _make_status_error(None, body=body, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "HTTP Status 400" in error.message


class TestMakeStatusErrorStatusCodes:
    """Comprehensive tests for all status code branches in _make_status_error."""
    
    def test_make_status_error_permission_denied_403(self):
        """Test status code 403 returning PermissionDeniedError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 403
        
        error = _make_status_error("Permission denied", body=None, response=mock_response)
        
        assert isinstance(error, PermissionDeniedError)
        assert "Permission denied" in error.message
    
    def test_make_status_error_conflict_409(self):
        """Test status code 409 returning ConflictError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 409
        
        error = _make_status_error("Resource already exists", body=None, response=mock_response)
        
        assert isinstance(error, ConflictError)
        assert "Resource already exists" in error.message
    
    def test_make_status_error_file_size_413(self):
        """Test status code 413 returning InvalidRequestError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 413
        
        error = _make_status_error("File too large", body=None, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "File too large" in error.message
    
    def test_make_status_error_content_type_415(self):
        """Test status code 415 returning InvalidRequestError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 415
        
        error = _make_status_error("Unsupported media type", body=None, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "Unsupported media type" in error.message
    
    def test_make_status_error_unprocessable_entity_422(self):
        """Test status code 422 returning UnprocessableEntityError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 422
        
        error = _make_status_error("Invalid data format", body=None, response=mock_response)
        
        assert isinstance(error, UnprocessableEntityError)
        assert "Invalid data format" in error.message
    
    def test_make_status_error_rate_limit_429(self):
        """Test status code 429 returning RateLimitError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        
        error = _make_status_error("Too many requests", body=None, response=mock_response)
        
        assert isinstance(error, RateLimitError)
        assert "Too many requests" in error.message
    
    def test_make_status_error_internal_server_error_5xx(self):
        """Test status codes 500+ returning InternalServerError."""
        # Test multiple 5xx status codes
        for status_code in [500, 502, 503, 504]:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = status_code
            
            error = _make_status_error(f"Server error {status_code}", body=None, response=mock_response)
            
            assert isinstance(error, InternalServerError)
            assert f"Server error {status_code}" in error.message
    
    def test_make_status_error_other_4xx(self):
        """Test other 4xx codes returning generic APIError."""
        # Test some unhandled 4xx status codes
        for status_code in [402, 405, 418, 451]:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = status_code
            
            error = _make_status_error(f"Error {status_code}", body=None, response=mock_response)
            
            assert isinstance(error, APIError)
            assert not isinstance(error, (
                InvalidRequestError, AuthenticationError, PermissionDeniedError,
                NotFoundError, ConflictError, UnprocessableEntityError, RateLimitError
            ))
            assert "Unhandled 4xx error" in error.message
    
    def test_make_status_error_unexpected_status(self):
        """Test any other status code returning generic APIError."""
        # Test with a status code outside the normal ranges
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 300  # Redirect
        
        error = _make_status_error("Unexpected status", body=None, response=mock_response)
        
        assert isinstance(error, APIError)
        assert not isinstance(error, (
            InvalidRequestError, AuthenticationError, PermissionDeniedError,
            NotFoundError, ConflictError, UnprocessableEntityError, RateLimitError,
            InternalServerError
        ))
        assert "Unexpected status" in error.message


class TestMakeStatusErrorComprehensive:
    """Comprehensive tests for complex scenarios in _make_status_error."""
    
    def test_make_status_error_with_complete_error_data(self):
        """Test with complete error data including code, message and custom message."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        
        body = {"error": {"code": "INVALID_PARAMETER", "message": "Parameter 'x' is invalid"}}
        
        error = _make_status_error("Bad request", body=body, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "Bad request" in error.message
        assert "Parameter 'x' is invalid" in error.message
        assert "INVALID_PARAMETER" in error.message
    
    def test_make_status_error_with_detail_instead_of_message(self):
        """Test error with detail field instead of message field."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        
        body = {"error": {"code": "VALIDATION_ERROR", "detail": "Validation failed"}}
        
        error = _make_status_error(None, body=body, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "HTTP Status 400" in error.message
        assert "Validation failed" in error.message
        assert "VALIDATION_ERROR" in error.message
    
    def test_make_status_error_null_body(self):
        """Test with null body to ensure fallback to default message."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        
        error = _make_status_error(None, body=None, response=mock_response)
        
        assert isinstance(error, InternalServerError)
        assert "HTTP Status 500" in error.message
import pytest
import httpx
import json
from unittest.mock import MagicMock, patch

from venice_ai.exceptions import (
    VeniceError, APIError, AuthenticationError, PermissionDeniedError,
    InvalidRequestError, NotFoundError, ConflictError, UnprocessableEntityError,
    RateLimitError, InternalServerError, _make_status_error
)

class TestVeniceError:
    def test_initialization(self):
        """Test basic initialization of VeniceError."""
        error = VeniceError("Test error message")
        assert error.message == "Test error message"
        assert error.response is None

    def test_initialization_with_response(self):
        """Test initialization with a response object."""
        mock_response = MagicMock(spec=httpx.Response)
        error = VeniceError("Test error message", response=mock_response)
        assert error.message == "Test error message"
        assert error.response == mock_response

    def test_str_representation(self):
        """Test string representation of the error."""
        error = VeniceError("Test error message")
        assert str(error) == "Test error message"

class TestAPIError:
    def test_initialization(self):
        """Test initialization of APIError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        error = APIError("Test API error", response=mock_response)
        assert error.message == "Test API error"
        assert error.response == mock_response
        assert error.status_code == 500

class TestMakeStatusError:
    @pytest.mark.parametrize("status_code,error_class", [
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
        (503, InternalServerError),
        (450, APIError),  # Unhandled 4xx
        (550, InternalServerError),  # Unhandled 5xx
    ])
    def test_make_status_error_status_codes(self, status_code, error_class):
        """Test _make_status_error produces correct error types for different status codes."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = status_code
        
        error = _make_status_error(None, body=None, response=mock_response)
        
        assert isinstance(error, error_class)
        assert mock_response.status_code == status_code

    def test_make_status_error_with_message(self):
        """Test providing a message to _make_status_error."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        
        error = _make_status_error("Custom error message", body=None, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "Custom error message" in str(error)

    def test_make_status_error_with_body_string(self):
        """Test _make_status_error with error body containing message."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        
        body = {"error": {"message": "Invalid parameter", "code": "INVALID_PARAMETER"}}
        
        error = _make_status_error(None, body=body, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "Invalid parameter" in str(error)
        assert "INVALID_PARAMETER" in str(error)

    def test_make_status_error_with_body_detail(self):
        """Test _make_status_error with error body containing detail instead of message."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        
        body = {"error": {"detail": "Parameter validation failed", "code": "VALIDATION_ERROR"}}
        
        error = _make_status_error(None, body=body, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "Parameter validation failed" in str(error)
        assert "VALIDATION_ERROR" in str(error)

    def test_make_status_error_with_invalid_body(self):
        """Test _make_status_error with invalid error body structure."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        
        body = {"message": "This is not in the expected format"}
        
        error = _make_status_error(None, body=body, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "HTTP Status 400" in str(error)  # Should fall back to default message

    def test_make_status_error_unhandled_4xx(self):
        """Test _make_status_error for an unhandled 4xx status code."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 418  # I'm a teapot - an unhandled 4xx
        
        error = _make_status_error(None, body=None, response=mock_response)
        
        assert isinstance(error, APIError)
        assert not isinstance(error, (InvalidRequestError, AuthenticationError, PermissionDeniedError, NotFoundError, ConflictError, UnprocessableEntityError, RateLimitError))
        assert "Unhandled 4xx error" in str(error)
        assert "HTTP Status 418" in str(error)
        assert error.status_code == 418

    def test_make_status_error_unhandled_5xx_not_internal_server_error(self):
        """Test _make_status_error for a 5xx status code that isn't 500 but should be InternalServerError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 502  # Bad Gateway
        
        error = _make_status_error(None, body=None, response=mock_response)
        
        assert isinstance(error, InternalServerError)
        assert "HTTP Status 502" in str(error)
        assert error.status_code == 502
        
    def test_make_status_error_with_non_dict_body(self):
        """Test _make_status_error with non-dictionary body."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        
        # Test with string body
        error = _make_status_error(None, body="This is a string, not a dict", response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "HTTP Status 400" in str(error)
        
        # Test with None body
        error = _make_status_error(None, body=None, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "HTTP Status 400" in str(error)
        
        # Test with list body
        error = _make_status_error(None, body=["item1", "item2"], response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "HTTP Status 400" in str(error)

    def test_make_status_error_with_incomplete_error_data(self):
        """Test _make_status_error with incomplete error data structure."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        
        # Error without 'message' or 'detail'
        body = {"error": {"code": "SOME_ERROR_CODE"}}
        
        error = _make_status_error(None, body=body, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "HTTP Status 400" in str(error)
        assert "SOME_ERROR_CODE" in str(error)
        
        # Error with empty error object
        body = {"error": {}}
        
        error = _make_status_error(None, body=body, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "HTTP Status 400" in str(error)
        
    def test_make_status_error_error_with_code_no_message(self):
        """Test _make_status_error when error has only a code but no message or detail."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        
        body = {"error": {"code": "ERROR_CODE_ONLY"}}
        
        error = _make_status_error(None, body=body, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "HTTP Status 400" in str(error)
        assert "ERROR_CODE_ONLY" in str(error)
        assert "Code: ERROR_CODE_ONLY" in str(error)
        
    def test_make_status_error_with_custom_message_and_body(self):
        """Test _make_status_error with both custom message and body."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404
        
        body = {"error": {"message": "Resource not found", "code": "NOT_FOUND"}}
        
        error = _make_status_error("Custom not found message", body=body, response=mock_response)
        
        assert isinstance(error, NotFoundError)
        assert "Custom not found message" in str(error)
        assert "Resource not found" in str(error)
        assert "NOT_FOUND" in str(error)
        
    def test_make_status_error_with_non_error_dict_body(self):
        """Test _make_status_error with a dict body that doesn't have the error key."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        
        body = {"message": "Server error occurred", "status": "failed"}
        
        error = _make_status_error(None, body=body, response=mock_response)
        
        assert isinstance(error, InternalServerError)
        assert "HTTP Status 500" in str(error)
        # Should not contain any parsed error details
        assert "Server error occurred" not in str(error)
    def test_api_error_direct_initialization(self):
        """Test direct initialization of APIError class."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        error = APIError("Direct initialization test", response=mock_response)
        assert isinstance(error, APIError)
        assert error.message == "Direct initialization test"
        assert error.response == mock_response
        assert error.status_code == 500
def test_make_status_error_non_json_response_body():
    """Test _make_status_error when response body is not valid JSON."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 400
    mock_response.text = "This is not JSON"

    # Pass None as body to simulate a case where json parsing failed
    error = _make_status_error(None, body=None, response=mock_response)

    assert isinstance(error, InvalidRequestError)
    # The error message should indicate the status code
    assert "HTTP Status 400" in str(error)
    # Verify the message doesn't contain any parsed error details
    assert "This is not JSON" not in str(error)  # The raw text shouldn't be in the error message
def test_api_error_subclasses_direct_initialization():
    """Test direct initialization of all APIError subclasses with a response."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 400
    
    # Test classes that inherit from VeniceError (response is optional)
    venice_subclasses = [
        AuthenticationError,
        PermissionDeniedError,
        InvalidRequestError,
        NotFoundError,
        ConflictError,
        UnprocessableEntityError,
        RateLimitError
    ]
    
    for error_class in venice_subclasses:
        # Test with response
        error = error_class(f"Direct {error_class.__name__} test", response=mock_response)
        assert isinstance(error, error_class)
        assert isinstance(error, VeniceError)
        assert f"Direct {error_class.__name__} test" in str(error)
        assert error.response == mock_response
    
    # Test InternalServerError separately as it inherits from APIError (response is required)
    error = InternalServerError(f"Direct InternalServerError test", response=mock_response)
    assert isinstance(error, InternalServerError)
    assert isinstance(error, APIError)
    assert "Direct InternalServerError test" in str(error)
    assert error.response == mock_response
    assert error.status_code == 400

# Add comprehensive tests for error class hierarchy and edge cases
class TestErrorHierarchy:
    """Tests for error class hierarchy and additional edge cases."""
    
    def test_venice_error_inheritance(self):
        """Test that all error classes proper inherit from VeniceError."""
        error_classes = [
            APIError,
            AuthenticationError,
            PermissionDeniedError,
            InvalidRequestError,
            NotFoundError,
            ConflictError,
            UnprocessableEntityError,
            RateLimitError,
            InternalServerError
        ]
        
        for error_class in error_classes:
            assert issubclass(error_class, VeniceError)
    
    def test_internal_server_error_api_error_inheritance(self):
        """Test that InternalServerError properly inherits from APIError."""
        assert issubclass(InternalServerError, APIError)
    
    def test_venice_error_exception_inheritance(self):
        """Test that VeniceError inherits from Exception."""
        assert issubclass(VeniceError, Exception)
        
    def test_error_str_representation_consistency(self):
        """Test string representation consistency across all error types."""
        test_message = "Test error message"
        
        # VeniceError - base class
        venice_error = VeniceError(test_message)
        assert str(venice_error) == test_message
        
        # APIError - requires response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        api_error = APIError(test_message, response=mock_response)
        assert str(api_error) == test_message
        
        # Other error classes
        error_classes = [
            AuthenticationError,
            PermissionDeniedError,
            InvalidRequestError,
            NotFoundError,
            ConflictError,
            UnprocessableEntityError,
            RateLimitError
        ]
        
        for error_class in error_classes:
            # Create a mock response with appropriate status code for each error class
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 500  # Default status code
            
            # Use the same mock response for all errors
            error = error_class(test_message, response=mock_response, body=None)
            assert str(error) == test_message