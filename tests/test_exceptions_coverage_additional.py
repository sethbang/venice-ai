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
    RateLimitError, InternalServerError, _make_status_error, PaymentRequiredError, ServiceUnavailableError
)


class TestAPIErrorConstructor:
    """Tests specifically for APIError constructor coverage."""
    
    def test_api_error_init_with_response(self):
        """Test APIError constructor with complete initialization flow."""
        # Create mock response with status code
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.headers = {}
        
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
        mock_response.headers = {}
        
        # Test with a string body
        error = _make_status_error(None, body="This is a string", response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "HTTP Status 400" in error.message
    
    def test_make_status_error_with_dict_without_error_key(self):
        """Test _make_status_error with a dict that doesn't have an 'error' key."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.headers = {}
        
        body = {"message": "Invalid request", "code": "INVALID_REQUEST"}
        
        error = _make_status_error(None, body=body, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "HTTP Status 400" in error.message
    
    def test_make_status_error_with_non_dict_error_value(self):
        """Test _make_status_error where 'error' value is not a dictionary."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.headers = {}
        
        body = {"error": "Something went wrong"}
        
        error = _make_status_error(None, body=body, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "HTTP Status 400" in error.message
    
    def test_make_status_error_with_error_missing_keys(self):
        """Test _make_status_error where 'error' dict doesn't have expected keys."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.headers = {}
        
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
        mock_response.headers = {}
        
        error = _make_status_error("Permission denied", body=None, response=mock_response)
        
        assert isinstance(error, PermissionDeniedError)
        assert "Permission denied" in error.message
    
    def test_make_status_error_conflict_409(self):
        """Test status code 409 returning ConflictError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 409
        mock_response.headers = {}
        
        error = _make_status_error("Resource already exists", body=None, response=mock_response)
        
        assert isinstance(error, ConflictError)
        assert "Resource already exists" in error.message
    
    def test_make_status_error_file_size_413(self):
        """Test status code 413 returning InvalidRequestError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 413
        mock_response.headers = {}
        
        error = _make_status_error("File too large", body=None, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "File too large" in error.message
    
    def test_make_status_error_content_type_415(self):
        """Test status code 415 returning InvalidRequestError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 415
        mock_response.headers = {}
        
        error = _make_status_error("Unsupported media type", body=None, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "Unsupported media type" in error.message
    
    def test_make_status_error_unprocessable_entity_422(self):
        """Test status code 422 returning UnprocessableEntityError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 422
        mock_response.headers = {}
        
        error = _make_status_error("Invalid data format", body=None, response=mock_response)
        
        assert isinstance(error, UnprocessableEntityError)
        assert "Invalid data format" in error.message
    
    def test_make_status_error_rate_limit_429(self):
        """Test status code 429 returning RateLimitError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.headers = {}
        
        error = _make_status_error("Too many requests", body=None, response=mock_response)
        
        assert isinstance(error, RateLimitError)
        assert "Too many requests" in error.message
    
    def test_make_status_error_internal_server_error_5xx(self):
        """Test status codes 500+ returning InternalServerError."""
        # Test multiple 5xx status codes
        for status_code in [500, 502, 504]: # Removed 503 as it's now ServiceUnavailableError
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = status_code
            mock_response.headers = {}
            
            error = _make_status_error(f"Server error {status_code}", body=None, response=mock_response)
            
            assert isinstance(error, InternalServerError)
            assert f"Server error {status_code}" in error.message
        
        # Test 503 specifically for ServiceUnavailableError
        mock_response_503 = MagicMock(spec=httpx.Response)
        mock_response_503.status_code = 503
        mock_response_503.headers = {}
        error_503 = _make_status_error("Server error 503", body=None, response=mock_response_503)
        assert isinstance(error_503, ServiceUnavailableError)
        assert "Server error 503" in error_503.message
    
    def test_make_status_error_other_4xx(self):
        """Test other 4xx codes returning generic APIError."""
        # Test some unhandled 4xx status codes
        for status_code in [405, 418, 451]: # Removed 402 as it's now PaymentRequiredError
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = status_code
            mock_response.headers = {}
            
            error = _make_status_error(f"Error {status_code}", body=None, response=mock_response)
            
            assert isinstance(error, APIError)
            assert not isinstance(error, (
                InvalidRequestError, AuthenticationError, PermissionDeniedError,
                NotFoundError, ConflictError, UnprocessableEntityError, RateLimitError, PaymentRequiredError
            ))
            # The "Unhandled 4xx error" message is no longer added by default for generic APIError
            # assert "Unhandled 4xx error" in error.message
            assert f"Error {status_code}" in error.message

        # Test 402 specifically for PaymentRequiredError
        mock_response_402 = MagicMock(spec=httpx.Response)
        mock_response_402.status_code = 402
        mock_response_402.headers = {}
        error_402 = _make_status_error("Error 402", body=None, response=mock_response_402)
        assert isinstance(error_402, PaymentRequiredError)
        assert "Error 402" in error_402.message
    
    def test_make_status_error_unexpected_status(self):
        """Test any other status code returning generic APIError."""
        # Test with a status code outside the normal ranges
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 300  # Redirect
        mock_response.headers = {}
        
        error = _make_status_error("Unexpected status", body=None, response=mock_response)
        
        assert isinstance(error, APIError)
        assert not isinstance(error, (
            InvalidRequestError, AuthenticationError, PermissionDeniedError,
            NotFoundError, ConflictError, UnprocessableEntityError, RateLimitError,
            InternalServerError
        ))
        assert "Unexpected status" in error.message
        
    def test_make_status_error_edge_cases(self):
        """Test edge case status codes to ensure full coverage."""
        # Test with a status code outside all handled ranges (to hit line 155)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 199  # Below 400, not a 4xx or 5xx
        mock_response.headers = {}
        
        error = _make_status_error("Edge case status", body=None, response=mock_response)
        
        assert isinstance(error, APIError)
        assert not isinstance(error, (
            InvalidRequestError, AuthenticationError, PermissionDeniedError,
            NotFoundError, ConflictError, UnprocessableEntityError, RateLimitError,
            InternalServerError
        ))
        assert "Edge case status" in error.message
        assert error.__class__ == APIError  # Not a subclass, but APIError itself


class TestMakeStatusErrorComprehensive:
    """Comprehensive tests for complex scenarios in _make_status_error."""
    
    def test_make_status_error_with_complete_error_data(self):
        """Test with complete error data including code, message and custom message."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.headers = {}
        
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
        mock_response.headers = {}
        
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
        mock_response.headers = {}
        
        error = _make_status_error(None, body=None, response=mock_response)
        
        assert isinstance(error, InternalServerError)
        assert "HTTP Status 500" in error.message
        
    def test_edge_case_combinations(self):
        """Test combinations of edge cases to ensure complete coverage."""
        # Test with no message but with complex error dict
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.headers = {}
        
        body = {"error": {"code": "COMPLEX_ERROR", "detail": None, "message": None}}
        
        error = _make_status_error(None, body=body, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "HTTP Status 400" in error.message
        assert "COMPLEX_ERROR" in error.message
        
        # Test with None values in error dict
        body = {"error": {"code": None}}
        error = _make_status_error(None, body=body, response=mock_response)
        assert isinstance(error, InvalidRequestError)
        assert "HTTP Status 400" in error.message
        
    def test_very_specific_edge_cases(self):
        """Test even more specific edge cases to achieve 100% coverage."""
        # Test with status code that falls through all conditionals
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 100  # Not 4xx or 5xx
        mock_response.headers = {}
        
        error = _make_status_error("Fallback error", body=None, response=mock_response)
        
        assert isinstance(error, APIError)
        assert error.__class__ is APIError  # Exactly APIError, not a subclass
        assert "Fallback error" in error.message
        
        # Test error dict with both message and detail being None
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.headers = {}
        
        body = {"error": {"message": None, "detail": None, "code": "TEST_CODE"}}
        
        error = _make_status_error("Error with empty details", body=body, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "Error with empty details" in error.message
        assert "TEST_CODE" in error.message