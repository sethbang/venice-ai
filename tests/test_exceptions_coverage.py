"""
Additional targeted tests for venice_ai.exceptions to improve code coverage.

This module focuses specifically on improving test coverage for the exceptions.py module,
especially the _make_status_error function and various exception classes.
"""

import pytest
import httpx
import json
from unittest.mock import MagicMock

from venice_ai.exceptions import (
    VeniceError,
    APIError,
    AuthenticationError,
    PermissionDeniedError,
    InvalidRequestError,
    NotFoundError,
    ConflictError,
    UnprocessableEntityError,
    RateLimitError,
    InternalServerError,
    _make_status_error
)


class TestExceptionsCoverage:
    """Tests specifically targeting coverage gaps in the exceptions module."""

    def test_api_error_init(self):
        """Test APIError constructor (lines 26-27)."""
        # Create mock response with status code
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 418  # I'm a teapot
        mock_response.headers = {}

        # Initialize APIError
        error = APIError("Test API error", response=mock_response)

        # Verify attributes were set correctly
        assert error.message == "Test API error"
        assert error.response == mock_response
        assert error.status_code == 418

    def test_make_status_error_with_error_dict(self):
        """Test _make_status_error with dict error data (lines 123-128)."""
        # Create mock response with status code
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.headers = {}

        # Create body with error dict containing code and message
        body = {
            "error": {
                "code": "INVALID_PARAMETER",
                "message": "The parameter 'x' is invalid"
            }
        }

        # Call _make_status_error
        error = _make_status_error(None, body=body, response=mock_response)

        # Verify the error message includes the details from the error dict
        assert isinstance(error, InvalidRequestError)
        assert "The parameter 'x' is invalid" in error.message
        assert "Code: INVALID_PARAMETER" in error.message

    def test_make_status_error_permission_denied(self):
        """Test _make_status_error with 403 status code (line 135)."""
        # Create mock response with 403 status code
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 403
        mock_response.headers = {}

        # Call _make_status_error
        error = _make_status_error("Permission denied", body=None, response=mock_response)

        # Verify correct error type is returned
        assert isinstance(error, PermissionDeniedError)
        assert "Permission denied" in error.message

    def test_make_status_error_conflict(self):
        """Test _make_status_error with 409 status code (line 139)."""
        # Create mock response with 409 status code
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 409
        mock_response.headers = {}

        # Call _make_status_error
        error = _make_status_error("Resource already exists", body=None, response=mock_response)

        # Verify correct error type is returned
        assert isinstance(error, ConflictError)
        assert "Resource already exists" in error.message

    def test_make_status_error_file_size(self):
        """Test _make_status_error with 413 status code (line 141)."""
        # Create mock response with 413 status code
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 413
        mock_response.headers = {}

        # Call _make_status_error
        error = _make_status_error("File too large", body=None, response=mock_response)

        # Verify correct error type is returned
        assert isinstance(error, InvalidRequestError)
        assert "File too large" in error.message

    def test_make_status_error_content_type(self):
        """Test _make_status_error with 415 status code (line 143)."""
        # Create mock response with 415 status code
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 415
        mock_response.headers = {}

        # Call _make_status_error
        error = _make_status_error("Unsupported media type", body=None, response=mock_response)

        # Verify correct error type is returned
        assert isinstance(error, InvalidRequestError)
        assert "Unsupported media type" in error.message

    def test_make_status_error_unprocessable_entity(self):
        """Test _make_status_error with 422 status code (line 145)."""
        # Create mock response with 422 status code
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 422
        mock_response.headers = {}

        # Call _make_status_error
        error = _make_status_error("Invalid data format", body=None, response=mock_response)

        # Verify correct error type is returned
        assert isinstance(error, UnprocessableEntityError)
        assert "Invalid data format" in error.message

    def test_make_status_error_rate_limit(self):
        """Test _make_status_error with 429 status code (line 147)."""
        # Create mock response with 429 status code
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.headers = {}

        # Call _make_status_error
        error = _make_status_error("Too many requests", body=None, response=mock_response)

        # Verify correct error type is returned
        assert isinstance(error, RateLimitError)
        assert "Too many requests" in error.message

    def test_make_status_error_server_error(self):
        """Test _make_status_error with 500+ status codes (line 149)."""
        # Create mock response with 500 status code
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.headers = {}

        # Call _make_status_error
        error = _make_status_error("Internal server error", body=None, response=mock_response)

        # Verify correct error type is returned
        assert isinstance(error, InternalServerError)
        assert "Internal server error" in error.message

        # Try with a different 5xx code
        mock_response.status_code = 503
        error = _make_status_error("Service unavailable", body=None, response=mock_response)
        assert isinstance(error, InternalServerError)
        assert "Service unavailable" in error.message

    def test_make_status_error_unhandled_4xx(self):
        """Test _make_status_error with unhandled 4xx status code (lines 152-153)."""
        # Create mock response with an unhandled 4xx status code (e.g., 418)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 418
        mock_response.headers = {}

        # Call _make_status_error
        error = _make_status_error("I'm a teapot", body=None, response=mock_response)

        # Verify correct error type is returned with proper message
        assert isinstance(error, APIError)
        assert "Unhandled 4xx error" in error.message
        assert "I'm a teapot" in error.message

    def test_make_status_error_fallback(self):
        """Test _make_status_error fallback case (line 155)."""
        # Create mock response with unusual status code (e.g., 999)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 999
        mock_response.headers = {}

        # Call _make_status_error
        error = _make_status_error("Unknown status code", body=None, response=mock_response)

        # Verify basic APIError is returned
        assert isinstance(error, APIError)
        assert error.__class__ == APIError  # Not a subclass, but APIError itself
        assert "Unknown status code" in error.message

    def test_make_status_error_with_error_detail(self):
        """Test handling error data with 'detail' field instead of 'message'."""
        # Create mock response with status code
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.headers = {}

        # Create body with error dict containing code and detail (not message)
        body = {
            "error": {
                "code": "VALIDATION_ERROR",
                "detail": "The input is malformed"
            }
        }

        # Call _make_status_error
        error = _make_status_error(None, body=body, response=mock_response)

        # Verify the error message includes the details from the error dict
        assert isinstance(error, InvalidRequestError)
        assert "The input is malformed" in error.message
        assert "Code: VALIDATION_ERROR" in error.message

    def test_make_status_error_with_error_code_only(self):
        """Test handling error data with only 'code' field."""
        # Create mock response with status code
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.headers = {}

        # Create body with error dict containing code only
        body = {
            "error": {
                "code": "MISSING_FIELD"
            }
        }

        # Call _make_status_error
        error = _make_status_error("Bad request", body=body, response=mock_response)

        # Verify the error message includes the code
        assert isinstance(error, InvalidRequestError)
        assert "Bad request" in error.message
        assert "Code: MISSING_FIELD" in error.message