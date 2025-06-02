import pytest
import httpx
from typing import Dict, Any
from unittest.mock import MagicMock, patch

from venice_ai.exceptions import (
    VeniceError, APIError, AuthenticationError, PermissionDeniedError,
    InvalidRequestError, NotFoundError, ConflictError, UnprocessableEntityError,
    RateLimitError, InternalServerError, _make_status_error
)

class TestMakeStatusErrorExtended:
    """Extended tests for _make_status_error to improve coverage."""
    
    def test_status_error_empty_error_dict(self):
        """Test error handling when error dict is empty."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        
        body: Dict[str, Dict[str, Any]] = {"error": {}}
        
        error = _make_status_error(None, body=body, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "HTTP Status 400" in str(error)
        
    def test_status_error_non_dict_error(self):
        """Test error handling when error is not a dictionary."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        
        body = {"error": "string instead of dict"}
        
        error = _make_status_error(None, body=body, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "HTTP Status 400" in str(error)
        
    def test_error_with_only_code_no_message_detail(self):
        """Test error with just a code field."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        
        body = {"error": {"code": "JUST_CODE"}}
        
        error = _make_status_error(None, body=body, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "HTTP Status 400" in str(error)
        assert "JUST_CODE" in str(error)
        
    def test_status_413_file_size(self):
        """Test specific handling of 413 status code (file size)."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 413
        
        error = _make_status_error(None, body=None, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "HTTP Status 413" in str(error)
        
    def test_status_415_content_type(self):
        """Test specific handling of 415 status code (content type)."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 415
        
        error = _make_status_error(None, body=None, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "HTTP Status 415" in str(error)
        
    @pytest.mark.parametrize("status_code", [
        501, 502, 503, 504, 505, 506, 507, 508, 510, 511
    ])
    def test_various_5xx_statuses(self, status_code):
        """Test handling of various 5xx status codes to ensure they all map to InternalServerError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = status_code
        
        error = _make_status_error(None, body=None, response=mock_response)
        
        assert isinstance(error, InternalServerError)
        assert f"HTTP Status {status_code}" in str(error)
        
    @pytest.mark.parametrize("status_code", [
        402, 405, 406, 407, 408, 410, 411, 412, 414, 416, 417, 418, 421, 423, 424, 426, 428
    ])
    def test_various_4xx_statuses(self, status_code):
        """Test handling of various 4xx status codes to ensure they map to general APIError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = status_code
        
        error = _make_status_error(None, body=None, response=mock_response)
        
        assert isinstance(error, APIError)
        assert not isinstance(error, (InvalidRequestError, AuthenticationError, PermissionDeniedError, NotFoundError, ConflictError, UnprocessableEntityError, RateLimitError))
        assert f"HTTP Status {status_code}" in str(error)
        assert "Unhandled 4xx error" in str(error)
        
    def test_custom_message_with_error_detail(self):
        """Test combining custom message with error details."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        
        body = {"error": {"message": "Parameter validation failed", "code": "VALIDATION_ERROR"}}
        
        error = _make_status_error("Base message", body=body, response=mock_response)
        
        assert isinstance(error, InvalidRequestError)
        assert "Base message" in str(error)
        assert "Parameter validation failed" in str(error)
        assert "VALIDATION_ERROR" in str(error)
        
    def test_unexpected_status_code(self):
        """Test handling of an unexpected status code (outside 400-599 range)."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 302  # Redirect
        
        error = _make_status_error(None, body=None, response=mock_response)
        
        assert isinstance(error, APIError)
        assert not isinstance(error, (AuthenticationError, PermissionDeniedError))
        assert "HTTP Status 302" in str(error)