"""Tests for retry-after header parsing in exceptions module."""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch
from venice_ai.exceptions import _parse_retry_after_header


class TestParseRetryAfterHeader:
    """Test the _parse_retry_after_header function for complete coverage."""

    def test_parse_retry_after_http_date_format(self):
        """Test parsing retry-after header with HTTP-date format."""
        # Test with a valid HTTP-date format
        http_date = "Wed, 21 Oct 2015 07:28:00 GMT"
        response_date = "Wed, 21 Oct 2015 07:27:00 GMT"  # 1 minute earlier
        
        result = _parse_retry_after_header(http_date, response_date)
        
        # Should return 60 seconds (1 minute difference)
        assert result == 60

    def test_parse_retry_after_http_date_no_response_date(self):
        """Test parsing retry-after header with HTTP-date format and no response date."""
        # Mock datetime.now to return a fixed time
        fixed_now = datetime(2015, 10, 21, 7, 27, 0, tzinfo=timezone.utc)
        
        with patch('venice_ai.exceptions.datetime') as mock_datetime:
            mock_datetime.now.return_value = fixed_now
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            http_date = "Wed, 21 Oct 2015 07:28:00 GMT"
            result = _parse_retry_after_header(http_date, None)
            
            # Should return 60 seconds (1 minute difference)
            assert result == 60

    def test_parse_retry_after_http_date_timezone_naive(self):
        """Test parsing retry-after header with timezone-naive HTTP-date."""
        # Test with timezone-naive date (should be treated as UTC)
        http_date = "Wed, 21 Oct 2015 07:28:00"  # No timezone
        response_date = "Wed, 21 Oct 2015 07:27:00"  # No timezone
        
        result = _parse_retry_after_header(http_date, response_date)
        
        # Should return 60 seconds
        assert result == 60

    def test_parse_retry_after_http_date_negative_delta(self):
        """Test parsing retry-after header with past HTTP-date (negative delta)."""
        # Test with a date in the past
        http_date = "Wed, 21 Oct 2015 07:26:00 GMT"  # 1 minute before response
        response_date = "Wed, 21 Oct 2015 07:27:00 GMT"
        
        result = _parse_retry_after_header(http_date, response_date)
        
        # Should return 0 (max(0, negative_value))
        assert result == 0

    def test_parse_retry_after_invalid_http_date_format(self):
        """Test parsing retry-after header with invalid HTTP-date format."""
        # Test with invalid date format that will cause ValueError
        invalid_date = "not-a-valid-date"
        response_date = "Wed, 21 Oct 2015 07:27:00 GMT"
        
        result = _parse_retry_after_header(invalid_date, response_date)
        
        # Should return None when parsing fails
        assert result is None

    def test_parse_retry_after_type_error_in_date_parsing(self):
        """Test parsing retry-after header that causes TypeError in date parsing."""
        # Test with a value that will cause TypeError in parsedate_to_datetime
        # Using an integer instead of string to trigger TypeError
        with patch('venice_ai.exceptions.parsedate_to_datetime') as mock_parse:
            mock_parse.side_effect = TypeError("Invalid type")
            
            result = _parse_retry_after_header("invalid", "Wed, 21 Oct 2015 07:27:00 GMT")
            
            # Should return None when TypeError occurs
            assert result is None

    def test_parse_retry_after_with_response_date_fallback(self):
        """Test parsing with response date that falls back to datetime.now()."""
        # Test the else branch where response_date_str is None
        http_date = "Wed, 21 Oct 2015 07:28:00 GMT"
        
        # Mock datetime.now to return a fixed time
        with patch('venice_ai.exceptions.datetime') as mock_datetime:
            # Keep the original datetime class for other operations
            original_datetime = datetime
            mock_datetime.side_effect = lambda *args, **kw: original_datetime(*args, **kw)
            mock_datetime.now.return_value = original_datetime(2015, 10, 21, 7, 27, 0, tzinfo=timezone.utc)
            
            result = _parse_retry_after_header(http_date, None)
            
            # Should return 60 seconds using datetime.now() fallback
            assert result == 60