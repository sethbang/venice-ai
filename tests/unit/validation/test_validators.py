"""Tests for validation utilities."""

import pytest

from venice_ai.validation.validators import (
    validate_api_key,
    validate_cache_size,
    validate_collection_size,
    validate_date_range,
    validate_interval,
    validate_model_id,
    validate_percentage,
    validate_positive_number,
    validate_priority,
    validate_text_length,
    validate_timeout,
    validate_ttl,
)


class TestValidateModelId:
    def test_valid_model_id(self):
        """Test valid model IDs pass."""
        validate_model_id("gpt-4")
        validate_model_id("claude-2")
        validate_model_id("model-123-abc")

    def test_none_model_id_raises(self):
        """Test None raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_model_id(None)

    def test_empty_model_id_raises(self):
        """Test empty string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_model_id("")
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_model_id("   ")

    def test_invalid_type_raises(self):
        """Test non-string raises TypeError."""
        with pytest.raises(TypeError, match="must be a string"):
            validate_model_id(123)  # type: ignore

    def test_invalid_characters_raise(self):
        """Test invalid characters raise ValueError."""
        with pytest.raises(ValueError, match="invalid character"):
            validate_model_id("model\nid")


class TestValidatePositiveNumber:
    def test_valid_positive_number(self):
        """Test valid numbers pass."""
        validate_positive_number(1.0, "test")
        validate_positive_number(100, "test")

    def test_negative_raises(self):
        """Test negative number raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            validate_positive_number(-1, "test")

    def test_exceeds_max_raises(self):
        """Test exceeding max raises ValueError."""
        with pytest.raises(ValueError, match="must be <="):
            validate_positive_number(100, "test", max_value=50)

    def test_invalid_type_raises(self):
        """Test non-number raises TypeError."""
        with pytest.raises(TypeError, match="must be a number"):
            validate_positive_number("not a number", "test")  # type: ignore


class TestValidateTTL:
    def test_valid_ttl_values(self):
        """Test valid TTL values pass."""
        assert validate_ttl(1) == 1
        assert validate_ttl(3600) == 3600
        assert validate_ttl(86400) == 86400
        assert validate_ttl(604800) == 604800

    def test_ttl_with_custom_bounds(self):
        """Test TTL validation with custom bounds."""
        assert validate_ttl(100, min_val=50, max_val=200) == 100
        assert validate_ttl(50, min_val=50, max_val=200) == 50
        assert validate_ttl(200, min_val=50, max_val=200) == 200

    def test_ttl_below_minimum_raises(self):
        """Test TTL below minimum raises ValueError."""
        with pytest.raises(ValueError, match="must be between 1 and 604800"):
            validate_ttl(0)
        with pytest.raises(ValueError, match="must be between 10 and 100"):
            validate_ttl(5, min_val=10, max_val=100)

    def test_ttl_above_maximum_raises(self):
        """Test TTL above maximum raises ValueError."""
        with pytest.raises(ValueError, match="must be between 1 and 604800"):
            validate_ttl(604801)
        with pytest.raises(ValueError, match="must be between 1 and 100"):
            validate_ttl(200, min_val=1, max_val=100)

    def test_ttl_invalid_type_raises(self):
        """Test non-integer TTL raises TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            validate_ttl(3600.5)  # type: ignore
        with pytest.raises(TypeError, match="must be an integer"):
            validate_ttl("3600")  # type: ignore

    def test_ttl_custom_param_name(self):
        """Test custom parameter name in error messages."""
        with pytest.raises(ValueError, match="cache_ttl must be between"):
            validate_ttl(0, param_name="cache_ttl")


class TestValidatePriority:
    def test_valid_priority_values(self):
        """Test valid priority values pass."""
        assert validate_priority(0) == 0
        assert validate_priority(5) == 5
        assert validate_priority(10) == 10

    def test_priority_with_custom_bounds(self):
        """Test priority validation with custom bounds."""
        assert validate_priority(50, min_val=0, max_val=100) == 50
        assert validate_priority(0, min_val=0, max_val=100) == 0
        assert validate_priority(100, min_val=0, max_val=100) == 100

    def test_priority_below_minimum_raises(self):
        """Test priority below minimum raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 10"):
            validate_priority(-1)
        with pytest.raises(ValueError, match="must be between 5 and 10"):
            validate_priority(3, min_val=5, max_val=10)

    def test_priority_above_maximum_raises(self):
        """Test priority above maximum raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 10"):
            validate_priority(11)
        with pytest.raises(ValueError, match="must be between 0 and 5"):
            validate_priority(10, min_val=0, max_val=5)

    def test_priority_invalid_type_raises(self):
        """Test non-integer priority raises TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            validate_priority(5.5)  # type: ignore
        with pytest.raises(TypeError, match="must be an integer"):
            validate_priority("5")  # type: ignore


class TestValidateCollectionSize:
    def test_valid_collection_sizes(self):
        """Test valid collection sizes pass."""
        assert validate_collection_size(1) == 1
        assert validate_collection_size(100) == 100
        assert validate_collection_size(10000) == 10000
        assert validate_collection_size(1000000) == 1000000

    def test_collection_size_with_custom_bounds(self):
        """Test collection size validation with custom bounds."""
        assert validate_collection_size(500, min_val=100, max_val=1000) == 500
        assert validate_collection_size(100, min_val=100, max_val=1000) == 100
        assert validate_collection_size(1000, min_val=100, max_val=1000) == 1000

    def test_collection_size_below_minimum_raises(self):
        """Test collection size below minimum raises ValueError."""
        with pytest.raises(ValueError, match="must be between 1 and 1000000"):
            validate_collection_size(0)
        with pytest.raises(ValueError, match="must be between 10 and 100"):
            validate_collection_size(5, min_val=10, max_val=100)

    def test_collection_size_above_maximum_raises(self):
        """Test collection size above maximum raises ValueError."""
        with pytest.raises(ValueError, match="must be between 1 and 1000000"):
            validate_collection_size(1000001)
        with pytest.raises(ValueError, match="must be between 1 and 100"):
            validate_collection_size(200, min_val=1, max_val=100)

    def test_collection_size_invalid_type_raises(self):
        """Test non-integer collection size raises TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            validate_collection_size(100.5)  # type: ignore
        with pytest.raises(TypeError, match="must be an integer"):
            validate_collection_size("100")  # type: ignore

    def test_collection_size_custom_param_name(self):
        """Test custom parameter name in error messages."""
        with pytest.raises(ValueError, match="max_cache_size must be between"):
            validate_collection_size(0, param_name="max_cache_size")


class TestValidateTimeout:
    def test_valid_timeout_values(self):
        """Test valid timeout values pass."""
        assert validate_timeout(0.1) == 0.1
        assert validate_timeout(30.0) == 30.0
        assert validate_timeout(300.0) == 300.0
        assert validate_timeout(60) == 60.0  # int converted to float

    def test_timeout_with_custom_bounds(self):
        """Test timeout validation with custom bounds."""
        assert validate_timeout(5.0, min_val=1.0, max_val=10.0) == 5.0
        assert validate_timeout(1.0, min_val=1.0, max_val=10.0) == 1.0
        assert validate_timeout(10.0, min_val=1.0, max_val=10.0) == 10.0

    def test_timeout_below_minimum_raises(self):
        """Test timeout below minimum raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0.1 and 300"):
            validate_timeout(0.05)
        with pytest.raises(ValueError, match="must be between 1.0 and 10.0"):
            validate_timeout(0.5, min_val=1.0, max_val=10.0)

    def test_timeout_above_maximum_raises(self):
        """Test timeout above maximum raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0.1 and 300"):
            validate_timeout(301.0)
        with pytest.raises(ValueError, match="must be between 1.0 and 10.0"):
            validate_timeout(15.0, min_val=1.0, max_val=10.0)

    def test_timeout_invalid_type_raises(self):
        """Test non-numeric timeout raises TypeError."""
        with pytest.raises(TypeError, match="must be a number"):
            validate_timeout("30")  # type: ignore

    def test_timeout_custom_param_name(self):
        """Test custom parameter name in error messages."""
        with pytest.raises(ValueError, match="connection_timeout must be between"):
            validate_timeout(0.05, param_name="connection_timeout")


class TestValidateInterval:
    def test_valid_interval_values(self):
        """Test valid interval values pass."""
        assert validate_interval(0.1) == 0.1
        assert validate_interval(60.0) == 60.0
        assert validate_interval(3600.0) == 3600.0
        assert validate_interval(86400.0) == 86400.0
        assert validate_interval(30) == 30.0  # int converted to float

    def test_interval_with_custom_bounds(self):
        """Test interval validation with custom bounds."""
        assert validate_interval(10.0, min_val=5.0, max_val=20.0) == 10.0
        assert validate_interval(5.0, min_val=5.0, max_val=20.0) == 5.0
        assert validate_interval(20.0, min_val=5.0, max_val=20.0) == 20.0

    def test_interval_below_minimum_raises(self):
        """Test interval below minimum raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0.1 and 86400"):
            validate_interval(0.05)
        with pytest.raises(ValueError, match="must be between 1.0 and 10.0"):
            validate_interval(0.5, min_val=1.0, max_val=10.0)

    def test_interval_above_maximum_raises(self):
        """Test interval above maximum raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0.1 and 86400"):
            validate_interval(86401.0)
        with pytest.raises(ValueError, match="must be between 1.0 and 10.0"):
            validate_interval(15.0, min_val=1.0, max_val=10.0)

    def test_interval_invalid_type_raises(self):
        """Test non-numeric interval raises TypeError."""
        with pytest.raises(TypeError, match="must be a number"):
            validate_interval("60")  # type: ignore

    def test_interval_custom_param_name(self):
        """Test custom parameter name in error messages."""
        with pytest.raises(ValueError, match="cleanup_interval must be between"):
            validate_interval(0.05, param_name="cleanup_interval")


class TestValidatePercentage:
    def test_valid_percentage_values(self):
        """Test valid percentage values pass."""
        assert validate_percentage(0.0) == 0.0
        assert validate_percentage(50.0) == 50.0
        assert validate_percentage(100.0) == 100.0
        assert validate_percentage(25) == 25.0  # int converted to float

    def test_percentage_with_custom_bounds(self):
        """Test percentage validation with custom bounds."""
        assert validate_percentage(0.5, min_val=0.0, max_val=1.0) == 0.5
        assert validate_percentage(0.0, min_val=0.0, max_val=1.0) == 0.0
        assert validate_percentage(1.0, min_val=0.0, max_val=1.0) == 1.0

    def test_percentage_below_minimum_raises(self):
        """Test percentage below minimum raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0.0 and 100"):
            validate_percentage(-1.0)
        with pytest.raises(ValueError, match="must be between 10.0 and 90.0"):
            validate_percentage(5.0, min_val=10.0, max_val=90.0)

    def test_percentage_above_maximum_raises(self):
        """Test percentage above maximum raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0.0 and 100"):
            validate_percentage(101.0)
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            validate_percentage(1.5, min_val=0.0, max_val=1.0)

    def test_percentage_invalid_type_raises(self):
        """Test non-numeric percentage raises TypeError."""
        with pytest.raises(TypeError, match="must be a number"):
            validate_percentage("50")  # type: ignore

    def test_percentage_custom_param_name(self):
        """Test custom parameter name in error messages."""
        with pytest.raises(ValueError, match="buffer_ratio must be between"):
            validate_percentage(101.0, param_name="buffer_ratio")


class TestValidateApiKey:
    """Tests for validate_api_key function - covers lines 212-234."""

    def test_valid_api_key(self):
        """Test valid API keys pass validation."""
        # At least 32 chars, alphanumeric with underscores/hyphens
        validate_api_key("vn_abcdefghijklmnopqrstuvwxyz12345")
        validate_api_key("sk_1234567890abcdefghijklmnopqrst")
        validate_api_key("a" * 32)  # Minimum length
        validate_api_key("abc-def_ghi-123456789012345678901")

    def test_api_key_non_string_raises_type_error(self):
        """Test non-string raises TypeError - covers line 212-213."""
        with pytest.raises(TypeError, match="api_key must be a string"):
            validate_api_key(12345)  # type: ignore
        with pytest.raises(TypeError, match="api_key must be a string"):
            validate_api_key(None)  # type: ignore
        with pytest.raises(TypeError, match="must be a string, got list"):
            validate_api_key(["key"])  # type: ignore

    def test_api_key_empty_raises_value_error(self):
        """Test empty string raises ValueError - covers lines 215-216."""
        with pytest.raises(ValueError, match="api_key cannot be empty"):
            validate_api_key("")
        with pytest.raises(ValueError, match="api_key cannot be empty"):
            validate_api_key("   ")
        with pytest.raises(ValueError, match="api_key cannot be empty"):
            validate_api_key("\t\n")

    def test_api_key_too_short_raises_value_error(self):
        """Test key shorter than 32 chars raises ValueError - covers lines 219-223."""
        with pytest.raises(ValueError, match="is too short"):
            validate_api_key("short_key")
        with pytest.raises(ValueError, match="at least 32 characters"):
            validate_api_key("a" * 31)  # One less than minimum

    def test_api_key_invalid_characters_raises_value_error(self):
        """Test invalid characters raise ValueError - covers lines 226-230."""
        # Keys with spaces (at least 32 chars to pass length check first)
        with pytest.raises(ValueError, match="contains invalid characters"):
            validate_api_key("api key with spaces here 1234567890")
        with pytest.raises(ValueError, match="contains invalid characters"):
            validate_api_key("key_with_special@chars12345678901234")
        with pytest.raises(ValueError, match="contains invalid characters"):
            validate_api_key("key!with#special$chars%^&*()+=1234")

    def test_api_key_with_whitespace_padding_raises_value_error(self):
        """Test whitespace padding raises ValueError - covers lines 233-237.

        Note: Leading/trailing whitespace is caught by the invalid character
        check (line 226) before the explicit whitespace check (line 233),
        so we verify this behavior instead.
        """
        # Leading/trailing whitespace caught by invalid character check
        with pytest.raises(ValueError, match="contains invalid characters"):
            validate_api_key(" vn_abcdefghijklmnopqrstuvwxyz123")
        with pytest.raises(ValueError, match="contains invalid characters"):
            validate_api_key("vn_abcdefghijklmnopqrstuvwxyz123 ")
        with pytest.raises(ValueError, match="contains invalid characters"):
            validate_api_key("\tvn_abcdefghijklmnopqrstuvwxyz12")

    def test_api_key_custom_param_name(self):
        """Test custom parameter name in error messages."""
        with pytest.raises(TypeError, match="custom_key must be a string"):
            validate_api_key(123, param_name="custom_key")  # type: ignore
        with pytest.raises(ValueError, match="venice_key cannot be empty"):
            validate_api_key("", param_name="venice_key")


class TestValidateDateRange:
    """Tests for validate_date_range function - covers lines 260-286."""

    def test_valid_date_range(self):
        """Test valid date ranges pass validation."""
        # Same date (single day query)
        validate_date_range("2025-01-01", "2025-01-01")
        # Multi-day range
        validate_date_range("2025-01-01", "2025-01-15")
        # With time
        validate_date_range("2025-01-01T00:00:00", "2025-01-15T23:59:59")
        # With timezone Z suffix
        validate_date_range("2025-01-01T00:00:00Z", "2025-01-15T23:59:59Z")
        # With timezone offset
        validate_date_range("2025-01-01T00:00:00+00:00", "2025-01-15T23:59:59+00:00")

    def test_start_date_non_string_raises_value_error(self):
        """Test non-string start date raises ValueError - covers line 260-261."""
        with pytest.raises(ValueError, match="start_date must be a non-empty string"):
            validate_date_range(None, "2025-01-15")  # type: ignore
        with pytest.raises(ValueError, match="start_date must be a non-empty string"):
            validate_date_range("", "2025-01-15")
        with pytest.raises(ValueError, match="start_date must be a non-empty string"):
            validate_date_range("   ", "2025-01-15")

    def test_end_date_non_string_raises_value_error(self):
        """Test non-string end date raises ValueError - covers lines 263-264."""
        with pytest.raises(ValueError, match="end_date must be a non-empty string"):
            validate_date_range("2025-01-01", None)  # type: ignore
        with pytest.raises(ValueError, match="end_date must be a non-empty string"):
            validate_date_range("2025-01-01", "")
        with pytest.raises(ValueError, match="end_date must be a non-empty string"):
            validate_date_range("2025-01-01", "   ")

    def test_invalid_date_format_raises_value_error(self):
        """Test invalid date format raises ValueError - covers lines 266-274."""
        with pytest.raises(ValueError, match="Invalid date format"):
            validate_date_range("not-a-date", "2025-01-15")
        with pytest.raises(ValueError, match="Invalid date format"):
            validate_date_range("2025-01-01", "also-not-a-date")
        with pytest.raises(ValueError, match="Invalid date format"):
            validate_date_range("01/01/2025", "01/15/2025")  # Wrong format
        with pytest.raises(ValueError, match="ISO 8601 format"):
            validate_date_range("2025-13-01", "2025-01-15")  # Invalid month

    def test_end_before_start_raises_value_error(self):
        """Test end date before start date raises ValueError - covers lines 277-280."""
        with pytest.raises(ValueError, match="cannot be before"):
            validate_date_range("2025-01-15", "2025-01-01")
        with pytest.raises(ValueError, match="cannot be before"):
            validate_date_range("2025-06-15T12:00:00Z", "2025-06-15T10:00:00Z")

    def test_date_range_exceeds_maximum_raises_value_error(self):
        """Test date range exceeds max days raises ValueError - covers lines 283-289."""
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            validate_date_range("2025-01-01", "2025-02-15", max_range_days=30)
        with pytest.raises(ValueError, match="45 days"):
            validate_date_range("2025-01-01", "2025-02-15", max_range_days=30)

    def test_valid_date_range_within_max_days(self):
        """Test valid date range within max days passes."""
        validate_date_range("2025-01-01", "2025-01-15", max_range_days=30)
        validate_date_range("2025-01-01", "2025-01-31", max_range_days=30)

    def test_custom_param_names_in_errors(self):
        """Test custom parameter names appear in error messages."""
        with pytest.raises(ValueError, match="from_date must be a non-empty string"):
            validate_date_range("", "2025-01-15", start_param="from_date")
        with pytest.raises(ValueError, match="to_date must be a non-empty string"):
            validate_date_range("2025-01-01", "", end_param="to_date")
        with pytest.raises(ValueError, match="to_date.*cannot be before.*from_date"):
            validate_date_range(
                "2025-01-15", "2025-01-01", start_param="from_date", end_param="to_date"
            )


class TestValidateTextLength:
    """Tests for validate_text_length function - covers lines 312, 320, 326."""

    def test_valid_text_length(self):
        """Test valid text length passes validation."""
        validate_text_length("Hello")
        validate_text_length("a" * 100)
        validate_text_length("a" * 5000)  # Max default length

    def test_text_non_string_raises_type_error(self):
        """Test non-string raises TypeError - covers line 312."""
        with pytest.raises(TypeError, match="text must be a string"):
            validate_text_length(123)  # type: ignore
        with pytest.raises(TypeError, match="must be a string, got list"):
            validate_text_length(["text"])  # type: ignore
        with pytest.raises(TypeError, match="must be a string, got NoneType"):
            validate_text_length(None)  # type: ignore

    def test_text_empty_raises_value_error(self):
        """Test empty text raises specific error - covers lines 318-319."""
        with pytest.raises(ValueError, match="Input text cannot be empty for speech generation"):
            validate_text_length("")

    def test_text_too_short_raises_value_error(self):
        """Test text shorter than min_length raises ValueError - covers lines 320-323."""
        with pytest.raises(ValueError, match="is too short"):
            validate_text_length("ab", min_length=5)
        with pytest.raises(ValueError, match="Minimum length is 10 characters"):
            validate_text_length("short", min_length=10)

    def test_text_too_long_raises_value_error(self):
        """Test text longer than max_length raises ValueError - covers lines 326-329."""
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            validate_text_length("a" * 5001)  # Exceeds default max of 5000
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            validate_text_length("a" * 101, max_length=100)
        with pytest.raises(ValueError, match="split into smaller chunks"):
            validate_text_length("a" * 200, max_length=100)

    def test_text_custom_bounds(self):
        """Test custom min/max bounds work correctly."""
        validate_text_length("hello", min_length=1, max_length=10)
        validate_text_length("a", min_length=1, max_length=10)
        validate_text_length("a" * 10, min_length=1, max_length=10)

    def test_text_custom_param_name(self):
        """Test custom parameter name in error messages."""
        with pytest.raises(TypeError, match="prompt must be a string"):
            validate_text_length(123, param_name="prompt")  # type: ignore
        with pytest.raises(ValueError, match="prompt is too short"):
            validate_text_length("ab", min_length=5, param_name="prompt")


class TestValidateCacheSize:
    """Tests for validate_cache_size function - covers lines 355, 358."""

    def test_valid_cache_size(self):
        """Test valid cache sizes pass validation."""
        assert validate_cache_size(1) == 1
        assert validate_cache_size(100) == 100
        assert validate_cache_size(10000) == 10000

    def test_cache_size_non_integer_raises_type_error(self):
        """Test non-integer raises TypeError - covers line 355."""
        with pytest.raises(TypeError, match="cache_size must be an integer"):
            validate_cache_size(100.5)  # type: ignore
        with pytest.raises(TypeError, match="cache_size must be an integer"):
            validate_cache_size("100")  # type: ignore
        with pytest.raises(TypeError, match="must be an integer, got NoneType"):
            validate_cache_size(None)  # type: ignore

    def test_cache_size_below_minimum_raises_value_error(self):
        """Test size below minimum raises ValueError - covers line 358."""
        with pytest.raises(ValueError, match="cache_size must be between 1 and 10000"):
            validate_cache_size(0)
        with pytest.raises(ValueError, match="must be between 10 and 100"):
            validate_cache_size(5, min_val=10, max_val=100)

    def test_cache_size_above_maximum_raises_value_error(self):
        """Test size above maximum raises ValueError - covers line 358."""
        with pytest.raises(ValueError, match="cache_size must be between 1 and 10000"):
            validate_cache_size(10001)
        with pytest.raises(ValueError, match="must be between 1 and 100"):
            validate_cache_size(200, min_val=1, max_val=100)

    def test_cache_size_custom_bounds(self):
        """Test custom min/max bounds work correctly."""
        assert validate_cache_size(50, min_val=10, max_val=100) == 50
        assert validate_cache_size(10, min_val=10, max_val=100) == 10
        assert validate_cache_size(100, min_val=10, max_val=100) == 100

    def test_cache_size_custom_param_name(self):
        """Test custom parameter name in error messages."""
        with pytest.raises(TypeError, match="max_entries must be an integer"):
            validate_cache_size(100.5, param_name="max_entries")  # type: ignore
        with pytest.raises(ValueError, match="max_entries must be between"):
            validate_cache_size(0, param_name="max_entries")
