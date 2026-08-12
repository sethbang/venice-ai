"""Tests for venice_ai.core.auth module."""

from venice_ai.core.auth import create_auth_headers, validate_api_key_format


class TestValidateApiKeyFormat:
    """Test validate_api_key_format edge cases for branch coverage."""

    def test_none_key(self):
        assert validate_api_key_format(None) is False  # type: ignore[arg-type]

    def test_empty_string(self):
        assert validate_api_key_format("") is False

    def test_non_string(self):
        assert validate_api_key_format(12345) is False  # type: ignore[arg-type]

    def test_short_key(self):
        assert validate_api_key_format("short") is False

    def test_exactly_10_chars(self):
        assert validate_api_key_format("1234567890") is False

    def test_11_chars(self):
        assert validate_api_key_format("12345678901") is True

    def test_whitespace_stripped(self):
        assert validate_api_key_format("  12345678901  ") is True

    def test_whitespace_only(self):
        assert validate_api_key_format("   ") is False

    def test_valid_key(self):
        assert validate_api_key_format("venice-api-key-abc123def456") is True


class TestCreateAuthHeaders:
    """Test create_auth_headers."""

    def test_basic(self):
        headers = create_auth_headers("test-key")
        assert headers == {"Authorization": "Bearer test-key"}

    def test_strips_whitespace(self):
        headers = create_auth_headers("  test-key  ")
        assert headers == {"Authorization": "Bearer test-key"}
