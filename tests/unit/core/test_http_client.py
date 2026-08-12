"""
Unit tests for VeniceHTTPClient in core/http_client.py.

This module provides comprehensive test coverage for the centralized HTTP client,
focusing on session management, header handling, request methods, and lifecycle management.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from venice_ai.core.config import VeniceAIConfig
from venice_ai.core.http_client import (
    VeniceHTTPClient,
    _extract_rate_limit_headers,
)
from venice_ai.middleware.retry import RetryOptions

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Create a valid test configuration."""
    return VeniceAIConfig.create_test_config()


@pytest.fixture
def http_client(config):
    """Create a VeniceHTTPClient instance for testing."""
    return VeniceHTTPClient(
        config=config,
        api_key="test-api-key-12345678901234567890",
    )


@pytest.fixture
def mock_response():
    """Create a mock aiohttp ClientResponse."""
    response = Mock(spec=aiohttp.ClientResponse)
    response.status = 200
    response.headers = {}
    return response


# =============================================================================
# Tests for _extract_rate_limit_headers (Lines 53-64)
# =============================================================================


class TestExtractRateLimitHeaders:
    """Tests for the _extract_rate_limit_headers helper function."""

    def test_extract_rate_limit_headers_basic(self):
        """Test extraction of basic rate limit headers (covers lines 53-60)."""
        mock_response = Mock(spec=aiohttp.ClientResponse)
        mock_response.headers = {
            "X-RateLimit-Limit": "100",
            "X-RateLimit-Remaining": "50",
            "X-RateLimit-Reset": "1234567890",
            "Retry-After": "30",
            "Content-Type": "application/json",  # Should be ignored
        }

        headers = _extract_rate_limit_headers(mock_response)

        assert "x-ratelimit-limit" in headers
        assert headers["x-ratelimit-limit"] == "100"
        assert "x-ratelimit-remaining" in headers
        assert headers["x-ratelimit-remaining"] == "50"
        assert "x-ratelimit-reset" in headers
        assert headers["x-ratelimit-reset"] == "1234567890"
        assert "retry-after" in headers
        assert headers["retry-after"] == "30"
        assert "content-type" not in headers

    def test_extract_rate_limit_headers_empty_response(self):
        """Test extraction with no rate limit headers (covers line 64)."""
        mock_response = Mock(spec=aiohttp.ClientResponse)
        mock_response.headers = {"Content-Type": "application/json"}

        headers = _extract_rate_limit_headers(mock_response)

        assert headers == {}

    def test_extract_rate_limit_headers_with_whitespace(self):
        """Test extraction with whitespace in values (covers line 60)."""
        mock_response = Mock(spec=aiohttp.ClientResponse)
        mock_response.headers = {
            "X-RateLimit-Limit": "  100  ",
            "Retry-After": " 30 ",
        }

        headers = _extract_rate_limit_headers(mock_response)

        assert headers["x-ratelimit-limit"] == "100"
        assert headers["retry-after"] == "30"

    def test_extract_rate_limit_headers_empty_values_ignored(self):
        """Test that empty values are ignored (covers line 59 branch)."""
        mock_response = Mock(spec=aiohttp.ClientResponse)
        mock_response.headers = {
            "X-RateLimit-Limit": "",
            "X-RateLimit-Remaining": "   ",  # Only whitespace
            "Retry-After": "30",
        }

        headers = _extract_rate_limit_headers(mock_response)

        assert "x-ratelimit-limit" not in headers
        assert "x-ratelimit-remaining" not in headers
        assert "retry-after" in headers

    def test_extract_rate_limit_headers_exception_handling(self):
        """Test exception handling when response is in bad state (covers lines 61-63)."""
        mock_response = Mock(spec=aiohttp.ClientResponse)
        # Simulate a bad response state by making headers access raise an exception
        mock_response.headers = Mock()
        mock_response.headers.items.side_effect = RuntimeError("Bad response state")

        headers = _extract_rate_limit_headers(mock_response)

        assert headers == {}

    def test_extract_rate_limit_headers_case_insensitive(self):
        """Test that headers are normalized to lowercase."""
        mock_response = Mock(spec=aiohttp.ClientResponse)
        mock_response.headers = {
            "X-RATELIMIT-LIMIT": "100",
            "x-ratelimit-remaining": "50",
            "RETRY-AFTER": "30",
        }

        headers = _extract_rate_limit_headers(mock_response)

        assert all(key == key.lower() for key in headers)


# =============================================================================
# Tests for VeniceHTTPClient Initialization
# =============================================================================


class TestVeniceHTTPClientInit:
    """Tests for VeniceHTTPClient initialization."""

    def test_init_with_basic_config(self, config):
        """Test basic initialization with config only."""
        client = VeniceHTTPClient(config=config)

        assert client._config == config
        assert client._base_url == config.api_base_url
        assert client._session is None
        assert client._is_closed is False

    def test_init_with_custom_base_url(self, config):
        """Test initialization with custom base URL."""
        custom_url = "https://custom.api.venice.ai"
        client = VeniceHTTPClient(config=config, base_url=custom_url)

        assert client._base_url == custom_url

    def test_init_with_api_key(self, config):
        """Test initialization with API key."""
        api_key = "test-api-key-12345678901234567890"
        client = VeniceHTTPClient(config=config, api_key=api_key)

        assert client._api_key == api_key

    def test_init_with_custom_headers(self, config):
        """Test initialization with custom headers."""
        headers = {"X-Custom-Header": "custom-value"}
        client = VeniceHTTPClient(config=config, headers=headers)

        assert client._custom_headers == headers
        assert "X-Custom-Header" in client._static_headers

    @pytest.mark.asyncio
    async def test_init_with_all_options(self, config):
        """Test initialization with all optional parameters."""
        cookie_jar = aiohttp.CookieJar()
        retry_options = RetryOptions(max_attempts=5)

        client = VeniceHTTPClient(
            config=config,
            api_key="test-key-1234567890123456789012",
            base_url="https://custom.api.com",
            headers={"X-Custom": "value"},
            trust_env=True,
            connector_limit=500,
            connector_limit_per_host=50,
            auto_decompress=False,
            cookie_jar=cookie_jar,
            skip_auto_headers=["User-Agent"],
            http_transport_options={"ssl": False},
            retry_options=retry_options,
        )

        assert client._trust_env is True
        assert client._connector_limit == 500
        assert client._connector_limit_per_host == 50
        assert client._auto_decompress is False
        assert client._cookie_jar is cookie_jar
        assert client._skip_auto_headers == ["User-Agent"]
        assert client._retry_options == retry_options


# =============================================================================
# Tests for Async Context Manager (Lines 142-143, 147)
# =============================================================================


class TestAsyncContextManager:
    """Tests for async context manager methods."""

    @pytest.mark.asyncio
    async def test_aenter_creates_session(self, config):
        """Test that __aenter__ creates a session (covers lines 142-143)."""
        client = VeniceHTTPClient(config=config, api_key="test-key-1234567890123456")

        async with client as entered_client:
            assert entered_client is client
            assert client._session is not None
            assert not client.is_closed

        # Cleanup
        await client.close()

    @pytest.mark.asyncio
    async def test_aexit_closes_client(self, config):
        """Test that __aexit__ closes the client (covers line 147)."""
        client = VeniceHTTPClient(config=config, api_key="test-key-1234567890123456")

        async with client:
            assert client._session is not None

        # After context manager exit, client should be closed
        assert client.is_closed

    @pytest.mark.asyncio
    async def test_aexit_handles_exception(self, config):
        """Test that __aexit__ properly handles exceptions."""
        client = VeniceHTTPClient(config=config, api_key="test-key-1234567890123456")

        with pytest.raises(ValueError):
            async with client:
                raise ValueError("Test exception")

        # Client should still be closed even after exception
        assert client.is_closed


# =============================================================================
# Tests for Session Building (Lines 212, 219, 222, 225, 230-231, 234)
# =============================================================================


class TestBuildSession:
    """Tests for _build_session method with various optional parameters."""

    @pytest.mark.asyncio
    async def test_build_session_with_trust_env(self, config):
        """Test session building with trust_env (covers line 212)."""
        client = VeniceHTTPClient(config=config, trust_env=True)

        session = await client.get_session()

        assert session is not None
        # Session should be built without errors
        assert not client.is_closed

        await client.close()

    @pytest.mark.asyncio
    async def test_build_session_with_auto_decompress(self, config):
        """Test session building with auto_decompress (covers line 219)."""
        client = VeniceHTTPClient(config=config, auto_decompress=False)

        session = await client.get_session()

        assert session is not None
        await client.close()

    @pytest.mark.asyncio
    async def test_build_session_with_cookie_jar(self, config):
        """Test session building with cookie_jar (covers line 222)."""
        cookie_jar = aiohttp.CookieJar()
        client = VeniceHTTPClient(config=config, cookie_jar=cookie_jar)

        session = await client.get_session()

        assert session is not None
        await client.close()

    @pytest.mark.asyncio
    async def test_build_session_with_skip_auto_headers(self, config):
        """Test session building with skip_auto_headers (covers line 225)."""
        client = VeniceHTTPClient(config=config, skip_auto_headers=["User-Agent"])

        session = await client.get_session()

        assert session is not None
        await client.close()

    @pytest.mark.asyncio
    async def test_build_session_with_retry_options(self, config):
        """Test session building with retry_options (covers lines 230-231, 234)."""
        retry_options = RetryOptions(max_attempts=5, base_delay=0.5)
        client = VeniceHTTPClient(config=config, retry_options=retry_options)

        session = await client.get_session()

        assert session is not None
        await client.close()

    @pytest.mark.asyncio
    async def test_build_session_trailing_slash_normalization(self, config):
        """Test that base URL is properly normalized with trailing slash."""
        # Base URL without trailing slash
        client_no_slash = VeniceHTTPClient(config=config, base_url="https://api.test.com")

        session = await client_no_slash.get_session()
        assert session is not None
        await client_no_slash.close()

        # Base URL with trailing slash
        client_with_slash = VeniceHTTPClient(config=config, base_url="https://api.test.com/")

        session = await client_with_slash.get_session()
        assert session is not None
        await client_with_slash.close()


# =============================================================================
# Tests for get_session (Lines 165→168 partial branch)
# =============================================================================


class TestGetSession:
    """Tests for get_session method."""

    @pytest.mark.asyncio
    async def test_get_session_creates_new(self, http_client):
        """Test session is created on first call."""
        assert http_client._session is None

        session = await http_client.get_session()

        assert session is not None
        assert http_client._session is session
        await http_client.close()

    @pytest.mark.asyncio
    async def test_get_session_returns_existing(self, http_client):
        """Test that same session is returned on subsequent calls (covers line 165→168)."""
        session1 = await http_client.get_session()
        session2 = await http_client.get_session()

        assert session1 is session2
        await http_client.close()

    @pytest.mark.asyncio
    async def test_get_session_raises_when_closed(self, http_client):
        """Test that RuntimeError is raised when client is closed."""
        await http_client.close()

        with pytest.raises(RuntimeError, match="HTTP client has been closed"):
            await http_client.get_session()

    @pytest.mark.asyncio
    async def test_get_session_concurrent_calls(self, config):
        """Test concurrent calls to get_session (covers double-check pattern)."""
        client = VeniceHTTPClient(config=config, api_key="test-key-1234567890123456789012")

        # Make concurrent calls to get_session
        sessions = await asyncio.gather(
            client.get_session(),
            client.get_session(),
            client.get_session(),
        )

        # All should return the same session
        assert all(s is sessions[0] for s in sessions)
        await client.close()


# =============================================================================
# Tests for Static Headers (Line 264→268 partial branch)
# =============================================================================


class TestStaticHeaders:
    """Tests for _build_static_headers method."""

    def test_static_headers_without_api_key(self, config):
        """Test static headers when no API key provided (covers line 264→268)."""
        client = VeniceHTTPClient(config=config)  # No API key

        assert "Accept" in client._static_headers
        assert client._static_headers["Accept"] == "application/json"
        assert "User-Agent" in client._static_headers
        # No Authorization header when no API key
        assert "Authorization" not in client._static_headers

    def test_static_headers_with_api_key(self, config):
        """Test static headers when API key is provided."""
        client = VeniceHTTPClient(config=config, api_key="test-key-1234567890123456789012")

        assert "Authorization" in client._static_headers
        assert client._static_headers["Authorization"].startswith("Bearer ")

    def test_static_headers_custom_override(self, config):
        """Test that custom headers override defaults."""
        custom_headers = {
            "Accept": "text/plain",
            "X-Custom": "value",
        }
        client = VeniceHTTPClient(config=config, headers=custom_headers)

        assert client._static_headers["Accept"] == "text/plain"
        assert client._static_headers["X-Custom"] == "value"


# =============================================================================
# Tests for request Method (Lines 318→323, 319→321 partial branches)
# =============================================================================


class TestRequestMethod:
    """Tests for the request method."""

    @pytest.mark.asyncio
    async def test_request_with_per_request_headers(self, config):
        """Test request with per-request headers override."""
        client = VeniceHTTPClient(config=config, api_key="test-key-1234567890123456789012")

        with patch.object(aiohttp.ClientSession, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = Mock(status=200)

            # Get session first to avoid mocking session creation
            await client.get_session()

            # Make request with per-request headers
            await client.request(
                "GET",
                "/test",
                headers={"X-Request-Header": "request-value"},
            )

            # Verify headers were merged
            call_args = mock_request.call_args
            assert "headers" in call_args.kwargs

        await client.close()

    @pytest.mark.asyncio
    async def test_request_with_float_timeout(self, config):
        """Test request with float timeout (covers lines 318-321)."""
        client = VeniceHTTPClient(config=config, api_key="test-key-1234567890123456789012")

        with patch.object(aiohttp.ClientSession, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = Mock(status=200)

            await client.get_session()

            # Make request with float timeout
            await client.request("GET", "/test", timeout=10.0)

            # Verify timeout was converted
            call_args = mock_request.call_args
            assert "timeout" in call_args.kwargs
            assert isinstance(call_args.kwargs["timeout"], aiohttp.ClientTimeout)

        await client.close()

    @pytest.mark.asyncio
    async def test_request_with_client_timeout(self, config):
        """Test request with ClientTimeout (covers line 318→323)."""
        client = VeniceHTTPClient(config=config, api_key="test-key-1234567890123456789012")

        with patch.object(aiohttp.ClientSession, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = Mock(status=200)

            await client.get_session()

            # Make request with aiohttp.ClientTimeout
            custom_timeout = aiohttp.ClientTimeout(total=15.0)
            await client.request("GET", "/test", timeout=custom_timeout)

            call_args = mock_request.call_args
            assert call_args.kwargs["timeout"] is custom_timeout

        await client.close()

    @pytest.mark.asyncio
    async def test_request_without_timeout(self, config):
        """Test request without explicit timeout (uses default)."""
        client = VeniceHTTPClient(config=config, api_key="test-key-1234567890123456789012")

        with patch.object(aiohttp.ClientSession, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = Mock(status=200)

            await client.get_session()

            # Make request without timeout
            await client.request("GET", "/test")

            call_args = mock_request.call_args
            # Timeout should not be in kwargs if not explicitly provided
            assert "timeout" not in call_args.kwargs

        await client.close()


# =============================================================================
# Tests for stream_request Method
# =============================================================================


class TestStreamRequest:
    """Tests for stream_request context manager."""

    @pytest.mark.asyncio
    async def test_stream_request_context_manager(self, config):
        """Test stream_request as async context manager."""
        client = VeniceHTTPClient(config=config, api_key="test-key-1234567890123456789012")

        mock_response = Mock(spec=aiohttp.ClientResponse)
        mock_response.status = 200
        mock_response.close = Mock()

        with patch.object(aiohttp.ClientSession, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            await client.get_session()

            async with client.stream_request("GET", "/stream") as response:
                assert response is mock_response

            # Verify response was closed
            mock_response.close.assert_called_once()

        await client.close()


# =============================================================================
# Tests for close Method (Line 490→497 partial branch)
# =============================================================================


class TestCloseMethod:
    """Tests for the close method."""

    @pytest.mark.asyncio
    async def test_close_fresh_client(self, http_client):
        """Test closing a client that was never used."""
        assert not http_client.is_closed
        assert http_client._session is None

        await http_client.close()

        assert http_client.is_closed
        assert http_client._session is None

    @pytest.mark.asyncio
    async def test_close_used_client(self, http_client):
        """Test closing a client that was used."""
        # Use the client to create a session
        session = await http_client.get_session()
        assert session is not None

        await http_client.close()

        assert http_client.is_closed
        assert http_client._session is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self, http_client):
        """Test that closing twice is safe (covers line 484-485)."""
        await http_client.get_session()
        await http_client.close()
        assert http_client.is_closed

        # Second close should be a no-op
        await http_client.close()
        assert http_client.is_closed

    @pytest.mark.asyncio
    async def test_close_with_already_closed_session(self, config):
        """Test close when internal session is already closed (covers line 490→497)."""
        client = VeniceHTTPClient(config=config, api_key="test-key-1234567890123456789012")

        session = await client.get_session()

        # Manually close the session to simulate already-closed state
        await session.close()

        # Now close the client - should handle gracefully
        await client.close()

        assert client.is_closed
        assert client._session is None


# =============================================================================
# Tests for Properties (Lines 509, 514)
# =============================================================================


class TestProperties:
    """Tests for client properties."""

    def test_is_closed_property(self, http_client):
        """Test is_closed property."""
        assert http_client.is_closed is False

    @pytest.mark.asyncio
    async def test_is_closed_after_close(self, http_client):
        """Test is_closed property after closing."""
        await http_client.close()
        assert http_client.is_closed is True

    def test_base_url_property(self, config):
        """Test base_url property (covers line 509)."""
        custom_url = "https://custom.api.venice.ai"
        client = VeniceHTTPClient(config=config, base_url=custom_url)

        assert client.base_url == custom_url

    def test_base_url_property_default(self, config):
        """Test base_url property with default value."""
        client = VeniceHTTPClient(config=config)

        assert client.base_url == config.api_base_url

    def test_default_timeout_property(self, config):
        """Test default_timeout property (covers line 514)."""
        client = VeniceHTTPClient(config=config)

        timeout = client.default_timeout

        assert isinstance(timeout, aiohttp.ClientTimeout)
        assert timeout.total == config.http_client.timeout


# =============================================================================
# Tests for Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in VeniceHTTPClient."""

    @pytest.mark.asyncio
    async def test_request_after_close_raises_error(self, http_client):
        """Test that requests after close raise RuntimeError."""
        await http_client.close()

        with pytest.raises(RuntimeError, match="HTTP client has been closed"):
            await http_client.request("GET", "/test")


# =============================================================================
# Tests for Connector Configuration
# =============================================================================


class TestConnectorConfiguration:
    """Tests for connector configuration options."""

    @pytest.mark.asyncio
    async def test_custom_connector_limits(self, config):
        """Test custom connector limits are applied."""
        client = VeniceHTTPClient(
            config=config,
            connector_limit=500,
            connector_limit_per_host=50,
        )

        session = await client.get_session()

        assert session is not None
        # The connector should be configured with our limits
        # We can verify the session was created successfully
        await client.close()

    @pytest.mark.asyncio
    async def test_default_connector_limits(self, config):
        """Test default connector limits are applied when not specified."""
        client = VeniceHTTPClient(config=config)

        session = await client.get_session()

        assert session is not None
        await client.close()

    @pytest.mark.asyncio
    async def test_http_transport_options(self, config):
        """Test custom HTTP transport options are applied."""
        client = VeniceHTTPClient(
            config=config,
            http_transport_options={"force_close": True},
        )

        session = await client.get_session()

        assert session is not None
        await client.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
