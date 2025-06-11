"""
Unit tests for VeniceClientWithRetries and AsyncVeniceClientWithRetries.

This module tests the retry logic functionality for both synchronous and asynchronous
Venice AI clients with retry capabilities.
"""

import asyncio
import email.utils
import time
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from typing import Any, Dict, List, Optional

import httpx
import pytest
from httpx_retries import Retry, RetryTransport

from venice_ai._client_with_retries import VeniceClientWithRetries, AsyncVeniceClientWithRetries
from venice_ai._constants import DEFAULT_MAX_RETRIES
from venice_ai.exceptions import (
    APITimeoutError,
    InvalidRequestError,
    InternalServerError,
    RateLimitError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    ConflictError,
    UnprocessableEntityError,
)


class TestVeniceClientWithRetries:
    """Test cases for the synchronous VeniceClientWithRetries class."""

    def test_initialization_default_max_retries(self, mocker):
        """Test initialization with default max_retries value."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        
        client = VeniceClientWithRetries(api_key="test-key")
        
        assert client._max_retries == DEFAULT_MAX_RETRIES
        
        # Verify Retry was configured with default parameters
        mock_retry.assert_called_once_with(
            total=DEFAULT_MAX_RETRIES,
            backoff_factor=2.0,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        )
        
        # Verify RetryTransport was created
        mock_retry_transport.assert_called_once()

    def test_initialization_custom_max_retries(self, mocker):
        """Test initialization with custom max_retries value."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        
        custom_retries = 5
        client = VeniceClientWithRetries(api_key="test-key", max_retries=custom_retries)
        
        assert client._max_retries == custom_retries
        
        # Verify Retry was configured with custom max_retries
        mock_retry.assert_called_once_with(
            total=custom_retries,
            backoff_factor=2.0,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        )

    def test_initialization_custom_retry_parameters(self, mocker):
        """Test initialization with custom retry parameters."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        
        client = VeniceClientWithRetries(
            api_key="test-key",
            max_retries=3,
            retry_backoff_factor=1.5,
            retry_status_codes=[429, 503],
            retry_respect_retry_after_header=False
        )
        
        # Verify Retry was configured with custom parameters
        mock_retry.assert_called_once_with(
            total=3,
            backoff_factor=1.5,
            status_forcelist=[429, 503],
            respect_retry_after_header=False,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        )

    def test_initialization_with_transport_passthrough(self, mocker):
        """Test that base client arguments are passed through correctly."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        mock_http_transport = mocker.patch('venice_ai._client_with_retries.httpx.HTTPTransport')
        
        mock_transport = MagicMock(spec=httpx.BaseTransport)
        timeout = httpx.Timeout(30.0)
        
        client = VeniceClientWithRetries(
            api_key="test-key",
            transport=mock_transport,
            timeout=timeout,
            max_retries=3
        )
        
        # Verify Retry was configured
        mock_retry.assert_called_once_with(
            total=3,
            backoff_factor=2.0,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        )
        
        # Verify RetryTransport was created with the provided transport
        mock_retry_transport.assert_called_once()
        retry_call_args = mock_retry_transport.call_args
        assert retry_call_args[1]['transport'] == mock_transport
        
        # Verify httpx.Client was called with correct timeout
        mock_client.assert_called_once()
        client_call_args = mock_client.call_args
        assert client_call_args[1]['timeout'] == timeout

    def test_default_retry_status_codes(self, mocker):
        """Test that default retry status codes are correctly set."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        
        client = VeniceClientWithRetries(api_key="test-key")
        
        # Verify default status codes
        mock_retry.assert_called_once()
        call_args = mock_retry.call_args
        assert call_args[1]['status_forcelist'] == [429, 500, 502, 503, 504]

    def test_custom_retry_status_codes(self, mocker):
        """Test that custom retry status codes are correctly set."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        
        custom_codes = [429, 503, 504]
        client = VeniceClientWithRetries(
            api_key="test-key",
            retry_status_codes=custom_codes
        )
        
        # Verify custom status codes
        mock_retry.assert_called_once()
        call_args = mock_retry.call_args
        assert call_args[1]['status_forcelist'] == custom_codes

    def test_retry_transport_integration(self, mocker):
        """Test that VeniceClient properly integrates RetryTransport with httpx.Client."""
        mock_retry_transport_instance = mocker.Mock()
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport', return_value=mock_retry_transport_instance)
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        
        client = VeniceClientWithRetries(api_key="test-key", max_retries=2)
        
        # Verify RetryTransport was created and passed to httpx.Client
        mock_retry_transport.assert_called_once()
        
        # Verify httpx.Client was called with the retry transport
        mock_client.assert_called_once()
        call_args = mock_client.call_args
        assert call_args[1]['transport'] == mock_retry_transport_instance

    def test_empty_retry_status_codes_fallback(self, mocker):
        """Test that empty retry_status_codes falls back to default."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        
        client = VeniceClientWithRetries(
            api_key="test-key",
            max_retries=2,
            retry_status_codes=[]  # Empty list should fall back to default
        )
        
        # Verify default status codes are used (empty list falls back to default)
        mock_retry.assert_called_once()
        call_args = mock_retry.call_args
        assert call_args[1]['status_forcelist'] == [429, 500, 502, 503, 504]


class TestAsyncVeniceClientWithRetries:
    """Test cases for the asynchronous AsyncVeniceClientWithRetries class."""

    def test_initialization_default_max_retries(self, mocker):
        """Test initialization with default max_retries value."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        
        client = AsyncVeniceClientWithRetries(api_key="test-key")
        
        assert client._max_retries == DEFAULT_MAX_RETRIES
        
        # Verify Retry was configured with default parameters
        mock_retry.assert_called_once_with(
            total=DEFAULT_MAX_RETRIES,
            backoff_factor=2.0,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        )
        
        # Verify RetryTransport was created
        mock_retry_transport.assert_called_once()

    def test_initialization_custom_max_retries(self, mocker):
        """Test initialization with custom max_retries value."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        
        custom_retries = 5
        client = AsyncVeniceClientWithRetries(api_key="test-key", max_retries=custom_retries)
        
        assert client._max_retries == custom_retries
        
        # Verify Retry was configured with custom max_retries
        mock_retry.assert_called_once_with(
            total=custom_retries,
            backoff_factor=2.0,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        )

    def test_initialization_custom_retry_parameters(self, mocker):
        """Test initialization with custom retry parameters."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        
        client = AsyncVeniceClientWithRetries(
            api_key="test-key",
            max_retries=3,
            retry_backoff_factor=1.5,
            retry_status_codes=[429, 503],
            retry_respect_retry_after_header=False
        )
        
        # Verify Retry was configured with custom parameters
        mock_retry.assert_called_once_with(
            total=3,
            backoff_factor=1.5,
            status_forcelist=[429, 503],
            respect_retry_after_header=False,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        )

    def test_initialization_with_transport_passthrough(self, mocker):
        """Test that base client arguments are passed through correctly."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        mock_async_http_transport = mocker.patch('venice_ai._client_with_retries.httpx.AsyncHTTPTransport')
        
        mock_async_transport = MagicMock(spec=httpx.AsyncBaseTransport)
        timeout = httpx.Timeout(30.0)
        
        client = AsyncVeniceClientWithRetries(
            api_key="test-key",
            async_transport=mock_async_transport,
            timeout=timeout,
            max_retries=3
        )
        
        # Verify Retry was configured
        mock_retry.assert_called_once_with(
            total=3,
            backoff_factor=2.0,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        )
        
        # Verify RetryTransport was created with the provided transport
        mock_retry_transport.assert_called_once()
        retry_call_args = mock_retry_transport.call_args
        assert retry_call_args[1]['transport'] == mock_async_transport
        
        # Verify httpx.AsyncClient was called with correct timeout
        mock_async_client.assert_called_once()
        client_call_args = mock_async_client.call_args
        assert client_call_args[1]['timeout'] == timeout

    def test_default_retry_status_codes(self, mocker):
        """Test that default retry status codes are correctly set."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        
        client = AsyncVeniceClientWithRetries(api_key="test-key")
        
        # Verify default status codes
        mock_retry.assert_called_once()
        call_args = mock_retry.call_args
        assert call_args[1]['status_forcelist'] == [429, 500, 502, 503, 504]

    def test_custom_retry_status_codes(self, mocker):
        """Test that custom retry status codes are correctly set."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        
        custom_codes = [429, 503, 504]
        client = AsyncVeniceClientWithRetries(
            api_key="test-key",
            retry_status_codes=custom_codes
        )
        
        # Verify custom status codes
        mock_retry.assert_called_once()
        call_args = mock_retry.call_args
        assert call_args[1]['status_forcelist'] == custom_codes

    def test_retry_transport_integration(self, mocker):
        """Test that AsyncVeniceClient properly integrates RetryTransport with httpx.AsyncClient."""
        mock_retry_transport_instance = mocker.Mock()
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport', return_value=mock_retry_transport_instance)
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        
        client = AsyncVeniceClientWithRetries(api_key="test-key", max_retries=2)
        
        # Verify RetryTransport was created and passed to httpx.AsyncClient
        mock_retry_transport.assert_called_once()
        
        # Verify httpx.AsyncClient was called with the retry transport
        mock_async_client.assert_called_once()
        call_args = mock_async_client.call_args
        assert call_args[1]['transport'] == mock_retry_transport_instance

    def test_empty_retry_status_codes_fallback(self, mocker):
        """Test that empty retry_status_codes falls back to default."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        
        client = AsyncVeniceClientWithRetries(
            api_key="test-key",
            max_retries=2,
            retry_status_codes=[]  # Empty list should fall back to default
        )
        
        # Verify default status codes are used (empty list falls back to default)
        mock_retry.assert_called_once()
        call_args = mock_retry.call_args
        assert call_args[1]['status_forcelist'] == [429, 500, 502, 503, 504]


class TestRetryConfigurationEdgeCases:
    """Test edge cases and specific retry configuration scenarios."""

    def test_sync_client_zero_retries(self, mocker):
        """Test VeniceClientWithRetries with max_retries=0 (no retries)."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        
        client = VeniceClientWithRetries(api_key="test-key", max_retries=0)
        
        # Verify Retry was configured with 0 retries
        mock_retry.assert_called_once_with(
            total=0,
            backoff_factor=2.0,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        )

    def test_async_client_zero_retries(self, mocker):
        """Test AsyncVeniceClientWithRetries with max_retries=0 (no retries)."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        
        client = AsyncVeniceClientWithRetries(api_key="test-key", max_retries=0)
        
        # Verify Retry was configured with 0 retries
        mock_retry.assert_called_once_with(
            total=0,
            backoff_factor=2.0,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        )

    def test_sync_client_custom_backoff_factor(self, mocker):
        """Test VeniceClientWithRetries with custom backoff factor."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        
        client = VeniceClientWithRetries(
            api_key="test-key",
            max_retries=3,
            retry_backoff_factor=0.5
        )
        
        # Verify Retry was configured with custom backoff factor
        mock_retry.assert_called_once_with(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        )

    def test_async_client_custom_backoff_factor(self, mocker):
        """Test AsyncVeniceClientWithRetries with custom backoff factor."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        
        client = AsyncVeniceClientWithRetries(
            api_key="test-key",
            max_retries=3,
            retry_backoff_factor=0.5
        )
        
        # Verify Retry was configured with custom backoff factor
        mock_retry.assert_called_once_with(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        )

    def test_sync_client_retry_after_header_disabled(self, mocker):
        """Test VeniceClientWithRetries with retry_respect_retry_after_header disabled."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        
        client = VeniceClientWithRetries(
            api_key="test-key",
            max_retries=3,
            retry_respect_retry_after_header=False
        )
        
        # Verify Retry was configured with retry-after header disabled
        mock_retry.assert_called_once_with(
            total=3,
            backoff_factor=2.0,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=False,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        )

    def test_async_client_comprehensive_custom_configuration(self, mocker):
        """Test AsyncVeniceClientWithRetries with all custom retry parameters."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        
        client = AsyncVeniceClientWithRetries(
            api_key="test-key",
            max_retries=5,
            retry_backoff_factor=1.5,
            retry_status_codes=[429, 500, 503],
            retry_respect_retry_after_header=False
        )
        
        # Verify Retry was configured with all custom parameters
        mock_retry.assert_called_once_with(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 503],
            respect_retry_after_header=False,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        )


class TestRetryTransportIntegration:
    """Integration tests for RetryTransport functionality."""

    def test_retry_transport_with_http_transport_options(self, mocker):
        """Test RetryTransport integration with http_transport_options."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        mock_http_transport = mocker.patch('venice_ai._client_with_retries.httpx.HTTPTransport')
        
        transport_options = {'retries': 5, 'verify': False}
        
        client = VeniceClientWithRetries(
            api_key="test-key",
            max_retries=3,
            http_transport_options=transport_options
        )
        
        # Verify HTTPTransport was created with the options
        mock_http_transport.assert_called_once_with(**transport_options)
        
        # Verify RetryTransport was created with the HTTPTransport
        mock_retry_transport.assert_called_once()
        retry_call_args = mock_retry_transport.call_args
        assert retry_call_args[1]['transport'] == mock_http_transport.return_value

    def test_async_retry_transport_with_http_transport_options(self, mocker):
        """Test async RetryTransport integration with http_transport_options."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        mock_async_http_transport = mocker.patch('venice_ai._client_with_retries.httpx.AsyncHTTPTransport')
        
        transport_options = {'retries': 5, 'verify': False}
        
        client = AsyncVeniceClientWithRetries(
            api_key="test-key",
            max_retries=3,
            http_transport_options=transport_options
        )
        
        # Verify AsyncHTTPTransport was created with the options
        mock_async_http_transport.assert_called_once_with(**transport_options)
        
        # Verify RetryTransport was created with the AsyncHTTPTransport
        mock_retry_transport.assert_called_once()
        retry_call_args = mock_retry_transport.call_args
        assert retry_call_args[1]['transport'] == mock_async_http_transport.return_value

    def test_retry_transport_fallback_when_no_retries_in_options(self, mocker):
        """Test that max_retries is added to transport options when not present."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        mock_http_transport = mocker.patch('venice_ai._client_with_retries.httpx.HTTPTransport')
        
        transport_options = {'verify': False}  # No 'retries' key
        
        client = VeniceClientWithRetries(
            api_key="test-key",
            max_retries=3,
            http_transport_options=transport_options
        )
        
        # Verify HTTPTransport was called with retries added
        expected_options = {'verify': False, 'retries': 3}
        mock_http_transport.assert_called_once_with(**expected_options)


class TestActualRetryClientInstantiation:
    """Integration tests that actually instantiate the retry clients to test real code execution."""

    def test_sync_client_instantiation_with_defaults(self):
        """Test that VeniceClientWithRetries can be instantiated with default parameters."""
        client = VeniceClientWithRetries(api_key="test-key")
        
        # Verify the client was created successfully
        assert client is not None
        assert client._max_retries == DEFAULT_MAX_RETRIES
        assert hasattr(client, '_client')
        
        # Verify retry configuration is stored correctly
        assert client._retry_backoff_factor == 2.0
        assert client._retry_status_codes == [429, 500, 502, 503, 504]
        assert client._retry_respect_retry_after_header is True
        
        # Verify the client can be used (basic functionality test)
        assert client._client is not None
        assert isinstance(client._client, httpx.Client)

    def test_sync_client_instantiation_with_custom_params(self):
        """Test that VeniceClientWithRetries can be instantiated with custom retry parameters."""
        client = VeniceClientWithRetries(
            api_key="test-key",
            max_retries=5,
            retry_backoff_factor=1.5,
            retry_status_codes=[429, 503],
            retry_respect_retry_after_header=False
        )
        
        # Verify the client was created successfully
        assert client is not None
        assert client._max_retries == 5
        
        # Verify custom retry configuration is stored correctly
        assert client._retry_backoff_factor == 1.5
        assert client._retry_status_codes == [429, 503]
        assert client._retry_respect_retry_after_header is False
        
        # Verify the client can be used (basic functionality test)
        assert client._client is not None
        assert isinstance(client._client, httpx.Client)

    def test_async_client_instantiation_with_defaults(self):
        """Test that AsyncVeniceClientWithRetries can be instantiated with default parameters."""
        client = AsyncVeniceClientWithRetries(api_key="test-key")
        
        # Verify the client was created successfully
        assert client is not None
        assert client._max_retries == DEFAULT_MAX_RETRIES
        assert hasattr(client, '_client')
        
        # Verify retry configuration is stored correctly
        assert client._retry_backoff_factor == 2.0
        assert client._retry_status_codes == [429, 500, 502, 503, 504]
        assert client._retry_respect_retry_after_header is True
        
        # Verify the client can be used (basic functionality test)
        assert client._client is not None
        assert isinstance(client._client, httpx.AsyncClient)

    def test_async_client_instantiation_with_custom_params(self):
        """Test that AsyncVeniceClientWithRetries can be instantiated with custom retry parameters."""
        client = AsyncVeniceClientWithRetries(
            api_key="test-key",
            max_retries=5,
            retry_backoff_factor=1.5,
            retry_status_codes=[429, 503],
            retry_respect_retry_after_header=False
        )
        
        # Verify the client was created successfully
        assert client is not None
        assert client._max_retries == 5
        
        # Verify custom retry configuration is stored correctly
        assert client._retry_backoff_factor == 1.5
        assert client._retry_status_codes == [429, 503]
        assert client._retry_respect_retry_after_header is False
        
        # Verify the client can be used (basic functionality test)
        assert client._client is not None
        assert isinstance(client._client, httpx.AsyncClient)

    def test_sync_client_with_zero_retries(self):
        """Test that VeniceClientWithRetries works with max_retries=0."""
        client = VeniceClientWithRetries(api_key="test-key", max_retries=0)
        
        # Verify the client was created successfully
        assert client is not None
        assert client._max_retries == 0
        
        # Verify the client can be used (basic functionality test)
        assert client._client is not None
        assert isinstance(client._client, httpx.Client)

    def test_async_client_with_zero_retries(self):
        """Test that AsyncVeniceClientWithRetries works with max_retries=0."""
        client = AsyncVeniceClientWithRetries(api_key="test-key", max_retries=0)
        
        # Verify the client was created successfully
        assert client is not None
        assert client._max_retries == 0
        
        # Verify the client can be used (basic functionality test)
        assert client._client is not None
        assert isinstance(client._client, httpx.AsyncClient)

    def test_sync_client_with_custom_transport(self):
        """Test that VeniceClientWithRetries works with a custom transport."""
        custom_transport = httpx.HTTPTransport()
        client = VeniceClientWithRetries(
            api_key="test-key",
            transport=custom_transport,
            max_retries=2
        )
        
        # Verify the client was created successfully
        assert client is not None
        assert client._max_retries == 2
        
        # Verify the client can be used (basic functionality test)
        assert client._client is not None
        assert isinstance(client._client, httpx.Client)

    def test_async_client_with_custom_transport(self):
        """Test that AsyncVeniceClientWithRetries works with a custom async transport."""
        custom_transport = httpx.AsyncHTTPTransport()
        client = AsyncVeniceClientWithRetries(
            api_key="test-key",
            async_transport=custom_transport,
            max_retries=2
        )
        
        # Verify the client was created successfully
        assert client is not None
        assert client._max_retries == 2
        
        # Verify the client can be used (basic functionality test)
        assert client._client is not None
        assert isinstance(client._client, httpx.AsyncClient)


class TestOptionalHttpxClientParameters:
    """Test coverage for optional httpx.Client and httpx.AsyncClient parameters (lines 107-128 and 224-245)."""

    def test_sync_client_proxy_parameter(self, mocker):
        """Test VeniceClientWithRetries with proxy parameter."""
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mocker.patch('venice_ai._client_with_retries.Retry')
        
        proxy_value = {"http://": "http://localhost:8080"}
        client = VeniceClientWithRetries(api_key="test-key", proxy=proxy_value)
        
        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args[1]
        assert "proxy" in call_kwargs
        assert call_kwargs["proxy"] == proxy_value

    def test_sync_client_limits_parameter(self, mocker):
        """Test VeniceClientWithRetries with limits parameter."""
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mocker.patch('venice_ai._client_with_retries.Retry')
        
        limits_value = httpx.Limits(max_connections=10)
        client = VeniceClientWithRetries(api_key="test-key", limits=limits_value)
        
        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args[1]
        assert "limits" in call_kwargs
        assert call_kwargs["limits"] == limits_value

    def test_sync_client_cert_parameter(self, mocker):
        """Test VeniceClientWithRetries with cert parameter."""
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mocker.patch('venice_ai._client_with_retries.Retry')
        
        cert_value = ("client.crt", "client.key")
        client = VeniceClientWithRetries(api_key="test-key", cert=cert_value)
        
        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args[1]
        assert "cert" in call_kwargs
        assert call_kwargs["cert"] == cert_value

    def test_sync_client_verify_parameter(self, mocker):
        """Test VeniceClientWithRetries with verify parameter."""
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mocker.patch('venice_ai._client_with_retries.Retry')
        
        verify_value = "/path/to/ca.pem"
        client = VeniceClientWithRetries(api_key="test-key", verify=verify_value)
        
        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args[1]
        assert "verify" in call_kwargs
        assert call_kwargs["verify"] == verify_value

    def test_sync_client_trust_env_parameter(self, mocker):
        """Test VeniceClientWithRetries with trust_env parameter."""
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mocker.patch('venice_ai._client_with_retries.Retry')
        
        trust_env_value = False
        client = VeniceClientWithRetries(api_key="test-key", trust_env=trust_env_value)
        
        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args[1]
        assert "trust_env" in call_kwargs
        assert call_kwargs["trust_env"] == trust_env_value

    def test_sync_client_http1_parameter(self, mocker):
        """Test VeniceClientWithRetries with http1 parameter."""
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mocker.patch('venice_ai._client_with_retries.Retry')
        
        http1_value = True
        client = VeniceClientWithRetries(api_key="test-key", http1=http1_value)
        
        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args[1]
        assert "http1" in call_kwargs
        assert call_kwargs["http1"] == http1_value

    def test_sync_client_http2_parameter(self, mocker):
        """Test VeniceClientWithRetries with http2 parameter."""
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mocker.patch('venice_ai._client_with_retries.Retry')
        
        http2_value = True
        client = VeniceClientWithRetries(api_key="test-key", http2=http2_value)
        
        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args[1]
        assert "http2" in call_kwargs
        assert call_kwargs["http2"] == http2_value

    def test_sync_client_follow_redirects_parameter(self, mocker):
        """Test VeniceClientWithRetries with follow_redirects parameter."""
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mocker.patch('venice_ai._client_with_retries.Retry')
        
        follow_redirects_value = True
        client = VeniceClientWithRetries(api_key="test-key", follow_redirects=follow_redirects_value)
        
        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args[1]
        assert "follow_redirects" in call_kwargs
        assert call_kwargs["follow_redirects"] == follow_redirects_value

    def test_sync_client_max_redirects_parameter(self, mocker):
        """Test VeniceClientWithRetries with max_redirects parameter."""
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mocker.patch('venice_ai._client_with_retries.Retry')
        
        max_redirects_value = 5
        client = VeniceClientWithRetries(api_key="test-key", max_redirects=max_redirects_value)
        
        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args[1]
        assert "max_redirects" in call_kwargs
        assert call_kwargs["max_redirects"] == max_redirects_value

    def test_sync_client_default_encoding_parameter(self, mocker):
        """Test VeniceClientWithRetries with default_encoding parameter."""
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mocker.patch('venice_ai._client_with_retries.Retry')
        
        default_encoding_value = "latin-1"
        client = VeniceClientWithRetries(api_key="test-key", default_encoding=default_encoding_value)
        
        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args[1]
        assert "default_encoding" in call_kwargs
        assert call_kwargs["default_encoding"] == default_encoding_value

    def test_sync_client_event_hooks_parameter(self, mocker):
        """Test VeniceClientWithRetries with event_hooks parameter."""
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mocker.patch('venice_ai._client_with_retries.Retry')
        
        event_hooks_value = {"request": [MagicMock()], "response": [MagicMock()]}
        client = VeniceClientWithRetries(api_key="test-key", event_hooks=event_hooks_value)
        
        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args[1]
        assert "event_hooks" in call_kwargs
        assert call_kwargs["event_hooks"] == event_hooks_value

    def test_async_client_proxy_parameter(self, mocker):
        """Test AsyncVeniceClientWithRetries with proxy parameter."""
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mocker.patch('venice_ai._client_with_retries.Retry')
        
        proxy_value = {"http://": "http://localhost:8080"}
        client = AsyncVeniceClientWithRetries(api_key="test-key", proxy=proxy_value)
        
        mock_async_client.assert_called_once()
        call_kwargs = mock_async_client.call_args[1]
        assert "proxy" in call_kwargs
        assert call_kwargs["proxy"] == proxy_value

    def test_async_client_limits_parameter(self, mocker):
        """Test AsyncVeniceClientWithRetries with limits parameter."""
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mocker.patch('venice_ai._client_with_retries.Retry')
        
        limits_value = httpx.Limits(max_connections=10)
        client = AsyncVeniceClientWithRetries(api_key="test-key", limits=limits_value)
        
        mock_async_client.assert_called_once()
        call_kwargs = mock_async_client.call_args[1]
        assert "limits" in call_kwargs
        assert call_kwargs["limits"] == limits_value

    def test_async_client_cert_parameter(self, mocker):
        """Test AsyncVeniceClientWithRetries with cert parameter."""
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mocker.patch('venice_ai._client_with_retries.Retry')
        
        cert_value = ("client.crt", "client.key")
        client = AsyncVeniceClientWithRetries(api_key="test-key", cert=cert_value)
        
        mock_async_client.assert_called_once()
        call_kwargs = mock_async_client.call_args[1]
        assert "cert" in call_kwargs
        assert call_kwargs["cert"] == cert_value

    def test_async_client_verify_parameter(self, mocker):
        """Test AsyncVeniceClientWithRetries with verify parameter."""
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mocker.patch('venice_ai._client_with_retries.Retry')
        
        verify_value = False
        client = AsyncVeniceClientWithRetries(api_key="test-key", verify=verify_value)
        
        mock_async_client.assert_called_once()
        call_kwargs = mock_async_client.call_args[1]
        assert "verify" in call_kwargs
        assert call_kwargs["verify"] == verify_value

    def test_async_client_trust_env_parameter(self, mocker):
        """Test AsyncVeniceClientWithRetries with trust_env parameter."""
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mocker.patch('venice_ai._client_with_retries.Retry')
        
        trust_env_value = False
        client = AsyncVeniceClientWithRetries(api_key="test-key", trust_env=trust_env_value)
        
        mock_async_client.assert_called_once()
        call_kwargs = mock_async_client.call_args[1]
        assert "trust_env" in call_kwargs
        assert call_kwargs["trust_env"] == trust_env_value

    def test_async_client_http1_parameter(self, mocker):
        """Test AsyncVeniceClientWithRetries with http1 parameter."""
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mocker.patch('venice_ai._client_with_retries.Retry')
        
        http1_value = True
        client = AsyncVeniceClientWithRetries(api_key="test-key", http1=http1_value)
        
        mock_async_client.assert_called_once()
        call_kwargs = mock_async_client.call_args[1]
        assert "http1" in call_kwargs
        assert call_kwargs["http1"] == http1_value

    def test_async_client_http2_parameter(self, mocker):
        """Test AsyncVeniceClientWithRetries with http2 parameter."""
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mocker.patch('venice_ai._client_with_retries.Retry')
        
        http2_value = True
        client = AsyncVeniceClientWithRetries(api_key="test-key", http2=http2_value)
        
        mock_async_client.assert_called_once()
        call_kwargs = mock_async_client.call_args[1]
        assert "http2" in call_kwargs
        assert call_kwargs["http2"] == http2_value

    def test_async_client_follow_redirects_parameter(self, mocker):
        """Test AsyncVeniceClientWithRetries with follow_redirects parameter."""
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mocker.patch('venice_ai._client_with_retries.Retry')
        
        follow_redirects_value = True
        client = AsyncVeniceClientWithRetries(api_key="test-key", follow_redirects=follow_redirects_value)
        
        mock_async_client.assert_called_once()
        call_kwargs = mock_async_client.call_args[1]
        assert "follow_redirects" in call_kwargs
        assert call_kwargs["follow_redirects"] == follow_redirects_value

    def test_async_client_max_redirects_parameter(self, mocker):
        """Test AsyncVeniceClientWithRetries with max_redirects parameter."""
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mocker.patch('venice_ai._client_with_retries.Retry')
        
        max_redirects_value = 5
        client = AsyncVeniceClientWithRetries(api_key="test-key", max_redirects=max_redirects_value)
        
        mock_async_client.assert_called_once()
        call_kwargs = mock_async_client.call_args[1]
        assert "max_redirects" in call_kwargs
        assert call_kwargs["max_redirects"] == max_redirects_value

    def test_async_client_default_encoding_parameter(self, mocker):
        """Test AsyncVeniceClientWithRetries with default_encoding parameter."""
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mocker.patch('venice_ai._client_with_retries.Retry')
        
        default_encoding_value = "latin-1"
        client = AsyncVeniceClientWithRetries(api_key="test-key", default_encoding=default_encoding_value)
        
        mock_async_client.assert_called_once()
        call_kwargs = mock_async_client.call_args[1]
        assert "default_encoding" in call_kwargs
        assert call_kwargs["default_encoding"] == default_encoding_value

    def test_async_client_event_hooks_parameter(self, mocker):
        """Test AsyncVeniceClientWithRetries with event_hooks parameter."""
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mocker.patch('venice_ai._client_with_retries.Retry')
        
        event_hooks_value = {"request": [MagicMock()], "response": [MagicMock()]}
        client = AsyncVeniceClientWithRetries(api_key="test-key", event_hooks=event_hooks_value)
        
        mock_async_client.assert_called_once()
        call_kwargs = mock_async_client.call_args[1]
        assert "event_hooks" in call_kwargs
        assert call_kwargs["event_hooks"] == event_hooks_value


class TestAlternativeTransportHandling:
    """Test coverage for alternative transport handling (lines 79, 86-89 for sync; lines 196, 203-206 for async)."""

    def test_sync_client_with_provided_transport(self, mocker):
        """Test VeniceClientWithRetries with pre-configured transport (else branch of line 79, lines 86-89)."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        mock_http_transport = mocker.patch('venice_ai._client_with_retries.httpx.HTTPTransport')
        
        # Create a mock transport
        custom_transport = MagicMock(spec=httpx.BaseTransport)
        
        client = VeniceClientWithRetries(
            api_key="test-key",
            transport=custom_transport,
            max_retries=3
        )
        
        # Verify RetryTransport was initialized with the provided transport
        mock_retry_transport.assert_called_once()
        retry_call_args = mock_retry_transport.call_args
        assert retry_call_args[1]['transport'] == custom_transport
        
        # Verify HTTPTransport was NOT called (since we provided a transport)
        mock_http_transport.assert_not_called()

    def test_async_client_with_provided_transport(self, mocker):
        """Test AsyncVeniceClientWithRetries with pre-configured async_transport (else branch of line 196, lines 203-206)."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        mock_async_http_transport = mocker.patch('venice_ai._client_with_retries.httpx.AsyncHTTPTransport')
        
        # Create a mock async transport
        custom_async_transport = AsyncMock(spec=httpx.AsyncBaseTransport)
        
        client = AsyncVeniceClientWithRetries(
            api_key="test-key",
            async_transport=custom_async_transport,
            max_retries=3
        )
        
        # Verify RetryTransport was initialized with the provided async transport
        mock_retry_transport.assert_called_once()
        retry_call_args = mock_retry_transport.call_args
        assert retry_call_args[1]['transport'] == custom_async_transport
        
        # Verify AsyncHTTPTransport was NOT called (since we provided a transport)
        mock_async_http_transport.assert_not_called()


class TestTransportOptionsWithRetriesKey:
    """Test coverage for transport_options with 'retries' key (line 83 for sync, line 200 for async)."""

    def test_sync_client_transport_options_with_retries_key(self, mocker):
        """Test VeniceClientWithRetries with 'retries' in http_transport_options (false condition for line 83)."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        mock_http_transport = mocker.patch('venice_ai._client_with_retries.httpx.HTTPTransport')
        
        # Provide transport options with 'retries' key
        transport_options = {"retries": 5, "verify": False}
        
        client = VeniceClientWithRetries(
            api_key="test-key",
            max_retries=3,  # This should be ignored since 'retries' is in transport_options
            http_transport_options=transport_options
        )
        
        # Verify HTTPTransport was called with the exact transport_options (including retries=5)
        mock_http_transport.assert_called_once_with(**transport_options)
        
        # Verify RetryTransport was initialized
        mock_retry_transport.assert_called_once()
        retry_call_args = mock_retry_transport.call_args
        assert retry_call_args[1]['transport'] == mock_http_transport.return_value

    def test_async_client_transport_options_with_retries_key(self, mocker):
        """Test AsyncVeniceClientWithRetries with 'retries' in http_transport_options (false condition for line 200)."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        mock_async_http_transport = mocker.patch('venice_ai._client_with_retries.httpx.AsyncHTTPTransport')
        
        # Provide transport options with 'retries' key
        transport_options = {"retries": 5, "verify": False}
        
        client = AsyncVeniceClientWithRetries(
            api_key="test-key",
            max_retries=3,  # This should be ignored since 'retries' is in transport_options
            http_transport_options=transport_options
        )
        
        # Verify AsyncHTTPTransport was called with the exact transport_options (including retries=5)
        mock_async_http_transport.assert_called_once_with(**transport_options)
        
        # Verify RetryTransport was initialized
        mock_retry_transport.assert_called_once()
        retry_call_args = mock_retry_transport.call_args
        assert retry_call_args[1]['transport'] == mock_async_http_transport.return_value