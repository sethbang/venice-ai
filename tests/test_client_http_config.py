import pytest
import httpx
import asyncio
import time
from typing import Optional
from unittest import mock

from venice_ai import VeniceClient, AsyncVeniceClient
from venice_ai._client_with_retries import VeniceClientWithRetries, AsyncVeniceClientWithRetries
from venice_ai.exceptions import (
    APITimeoutError,
    APIConnectionError,
    APIError,
    InternalServerError,
    InvalidRequestError
)
from venice_ai.utils import NOT_GIVEN


class TestSyncClientHttpConfig:
    """Test HTTP client configuration for synchronous VeniceClient."""

    def test_sync_client_uses_provided_http_client(self):
        """Test that VeniceClient uses a provided httpx.Client and doesn't close it."""
        # Create a mock httpx.Client instance
        mock_custom_client = mock.Mock(spec=httpx.Client)
        mock_custom_client.is_closed = False
        # Make headers support item assignment
        mock_custom_client.headers = {}
        
        # Instantiate VeniceClient with the mock client
        client = VeniceClient(api_key="fake", http_client=mock_custom_client)
        
        # Assert that the SDK client's internal client is the one provided
        assert client._client is mock_custom_client
        
        # Verify base_url was set on the mock
        assert mock_custom_client.base_url == client._base_url
        
        # Verify timeout was set on the mock
        assert mock_custom_client.timeout == client._timeout
        
        # Verify Authorization header was set
        assert mock_custom_client.headers["Authorization"] == f"Bearer {client._api_key}"
        
        # Call client.close()
        client.close()
        
        # Assert that mock_custom_client.close() was NOT called
        mock_custom_client.close.assert_not_called()

    def test_sync_client_initializes_with_custom_httpx_settings(self):
        """Test that VeniceClient initializes httpx.Client with custom settings."""
        # Define some custom settings
        custom_proxy = "http://foo.bar"
        custom_limits = httpx.Limits(max_connections=5)
        custom_verify = False
        custom_trust_env = True
        
        with mock.patch("httpx.Client", autospec=True) as mock_httpx_client_constructor:
            # Store the original instance to allow methods like .close() to be called on it
            mock_internal_client_instance = mock.MagicMock()
            mock_internal_client_instance.is_closed = False  # Simulate it's open
            mock_internal_client_instance.headers = {}  # Support item assignment
            
            def side_effect(*args, **kwargs):
                # Return the instance that can have methods called on it
                return mock_internal_client_instance
            
            mock_httpx_client_constructor.side_effect = side_effect

            client = VeniceClient(
                api_key="fake",
                proxy=custom_proxy,
                limits=custom_limits,
                verify=custom_verify,
                trust_env=custom_trust_env
            )
            
            # Ensure the constructor was called
            mock_httpx_client_constructor.assert_called_once()
            
            # Inspect the arguments passed to httpx.Client constructor
            args, kwargs = mock_httpx_client_constructor.call_args
            assert kwargs.get("proxy") == custom_proxy
            assert kwargs.get("limits") == custom_limits
            assert kwargs.get("verify") == custom_verify
            assert kwargs.get("trust_env") == custom_trust_env
            
            # Verify base_url and timeout are set
            assert kwargs.get("base_url") == client._base_url
            assert kwargs.get("timeout") == client._timeout
            
            # Verify headers are set
            headers = kwargs.get("headers", {})
            assert headers.get("Authorization") == f"Bearer fake"
            assert headers.get("Accept") == "application/json"

            # Test that the SDK still closes its *own* client
            client.close()
            mock_internal_client_instance.close.assert_called_once()

    def test_sync_client_default_initialization(self):
        """Test VeniceClient default initialization without custom httpx settings."""
        with mock.patch("httpx.Client", autospec=True) as mock_httpx_client_constructor:
            mock_internal_client_instance = mock.MagicMock()
            mock_internal_client_instance.is_closed = False
            mock_internal_client_instance.headers = {}  # Support item assignment
            
            def side_effect(*args, **kwargs):
                return mock_internal_client_instance
            
            mock_httpx_client_constructor.side_effect = side_effect

            client = VeniceClient(api_key="fake")
            
            # Assert httpx.Client constructor was called
            mock_httpx_client_constructor.assert_called_once()
            
            # Check that default values are used (i.e., custom settings are not present)
            args, kwargs = mock_httpx_client_constructor.call_args
            assert "proxy" not in kwargs
            assert "limits" not in kwargs
            assert "verify" not in kwargs
            assert "trust_env" not in kwargs
            
            # But base settings should be present
            assert kwargs.get("base_url") == client._base_url
            assert kwargs.get("timeout") == client._timeout
            
            client.close()


class TestAsyncClientHttpConfig:
    """Test HTTP client configuration for asynchronous AsyncVeniceClient."""

    @pytest.mark.asyncio
    async def test_async_client_uses_provided_http_client(self):
        """Test that AsyncVeniceClient uses a provided httpx.AsyncClient and doesn't close it."""
        # Create a mock httpx.AsyncClient instance
        mock_custom_async_client = mock.AsyncMock(spec=httpx.AsyncClient)
        mock_custom_async_client.is_closed = False
        
        # Mock the headers attribute to be a mutable dict-like object
        mock_custom_async_client.headers = httpx.Headers()
        
        # Instantiate AsyncVeniceClient with the mock client
        async_client = AsyncVeniceClient(api_key="fake", http_client=mock_custom_async_client)
        
        # Assert that the SDK client's internal client is the one provided
        assert async_client._client is mock_custom_async_client
        
        # Verify base_url was set on the mock
        assert mock_custom_async_client.base_url == async_client._base_url
        
        # Verify timeout was set on the mock
        assert mock_custom_async_client.timeout == async_client._timeout
        
        # Verify Authorization header was set
        assert async_client._client.headers["Authorization"] == f"Bearer {async_client._api_key}"
        
        # Call await async_client.aclose()
        await async_client.aclose()
        
        # Assert that mock_custom_async_client.aclose() was NOT called
        mock_custom_async_client.aclose.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_client_initializes_with_custom_httpx_settings(self):
        """Test that AsyncVeniceClient initializes httpx.AsyncClient with custom settings."""
        # Define some custom settings
        custom_proxy = "http://foo.bar"
        custom_limits = httpx.Limits(max_connections=5)
        custom_verify = False
        custom_trust_env = True
        
        with mock.patch("httpx.AsyncClient", autospec=True) as mock_httpx_async_client_constructor:
            # Store the original instance to allow methods like .aclose() to be called on it
            mock_internal_async_client_instance = mock.AsyncMock()
            mock_internal_async_client_instance.is_closed = False  # Simulate it's open
            mock_internal_async_client_instance.headers = {}  # Support item assignment
            
            def side_effect(*args, **kwargs):
                # Return the instance that can have methods called on it
                return mock_internal_async_client_instance
            
            mock_httpx_async_client_constructor.side_effect = side_effect

            async_client = AsyncVeniceClient(
                api_key="fake",
                proxy=custom_proxy,
                limits=custom_limits,
                verify=custom_verify,
                trust_env=custom_trust_env
            )
            
            # Ensure the constructor was called
            mock_httpx_async_client_constructor.assert_called_once()
            
            # Inspect the arguments passed to httpx.AsyncClient constructor
            args, kwargs = mock_httpx_async_client_constructor.call_args
            assert kwargs.get("proxy") == custom_proxy
            assert kwargs.get("limits") == custom_limits
            assert kwargs.get("verify") == custom_verify
            assert kwargs.get("trust_env") == custom_trust_env
            
            # Verify base_url and timeout are set
            assert kwargs.get("base_url") == async_client._base_url
            assert kwargs.get("timeout") == async_client._timeout
            
            # Verify headers are set
            headers = kwargs.get("headers", {})
            assert headers.get("Authorization") == f"Bearer fake"
            assert headers.get("Accept") == "application/json"

            # Test that the SDK still closes its *own* client
            await async_client.aclose()
            mock_internal_async_client_instance.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_client_default_initialization(self):
        """Test AsyncVeniceClient default initialization without custom httpx settings."""
        with mock.patch("httpx.AsyncClient", autospec=True) as mock_httpx_async_client_constructor:
            mock_internal_async_client_instance = mock.AsyncMock()
            mock_internal_async_client_instance.is_closed = False
            mock_internal_async_client_instance.headers = {}  # Support item assignment
            
            def side_effect(*args, **kwargs):
                return mock_internal_async_client_instance
            
            mock_httpx_async_client_constructor.side_effect = side_effect

            async_client = AsyncVeniceClient(api_key="fake")
            
            # Assert httpx.AsyncClient constructor was called
            mock_httpx_async_client_constructor.assert_called_once()
            
            # Check that default values are used (i.e., custom settings are not present)
            args, kwargs = mock_httpx_async_client_constructor.call_args
            assert "proxy" not in kwargs
            assert "limits" not in kwargs
            assert "verify" not in kwargs
            assert "trust_env" not in kwargs
            
            # But base settings should be present
            assert kwargs.get("base_url") == async_client._base_url
            assert kwargs.get("timeout") == async_client._timeout
            
            await async_client.aclose()


class TestClientLifecycleManagement:
    """Test client lifecycle management for both sync and async clients."""

    def test_sync_client_should_close_session_flag_with_provided_client(self):
        """Test that _should_close_session is False when http_client is provided."""
        mock_custom_client = mock.Mock(spec=httpx.Client)
        mock_custom_client.is_closed = False
        mock_custom_client.headers = {}  # Support item assignment
        
        client = VeniceClient(api_key="fake", http_client=mock_custom_client)
        
        # Should not close user-provided client
        assert client._should_close_session is False

    def test_sync_client_should_close_session_flag_with_internal_client(self):
        """Test that _should_close_session is True when client creates its own httpx.Client."""
        with mock.patch("httpx.Client", autospec=True) as mock_httpx_client_constructor:
            mock_internal_client_instance = mock.MagicMock()
            mock_internal_client_instance.is_closed = False
            mock_internal_client_instance.headers = {}  # Support item assignment
            mock_httpx_client_constructor.return_value = mock_internal_client_instance

            client = VeniceClient(api_key="fake")
            
            # Should close internally created client
            assert client._should_close_session is True

    @pytest.mark.asyncio
    async def test_async_client_should_close_session_flag_with_provided_client(self):
        """Test that _should_close_session is False when http_client is provided."""
        mock_custom_async_client = mock.AsyncMock(spec=httpx.AsyncClient)
        mock_custom_async_client.is_closed = False
        mock_custom_async_client.headers = {}  # Support item assignment
        
        async_client = AsyncVeniceClient(api_key="fake", http_client=mock_custom_async_client)
        
        # Should not close user-provided client
        assert async_client._should_close_session is False

    @pytest.mark.asyncio
    async def test_async_client_should_close_session_flag_with_internal_client(self):
        """Test that _should_close_session is True when client creates its own httpx.AsyncClient."""
        with mock.patch("httpx.AsyncClient", autospec=True) as mock_httpx_async_client_constructor:
            mock_internal_async_client_instance = mock.AsyncMock()
            mock_internal_async_client_instance.is_closed = False
            mock_internal_async_client_instance.headers = {}  # Support item assignment
            mock_httpx_async_client_constructor.return_value = mock_internal_async_client_instance

            async_client = AsyncVeniceClient(api_key="fake")
            
            # Should close internally created client
            assert async_client._should_close_session is True


class TestSyncClientDefaultTimeout:
    """Test default timeout functionality for synchronous VeniceClient."""

    def slow_request_effect(self, *args, **kwargs):
        """Mock side effect that simulates a slow request."""
        time.sleep(0.2)  # Sleep for 0.2s
        # This response will be returned if no timeout occurs before 0.2s
        mock_resp = mock.MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"object": "list", "data": []}
        mock_resp.content = b'{"object": "list", "data": []}'
        mock_resp.text = '{"object": "list", "data": []}'
        mock_resp.headers = httpx.Headers({'content-type': 'application/json'})
        return mock_resp

    def fast_request_effect(self, *args, **kwargs):
        """Mock side effect that simulates a fast, successful request."""
        mock_resp = mock.MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"object": "list", "data": []}
        mock_resp.content = b'{"object": "list", "data": []}'
        mock_resp.text = '{"object": "list", "data": []}'
        mock_resp.headers = httpx.Headers({'content-type': 'application/json'})
        return mock_resp

    def test_sync_client_default_timeout_applied(self, mocker):
        """Test that default timeout is applied when no per-request timeout is specified."""
        # Mock httpx.Client.request to raise TimeoutException when called with short timeout
        def mock_request(*args, **kwargs):
            timeout = kwargs.get('timeout')
            # Check if the timeout is short - httpx.Timeout objects have read/connect attributes
            if timeout and isinstance(timeout, httpx.Timeout):
                # For a simple timeout like httpx.Timeout(0.01), all timeout values are set to 0.01
                if (hasattr(timeout, 'read') and timeout.read and timeout.read <= 0.01) or \
                   (hasattr(timeout, 'connect') and timeout.connect and timeout.connect <= 0.01):
                    raise httpx.TimeoutException("Request timed out")
            return self.fast_request_effect(*args, **kwargs)
        
        mocker.patch('httpx.Client.request', side_effect=mock_request)

        client = VeniceClient(
            base_url="http://localhost:12345",
            api_key="test_key",
            default_timeout=httpx.Timeout(0.01)  # Very short timeout
        )
        
        with pytest.raises((httpx.TimeoutException, APITimeoutError)):
            client.models.list()

    def test_sync_client_request_timeout_overrides_default(self, mocker):
        """Test that per-request timeout overrides default timeout."""
        # Mock httpx.Client.request to raise TimeoutException when called with short timeout
        def mock_request(*args, **kwargs):
            timeout = kwargs.get('timeout')
            # Check if the timeout is short (per-request override)
            if timeout and isinstance(timeout, httpx.Timeout):
                if (hasattr(timeout, 'read') and timeout.read and timeout.read <= 0.01) or \
                   (hasattr(timeout, 'connect') and timeout.connect and timeout.connect <= 0.01):
                    raise httpx.TimeoutException("Request timed out")
            return self.fast_request_effect(*args, **kwargs)
        
        mocker.patch('httpx.Client.request', side_effect=mock_request)

        client = VeniceClient(
            base_url="http://localhost:12345",
            api_key="test_key",
            default_timeout=httpx.Timeout(60.0)  # Long default timeout
        )
        
        # Make a request with a short timeout that should override the default
        with pytest.raises((httpx.TimeoutException, APITimeoutError)):
            # Call _request directly to test timeout override
            client._request("GET", "models", timeout=httpx.Timeout(0.01))

    def test_sync_client_no_timeout_specified(self, mocker):
        """Test that requests complete successfully when no timeout is specified."""
        # Patch the underlying httpx.Client's request method
        mocker.patch('httpx.Client.request', side_effect=self.fast_request_effect)

        client = VeniceClient(
            base_url="http://localhost:12345",
            api_key="test_key"
            # No default_timeout specified
        )
        
        # Should complete successfully without timeout
        result = client.models.list()
        assert result == {"object": "list", "data": []}

    def test_sync_client_default_timeout_none(self, mocker):
        """Test that requests complete successfully when default_timeout is explicitly None."""
        # Patch the underlying httpx.Client's request method
        mocker.patch('httpx.Client.request', side_effect=self.fast_request_effect)

        client = VeniceClient(
            base_url="http://localhost:12345",
            api_key="test_key",
            default_timeout=None
        )
        
        # Should complete successfully without timeout
        result = client.models.list()
        assert result == {"object": "list", "data": []}


class TestAsyncClientDefaultTimeout:
    """Test default timeout functionality for asynchronous AsyncVeniceClient."""

    async def slow_async_request_effect(self, *args, **kwargs):
        """Mock side effect that simulates a slow async request."""
        await asyncio.sleep(0.2)  # Sleep for 0.2s
        # This response will be returned if no timeout occurs before 0.2s
        mock_resp = mock.AsyncMock(spec=httpx.Response)
        mock_resp.headers = {}
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"object": "list", "data": []}
        mock_resp.content = b'{"object": "list", "data": []}'
        mock_resp.text = '{"object": "list", "data": []}'
        mock_resp.headers = httpx.Headers({'content-type': 'application/json'})
        return mock_resp

    async def fast_async_request_effect(self, *args, **kwargs):
        """Mock side effect that simulates a fast, successful async request."""
        mock_resp = mock.AsyncMock(spec=httpx.Response)
        mock_resp.headers = {}
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"object": "list", "data": []}
        mock_resp.content = b'{"object": "list", "data": []}'
        mock_resp.text = '{"object": "list", "data": []}'
        mock_resp.headers = httpx.Headers({'content-type': 'application/json'})
        return mock_resp

    @pytest.mark.asyncio
    async def test_async_client_default_timeout_applied(self, mocker):
        """Test that default timeout is applied when no per-request timeout is specified."""
        # Mock httpx.AsyncClient.request to raise TimeoutException when called with short timeout
        async def mock_async_request(*args, **kwargs):
            timeout = kwargs.get('timeout')
            # Check if the timeout is short
            if timeout and isinstance(timeout, httpx.Timeout):
                if (hasattr(timeout, 'read') and timeout.read and timeout.read <= 0.01) or \
                   (hasattr(timeout, 'connect') and timeout.connect and timeout.connect <= 0.01):
                    raise httpx.TimeoutException("Request timed out")
            return await self.fast_async_request_effect(*args, **kwargs)
        
        mocker.patch('httpx.AsyncClient.request', side_effect=mock_async_request)

        async_client = AsyncVeniceClient(
            base_url="http://localhost:12345",
            api_key="test_key",
            default_timeout=httpx.Timeout(0.01)  # Very short timeout
        )
        
        with pytest.raises((httpx.TimeoutException, APITimeoutError)):
            await async_client.models.list()

    @pytest.mark.asyncio
    async def test_async_client_request_timeout_overrides_default(self, mocker):
        """Test that per-request timeout overrides default timeout."""
        # Mock httpx.AsyncClient.request to raise TimeoutException when called with short timeout
        async def mock_async_request(*args, **kwargs):
            timeout = kwargs.get('timeout')
            # Check if the timeout is short (per-request override)
            if timeout and isinstance(timeout, httpx.Timeout):
                if (hasattr(timeout, 'read') and timeout.read and timeout.read <= 0.01) or \
                   (hasattr(timeout, 'connect') and timeout.connect and timeout.connect <= 0.01):
                    raise httpx.TimeoutException("Request timed out")
            return await self.fast_async_request_effect(*args, **kwargs)
        
        mocker.patch('httpx.AsyncClient.request', side_effect=mock_async_request)

        async_client = AsyncVeniceClient(
            base_url="http://localhost:12345",
            api_key="test_key",
            default_timeout=httpx.Timeout(60.0)  # Long default timeout
        )
        
        # Make a request with a short timeout that should override the default
        with pytest.raises((httpx.TimeoutException, APITimeoutError)):
            # Call _request directly to test timeout override
            await async_client._request("GET", "models", timeout=httpx.Timeout(0.01))

    @pytest.mark.asyncio
    async def test_async_client_no_timeout_specified(self, mocker):
        """Test that requests complete successfully when no timeout is specified."""
        # Patch the underlying httpx.AsyncClient's request method
        mocker.patch('httpx.AsyncClient.request', side_effect=self.fast_async_request_effect)

        async_client = AsyncVeniceClient(
            base_url="http://localhost:12345",
            api_key="test_key"
            # No default_timeout specified
        )
        
        # Should complete successfully without timeout
        result = await async_client.models.list()
        assert result == {"object": "list", "data": []}


class TestSyncClientRetryFunctionality:
    """Test automatic retry functionality for synchronous VeniceClient."""

    def create_mock_response(self, status_code: int, json_data: Optional[dict] = None, request_url: str = "http://test") -> httpx.Response:
        """Helper to create mock httpx.Response objects."""
        request = httpx.Request("GET", request_url)
        if json_data is None:
            json_data = {"object": "list", "data": []}
        
        import json
        content = json.dumps(json_data).encode() if json_data else b''
        
        response = httpx.Response(
            status_code=status_code,
            request=request,
            content=content,
            headers={"content-type": "application/json"}
        )
        return response

    def test_sync_client_retry_configuration_applied(self, mocker):
        """Test that VeniceClientWithRetries applies retry configuration to the underlying transport."""
        # Mock the RetryTransport constructor to verify it's called with correct parameters
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        
        # Mock httpx.Client to avoid actual HTTP calls
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        
        client = VeniceClientWithRetries(
            base_url="http://localhost:12345",
            api_key="test_key",
            max_retries=3,
            retry_backoff_factor=0.8,
            retry_status_codes=[500, 502]
        )
        
        # Verify Retry was configured with correct parameters
        mock_retry.assert_called_once_with(
            total=3,
            backoff_factor=0.8,
            status_forcelist=[500, 502],
            respect_retry_after_header=True,  # Default value
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        )
        
        # Verify RetryTransport was created
        mock_retry_transport.assert_called_once()

    def test_sync_client_default_retry_configuration(self, mocker):
        """Test that VeniceClientWithRetries uses default retry configuration when none specified."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        
        client = VeniceClientWithRetries(
            base_url="http://localhost:12345",
            api_key="test_key",
            max_retries=3
        )
        
        # Verify Retry was configured with default parameters
        mock_retry.assert_called_once()
        call_args = mock_retry.call_args
        assert call_args[1]['total'] == 3
        assert call_args[1]['backoff_factor'] == 2.0  # Default for VeniceClientWithRetries
        assert call_args[1]['status_forcelist'] == [429, 500, 502, 503, 504]  # Default status list
        assert call_args[1]['respect_retry_after_header'] == True
        assert set(call_args[1]['allowed_methods']) == set(["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "PATCH", "POST"])

    def test_sync_client_zero_max_retries_configuration(self, mocker):
        """Test VeniceClientWithRetries with max_retries=0 (no retries)."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        
        client = VeniceClientWithRetries(
            base_url="http://localhost:12345",
            api_key="test_key",
            max_retries=0  # This test specifically validates zero retries
        )
        
        # Verify Retry was configured with 0 retries
        mock_retry.assert_called_once_with(
            total=0,
            backoff_factor=2.0,  # Updated to match our default
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]  # Updated to match our implementation
        )

    def test_sync_client_custom_status_forcelist_configuration(self, mocker):
        """Test VeniceClientWithRetries with custom retry_status_codes."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        
        custom_status_list = [500, 502]
        client = VeniceClientWithRetries(
            base_url="http://localhost:12345",
            api_key="test_key",
            max_retries=3,  # Explicitly enable retries for this test
            retry_status_codes=custom_status_list
        )
        
        # Verify Retry was configured with custom status list
        mock_retry.assert_called_once_with(
            total=3,
            backoff_factor=2.0,  # Updated to match our default
            status_forcelist=[500, 502],  # Should match the custom_status_list
            respect_retry_after_header=True,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]  # Updated to match our implementation
        )

    def test_sync_client_retry_transport_integration(self, mocker):
        """Test that VeniceClient properly integrates RetryTransport with httpx.Client."""
        mock_retry_transport_instance = mocker.Mock()
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport', return_value=mock_retry_transport_instance)
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        
        client = VeniceClientWithRetries(
            base_url="http://localhost:12345",
            api_key="test_key",
            max_retries=1  # Explicitly enable retries for this test
        )
        
        # Verify RetryTransport was created and passed to httpx.Client
        mock_retry_transport.assert_called_once()
        
        # Verify httpx.Client was called with the retry transport
        call_args = mock_client.call_args
        assert call_args is not None
        assert call_args[1]['transport'] == mock_retry_transport_instance

    def test_sync_client_retry_on_status_code_and_succeed(self, mocker):
        """Test that sync client retries on 503 status code and succeeds on second attempt."""
        # Create mock responses: first 503, then 200
        first_response = self.create_mock_response(503, {"error": "Service unavailable"})
        second_response = self.create_mock_response(200, {"object": "list", "data": []})
        
        # Mock the transport's handle_request method which is what RetryTransport calls
        mock_handle_request = mocker.patch('httpx.HTTPTransport.handle_request', side_effect=[first_response, second_response])
        
        client = VeniceClientWithRetries(
            base_url="http://localhost:12345",
            api_key="test_key",
            max_retries=2  # Explicitly enable retries for this test
        )
        
        # Should succeed after retry
        result = client.models.list()
        assert result == {"object": "list", "data": []}
        
        # Verify transport was called twice (original + 1 retry)
        assert mock_handle_request.call_count == 2

    def test_sync_client_retry_on_exception_and_succeed(self, mocker):
        """Test that sync client retries on ConnectError and succeeds on second attempt."""
        # Create successful response for second attempt
        success_response = self.create_mock_response(200, {"object": "list", "data": []})
        
        # Mock the transport's handle_request method to raise exception first, then succeed
        mock_handle_request = mocker.patch('httpx.HTTPTransport.handle_request', side_effect=[
            httpx.ConnectError("Connection failed"),
            success_response
        ])
        
        client = VeniceClientWithRetries(
            base_url="http://localhost:12345",
            api_key="test_key",
            max_retries=2  # Explicitly enable retries for this test
        )
        
        # Should succeed after retry
        result = client.models.list()
        assert result == {"object": "list", "data": []}
        
        # Verify transport was called twice (original + 1 retry)
        assert mock_handle_request.call_count == 2

    def test_sync_client_max_retries_exhausted_status_code(self, mocker):
        """Test that sync client exhausts retries on repeated 503 status codes."""
        # Create mock responses: all 503
        error_response = self.create_mock_response(503, {"error": "Service unavailable"})
        
        # Mock the transport's handle_request method to always return 503
        mock_handle_request = mocker.patch('httpx.HTTPTransport.handle_request', side_effect=[error_response, error_response])
        
        client = VeniceClientWithRetries(
            base_url="http://localhost:12345",
            api_key="test_key",
            max_retries=1  # Only 1 retry allowed - explicitly enable retries for this test
        )
        
        # Should raise InternalServerError after exhausting retries
        with pytest.raises(InternalServerError):
            client.models.list()
        
        # Verify transport was called max_retries + 1 times (original + retries)
        assert mock_handle_request.call_count == 2

    def test_sync_client_max_retries_exhausted_exception(self, mocker):
        """Test that sync client exhausts retries on repeated ConnectError exceptions."""
        # Mock the transport's handle_request method to always raise ConnectError
        mock_handle_request = mocker.patch('httpx.HTTPTransport.handle_request', side_effect=[
            httpx.ConnectError("Connection failed"),
            httpx.ConnectError("Connection failed")
        ])
        
        client = VeniceClientWithRetries(
            base_url="http://localhost:12345",
            api_key="test_key",
            max_retries=1  # Only 1 retry allowed - explicitly enable retries for this test
        )
        
        # Should raise APIConnectionError after exhausting retries
        with pytest.raises(APIConnectionError):
            client.models.list()
        
        # Verify transport was called max_retries + 1 times (original + retries)
        assert mock_handle_request.call_count == 2

    def test_sync_client_no_retry_for_non_whitelisted_status_code(self, mocker):
        """Test that sync client doesn't retry on 400 Bad Request (non-whitelisted status)."""
        # Create mock response with 400 status
        error_response = self.create_mock_response(400, {"error": "Bad request"})
        
        # Mock the transport's handle_request method to return 400
        mock_handle_request = mocker.patch('httpx.HTTPTransport.handle_request', return_value=error_response)
        
        client = VeniceClientWithRetries(
            base_url="http://localhost:12345",
            api_key="test_key",
            max_retries=2  # Explicitly enable retries for this test
        )
        
        # Should raise InvalidRequestError without retrying
        with pytest.raises(InvalidRequestError):
            client.models.list()
        
        # Verify transport was called only once (no retries for 400)
        assert mock_handle_request.call_count == 1


class TestAsyncClientRetryFunctionality:
    """Test automatic retry functionality for asynchronous AsyncVeniceClient."""

    def create_mock_response(self, status_code: int, json_data: Optional[dict] = None, request_url: str = "http://test") -> httpx.Response:
        """Helper to create mock httpx.Response objects."""
        request = httpx.Request("GET", request_url)
        if json_data is None:
            json_data = {"object": "list", "data": []}
        
        import json
        content = json.dumps(json_data).encode() if json_data else b''
        
        response = httpx.Response(
            status_code=status_code,
            request=request,
            content=content,
            headers={"content-type": "application/json"}
        )
        return response

    @pytest.mark.asyncio
    async def test_async_client_retry_configuration_applied(self, mocker):
        """Test that AsyncVeniceClient applies retry configuration to the underlying transport."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        
        async_client = AsyncVeniceClientWithRetries(
            base_url="http://localhost:12345",
            api_key="test_key",
            max_retries=3,  # Explicitly enable retries for this test
            retry_backoff_factor=0.8,
            retry_status_codes=[500, 502],
            retry_respect_retry_after_header=False,
        )
        
        # Verify Retry was configured with correct parameters
        mock_retry.assert_called_once_with(
            total=3,
            backoff_factor=0.8,
            status_forcelist=[500, 502],
            respect_retry_after_header=False,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]  # Updated to match our implementation
        )
        
        # Verify RetryTransport was created
        mock_retry_transport.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_client_default_retry_configuration(self, mocker):
        """Test that AsyncVeniceClient uses default retry configuration when none specified."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        
        async_client = AsyncVeniceClientWithRetries(
            base_url="http://localhost:12345",
            api_key="test_key",
            max_retries=3  # Explicitly enable retries for this test
        )
        
        # Verify Retry was configured with default parameters
        mock_retry.assert_called_once()
        call_args = mock_retry.call_args
        assert call_args[1]['total'] == 3  # DEFAULT_MAX_RETRIES
        assert call_args[1]['backoff_factor'] == 2.0  # Updated to match our default
        assert call_args[1]['status_forcelist'] == [429, 500, 502, 503, 504]  # Default status list
        assert call_args[1]['respect_retry_after_header'] == True
        assert set(call_args[1]['allowed_methods']) == set(["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"])  # Updated to match our implementation

    @pytest.mark.asyncio
    async def test_async_client_zero_max_retries_configuration(self, mocker):
        """Test AsyncVeniceClient with max_retries=0 (no retries)."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        
        async_client = AsyncVeniceClientWithRetries(
            base_url="http://localhost:12345",
            api_key="test_key",
            max_retries=0  # This test specifically validates zero retries
        )
        
        # Verify Retry was configured with 0 retries
        mock_retry.assert_called_once_with(
            total=0,
            backoff_factor=2.0,  # Updated to match our default
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]  # Updated to match our implementation
        )

    @pytest.mark.asyncio
    async def test_async_client_retry_transport_integration(self, mocker):
        """Test that AsyncVeniceClient properly integrates RetryTransport with httpx.AsyncClient."""
        mock_retry_transport_instance = mocker.Mock()
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport', return_value=mock_retry_transport_instance)
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        
        async_client = AsyncVeniceClientWithRetries(
            base_url="http://localhost:12345",
            api_key="test_key",
            max_retries=1  # Explicitly enable retries for this test
        )
        
        # Verify RetryTransport was created and passed to httpx.AsyncClient
        mock_retry_transport.assert_called_once()
        
        # Verify httpx.AsyncClient was called with the retry transport
        call_args = mock_async_client.call_args
        assert call_args is not None
        assert call_args[1]['transport'] == mock_retry_transport_instance

    @pytest.mark.asyncio
    async def test_async_client_retry_on_status_code_and_succeed(self, mocker):
        """Test that async client retries on 503 status code and succeeds on second attempt."""
        # Create mock responses: first 503, then 200
        first_response = self.create_mock_response(503, {"error": "Service unavailable"})
        second_response = self.create_mock_response(200, {"object": "list", "data": []})
        
        # Mock the async transport's handle_async_request method
        mock_handle_async_request = mocker.patch('httpx.AsyncHTTPTransport.handle_async_request', side_effect=[first_response, second_response])
        
        async_client = AsyncVeniceClientWithRetries(
            base_url="http://localhost:12345",
            api_key="test_key",
            max_retries=2  # Explicitly enable retries for this test
        )
        
        # Should succeed after retry
        result = await async_client.models.list()
        assert result == {"object": "list", "data": []}
        
        # Verify async transport was called twice (original + 1 retry)
        assert mock_handle_async_request.call_count == 2

    @pytest.mark.asyncio
    async def test_async_client_retry_on_exception_and_succeed(self, mocker):
        """Test that async client retries on ConnectError and succeeds on second attempt."""
        # Create successful response for second attempt
        success_response = self.create_mock_response(200, {"object": "list", "data": []})
        
        # Mock the async transport's handle_async_request method to raise exception first, then succeed
        mock_handle_async_request = mocker.patch('httpx.AsyncHTTPTransport.handle_async_request', side_effect=[
            httpx.ConnectError("Connection failed"),
            success_response
        ])
        
        async_client = AsyncVeniceClientWithRetries(
            base_url="http://localhost:12345",
            api_key="test_key",
            max_retries=2  # Explicitly enable retries for this test
        )
        
        # Should succeed after retry
        result = await async_client.models.list()
        assert result == {"object": "list", "data": []}
        
        # Verify async transport was called twice (original + 1 retry)
        assert mock_handle_async_request.call_count == 2

    @pytest.mark.asyncio
    async def test_async_client_max_retries_exhausted_status_code(self, mocker):
        """Test that async client exhausts retries on repeated 503 status codes."""
        # Create mock responses: all 503
        error_response = self.create_mock_response(503, {"error": "Service unavailable"})
        
        # Mock the async transport's handle_async_request method to always return 503
        mock_handle_async_request = mocker.patch('httpx.AsyncHTTPTransport.handle_async_request', side_effect=[error_response, error_response])
        
        async_client = AsyncVeniceClientWithRetries(
            base_url="http://localhost:12345",
            api_key="test_key",
            max_retries=1  # Only 1 retry allowed - explicitly enable retries for this test
        )
        
        # Should raise InternalServerError after exhausting retries
        with pytest.raises(InternalServerError):
            await async_client.models.list()
        
        # Verify async transport was called max_retries + 1 times (original + retries)
        assert mock_handle_async_request.call_count == 2

    @pytest.mark.asyncio
    async def test_async_client_max_retries_exhausted_exception(self, mocker):
        """Test that async client exhausts retries on repeated ConnectError exceptions."""
        # Mock the async transport's handle_async_request method to always raise ConnectError
        mock_handle_async_request = mocker.patch('httpx.AsyncHTTPTransport.handle_async_request', side_effect=[
            httpx.ConnectError("Connection failed"),
            httpx.ConnectError("Connection failed")
        ])
        
        async_client = AsyncVeniceClientWithRetries(
            base_url="http://localhost:12345",
            api_key="test_key",
            max_retries=1  # Only 1 retry allowed - explicitly enable retries for this test
        )
        
        # Should raise APIConnectionError after exhausting retries
        with pytest.raises(APIConnectionError):
            await async_client.models.list()
        
        # Verify async transport was called max_retries + 1 times (original + retries)
        assert mock_handle_async_request.call_count == 2

    @pytest.mark.asyncio
    async def test_async_client_no_retry_for_non_whitelisted_status_code(self, mocker):
        """Test that async client doesn't retry on 400 Bad Request (non-whitelisted status)."""
        # Create mock response with 400 status
        error_response = self.create_mock_response(400, {"error": "Bad request"})
        
        # Mock the async transport's handle_async_request method to return 400
        mock_handle_async_request = mocker.patch('httpx.AsyncHTTPTransport.handle_async_request', return_value=error_response)
        
        async_client = AsyncVeniceClientWithRetries(
            base_url="http://localhost:12345",
            api_key="test_key",
            max_retries=2  # Explicitly enable retries for this test
        )
        
        # Should raise InvalidRequestError without retrying
        with pytest.raises(InvalidRequestError):
            await async_client.models.list()
        
        # Verify async transport was called only once (no retries for 400)
        assert mock_handle_async_request.call_count == 1


class TestRetryConfigurationEdgeCases:
    """Test edge cases and specific retry configuration scenarios."""

    def test_sync_client_empty_retry_status_forcelist_configuration(self, mocker):
        """Test VeniceClientWithRetries with empty retry_status_codes configuration.
        
        Note: Due to the implementation using `retry_status_codes or [default_list]`,
        an empty list will fall back to the default status codes.
        """
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')

        client = VeniceClientWithRetries(
            base_url="http://localhost:12345",
            api_key="test_key",
            max_retries=2,  # Explicitly enable retries for this test
            retry_status_codes=[]  # Empty list - falls back to default
        )

        # Verify Retry was configured with default status list (due to empty list fallback)
        mock_retry.assert_called_once_with(
            total=2,  # Should match max_retries parameter
            backoff_factor=2.0,  # Updated to match our default
            status_forcelist=[429, 500, 502, 503, 504],  # Default list used (empty list falls back to default)
            respect_retry_after_header=True,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]  # Updated to match our implementation
        )

    @pytest.mark.asyncio
    async def test_async_client_custom_backoff_factor_configuration(self, mocker):
        """Test AsyncVeniceClient with custom backoff factor configuration."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        
        async_client = AsyncVeniceClientWithRetries(
            base_url="http://localhost:12345",
            api_key="test_key",
            max_retries=3,  # Explicitly enable retries for this test
            retry_backoff_factor=0.5  # Custom backoff factor
        )
        
        # Verify Retry was configured with custom backoff factor
        mock_retry.assert_called_once_with(
            total=3,
            backoff_factor=0.5,  # Custom value
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        )

    def test_sync_client_retry_after_header_disabled_configuration(self, mocker):
        """Test VeniceClientWithRetries with retry_respect_retry_after_header disabled."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_client = mocker.patch('venice_ai._client_with_retries.httpx.Client')
        
        client = VeniceClientWithRetries(
            base_url="http://localhost:12345",
            api_key="test_key",
            max_retries=3,  # Explicitly enable retries for this test
            retry_backoff_factor=0.1,
            retry_respect_retry_after_header=False,  # Disable retry-after header respect
        )
        
        # Verify Retry was configured with retry-after header disabled
        mock_retry.assert_called_once_with(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=False,  # Disabled
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        )

    @pytest.mark.asyncio
    async def test_async_client_comprehensive_custom_configuration(self, mocker):
        """Test AsyncVeniceClient with all custom retry parameters."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        mock_async_client = mocker.patch('venice_ai._client_with_retries.httpx.AsyncClient')
        
        async_client = AsyncVeniceClientWithRetries(
            base_url="http://localhost:12345",
            api_key="test_key",
            max_retries=5,  # Explicitly enable retries for this test
            retry_backoff_factor=1.5,
            retry_status_codes=[429, 500, 503],
            retry_respect_retry_after_header=False,
        )
        
        # Verify Retry was configured with all custom parameters
        mock_retry.assert_called_once_with(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 503],
            respect_retry_after_header=False,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]  # Updated to match our implementation
        )

    def test_sync_client_with_provided_http_client_no_retry_transport(self, mocker):
        """Test that VeniceClient with provided http_client doesn't create RetryTransport."""
        mock_retry_transport = mocker.patch('venice_ai._client_with_retries.RetryTransport')
        mock_retry = mocker.patch('venice_ai._client_with_retries.Retry')
        
        # Create a mock external client
        mock_external_client = mock.Mock(spec=httpx.Client)
        mock_external_client.is_closed = False
        mock_external_client.headers = {}
        
        client = VeniceClient(
            api_key="test_key",
            http_client=mock_external_client  # Provide external client
        )
        
        # Verify RetryTransport and Retry were NOT called when external client is provided
        mock_retry_transport.assert_not_called()
        mock_retry.assert_not_called()
        
        # Verify the external client is used
        assert client._client is mock_external_client

    @pytest.mark.asyncio
    async def test_async_client_default_timeout_none(self, mocker):
        """Test that requests complete successfully when default_timeout is explicitly None."""
        # Patch the underlying httpx.AsyncClient's request method
        async def fast_async_request_effect(*args, **kwargs):
            """Mock side effect that simulates a fast, successful async request."""
            mock_resp = mock.AsyncMock(spec=httpx.Response)
            mock_resp.headers = {}
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"object": "list", "data": []}
            mock_resp.content = b'{"object": "list", "data": []}'
            mock_resp.text = '{"object": "list", "data": []}'
            mock_resp.headers = httpx.Headers({'content-type': 'application/json'})
            return mock_resp
        
        mocker.patch('httpx.AsyncClient.request', side_effect=fast_async_request_effect)

        async_client = AsyncVeniceClient(
            base_url="http://localhost:12345",
            api_key="test_key",
            default_timeout=None
        )
        
        # Should complete successfully without timeout
        result = await async_client.models.list()
        assert result == {"object": "list", "data": []}