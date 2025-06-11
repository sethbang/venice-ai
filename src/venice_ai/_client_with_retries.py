"""
Venice AI Client with Retries

This module provides Venice AI client classes that include retry logic using httpx-retries.
These classes extend the standard VeniceClient and AsyncVeniceClient with retry capabilities.
"""

from typing import Optional, Union, Dict, Any, List
import httpx
from httpx_retries import Retry, RetryTransport

from ._client import VeniceClient
from ._async_client import AsyncVeniceClient


class VeniceClientWithRetries(VeniceClient):
    """
    Synchronous Venice AI client with retry capabilities.
    
    This client extends the standard VeniceClient with automatic retry logic
    for failed HTTP requests using httpx-retries.
    """
    
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Union[float, httpx.Timeout, None] = None,
        max_retries: int = 2,
        retry_backoff_factor: float = 2.0,
        retry_status_codes: Optional[List[int]] = None,
        retry_respect_retry_after_header: bool = True,
        http_client: Optional[httpx.Client] = None,
        **kwargs
    ):
        """
        Initialize the Venice AI client with retry capabilities.
        
        Args:
            api_key: Venice AI API key
            base_url: Base URL for the Venice AI API
            timeout: Request timeout configuration
            max_retries: Maximum number of retry attempts (default: 2)
            retry_backoff_factor: Backoff factor for retry delays (default: 2.0)
            retry_status_codes: HTTP status codes to retry on (default: [429, 500, 502, 503, 504])
            retry_respect_retry_after_header: Whether to respect Retry-After headers (default: True)
            http_client: Optional pre-configured httpx.Client
            **kwargs: Additional arguments passed to the base client
        """
        # Store retry configuration
        self._max_retries = max_retries
        self._retry_backoff_factor = retry_backoff_factor
        self._retry_status_codes = retry_status_codes or [429, 500, 502, 503, 504]
        self._retry_respect_retry_after_header = retry_respect_retry_after_header
        
        # Initialize the base client
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            http_client=http_client,
            **kwargs
        )
    
    def _build_raw_client(self) -> httpx.Client:
        """Build the raw httpx.Client with retry transport."""
        # Create retry strategy
        retry_strategy = Retry(
            total=self._max_retries,
            backoff_factor=self._retry_backoff_factor,
            status_forcelist=self._retry_status_codes,
            respect_retry_after_header=self._retry_respect_retry_after_header,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        )
        
        # Determine the base transport
        from .utils import NOT_GIVEN
        if isinstance(self._transport, type(NOT_GIVEN)) or self._transport is None:
            # Prepare transport options with max_retries fallback
            transport_options = dict(self._http_transport_options or {})
            # If 'retries' is not specified in http_transport_options, use max_retries as fallback
            if 'retries' not in transport_options:
                transport_options['retries'] = self._max_retries
            base_sync_transport: httpx.BaseTransport = httpx.HTTPTransport(**transport_options)
        else:
            # Type cast since we know it's not NOT_GIVEN at this point
            from typing import cast
            base_sync_transport = cast(httpx.BaseTransport, self._transport)
        
        # Wrap the base transport with RetryTransport
        effective_transport = RetryTransport(transport=base_sync_transport, retry=retry_strategy)
        
        # Build kwargs for httpx.Client, only including non-NOT_GIVEN values
        client_kwargs: Dict[str, Any] = {
            "base_url": self._base_url,
            "timeout": self._timeout,
            "headers": {
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            "transport": effective_transport,
        }
        
        # Add other httpx parameters if they are not NOT_GIVEN
        from .utils import NOT_GIVEN
        if not isinstance(self._proxy, type(NOT_GIVEN)):
            client_kwargs["proxy"] = self._proxy
        if not isinstance(self._limits, type(NOT_GIVEN)):
            client_kwargs["limits"] = self._limits
        if not isinstance(self._cert, type(NOT_GIVEN)):
            client_kwargs["cert"] = self._cert
        if not isinstance(self._verify, type(NOT_GIVEN)):
            client_kwargs["verify"] = self._verify
        if not isinstance(self._trust_env, type(NOT_GIVEN)):
            client_kwargs["trust_env"] = self._trust_env
        if not isinstance(self._http1, type(NOT_GIVEN)):
            client_kwargs["http1"] = self._http1
        if not isinstance(self._http2, type(NOT_GIVEN)):
            client_kwargs["http2"] = self._http2
        if not isinstance(self._follow_redirects, type(NOT_GIVEN)):
            client_kwargs["follow_redirects"] = self._follow_redirects
        if not isinstance(self._max_redirects, type(NOT_GIVEN)):
            client_kwargs["max_redirects"] = self._max_redirects
        if not isinstance(self._default_encoding, type(NOT_GIVEN)):
            client_kwargs["default_encoding"] = self._default_encoding
        if not isinstance(self._event_hooks, type(NOT_GIVEN)):
            client_kwargs["event_hooks"] = self._event_hooks
        
        return httpx.Client(**client_kwargs)


class AsyncVeniceClientWithRetries(AsyncVeniceClient):
    """
    Asynchronous Venice AI client with retry capabilities.
    
    This client extends the standard AsyncVeniceClient with automatic retry logic
    for failed HTTP requests using httpx-retries.
    """
    
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Union[float, httpx.Timeout, None] = None,
        max_retries: int = 2,
        retry_backoff_factor: float = 2.0,
        retry_status_codes: Optional[List[int]] = None,
        retry_respect_retry_after_header: bool = True,
        http_client: Optional[httpx.AsyncClient] = None,
        **kwargs
    ):
        """
        Initialize the async Venice AI client with retry capabilities.
        
        Args:
            api_key: Venice AI API key
            base_url: Base URL for the Venice AI API
            timeout: Request timeout configuration
            max_retries: Maximum number of retry attempts (default: 2)
            retry_backoff_factor: Backoff factor for retry delays (default: 2.0)
            retry_status_codes: HTTP status codes to retry on (default: [429, 500, 502, 503, 504])
            retry_respect_retry_after_header: Whether to respect Retry-After headers (default: True)
            http_client: Optional pre-configured httpx.AsyncClient
            **kwargs: Additional arguments passed to the base client
        """
        # Store retry configuration
        self._max_retries = max_retries
        self._retry_backoff_factor = retry_backoff_factor
        self._retry_status_codes = retry_status_codes or [429, 500, 502, 503, 504]
        self._retry_respect_retry_after_header = retry_respect_retry_after_header
        
        # Initialize the base client
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            http_client=http_client,
            **kwargs
        )
    
    def _build_async_raw_client(self) -> httpx.AsyncClient:
        """Build the raw httpx.AsyncClient with retry transport."""
        # Create retry strategy
        retry_strategy = Retry(
            total=self._max_retries,
            backoff_factor=self._retry_backoff_factor,
            status_forcelist=self._retry_status_codes,
            respect_retry_after_header=self._retry_respect_retry_after_header,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        )
        
        # Determine the base async transport
        from .utils import NOT_GIVEN
        if isinstance(self._async_transport, type(NOT_GIVEN)) or self._async_transport is None:
            # Prepare transport options with max_retries fallback
            transport_options = dict(self._http_transport_options or {})
            # If 'retries' is not specified in http_transport_options, use max_retries as fallback
            if 'retries' not in transport_options:
                transport_options['retries'] = self._max_retries
            base_async_transport: httpx.AsyncBaseTransport = httpx.AsyncHTTPTransport(**transport_options)
        else:
            # Type cast since we know it's not NOT_GIVEN at this point
            from typing import cast
            base_async_transport = cast(httpx.AsyncBaseTransport, self._async_transport)
        
        # Wrap the base transport with RetryTransport
        effective_async_transport = RetryTransport(transport=base_async_transport, retry=retry_strategy)
        
        # Build kwargs for httpx.AsyncClient, only including non-NOT_GIVEN values
        client_kwargs: Dict[str, Any] = {
            "base_url": self._base_url,
            "timeout": self._timeout,
            "headers": {
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            "transport": effective_async_transport,
        }
        
        # Add other httpx parameters if they are not NOT_GIVEN
        from .utils import NOT_GIVEN
        if not isinstance(self._proxy, type(NOT_GIVEN)):
            client_kwargs["proxy"] = self._proxy
        if not isinstance(self._limits, type(NOT_GIVEN)):
            client_kwargs["limits"] = self._limits
        if not isinstance(self._cert, type(NOT_GIVEN)):
            client_kwargs["cert"] = self._cert
        if not isinstance(self._verify, type(NOT_GIVEN)):
            client_kwargs["verify"] = self._verify
        if not isinstance(self._trust_env, type(NOT_GIVEN)):
            client_kwargs["trust_env"] = self._trust_env
        if not isinstance(self._http1, type(NOT_GIVEN)):
            client_kwargs["http1"] = self._http1
        if not isinstance(self._http2, type(NOT_GIVEN)):
            client_kwargs["http2"] = self._http2
        if not isinstance(self._follow_redirects, type(NOT_GIVEN)):
            client_kwargs["follow_redirects"] = self._follow_redirects
        if not isinstance(self._max_redirects, type(NOT_GIVEN)):
            client_kwargs["max_redirects"] = self._max_redirects
        if not isinstance(self._default_encoding, type(NOT_GIVEN)):
            client_kwargs["default_encoding"] = self._default_encoding
        if not isinstance(self._event_hooks, type(NOT_GIVEN)):
            client_kwargs["event_hooks"] = self._event_hooks
        
        return httpx.AsyncClient(**client_kwargs)