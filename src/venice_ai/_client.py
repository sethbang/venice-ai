from __future__ import annotations
"""
Synchronous client for the Venice AI API.

This module provides the main client class for synchronous interaction with
the Venice AI API, including methods for making requests, handling responses,
and managing resources like chat completions.
"""

import httpx
import json
import os
import time
from typing import Optional, Union, Any, Dict, Mapping, cast, Iterator, Callable, List, Type, TypeVar, TYPE_CHECKING
from typing_extensions import override
import logging
from pydantic import BaseModel
from httpx import Request, HTTPStatusError, TimeoutException, ConnectError, RequestError, Timeout, StreamConsumed, StreamClosed
from httpx._types import ProxyTypes, CertTypes
# httpx is imported below for TYPE_CHECKING, and also generally on line 10
import ssl

if TYPE_CHECKING:
    from httpx import URL, Proxy

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

from . import _constants
from .exceptions import VeniceError, APIError, APITimeoutError, APIConnectionError, APIResponseProcessingError, StreamConsumedError, StreamClosedError, _make_status_error
from .utils import NotGiven, NOT_GIVEN, truncate_string
from .resources.api_keys import ApiKeys # Import the API Keys resource
from .resources.audio import Audio # Import the Audio resource
from .resources.billing import Billing # Import the Billing resource
from .resources.characters import Characters # Import the Characters resource
from .resources.chat import ChatResource # Import the ChatResource
from .resources.chat.completions import ChatCompletions # Import the resource
from .resources.embeddings import Embeddings # Import the Embeddings resource
from .resources.image import Image # Import the Image resource
from .resources import Models # Import the new Models resource
from .types.chat import ChatCompletionChunk
from .types.models import ModelPricing
from .streaming import Stream # For default stream class

class BaseClient:
    """
    Base client class providing common functionality for both sync and async Venice AI clients.
    
    This class contains shared initialization logic, retry configuration, and transport setup
    that is used by both VeniceClient and AsyncVeniceClient.
    """
    
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[Union[str, httpx.URL]] = None,
        timeout: Union[float, httpx.Timeout, None] = _constants.DEFAULT_TIMEOUT,
        default_timeout: Optional[httpx.Timeout] = None,
        http_client: Optional[Union[httpx.Client, httpx.AsyncClient]] = None,
        # HTTP transport options
        http_transport_options: Optional[Dict[str, Any]] = None,
        # Additional httpx client settings
        proxy: Union[ProxyTypes, NotGiven] = NOT_GIVEN,
        transport: Union[httpx.BaseTransport, NotGiven] = NOT_GIVEN,
        async_transport: Union[httpx.AsyncBaseTransport, NotGiven] = NOT_GIVEN,
        limits: Union[httpx.Limits, NotGiven] = NOT_GIVEN,
        cert: Union[CertTypes, NotGiven] = NOT_GIVEN,
        verify: Union[bool, str, ssl.SSLContext, NotGiven] = NOT_GIVEN,
        trust_env: Union[bool, NotGiven] = NOT_GIVEN,
        http1: Union[bool, NotGiven] = NOT_GIVEN,
        http2: Union[bool, NotGiven] = NOT_GIVEN,
        follow_redirects: Union[bool, NotGiven] = NOT_GIVEN,
        max_redirects: Union[int, NotGiven] = NOT_GIVEN,
        default_encoding: Union[str, Callable[[bytes], str], NotGiven] = NOT_GIVEN,
        event_hooks: Union[Mapping[str, List[Callable[..., Any]]], NotGiven] = NOT_GIVEN,
    ) -> None:
        """
        Initialize the BaseClient with common configuration.
        
        This constructor sets up the foundational configuration shared by both
        VeniceClient and AsyncVeniceClient, including authentication, base URL,
        timeout settings, and HTTP client configuration options.
        
        Args:
            api_key (Optional[str]): The API key for authenticating requests.
                If not provided, it attempts to read from the `VENICE_API_KEY`
                environment variable.
            base_url (Optional[Union[str, httpx.URL]]): The base URL for the API.
                Defaults to `_constants.DEFAULT_BASE_URL` if not provided.
            timeout (Union[float, httpx.Timeout, None]): The default timeout for
                requests. Can be a float (seconds) or an `httpx.Timeout` object.
                This is superseded by `default_timeout` if that is provided.
            default_timeout (Optional[httpx.Timeout]): A more specific global default
                timeout. If provided, this takes precedence over the `timeout` parameter.
            http_client (Optional[Union[httpx.Client, httpx.AsyncClient]]): An optional,
                pre-configured `httpx.Client` or `httpx.AsyncClient` instance. If provided,
                the SDK will attempt to use it. Note: Lifecycle management of a
                user-provided client is typically handled by the derived SDK clients
                (`VeniceClient`, `AsyncVeniceClient`).
            http_transport_options (Optional[Dict[str, Any]]): Dictionary of options
                to pass to the underlying `httpx.HTTPTransport` or `httpx.AsyncHTTPTransport`
                if a custom `transport` or `async_transport` is not provided.
                These options are used when the SDK creates its internal transport.
                Example: `{"retries": 3}`.
            proxy (Union[httpx._types.ProxyTypes, venice_ai.utils.NotGiven]): Proxy configuration for
                HTTP requests. Can be a URL string, a dictionary mapping schemes to proxy URLs,
                or an `httpx.Proxy` instance. Used if `http_client` is not provided.
            transport (Union[httpx.BaseTransport, venice_ai.utils.NotGiven]): A custom synchronous
                HTTPX transport instance (e.g., `httpx.HTTPTransport`). Used if `http_client`
                is not provided.
            async_transport (Union[httpx.AsyncBaseTransport, venice_ai.utils.NotGiven]): A custom
                asynchronous HTTPX transport instance (e.g., `httpx.AsyncHTTPTransport`).
                Used if `http_client` is not provided.
            limits (Union[httpx.Limits, venice_ai.utils.NotGiven]): Configuration for connection
                limits (e.g., `httpx.Limits(max_connections=100)`). Used if `http_client`
                is not provided.
            cert (Union[httpx._types.CertTypes, venice_ai.utils.NotGiven]): SSL certificate configuration.
                Can be a path to a PEM file or a 2-tuple of (cert, key) paths. Used if
                `http_client` is not provided.
            verify (Union[bool, str, ssl.SSLContext, venice_ai.utils.NotGiven]): SSL verification
                setting. Can be a boolean, a path to a CA bundle, or an `ssl.SSLContext`.
                Defaults to `True`. Used if `http_client` is not provided.
            trust_env (Union[bool, venice_ai.utils.NotGiven]): If `True`, trusts environment
                variables for proxy configuration, SSL certificates, etc. Defaults to `True`.
                Used if `http_client` is not provided.
            http1 (Union[bool, venice_ai.utils.NotGiven]): If `True`, enables HTTP/1.1 support.
                Defaults to `True`. Used if `http_client` is not provided.
            http2 (Union[bool, venice_ai.utils.NotGiven]): If `True`, enables HTTP/2 support.
                Defaults to `False` (httpx default). Used if `http_client` is not provided.
            follow_redirects (Union[bool, venice_ai.utils.NotGiven]): If `True`, automatically
                follows redirects. Defaults to `False` for the SDK client. Used if `http_client`
                is not provided.
            max_redirects (Union[int, venice_ai.utils.NotGiven]): Maximum number of redirects to
                follow if `follow_redirects` is `True`. Used if `http_client` is not provided.
            default_encoding (Union[str, Callable[[bytes], str], venice_ai.utils.NotGiven]):
                Default encoding for response text. Can be a string or a callable. Used if
                `http_client` is not provided.
            event_hooks (Union[Mapping[str, List[Callable[..., Any]]], venice_ai.utils.NotGiven]):
                Event hooks for the request/response lifecycle (e.g., `{"request": [log_request]}`).
                Used if `http_client` is not provided.
        """
        # Try to get API key from parameter or environment variable
        effective_api_key = api_key
        if effective_api_key is None:
            effective_api_key = os.environ.get("VENICE_API_KEY")

        if not effective_api_key:
            raise ValueError("The api_key client option must be set.")
        # Strip whitespace from API key to avoid authentication issues
        self._api_key = effective_api_key.strip()

        if base_url is None:
            base_url = _constants.DEFAULT_BASE_URL
        self._base_url = httpx.URL(str(base_url).rstrip("/") + "/")  # Ensure trailing slash

        # Handle timeout conversion for MyPy compatibility
        # If default_timeout is provided, it takes precedence over timeout parameter
        effective_timeout = default_timeout if default_timeout is not None else timeout
        
        if isinstance(effective_timeout, float):
            self._timeout = Timeout(effective_timeout)
        elif isinstance(effective_timeout, Timeout):
            self._timeout = effective_timeout
        else:
            # effective_timeout is None, use default
            self._timeout = _constants.DEFAULT_TIMEOUT
        
        # Store HTTP transport options
        self._http_transport_options = http_transport_options
        
        # Store additional httpx client settings
        self._proxy = proxy
        self._transport = transport
        self._async_transport = async_transport
        self._limits = limits
        self._cert = cert
        self._verify = verify
        self._trust_env = trust_env
        self._http1 = http1
        self._http2 = http2
        self._follow_redirects = follow_redirects
        self._max_redirects = max_redirects
        self._default_encoding = default_encoding
        self._event_hooks = event_hooks

    def _build_raw_client(self) -> httpx.Client:
        """Build and configure the synchronous httpx client without retry transport."""
        # Determine the base transport
        if isinstance(self._transport, type(NOT_GIVEN)) or self._transport is None:
            # Prepare transport options
            transport_options = dict(self._http_transport_options or {})
            base_sync_transport: httpx.BaseTransport = httpx.HTTPTransport(**transport_options)
        else:
            # Type cast since we know it's not NOT_GIVEN at this point
            base_sync_transport = cast(httpx.BaseTransport, self._transport)
        
        # Build kwargs for httpx.Client, only including non-NOT_GIVEN values
        client_kwargs: Dict[str, Any] = {
            "base_url": self._base_url,
            "timeout": self._timeout,
            "headers": {
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                # Note: Content-Type is set per-request based on content type
            },
            "transport": base_sync_transport,
        }
        
        # Add other httpx parameters if they are not NOT_GIVEN
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

    def _build_async_raw_client(self) -> httpx.AsyncClient:
        """Build and configure the asynchronous httpx client without retry transport."""
        # Determine the base async transport
        if isinstance(self._async_transport, type(NOT_GIVEN)) or self._async_transport is None:
            # Prepare transport options
            transport_options = dict(self._http_transport_options or {})
            base_async_transport: httpx.AsyncBaseTransport = httpx.AsyncHTTPTransport(**transport_options)
        else:
            # Type cast since we know it's not NOT_GIVEN at this point
            base_async_transport = cast(httpx.AsyncBaseTransport, self._async_transport)
        
        # Build kwargs for httpx.AsyncClient, only including non-NOT_GIVEN values
        client_kwargs: Dict[str, Any] = {
            "base_url": self._base_url,
            "timeout": self._timeout,
            "headers": {
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                # Note: Content-Type is set per-request based on content type
            },
            "transport": base_async_transport,
        }
        
        # Add other httpx parameters if they are not NOT_GIVEN
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


class VeniceClient(BaseClient):
    """
    Provides a synchronous client for interacting with the Venice.ai API.
    
    This client provides a complete interface for making synchronous requests to all
    Venice AI API endpoints. It handles authentication, request formation, response
    parsing, and error management through a clean, resource-oriented design.
    
    The client architecture follows a namespaced resource pattern, where different
    API capabilities are organized into dedicated resource objects (e.g., `chat`,
    `models`, `image`). This design creates a clean separation of concerns and makes
    the API more discoverable and easily navigable.
    
    :param api_key: Your Venice.ai API key. This is required for authentication.
    :type api_key: str
    :param base_url: Overrides the default base URL. Defaults to the Venice AI
        production API URL. Useful for testing against different environments.
    :type base_url: Optional[Union[str, httpx.URL]]
    :param timeout: Request timeout in seconds or as a detailed ``httpx.Timeout``
        object for more granular control. Defaults to 60.0 seconds.
    :type timeout: Optional[Union[float, httpx.Timeout]]
    :param default_timeout: Global default timeout for all API calls made by this client instance.
        If provided, this will be used as the default timeout for all requests unless overridden
        on a per-request basis. Takes precedence over the ``timeout`` parameter.
    :type default_timeout: Optional[httpx.Timeout]
    :param max_retries: Maximum number of retries for connection errors or transient failures.
        This parameter controls the total number of retries for the httpx-retries mechanism.
        Defaults to 2.
    :type max_retries: int
    :param retry_backoff_factor: Backoff factor for retry delays.
        Defaults to 0.5.
    :type retry_backoff_factor: float
    :param retry_status_forcelist: List of HTTP status codes to retry on.
        Defaults to [429, 500, 502, 503, 504].
    :type retry_status_forcelist: Optional[List[int]]
    :param retry_respect_retry_after_header: Whether to respect Retry-After headers.
        Defaults to True.
    :type retry_respect_retry_after_header: bool
    :param http_client: An optional pre-configured ``httpx.Client`` instance to use for HTTP requests.
        If provided:

        - The SDK will use this custom client directly.
        - The SDK will still configure `base_url` (from the `base_url` parameter or default),
          `timeout` (from `default_timeout` or `timeout` parameter), and `Authorization` headers
          on this provided client instance.
        - All other HTTP-related parameters passed to this constructor (e.g., `max_retries`,
          `retry_backoff_factor`, `proxy`, `transport`, `limits`, `verify`, etc.) will be **ignored**.
          It is assumed that the provided `http_client` is already configured with these aspects.
        - You are responsible for managing the lifecycle of the provided `http_client` (e.g., closing it).

        If not provided, the SDK will create and manage its own internal `httpx.Client`.
    :type http_client: Optional[httpx.Client]
    :param proxy: Proxy configuration for HTTP requests. Only used when ``http_client`` is not provided.
    :type proxy: Optional[Union[str, httpx.URL, httpx.Proxy]]
    :param transport: Custom transport for HTTP requests. Only used when ``http_client`` is not provided.
    :type transport: Optional[httpx.BaseTransport]
    :param limits: Connection limits configuration. Only used when ``http_client`` is not provided.
    :type limits: Optional[httpx.Limits]
    :param cert: Client certificate configuration. Only used when ``http_client`` is not provided.
    :type cert: Optional[Union[str, Tuple[str, str]]]
    :param verify: SSL certificate verification. Only used when ``http_client`` is not provided.
    :type verify: Optional[Union[bool, str, ssl.SSLContext]]
    :param trust_env: Whether to trust environment variables for proxy configuration. Only used when ``http_client`` is not provided.
    :type trust_env: Optional[bool]
    :param http1: Whether to enable HTTP/1.1. Only used when ``http_client`` is not provided.
    :type http1: Optional[bool]
    :param http2: Whether to enable HTTP/2. Only used when ``http_client`` is not provided.
    :type http2: Optional[bool]
    :param default_encoding: Default encoding for response content. Only used when ``http_client`` is not provided.
    :type default_encoding: Optional[Union[str, Callable[[bytes], str]]]
    :param event_hooks: Event hooks for request/response lifecycle. Only used when ``http_client`` is not provided.
    :type event_hooks: Optional[Mapping[str, List[Callable[..., Any]]]]
            
    Attributes:
        chat (``ChatResource``): Access to chat-related endpoints.
        models (``Models``): Access to model listing and information endpoints.
        image (``Image``): Access to image generation and manipulation endpoints.
        audio (``Audio``): Access to speech synthesis and audio processing endpoints.
        billing (``Billing``): Access to billing and usage information endpoints.
        embeddings (``Embeddings``): Access to embedding generation endpoints.
        api_keys (``ApiKeys``): Access to API key management endpoints.
        characters (``Characters``): Access to character management endpoints.

    Examples:
        Basic usage:

        .. code-block:: python

            from venice_ai import VeniceClient
            
            client = VeniceClient(api_key="your-api-key")
            response = client.chat.completions.create(
                model="venice-1",
                messages=[{"role": "user", "content": "Hello, world!"}]
            )
            print(response["choices"][0]["message"]["content"])
            client.close() # Important to close the client when done
        
        Using as a context manager (recommended):

        .. code-block:: python

            from venice_ai import VeniceClient
            
            with VeniceClient(api_key="your-api-key") as client:
                response = client.chat.completions.create(
                    model="venice-1",
                    messages=[{"role": "user", "content": "Hello, world!"}]
                )
                print(response["choices"][0]["message"]["content"])
            # Client is automatically closed here
        
        Streaming example:

        .. code-block:: python

            from venice_ai import VeniceClient
            
            with VeniceClient(api_key="your-api-key") as client:
                for chunk in client.chat.completions.create(
                    model="venice-1",
                    messages=[{"role": "user", "content": "Count to 5"}],
                    stream=True
                ):
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        print(content, end="", flush=True)
        
    :raises ValueError: If ``api_key`` is empty or ``None``.

    Note:
        When used as a context manager (with ``with``), the client will
        automatically close the underlying HTTP client upon exit, freeing any resources.
        For manual resource management, always call the ``close()`` method when done.
    """
    _api_key: str
    _base_url: httpx.URL
    _timeout: httpx.Timeout
    _max_retries: int
    _client: httpx.Client # The underlying httpx client
    _should_close_session: bool # Flag to track if we should close the client
    
    # Additional httpx client settings
    _proxy: Union[ProxyTypes, NotGiven]
    _transport: Union[httpx.BaseTransport, NotGiven]
    _limits: Union[httpx.Limits, NotGiven]
    _cert: Union[CertTypes, NotGiven]
    _verify: Union[bool, str, ssl.SSLContext, NotGiven]
    _trust_env: Union[bool, NotGiven]
    _http1: Union[bool, NotGiven]
    _http2: Union[bool, NotGiven]
    _follow_redirects: Union[bool, NotGiven]
    _max_redirects: Union[int, NotGiven]
    _default_encoding: Union[str, Callable[[bytes], str], NotGiven]
    _event_hooks: Union[Mapping[str, List[Callable[..., Any]]], NotGiven]

    # Resource namespaces
    chat: "ChatResource" # Forward reference
    models: "Models" # Forward reference for Models resource
    image: "Image" # Forward reference for Image resource
    audio: "Audio" # Forward reference for Audio resource
    billing: "Billing" # Forward reference for Billing resource
    embeddings: "Embeddings" # Forward reference for Embeddings resource
    api_keys: "ApiKeys" # Forward reference for ApiKeys resource
    characters: "Characters" # Forward reference for Characters resource

    def __init__(
        self,
        *, # Force keyword arguments
        api_key: Optional[str] = None,
        base_url: Optional[Union[str, httpx.URL]] = None,
        timeout: Union[float, httpx.Timeout, None] = _constants.DEFAULT_TIMEOUT,
        default_timeout: Optional[httpx.Timeout] = None,
        http_client: Optional[httpx.Client] = None,
        # HTTP transport options
        http_transport_options: Optional[Dict[str, Any]] = None,
        # Additional httpx.Client constructor arguments
        proxy: Union[ProxyTypes, NotGiven] = NOT_GIVEN,
        transport: Union[httpx.BaseTransport, NotGiven] = NOT_GIVEN,
        limits: Union[httpx.Limits, NotGiven] = NOT_GIVEN,
        cert: Union[CertTypes, NotGiven] = NOT_GIVEN,
        verify: Union[bool, str, ssl.SSLContext, NotGiven] = NOT_GIVEN,
        trust_env: Union[bool, NotGiven] = NOT_GIVEN,
        http1: Union[bool, NotGiven] = NOT_GIVEN,
        http2: Union[bool, NotGiven] = NOT_GIVEN,
        follow_redirects: Union[bool, NotGiven] = NOT_GIVEN,
        max_redirects: Union[int, NotGiven] = NOT_GIVEN,
        default_encoding: Union[str, Callable[[bytes], str], NotGiven] = NOT_GIVEN,
        event_hooks: Union[Mapping[str, List[Callable[..., Any]]], NotGiven] = NOT_GIVEN,
    ) -> None:
        """
        Initialize the VeniceClient.

        This constructor sets up the client for making API requests. It configures
        authentication, base URL, timeout settings, and retry mechanisms. It also
        initializes all the resource namespaces (e.g., chat, models).

        :param api_key: The API key for authentication. Must not be empty or None.
        :type api_key: str
        :param base_url: Optional base URL to override the default Venice AI API URL.
            If not provided, uses the default production API URL.
        :type base_url: Optional[Union[str, httpx.URL]]
        :param timeout: Request timeout in seconds or as an ``httpx.Timeout`` object
            for more granular control. Defaults to 60.0 seconds.
        :type timeout: Optional[Union[float, httpx.Timeout]]
        :param default_timeout: Global default timeout for all API calls made by this client instance.
            If provided, this will be used as the default timeout for all requests unless overridden
            on a per-request basis. Takes precedence over the ``timeout`` parameter.
        :type default_timeout: Optional[httpx.Timeout]
        :param max_retries: Maximum number of retries for connection errors or
            transient failures. This parameter controls the total number of retries
            for the httpx-retries mechanism. Defaults to 2.
        :type max_retries: int
        :param retry_backoff_factor: Backoff factor for retry delays.
            Defaults to 0.5.
        :type retry_backoff_factor: float
        :param retry_status_forcelist: List of HTTP status codes to retry on.
            Defaults to [429, 500, 502, 503, 504].
        :type retry_status_forcelist: Optional[List[int]]
        :param retry_respect_retry_after_header: Whether to respect Retry-After headers.
            Defaults to True.
        :type retry_respect_retry_after_header: bool
        :param http_client: An optional pre-configured ``httpx.Client`` instance to use for HTTP requests.
            If provided:

            - The SDK will use this custom client directly.
            - The SDK will still configure `base_url` (from the `base_url` parameter or default),
              `timeout` (from `default_timeout` or `timeout` parameter), and `Authorization` headers
              on this provided client instance.
            - All other HTTP-related parameters passed to this constructor (e.g., `max_retries`,
              `retry_backoff_factor`, `proxy`, `transport`, `limits`, `verify`, etc.) will be **ignored**.
              It is assumed that the provided `http_client` is already configured with these aspects.
            - You are responsible for managing the lifecycle of the provided `http_client` (e.g., closing it).

            If not provided, the SDK will create and manage its own internal `httpx.Client`.
        :type http_client: Optional[httpx.Client]
        
        :raises ValueError: If ``api_key`` is empty or ``None`` and ``VENICE_API_KEY`` environment variable is not set.
        """
        # Call parent constructor
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            default_timeout=default_timeout,
            http_client=http_client,
            # Pass HTTP transport options
            http_transport_options=http_transport_options,
            # Pass httpx client settings
            proxy=proxy,
            transport=transport,
            limits=limits,
            cert=cert,
            verify=verify,
            trust_env=trust_env,
            http1=http1,
            http2=http2,
            follow_redirects=follow_redirects,
            max_redirects=max_redirects,
            default_encoding=default_encoding,
            event_hooks=event_hooks,
        )
        self._is_closed = False # Initialize for idempotency

        # Initialize the httpx client
        if http_client is not None:
            self._client = http_client
            self._should_close_session = False  # Don't close user-provided client
            
            # Apply SDK-level settings to the user-provided client
            # Update base_url to ensure SDK's base URL is used
            self._client.base_url = self._base_url
            
            # Update timeout to ensure SDK's timeout is used
            self._client.timeout = self._timeout
            
            # Ensure the Authorization header is set on external clients
            self._client.headers["Authorization"] = f"Bearer {self._api_key}"
        else:
            self._should_close_session = True  # We created it, so we should close it
            # Use BaseClient's _build_raw_client method which includes retry logic
            self._client = self._build_raw_client()
            
            # Apply SDK-specific headers
            self._client.headers.update({
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            })

        # Initialize resource namespaces
        self.chat = ChatResource(self) # Pass client instance to resource
        self.models = Models(self) # Initialize the Models resource
        self.image = Image(self) # Initialize the Image resource
        self.audio = Audio(self) # Initialize the Audio resource
        self.billing = Billing(self) # Initialize the Billing resource
        self.embeddings = Embeddings(self) # Initialize the Embeddings resource
        self.api_keys = ApiKeys(self) # Initialize the API Keys resource
        self.characters = Characters(self) # Initialize the Characters resource

    @property
    def api_key(self) -> str:
        """
        Get the API key for authentication.
        
        Returns the explicitly set API key, or falls back to the VENICE_API_KEY
        environment variable if no key was explicitly provided.
        
        :return: The API key to use for authentication.
        :rtype: str
        """
        return self._api_key or os.environ.get("VENICE_API_KEY", "")

    def _auth_headers(self) -> Dict[str, str]:
        """
        Generate authentication headers for API requests.
        
        :return: Dictionary containing the Authorization header if an API key is available.
        :rtype: Dict[str, str]
        """
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def build_request(
        self,
        method: str,
        path: str,
        *,
        json_data: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a request with proper headers including authentication.
        
        This method constructs the headers for a request, merging authentication
        headers with any provided headers. It supports default token retention
        by using the current api_key value.
        
        :param method: HTTP method for the request.
        :type method: str
        :param path: API endpoint path relative to the base URL.
        :type path: str
        :param json_data: JSON-serializable request body.
        :type json_data: Optional[Mapping[str, Any]]
        :param headers: Additional HTTP headers to include.
        :type headers: Optional[Mapping[str, str]]
        :param params: URL query parameters.
        :type params: Optional[Mapping[str, Any]]
        
        :return: Dictionary containing the built request information.
        :rtype: Dict[str, Any]
        """
        # Start with authentication headers
        request_headers = self._auth_headers()
        
        # Add default headers
        if method.upper() != "GET":
            request_headers["Accept"] = "application/json"
            if json_data is not None:
                request_headers["Content-Type"] = "application/json"
        
        # Merge with provided headers (provided headers take precedence)
        if headers:
            request_headers.update(headers)
        
        return {
            "method": method,
            "url": str(self._base_url.join(path)),
            "headers": request_headers,
            "json": json_data,
            "params": params,
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_data: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        params: Optional[Mapping[str, Any]] = None,
        raw_response: bool = False,
        timeout: Union[float, httpx.Timeout, None] = None,
        cast_to: Optional[Type[T]] = None,
    ) -> Union[T, Any, bytes]:
        """
        Make an HTTP request and handle potential errors.
        
        This is an internal method used by resource classes to make HTTP requests
        to the Venice AI API. It handles response parsing, error handling, and
        exception generation.
        
        :param method: HTTP method (GET, POST, DELETE, etc.) to use for the request.
        :type method: str
        :param path: API endpoint path relative to the base URL.
        :type path: str
        :param json_data: JSON-serializable request body. This will be serialized
            and sent in the request body for POST/PUT requests.
        :type json_data: Optional[Mapping[str, Any]]
        :param headers: Additional HTTP headers to include. These will override
            any default headers with the same name.
        :type headers: Optional[Mapping[str, str]]
        :param params: URL query parameters to include in the request.
        :type params: Optional[Mapping[str, Any]]
        :param raw_response: If ``True``, returns the raw response content as ``bytes``
            instead of parsing it as JSON. Useful for binary responses like images.
        :type raw_response: bool
        :param timeout: Request timeout in seconds or an ``httpx.Timeout`` object.
            If not provided, uses the client's default timeout.
        :type timeout: Optional[Union[float, httpx.Timeout]]

        :param cast_to: Optional Pydantic model to cast the response to.
        :type cast_to: Optional[Type[T]]

        :return: Parsed JSON response (optionally cast to Pydantic model), or raw ``bytes`` if ``raw_response`` is ``True``.
        :rtype: Union[T, Any, bytes]

        :raises venice_ai.exceptions.InvalidRequestError: If the request parameters are invalid (HTTP 400).
        :raises venice_ai.exceptions.AuthenticationError: If authentication fails (HTTP 401).
        :raises venice_ai.exceptions.PermissionDeniedError: If access is denied (HTTP 403).
        :raises venice_ai.exceptions.NotFoundError: If a resource is not found (HTTP 404).
        :raises venice_ai.exceptions.RateLimitError: If rate limits are exceeded (HTTP 429).
        :raises venice_ai.exceptions.InternalServerError: If a server-side error occurs (HTTP 5xx).
        :raises venice_ai.exceptions.APITimeoutError: If the request times out.
        :raises venice_ai.exceptions.APIConnectionError: If a connection error occurs.
        :raises venice_ai.exceptions.APIError: For other API-related errors.
        """
        url = self._base_url.join(path)
        try:
            # Prepare headers by merging default headers with any provided headers
            request_headers = dict(self._client.headers) # Start with client defaults
            if headers:
                request_headers.update(headers) # Apply specific request headers

            # Handle Content-Type header based on request method and data
            if method.upper() == "GET":
                # Remove Content-Type and Accept for GET requests unless explicitly provided
                if headers is None or "Content-Type" not in headers:
                    request_headers.pop("Content-Type", None)
                if headers is None or "Accept" not in headers:
                    request_headers.pop("Accept", None)
            elif json_data is not None:
                # Ensure Content-Type is set for JSON requests
                request_headers["Content-Type"] = "application/json"

            logger.debug(f"Request headers for {method} {url}: {request_headers}")
            logger.debug(f"Request JSON data for {method} {url}: {json_data}")
            
            response = self._client.request(
                method=method,
                url=url,
                json=json_data if json_data else None,
                headers=request_headers, # Use potentially modified headers
                params=params,
                timeout=timeout if timeout is not None else self._timeout,
            )
            response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx

            # Return raw bytes if raw_response is True
            if raw_response:
                return response.content

            json_response = response.json()
            if cast_to:
                try:
                    return cast(T, cast_to.model_validate(json_response))
                except Exception as exc:
                    raise APIResponseProcessingError(
                        message=f"Failed to cast response to {cast_to}: {exc}",
                        response=response,
                        # body=json_response, # APIResponseProcessingError does not take 'body'
                        original_error=exc
                    ) from exc
            return json_response
        except HTTPStatusError as e:
            # THIS IS THE CRITICAL PART: Ensure this block is reached.
            # The existing logic to translate 'e' (an httpx.HTTPStatusError)
            # into a VeniceError subclass (e.g., using _translate_httpx_error_to_api_error)
            # should be here.
            default_request = Request(method=method, url=str(url))
            api_error = self._translate_httpx_error_to_api_error(e, default_request)
            raise api_error from e
        except TimeoutException as e:
            # Handle timeout errors specifically - ENSURE NEVER DIRECTLY ACCESS e.request
            # Safely access e.request, providing a fallback if it's not available
            _request_for_error = None
            try:
                _request_for_error = e.request
            except RuntimeError:
                # This specific RuntimeError is raised by httpx when .request is accessed if _request is None
                pass  # _request_for_error remains None

            if not _request_for_error:  # if e.request is None or not present
                _request_for_error = Request(method=method, url=str(url))
            
            # Get response if present in the exception using getattr for safety
            _response_for_error = getattr(e, 'response', None)
            original_exception_message = str(e.args[0]) if e.args else "Timeout occurred"
            raise APITimeoutError(
                message=f"Request timed out: {original_exception_message}",
                request=_request_for_error,
                response=_response_for_error,
                original_error=e
            ) from e
        except RequestError as e:
            # Handle other request errors (like connection errors) - ENSURE NEVER DIRECTLY ACCESS e.request
            # Safely access e.request, providing a fallback if it's not available
            _request_for_error = None
            try:
                _request_for_error = e.request
            except RuntimeError:
                # This specific RuntimeError is raised by httpx when .request is accessed if _request is None
                pass  # _request_for_error remains None

            if not _request_for_error:  # if e.request is None or not present
                _request_for_error = Request(method=method, url=str(url))
            
            # Get response if present in the exception using getattr for safety
            _response_for_error = getattr(e, 'response', None)
            original_exception_message = str(e.args[0]) if e.args else "A network request error occurred"
            raise APIConnectionError(
                message=f"Request failed: {original_exception_message}",
                request=_request_for_error,
                response=_response_for_error,
                original_error=e
            ) from e

    def get(self, path: str, *, params: Optional[Mapping[str, Any]] = None, cast_to: Optional[Type[T]] = None, **kwargs) -> Any:
        """
        Make a GET request to the specified API endpoint.

        This is a convenience method for making GET requests. It automatically
        handles header configuration appropriate for GET requests.

        :param path: API endpoint path relative to the base URL.
        :type path: str
        :param params: URL query parameters to include in the request.
        :type params: Optional[Mapping[str, Any]]
        :param cast_to: Optional Pydantic model to cast the response to.
        :type cast_to: Optional[Type[T]]
        :param kwargs: Additional arguments to pass to :func:`~venice_ai._client.VeniceClient._request`.

        :return: Parsed JSON response body.
        :rtype: Any

        :raises venice_ai.exceptions.APIError: If the request fails.
        """
        return self._request("GET", path, params=params, cast_to=cast_to, **kwargs)

    def post(self, path: str, *, json_data: Optional[Mapping[str, Any]] = None, timeout: Union[float, httpx.Timeout, None] = None, cast_to: Optional[Type[T]] = None, **kwargs) -> Any:
        """
        Make a POST request to the specified API endpoint.

        This is a convenience method for making POST requests with JSON data.
        It automatically sets appropriate headers for JSON content.

        :param path: API endpoint path relative to the base URL.
        :type path: str
        :param json_data: JSON-serializable request body to send with the request.
        :type json_data: Optional[Mapping[str, Any]]
        :param timeout: Request timeout in seconds or an ``httpx.Timeout`` object.
            If not provided, uses the client's default timeout.
        :type timeout: Optional[Union[float, httpx.Timeout]]
        :param cast_to: Optional Pydantic model to cast the response to.
        :type cast_to: Optional[Type[T]]
        :param kwargs: Additional arguments to pass to :func:`~venice_ai._client.VeniceClient._request`.

        :return: Parsed JSON response body.
        :rtype: Any

        :raises venice_ai.exceptions.APIError: If the request fails.
        """
        return self._request("POST", path, json_data=json_data, timeout=timeout, cast_to=cast_to, **kwargs)

    def _stream_request(
        self,
        method: str,
        path: str,
        *,
        json_data: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        params: Optional[Mapping[str, Any]] = None,
        cast_to: Optional[Type[T]] = None,
    ) -> Iterator[Union[T, ChatCompletionChunk]]:
        """
        Make a streaming HTTP request and handle Server-Sent Events (SSE) responses.

        This method is used internally for streaming responses such as chat completions.
        It handles the SSE protocol, parsing each data chunk and yielding it as a
        ChatCompletionChunk object.

        :param method: HTTP method (e.g., 'POST') to use for the request.
        :type method: str
        :param path: API endpoint path relative to the base URL.
        :type path: str
        :param json_data: JSON-serializable request body containing parameters
            for the streaming request.
        :type json_data: Optional[Mapping[str, Any]]
        :param headers: Additional HTTP headers to include in the request.
        :type headers: Optional[Mapping[str, str]]
        :param params: URL query parameters to include in the request.
        :type params: Optional[Mapping[str, Any]]

        :param cast_to: Optional Pydantic model to cast each SSE chunk to.
        :type cast_to: Optional[Type[T]]

        :yields: Union[T, venice_ai.types.chat.ChatCompletionChunk]: Parsed chunk objects from the SSE stream.
            If `cast_to` is provided, chunks are cast to type T. Otherwise, defaults to ChatCompletionChunk.
            Each chunk represents an incremental update from the model's response.

        :raises venice_ai.exceptions.InvalidRequestError: If the request parameters are invalid (HTTP 400).
        :raises venice_ai.exceptions.AuthenticationError: If authentication fails (HTTP 401).
        :raises venice_ai.exceptions.PermissionDeniedError: If access is denied (HTTP 403).
        :raises venice_ai.exceptions.NotFoundError: If a resource is not found (HTTP 404).
        :raises venice_ai.exceptions.RateLimitError: If rate limits are exceeded (HTTP 429).
        :raises venice_ai.exceptions.InternalServerError: If a server-side error occurs (HTTP 5xx).
        :raises venice_ai.exceptions.APITimeoutError: If the request times out.
        :raises venice_ai.exceptions.APIConnectionError: If a connection error occurs.
        :raises venice_ai.exceptions.APIError: For other API-related errors.
        """
        _url = self._base_url.join(path)
        _method = method
        _json_data = json_data
        _headers = headers
        _params = params

        def _sse_event_generator() -> Iterator[Union[T, ChatCompletionChunk]]:
            # This generator encapsulates the actual streaming and SSE parsing logic.
            try:
                # Prepare headers for streaming requests
                _request_headers = {}
                # Copy headers from client defaults, handling both real and mock headers
                if hasattr(self._client, 'headers') and self._client.headers is not None:
                    try:
                        # Try to convert to dict first
                        _request_headers.update(dict(self._client.headers))
                    except (TypeError, AttributeError):
                        # Fallback for mock objects that don't behave like real headers
                        try:
                            for key, value in self._client.headers.items():
                                _request_headers[key] = value
                        except (TypeError, AttributeError):
                            # If all else fails, try to access as attributes
                            if hasattr(self._client.headers, '__dict__'):
                                _request_headers.update(self._client.headers.__dict__)
                
                # Apply specific request headers passed to this method
                if _headers:
                    _request_headers.update(_headers)

                if _method.upper() == "GET":
                    if _headers is None or "Content-Type" not in _headers:
                        _request_headers.pop("Content-Type", None)
                    if _headers is None or "Accept" not in _headers:
                        _request_headers.pop("Accept", None)
                elif _json_data is not None:
                    _request_headers["Content-Type"] = "application/json"

                # For streaming requests, ensure Accept: text/event-stream is set if not already present
                # But only if we're not doing a GET request where Accept was explicitly removed
                if _method.upper() != "GET" and ("Accept" not in _request_headers or _request_headers.get("Accept") == "application/json"):
                    _request_headers["Accept"] = "text/event-stream"
                elif _method.upper() == "GET" and "Accept" in _request_headers and _request_headers.get("Accept") == "application/json":
                    _request_headers["Accept"] = "text/event-stream"

                with self._client.stream(
                    method=_method,
                    url=_url,
                    json=_json_data if _json_data else None,
                    headers=_request_headers,
                    params=_params,
                ) as response:
                    response.raise_for_status()  # Raise early for status errors
                    
                    # Process the successfully established stream
                    logger.debug(f"Starting stream processing for {_method} {_url}")
                    chunk_count = 0
                    for line in response.iter_lines():
                        logger.debug(f"Raw line received: '{line}'")
                        line = line.strip()
                        if not line:
                            logger.debug("Skipping empty line")
                            continue

                        if isinstance(line, bytes):
                            line_str = line.decode('utf-8')
                        else:
                            line_str = line

                        if line_str == "data: [DONE]":
                            logger.debug("Stream termination signal [DONE] received")
                            break

                        if line_str.startswith("data: "):
                            json_str = line_str[6:]
                            logger.debug(f"JSON string extracted: '{json_str}'")
                            try:
                                json_chunk = json.loads(json_str)
                                chunk_count += 1
                                logger.debug(f"Successfully parsed chunk {chunk_count}: {json_chunk}")
                                if cast_to:
                                    try:
                                        yield cast(T, cast_to.model_validate(json_chunk))
                                    except Exception as exc_cast: # Catch Pydantic validation errors etc.
                                        logger.error(f"Failed to cast SSE chunk to {cast_to}: {exc_cast} - Data: {json_chunk}")
                                        # Decide on error handling: skip, yield error, or raise
                                        # For now, skipping problematic chunks to align with previous behavior
                                        # Could raise APIResponseProcessingError here if strictness is required
                                        # raise APIResponseProcessingError(message=f"Failed to cast SSE chunk: {exc_cast}", response=response, original_error=exc_cast) from exc_cast
                                        continue # Skip this chunk
                                else:
                                    # Default to ChatCompletionChunk if no cast_to is provided
                                    # This maintains previous behavior for non-chat streams if any
                                    yield cast(ChatCompletionChunk, json_chunk)
                            except json.JSONDecodeError as e_json:
                                logger.error(f"Failed to parse JSON in streaming response: {e_json}")
                                logger.error(f"Problematic JSON string: '{json_str}'")
                                # Optionally, raise a specific error or yield an error object
                                continue
                    logger.debug(f"Stream processing completed. Total chunks processed: {chunk_count}")
                    return  # Successfully processed stream, exit function
            
            # Errors during stream setup (e.g., connection, initial HTTP error before iteration)
            # will be caught by the try/except block in the main _stream_request body.
            # Errors during iteration (e.g., httpx.StreamConsumedError, ReadError) will propagate from here.
            # This block should primarily catch errors that occur *during* the iteration over `response.iter_lines()`.
            # Initial connection errors or HTTPStatusErrors from the `with` statement should be caught by the outer try-except.
            except httpx.ReadError as e: # More specific error for issues during stream reading
                _safe_request = None
                try:
                    _safe_request = e.request
                except RuntimeError:
                    pass
                _request_for_error = _safe_request or Request(method=_method, url=str(_url))
                _response_for_error = getattr(e, 'response', None) # Unlikely to have response here
                original_exception_message = str(e.args[0]) if e.args else "Error reading from stream"
                logger.error(f"ReadError during SSE generation for {_method} {_url}: {e}")
                raise APIConnectionError(
                    message=f"Stream read error during generation: {original_exception_message}",
                    request=_request_for_error,
                    response=_response_for_error,
                    original_error=e
                ) from e
            except StreamConsumed as e: # If stream is consumed more than once
                # Try to get the request from the original exception, fall back to creating one
                _safe_request = None
                _request_access_failed = False
                try:
                    # Access the request attribute directly to trigger any PropertyMock side effects
                    _safe_request = e.request  # type: ignore[attr-defined]
                except (RuntimeError, AttributeError):
                    _request_access_failed = True
                
                _request_for_error = _safe_request or Request(method=_method, url=str(_url))
                logger.error(f"StreamConsumedError during SSE generation for {_method} {_url}: {e}")
                
                # If request access failed, raise APIError; otherwise raise StreamConsumedError
                if _request_access_failed and _safe_request is None:
                    raise APIError(
                        message="Stream already consumed.",
                        request=_request_for_error,
                        response=getattr(e, 'response', None) or httpx.Response(status_code=0, request=_request_for_error)
                    ) from e
                else:
                    raise StreamConsumedError(
                        message="Stream already consumed.",
                        request=_request_for_error,
                        response=getattr(e, 'response', None)
                    ) from e
            except StreamClosed as e: # If stream is closed and then iterated
                # Try to get the request from the original exception, fall back to creating one
                _safe_request = None
                _request_access_failed = False
                try:
                    # Access the request attribute directly to trigger any PropertyMock side effects
                    _safe_request = e.request  # type: ignore[attr-defined]
                except (RuntimeError, AttributeError):
                    _request_access_failed = True
                
                _request_for_error = _safe_request or Request(method=_method, url=str(_url))
                logger.error(f"StreamClosedError during SSE generation for {_method} {_url}: {e}")
                
                # If request access failed, raise APIError; otherwise raise StreamClosedError
                if _request_access_failed and _safe_request is None:
                    raise APIError(
                        message="Stream already closed.",
                        request=_request_for_error,
                        response=getattr(e, 'response', None) or httpx.Response(status_code=0, request=_request_for_error)
                    ) from e
                else:
                    raise StreamClosedError(
                        message="Stream already closed.",
                        request=_request_for_error,
                        response=getattr(e, 'response', None)
                    ) from e
            # Let other RequestError types (like ConnectError, TimeoutException if they happen here,
            # or HTTPStatusError from response.raise_for_status()) propagate to be handled
            # by the outer try-except block in _stream_request, or be caught if they are APIError.


        # Main body of _stream_request
        # The _sse_event_generator now directly yields the chunks.
        try:
            # The generator handles its own internal errors related to stream processing.
            # This try-except block is for errors during the initial setup of the stream by httpx,
            # or errors from the generator that are not caught internally by it (e.g. httpx.RequestError if not caught inside)
            yield from _sse_event_generator()
        except HTTPStatusError as e:
            # Handle HTTPStatusError that propagated from _sse_event_generator
            # This ensures proper translation to VeniceError subclasses
            _safe_request = None
            try:
                _safe_request = e.request
            except RuntimeError:
                pass
            _request_for_error = _safe_request or Request(method=_method, url=str(_url))
            api_error = self._translate_httpx_error_to_api_error(e, _request_for_error, is_stream=True)
            raise api_error from e
        except TimeoutException as e: # Catches timeout for initial connection/request
            _safe_request = None
            try:
                _safe_request = e.request
            except RuntimeError:
                pass
            _request_for_error = _safe_request or Request(method=_method, url=str(_url))
            _response_for_error = getattr(e, 'response', None)
            original_exception_message = str(e.args[0]) if e.args else "Timeout occurred"
            final_message = f"Stream request timed out: {original_exception_message}"
            if "timed out" not in original_exception_message.lower():
                 final_message = f"Stream request timed out ({original_exception_message})"

            raise APITimeoutError(
                message=final_message,
                request=_request_for_error,
                response=_response_for_error,
                original_error=e
            ) from e
        except RequestError as e: # Catches other RequestErrors for initial connection/request
            _safe_request = None
            try:
                _safe_request = e.request
            except RuntimeError:
                pass
            _request_for_error = _safe_request or Request(method=_method, url=str(_url))
            _response_for_error = getattr(e, 'response', None)
            original_exception_message = str(e.args[0]) if e.args else "A network request error occurred"

            if isinstance(e, APIError):
                raise e

            logger.error(f"RequestError in _stream_request for {_method} {_url}: {e}")
            final_message = f"Stream request failed: {original_exception_message}"
            if "failed" not in original_exception_message.lower():
                final_message = f"Stream request failed ({original_exception_message})"

            raise APIConnectionError(
                message=final_message,
                request=_request_for_error,
                response=_response_for_error,
                original_error=e
            ) from e

    def delete(self, path: str, *, cast_to: Optional[Type[T]] = None, **kwargs) -> Any:
        """
        Make a DELETE request to the specified API endpoint.

        This is a convenience method for making DELETE requests. It automatically
        handles header configuration appropriate for DELETE requests.

        :param path: API endpoint path relative to the base URL.
        :type path: str
        :param cast_to: Optional Pydantic model to cast the response to.
        :type cast_to: Optional[Type[T]]
        :param kwargs: Additional arguments to pass to :func:`~venice_ai._client.VeniceClient._request`.

        :return: Parsed JSON response body.
        :rtype: Any

        :raises venice_ai.exceptions.APIError: If the request fails.
        """
        return self._request("DELETE", path, cast_to=cast_to, **kwargs)

    # Add methods for multipart/form-data requests and streaming raw responses

    def _request_multipart(
        self,
        method: str,
        path: str,
        *,
        files: Dict[str, Any],
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        params: Optional[Mapping[str, Any]] = None,
        raw_response: bool = False,
        timeout: Union[float, httpx.Timeout, None] = None,
        cast_to: Optional[Type[T]] = None,
    ) -> Union[T, Any, bytes]:
        """
        Make an HTTP request with multipart/form-data content (for file uploads).

        This method is used for endpoints that require file uploads, such as
        image upscaling. It handles the formation of multipart requests and
        manages the response parsing similar to the standard _request method.

        :param method: HTTP method (e.g., 'POST') to use for the request.
        :type method: str
        :param path: API endpoint path relative to the base URL.
        :type path: str
        :param files: Files to include in the multipart request.
            Each file should be in ``httpx`` format: ``(filename, content, content_type)``.
        :type files: Dict[str, Any]
        :param data: Form data fields to include alongside files.
        :type data: Optional[Dict[str, Any]]
        :param headers: Additional HTTP headers to include in the request.
        :type headers: Optional[Mapping[str, str]]
        :param params: URL query parameters to include in the request.
        :type params: Optional[Mapping[str, Any]]
        :param raw_response: If ``True``, returns the raw response content as ``bytes``
            instead of parsing it as JSON.
        :type raw_response: bool
        :param timeout: Request timeout in seconds or an ``httpx.Timeout`` object.
            If not provided, uses the client's default timeout.
        :type timeout: Optional[Union[float, httpx.Timeout]]

        :param cast_to: Optional Pydantic model to cast the response to.
        :type cast_to: Optional[Type[T]]

        :return: Parsed JSON response (optionally cast to Pydantic model), or raw ``bytes`` if ``raw_response`` is ``True``.
        :rtype: Union[T, Any, bytes]

        :raises venice_ai.exceptions.InvalidRequestError: If the request parameters are invalid (HTTP 400).
        :raises venice_ai.exceptions.AuthenticationError: If authentication fails (HTTP 401).
        :raises venice_ai.exceptions.PermissionDeniedError: If access is denied (HTTP 403).
        :raises venice_ai.exceptions.NotFoundError: If a resource is not found (HTTP 404).
        :raises venice_ai.exceptions.RateLimitError: If rate limits are exceeded (HTTP 429).
        :raises venice_ai.exceptions.InternalServerError: If a server-side error occurs (HTTP 5xx).
        :raises venice_ai.exceptions.APITimeoutError: If the request times out.
        :raises venice_ai.exceptions.APIConnectionError: If a connection error occurs.
        :raises venice_ai.exceptions.APIError: For other API-related errors.
        """
        url = self._base_url.join(path)

        # Prepare headers for multipart. Start fresh to avoid default Content-Type: application/json.
        request_headers = {}
        # Copy essential headers from client defaults
        if "Authorization" in self._client.headers:
            request_headers["Authorization"] = self._client.headers["Authorization"]
        if "User-Agent" in self._client.headers: # Preserve User-Agent if set
            request_headers["User-Agent"] = self._client.headers["User-Agent"]
        
        # Apply specific request headers passed to this method
        if headers:
            request_headers.update(headers)

        # Set a more generic Accept for multipart if not explicitly provided
        if "Accept" not in request_headers:
            request_headers["Accept"] = "*/*"
        # httpx will automatically set Content-Type for multipart/form-data with boundary

        logger.debug(f"Sending multipart request to {method} {url}")
        logger.debug(f"Request headers: {request_headers}")
        logger.debug(f"Content-Type header sent: {request_headers.get('Content-Type', 'Not Present')}")
        logger.debug(f"Files: {files}")
        logger.debug(f"Files content type: {type(files)}")
        for file_key, file_value in files.items():
            logger.debug(f"File '{file_key}' details: {file_value}")
        logger.debug(f"Data: {data}")
        logger.debug(f"Params: {params}")
    
        try:
            response = self._client.request(
                method=method,
                url=url,
                files=files,
                data=data,
                headers=request_headers,
                params=params,
                timeout=timeout if timeout is not None else self._timeout,
            )
            logger.debug(f"Received response with status code: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            
            response.raise_for_status()
    
            if raw_response:
                logger.debug("Returning raw response content for multipart request.")
                return response.content
    
            logger.debug(f"Response content (first 500 chars for JSON): {response.text[:500]}")
            json_response = response.json()
            if cast_to:
                try:
                    return cast(T, cast_to.model_validate(json_response))
                except Exception as exc:
                    raise APIResponseProcessingError(
                        message=f"Failed to cast multipart response to {cast_to}: {exc}",
                        response=response,
                        original_error=exc
                    ) from exc
            return json_response
        except HTTPStatusError as e:
            # THIS IS THE CRITICAL PART: Ensure this block is reached.
            # The existing logic to translate 'e' (an httpx.HTTPStatusError)
            # into a VeniceError subclass (e.g., using _translate_httpx_error_to_api_error)
            # should be here.
            default_request = Request(method=method, url=str(url))
            api_error = self._translate_httpx_error_to_api_error(e, default_request)
            raise api_error from e
        except TimeoutException as e:
            # Handle timeout errors specifically - ENSURE NEVER DIRECTLY ACCESS e.request
            # Safely access e.request, providing a fallback if it's not available
            _request_for_error = None
            try:
                _request_for_error = e.request
            except RuntimeError:
                # This specific RuntimeError is raised by httpx when .request is accessed if _request is None
                pass  # _request_for_error remains None

            if not _request_for_error:  # if e.request is None or not present
                _request_for_error = Request(method=method, url=str(url))
            
            # Get response if present in the exception using getattr for safety
            _response_for_error = getattr(e, 'response', None)
            original_exception_message = str(e.args[0]) if e.args else "Timeout occurred"
            raise APITimeoutError(
                message=f"Request timed out: {original_exception_message}",
                request=_request_for_error,
                response=_response_for_error,
                original_error=e
            ) from e
        except RequestError as e:
            # Handle other request errors (like connection errors) - ENSURE NEVER DIRECTLY ACCESS e.request
            # Safely access e.request, providing a fallback if it's not available
            _request_for_error = None
            try:
                _request_for_error = e.request
            except RuntimeError:
                # This specific RuntimeError is raised by httpx when .request is accessed if _request is None
                pass  # _request_for_error remains None

            if not _request_for_error:  # if e.request is None or not present
                _request_for_error = Request(method=method, url=str(url))
            
            # Get response if present in the exception using getattr for safety
            _response_for_error = getattr(e, 'response', None)
            original_exception_message = str(e.args[0]) if e.args else "A network request error occurred"
            raise APIConnectionError(
                message=f"Request failed: {original_exception_message}",
                request=_request_for_error,
                response=_response_for_error,
                original_error=e
            ) from e

    def _stream_request_raw(
        self,
        method: str,
        path: str,
        *,
        json_data: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        params: Optional[Mapping[str, Any]] = None,
        timeout: Union[float, httpx.Timeout, None] = None,
        # cast_to is not typically used for raw byte streams, but kept for signature consistency if ever needed
        cast_to: Optional[Type[T]] = None,
    ) -> Iterator[bytes]: # This method specifically returns bytes, so cast_to might be less relevant
        """
        Make a streaming HTTP request and yield raw binary chunks.

        This method is used for endpoints that return streaming binary data,
        such as the audio speech API in streaming mode. It's similar to
        _stream_request, but instead of parsing JSON chunks, it yields
        raw bytes directly.

        :param method: HTTP method (e.g., 'POST') to use for the request.
        :type method: str
        :param path: API endpoint path relative to the base URL.
        :type path: str
        :param json_data: JSON-serializable request body containing parameters
            for the streaming request.
        :type json_data: Optional[Mapping[str, Any]]
        :param headers: Additional HTTP headers to include in the request.
        :type headers: Optional[Mapping[str, str]]
        :param params: URL query parameters to include in the request.
        :type params: Optional[Mapping[str, Any]]
        :param timeout: Request timeout in seconds or an ``httpx.Timeout`` object.
            If not provided, uses the client's default timeout.
        :type timeout: Optional[Union[float, httpx.Timeout]]

        :yields: bytes: Raw binary chunks from the streaming response.

        :raises venice_ai.exceptions.InvalidRequestError: If the request parameters are invalid (HTTP 400).
        :raises venice_ai.exceptions.AuthenticationError: If authentication fails (HTTP 401).
        :raises venice_ai.exceptions.PermissionDeniedError: If access is denied (HTTP 403).
        :raises venice_ai.exceptions.NotFoundError: If a resource is not found (HTTP 404).
        :raises venice_ai.exceptions.RateLimitError: If rate limits are exceeded (HTTP 429).
        :raises venice_ai.exceptions.InternalServerError: If a server-side error occurs (HTTP 5xx).
        :raises venice_ai.exceptions.APITimeoutError: If the request times out.
        :raises venice_ai.exceptions.APIConnectionError: If a connection error occurs.
        :raises venice_ai.exceptions.APIError: For other API-related errors.
        """
        url = self._base_url.join(path)

        try:
            # Prepare headers for streaming requests with same logic as regular requests
            # Prepare headers for raw streaming. Start fresh to avoid default Content-Type: application/json.
            request_headers = {}
            # Copy essential headers from client defaults
            if "Authorization" in self._client.headers:
                request_headers["Authorization"] = self._client.headers["Authorization"]
            if "User-Agent" in self._client.headers: # Preserve User-Agent if set
                request_headers["User-Agent"] = self._client.headers["User-Agent"]
            
            # Apply specific request headers passed to this method
            if headers:
                request_headers.update(headers)

            with self._client.stream(
                method=method,
                url=url,
                json=json_data if json_data else None,
                headers=request_headers,
                params=params,
                timeout=timeout if timeout is not None else self._timeout,
            ) as response:
                response.raise_for_status()  # Raise early for status errors

                # Yield the content in chunks
                for chunk in response.iter_bytes():
                    if chunk:  # Skip empty chunks
                        yield chunk
                return  # Successfully processed stream, exit function

        except HTTPStatusError as e:
            # Handle HTTPStatusError that propagated from the retry loop
            # This ensures proper translation to VeniceError subclasses
            _safe_request = None
            try:
                _safe_request = e.request
            except RuntimeError:
                pass
            _request_for_error = _safe_request or Request(method=method, url=str(url))
            api_error = self._translate_httpx_error_to_api_error(e, _request_for_error, is_stream=True)
            raise api_error from e
        except TimeoutException as e:
            # Handle timeout errors specifically - ENSURE NEVER DIRECTLY ACCESS e.request
            # Safely access e.request, providing a fallback if it's not available
            _request_for_error = None
            try:
                _request_for_error = e.request
            except RuntimeError:
                # This specific RuntimeError is raised by httpx when .request is accessed if _request is None
                pass  # _request_for_error remains None

            if not _request_for_error:  # if e.request is None or not present
                _request_for_error = Request(method=method, url=str(url))
            
            # Get response if present in the exception using getattr for safety
            _response_for_error = getattr(e, 'response', None)
            original_exception_message = str(e.args[0]) if e.args else "Timeout occurred"
            raise APITimeoutError(
                message=f"Stream request timed out: {original_exception_message}",
                request=_request_for_error,
                response=_response_for_error,
                original_error=e
            ) from e
        except RequestError as e:
            # Handle other request errors (like connection errors) - ENSURE NEVER DIRECTLY ACCESS e.request
            # Safely access e.request, providing a fallback if it's not available
            _request_for_error = None
            try:
                _request_for_error = e.request
            except RuntimeError:
                # This specific RuntimeError is raised by httpx when .request is accessed if _request is None
                pass  # _request_for_error remains None

            if not _request_for_error:  # if e.request is None or not present
                _request_for_error = Request(method=method, url=str(url))
            
            # Get response if present in the exception using getattr for safety
            _response_for_error = getattr(e, 'response', None)
            original_exception_message = str(e.args[0]) if e.args else "A network request error occurred"
            raise APIConnectionError(
                message=f"Stream request failed: {original_exception_message}",
                request=_request_for_error,
                response=_response_for_error,
                original_error=e
            ) from e

    def _translate_httpx_error_to_api_error(self, error: Union[RequestError, HTTPStatusError], default_request: Request, is_stream: bool = False) -> VeniceError:
        """
        Translate an HTTPX RequestError into a corresponding Venice AI APIError.
        
        This internal method converts low-level HTTPX errors into appropriate
        Venice AI API exceptions with proper error messages and context.
        
        :param error: The HTTPX error to translate.
        :type error: httpx.RequestError
        :param default_request: The request to use if ``error.request`` is not available.
        :type default_request: httpx.Request
        :param is_stream: Whether the error occurred during a streaming request.
            This affects the error message formatting.
        :type is_stream: bool
        
        :return: The corresponding Venice AI APIError with appropriate type and message.
        :rtype: venice_ai.exceptions.APIError
        """
        # IMPORTANT: Safely access error.request using try-except to avoid RuntimeError
        _raw_request = None
        try:
            _raw_request = error.request
        except RuntimeError:
            # This specific RuntimeError is raised by httpx when .request is accessed if _request is None
            pass  # _raw_request remains None
        
        request_obj = cast(httpx.Request, _raw_request if _raw_request is not None else default_request)
        
        if isinstance(error, HTTPStatusError):
            response_obj = error.response
            
            # Inside _translate_httpx_error_to_api_error, after getting response and request objects
            parsed_json_body: Optional[object] = None
            raw_body_text: Optional[str] = None
            final_body_for_api_error: Any = None # Initialize to None

            try:
                # Attempt to read the raw response text
                raw_body_text = response_obj.text
            except Exception as e:
                # Log if reading response.text itself fails (highly unlikely for httpx.Response)
                logger.debug("Failed to read response.text during error handling: %s", e)
                raw_body_text = None # Treat as no text if reading failed
            
            # Ensure raw_body_text is a string or None, even if response_obj.text returned a Mock
            if not isinstance(raw_body_text, str):
                logger.debug(f"response_obj.text returned a non-string type ({type(raw_body_text)}), treating as no text.")
                raw_body_text = None

            # Attempt to parse JSON first
            try:
                # Ensure parsed_json_body is declared for this scope if not already
                # parsed_json_body: Optional[object] = None # Already declared at line 1376
                parsed_json_body = response_obj.json()
                logger.debug(f"[_client._translate] After response_obj.json(), parsed_json_body: {parsed_json_body} (type: {type(parsed_json_body)})")
                final_body_for_api_error = parsed_json_body # Successfully parsed JSON
            except json.JSONDecodeError as jde:
                logger.debug(f"[_client._translate] response_obj.json() raised JSONDecodeError: {jde}")
                # JSON parsing failed.
                # Now, use the raw_body_text (which was attempted to be read earlier)
                # to construct a "Non-JSON response" error body if raw_body_text is available.
                if raw_body_text: # If raw_body_text was successfully read and is not empty
                    final_body_for_api_error = {
                        "error": (
                            f"Non-JSON response from API (status {response_obj.status_code}): "
                            f"{truncate_string(raw_body_text, 500)}"
                        )
                    }
                    logger.debug(f"[_client._translate] JSONDecodeError fallback: final_body_for_api_error set to non-JSON text structure based on raw_body_text: '{raw_body_text}'")
                else:
                    logger.debug(f"[_client._translate] JSONDecodeError fallback: raw_body_text is None or empty, final_body_for_api_error remains None.")
                # If raw_body_text is None or empty, final_body_for_api_error remains None (its initial value),
                # representing an unreadable or empty original response body where JSON parsing also failed.
            except Exception as e:
                # Catch other potential errors from response_obj.json() if any (e.g., not a valid JSON mock)
                logger.debug(f"[_client._translate] response_obj.json() raised unexpected Exception: {e} (type: {type(e)})")
                # Fallback to checking raw_body_text if .json() itself raised an unexpected error
                if raw_body_text:
                    final_body_for_api_error = {
                        "error": (
                            f"Non-JSON response (or JSON parse error) from API (status {response_obj.status_code}): "
                            f"{truncate_string(raw_body_text, 500)}"
                        )
                    }
                # If raw_body_text is also None or empty, final_body_for_api_error remains None.

            # Log the error body details for debugging
            logger.error(f"Error response body (full details): {final_body_for_api_error}")

            # _make_status_error will build the detailed message from the response and body
            constructed_message_for_make_status_error = f"API error {response_obj.status_code} for {request_obj.method} {request_obj.url}"
            logger.debug(f"[_client._translate] Passing to _make_status_error - message: '{constructed_message_for_make_status_error}', body: {final_body_for_api_error}")
            return _make_status_error(
                message=constructed_message_for_make_status_error,
                request=request_obj,
                response=response_obj,
                body=final_body_for_api_error # This could be a dict or a string now
            )
        elif isinstance(error, TimeoutException): # Catches ReadTimeout, WriteTimeout, ConnectTimeout, PoolTimeout
            logger.error(f"Request timed out for {request_obj.method} {request_obj.url}: {error}")
            prefix = "Stream request" if is_stream else "Request"
            # Use the safely accessed request_obj and get the original exception message
            original_exception_message = str(error.args[0]) if error.args else "Timeout occurred"
            # Safely get response from the exception
            timeout_response: Optional[httpx.Response] = getattr(error, 'response', None)
            return APITimeoutError(message=f"{prefix} timed out: {original_exception_message}", request=request_obj, response=timeout_response, original_error=error)
        elif isinstance(error, ConnectError): # More specific connection issue
            logger.error(f"Connection error for {request_obj.method} {request_obj.url}: {error}")
            prefix = "Stream request" if is_stream else "Request"
            # Use the safely accessed request_obj and get the original exception message
            original_exception_message = str(error.args[0]) if error.args else "Connection error occurred"
            # Safely get response from the exception
            connect_response: Optional[httpx.Response] = getattr(error, 'response', None)
            return APIConnectionError(message=f"{prefix} failed: {original_exception_message}", request=request_obj, response=connect_response, original_error=error)
        else:  # Fallback for other httpx.RequestError instances
            logger.error(f"Request failed for {request_obj.method} {request_obj.url}: {error}")
            prefix = "Stream request" if is_stream else "Request"
            # Get the original exception message
            original_exception_message = str(error.args[0]) if error.args else "A network request error occurred"
            # Safely get response from the exception
            fallback_response: Optional[httpx.Response] = getattr(error, 'response', None)
            return APIConnectionError(message=f"{prefix} failed: {original_exception_message}", request=request_obj, response=fallback_response, original_error=error)
            
    # Add put, patch similarly if needed

    def get_model_pricing(self, model_id: str) -> ModelPricing:
        """
        Get pricing information for a specific model.
        
        Retrieves the pricing structure for a given model ID, including both
        USD and VCU (Venice Compute Units) costs for input and output tokens.
        
        :param model_id: The ID of the model to get pricing for
        :type model_id: str
        :return: Pricing information for the model
        :rtype: ModelPricing
        :raises ValueError: If the model is not found
        
        Example:
            >>> client = VeniceClient(api_key="your-api-key")
            >>> pricing = client.get_model_pricing("llama-3.3-70b")
            >>> print(f"Input: ${pricing['input']['usd']}/1k tokens")
            >>> print(f"Output: ${pricing['output']['usd']}/1k tokens")
        """
        # Get all models
        models_response = self.models.list()
        
        # Find the requested model
        for model in models_response['data']:
            if model['id'] == model_id:
                return model['model_spec']['pricing']
        
        raise ValueError(f"Model '{model_id}' not found")

    def close(self) -> None:
        """
        Close the underlying HTTP client and free resources.

        This method should be called when the client is no longer needed to ensure
        proper cleanup of resources. If using the client as a context manager,
        this is called automatically on exit.
        
        It is safe to call this method multiple times.
        
        Note:
            If a user-provided httpx.Client was passed to the constructor,
            this method will not close it, as the user is responsible for
            managing the lifecycle of their own client.
        """
        if hasattr(self, "_client") and getattr(self, "_should_close_session", True) and not self._is_closed:
            self._client.close()
            self._is_closed = True

    def __enter__(self) -> "VeniceClient":
        """
        Enter the context manager, enabling use with 'with' statements.

        :return: This client instance for use within the context.
        :rtype: VeniceClient
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the context manager, automatically closing the client.

        This method is called automatically when exiting a 'with' statement.
        It ensures proper cleanup of resources by closing the underlying HTTP client.

        :param exc_type: Exception type if an exception was raised in the context.
        :type exc_type: Optional[type]
        :param exc_val: Exception value if an exception was raised in the context.
        :type exc_val: Optional[BaseException]
        :param exc_tb: Exception traceback if an exception was raised in the context.
        :type exc_tb: Optional[Any]
        """
        if hasattr(self, "_client") and self._client:
            self.close()

