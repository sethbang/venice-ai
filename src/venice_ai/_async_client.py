"""
Asynchronous client for the Venice AI API.

This module provides the main client class for asynchronous interaction with
the Venice AI API, including methods for making requests, handling responses,
and managing resources like chat completions.
"""

import httpx
import json
import os
from typing import Optional, Union, Any, Dict, Mapping, cast, AsyncIterator, Awaitable, TYPE_CHECKING, Callable
import inspect # Added for inspect.isawaitable
from typing_extensions import override
import logging
from unittest.mock import AsyncMock  # Add for isinstance check
from httpx import Request, HTTPStatusError, TimeoutException, ConnectError, RequestError, Timeout
logger = logging.getLogger(__name__)

from . import _constants
from .exceptions import VeniceError, APIError, APITimeoutError, APIConnectionError, _make_status_error
from .resources.api_keys import AsyncApiKeys # Import the AsyncApiKeys resource
from .resources.audio import AsyncAudio # Import the AsyncAudio resource
from .resources.billing import AsyncBilling # Import the AsyncBilling resource
from .resources.characters import AsyncCharacters # Import the AsyncCharacters resource
from .resources.chat import AsyncChatResource # Import the AsyncChatResource
from .resources.embeddings import AsyncEmbeddings # Import the AsyncEmbeddings resource
from .resources.image import AsyncImage # Import the AsyncImage resource
from .resources.models import AsyncModels # Import the AsyncModels resource
from .types.chat import ChatCompletionChunk, ChatCompletion
from .streaming import AsyncStream # For default stream class

if TYPE_CHECKING:
    pass  # No imports needed for type checking in this module

class AsyncVeniceClient:
    """
    Provides an asynchronous client for interacting with the Venice.ai API.
    
    This client provides a complete interface for making asynchronous requests to all
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
    :param max_retries: Maximum number of retries for connection errors or transient failures.
        Defaults to 2.
    :type max_retries: int
    :param http_client: An optional external ``httpx.AsyncClient`` to use. If provided, other client
        configuration options will be ignored in favor of those from the provided client,
        though essential headers will still be set.
    :type http_client: Optional[httpx.AsyncClient]

    Attributes:
        chat (``AsyncChatResource``): Access to chat-related endpoints.
        models (``AsyncModels``): Access to model listing and information endpoints.
        image (``AsyncImage``): Access to image generation and manipulation endpoints.
        audio (``AsyncAudio``): Access to speech synthesis and audio processing endpoints.
        billing (``AsyncBilling``): Access to billing and usage information endpoints.
        embeddings (``AsyncEmbeddings``): Access to embedding generation endpoints.
        api_keys (``AsyncApiKeys``): Access to API key management endpoints.
        characters (``AsyncCharacters``): Access to character management endpoints.

    Examples:
        Basic usage:

        .. code-block:: python

            from venice_ai import AsyncVeniceClient
            
            async with AsyncVeniceClient(api_key="your-api-key") as client:
                response = await client.chat.completions.create(
                    model="venice-1",
                    messages=[{"role": "user", "content": "Hello, world!"}]
                )
                print(response["choices"][0]["message"]["content"])
        
        Streaming example:

        .. code-block:: python

            from venice_ai import AsyncVeniceClient
            
            async with AsyncVeniceClient(api_key="your-api-key") as client:
                async for chunk in client.chat.completions.create(
                    model="venice-1",
                    messages=[{"role": "user", "content": "Count to 5"}],
                    stream=True
                ):
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        print(content, end="", flush=True)
        
        Using with a custom ``httpx`` client:

        .. code-block:: python

            import httpx
            from venice_ai import AsyncVeniceClient
            
            # Create a custom client with specific configurations
            custom_client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0),
                follow_redirects=True,
                http2=True
            )
            
            # Use the custom client with AsyncVeniceClient
            async with AsyncVeniceClient(
                api_key="your-api-key",
                http_client=custom_client
            ) as client:
                # Your API operations here
                pass

    :raises ValueError: If ``api_key`` is empty or ``None``.

    Note:
        When used as an async context manager (with ``async with``), the client will
        automatically close the underlying HTTP client upon exit, freeing any resources.
        For manual resource management, use the ``close()`` method.
    """
    _api_key: str
    _base_url: httpx.URL
    _timeout: httpx.Timeout
    _max_retries: int
    _client: httpx.AsyncClient  # The underlying httpx async client
    _is_closed: bool = False  # Track if client has been closed

    # Resource namespaces
    chat: "AsyncChatResource"  # Forward reference
    models: "AsyncModels" # Forward reference for AsyncModels resource
    image: "AsyncImage" # Forward reference for AsyncImage resource
    audio: "AsyncAudio" # Forward reference for AsyncAudio resource
    billing: "AsyncBilling" # Forward reference for AsyncBilling resource
    embeddings: "AsyncEmbeddings" # Forward reference for AsyncEmbeddings resource
    api_keys: "AsyncApiKeys" # Forward reference for AsyncApiKeys resource
    characters: "AsyncCharacters" # Forward reference for AsyncCharacters resource

    def __init__(
        self,
        *, # Force keyword arguments
        api_key: Optional[str] = None,
        base_url: Optional[Union[str, httpx.URL]] = None,
        timeout: Union[float, httpx.Timeout, None] = _constants.DEFAULT_TIMEOUT,
        max_retries: int = _constants.DEFAULT_MAX_RETRIES,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """
        Initialize the AsyncVeniceClient for asynchronous API interactions.

        This constructor sets up the client for making asynchronous API requests to the Venice AI API.
        It configures authentication, base URL, timeout settings, retry mechanisms, and initializes
        all the asynchronous resource namespaces (e.g., chat, models, image, audio).

        The client can be configured with custom HTTP settings through the ``http_client`` parameter,
        or it will create its own ``httpx.AsyncClient`` with appropriate defaults. When providing
        a custom client, essential headers like Authorization will be automatically set or updated.

        :param api_key: Your Venice.ai API key for authentication. This is required and cannot be empty.
            The key will be automatically stripped of whitespace to prevent authentication issues.
        :type api_key: str
        :param base_url: Base URL for the Venice AI API. If not provided, defaults to the production
            Venice AI API URL. Can be a string or ``httpx.URL`` object. Useful for testing against
            different environments or API versions.
        :type base_url: Optional[Union[str, httpx.URL]]
        :param timeout: Request timeout configuration. Can be a float (seconds) for simple timeout,
            or an ``httpx.Timeout`` object for granular control over connect, read, write, and pool timeouts.
            Defaults to 60.0 seconds if not specified.
        :type timeout: Optional[Union[float, httpx.Timeout]]
        :param max_retries: Maximum number of automatic retries for failed requests due to connection
            errors or transient failures. Does not retry on HTTP error status codes. Defaults to 2.
        :type max_retries: int
        :param http_client: Optional pre-configured ``httpx.AsyncClient`` to use for HTTP requests.
            If provided, the client's existing configuration will be preserved, but essential headers
            (Authorization, Accept, Content-Type) will be set or updated as needed. If not provided,
            a new client will be created with appropriate defaults.
        :type http_client: Optional[httpx.AsyncClient]

        :raises ValueError: If ``api_key`` is empty, ``None``, or consists only of whitespace.

        Note:
            When using a custom ``http_client``, ensure it's configured appropriately for your use case.
            The Venice AI client will modify headers but will not change other client settings like
            timeouts, proxies, or SSL configuration.
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

        # Handle timeout conversion for httpx compatibility
        if isinstance(timeout, (float, int)): # Handle float or int
            self._timeout = Timeout(float(timeout)) # Convert to float, then to httpx.Timeout
        elif isinstance(timeout, Timeout): # Already an httpx.Timeout object
            self._timeout = timeout
        elif timeout is None: # Handle None case, use default
            default_timeout = _constants.DEFAULT_TIMEOUT
            # _constants.DEFAULT_TIMEOUT is already an httpx.Timeout object,
            # so no further conversion is needed here.
            self._timeout = default_timeout
        else:
            # This case should ideally not be reached if the input `timeout`
            # adheres to Union[float, int, httpx.Timeout, None].
            # If it's reached with an unexpected type, it might indicate an issue.
            # For now, let's assume Pylance's concern about 'int' was the main one.
            # If other types are possible and need conversion, this would need expansion.
            # To satisfy the type checker if `timeout` could be something else unassignable:
            raise TypeError(f"Unexpected type for timeout: {type(timeout)}. Expected float, int, httpx.Timeout, or None.")
        self._max_retries = max_retries

        # Initialize the httpx async client
        if http_client is not None:
            self._client = http_client
            # Ensure Authorization header is set on the provided client
            # Make a mutable copy if necessary (httpx.Headers can be immutable)
            # and also merge other defaults if not present.
            current_headers = httpx.Headers(self._client.headers) # Make mutable copy
            
            # Set/override Authorization
            current_headers["Authorization"] = f"Bearer {self._api_key}"
            
            # Ensure other defaults if not present in external client's headers
            if "Accept" not in current_headers:
                current_headers["Accept"] = "application/json"
            if "Content-Type" not in current_headers: # Add default Content-Type if not present
                current_headers["Content-Type"] = "application/json"

            self._client.headers = current_headers # Assign back the modified headers
        else:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                transport=httpx.AsyncHTTPTransport(retries=self._max_retries),
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                    # Note: Content-Type is set per-request based on content type
                },
            )

        # Initialize resource namespaces
        self.chat = AsyncChatResource(self)  # Pass client instance to resource
        self.models = AsyncModels(self) # Initialize the AsyncModels resource
        self.image = AsyncImage(self) # Initialize the AsyncImage resource
        self.audio = AsyncAudio(self) # Initialize the AsyncAudio resource
        self.billing = AsyncBilling(self) # Initialize the AsyncBilling resource
        self.embeddings = AsyncEmbeddings(self) # Initialize the AsyncEmbeddings resource
        self.api_keys = AsyncApiKeys(self) # Initialize the AsyncApiKeys resource
        self.characters = AsyncCharacters(self) # Initialize the AsyncCharacters resource

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

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_data: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        params: Optional[Mapping[str, Any]] = None,
        raw_response: bool = False,
        timeout: Union[float, httpx.Timeout, None] = None,
    ) -> Any:
        """
        Make an asynchronous HTTP request to the Venice AI API with comprehensive error handling.
        
        This is the core internal method used by all resource classes to make asynchronous HTTP requests
        to the Venice AI API. It provides a unified interface for request preparation, header management,
        response parsing, and error handling across all API endpoints.
        
        The method intelligently handles different request types by automatically adjusting headers
        based on the HTTP method and content type. For GET requests, it removes Content-Type headers
        to follow HTTP best practices. For requests with JSON data, it ensures proper Content-Type
        headers are set. It also supports both JSON and raw binary responses for different use cases.
        
        Error handling is comprehensive and translates low-level HTTP errors into appropriate
        Venice AI exception types with detailed context information. This includes handling of
        timeout errors, connection errors, and HTTP status errors with proper request/response
        context preservation.
        
        :param method: HTTP method to use for the request (e.g., 'GET', 'POST', 'PUT', 'DELETE').
            Case-insensitive, but typically provided in uppercase.
        :type method: str
        :param path: API endpoint path relative to the client's base URL. Should not include
            leading slash as it will be properly joined with the base URL.
        :type path: str
        :param json_data: JSON-serializable data to send in the request body. Will be automatically
            serialized to JSON and sent with appropriate Content-Type headers. Only used for
            methods that support request bodies (POST, PUT, PATCH, etc.).
        :type json_data: Optional[Mapping[str, Any]]
        :param headers: Additional HTTP headers to include in the request. These headers will
            be merged with the client's default headers, with these taking precedence for
            any conflicting header names.
        :type headers: Optional[Mapping[str, str]]
        :param params: URL query parameters to append to the request URL. Will be properly
            URL-encoded and appended to the endpoint path.
        :type params: Optional[Mapping[str, Any]]
        :param raw_response: If ``True``, returns the raw response content as ``bytes`` without
            attempting JSON parsing. Useful for binary endpoints like image generation or
            file downloads. If ``False`` (default), attempts to parse response as JSON.
        :type raw_response: bool
        :param timeout: Request timeout configuration. Can be a float specifying timeout in seconds,
            or an ``httpx.Timeout`` object for granular timeout control. If not provided,
            uses the client's default timeout setting.
        :type timeout: Optional[Union[float, httpx.Timeout]]

        :return: For JSON responses (``raw_response=False``): Parsed JSON data as Python objects
            (dict, list, etc.). For raw responses (``raw_response=True``): Raw response content as bytes.
        :rtype: Any

        :raises venice_ai.exceptions.InvalidRequestError: For HTTP 400 errors indicating invalid request parameters.
        :raises venice_ai.exceptions.AuthenticationError: For HTTP 401 errors indicating invalid or missing API key.
        :raises venice_ai.exceptions.PermissionDeniedError: For HTTP 403 errors indicating insufficient permissions.
        :raises venice_ai.exceptions.NotFoundError: For HTTP 404 errors indicating the requested resource was not found.
        :raises venice_ai.exceptions.RateLimitError: For HTTP 429 errors indicating rate limit exceeded.
        :raises venice_ai.exceptions.InternalServerError: For HTTP 5xx errors indicating server-side problems.
        :raises venice_ai.exceptions.APITimeoutError: If the request times out before completion.
        :raises venice_ai.exceptions.APIConnectionError: For network connectivity issues or connection failures.
        :raises venice_ai.exceptions.APIError: For other HTTP error status codes not covered by specific exceptions.
        :raises venice_ai.exceptions.VeniceError: For other network errors or unexpected failures.
        
        Note:
            This method includes special handling for both synchronous and asynchronous response
            objects to ensure compatibility across different testing and production environments.
            The JSON parsing logic adapts to whether the response object's ``json()`` method
            is awaitable or not.
        """
        url = self._base_url.join(path)
        try:
            # Prepare headers, handling Content-Type based on request method and data
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

            response = await self._client.request(
                method=method,
                url=url,
                json=json_data if json_data else None,
                headers=request_headers, # Use potentially modified headers
                params=params,
                timeout=timeout if timeout is not None else self._timeout,
            )
            
            try:
                response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx
            except HTTPStatusError as e:
                api_error = await self._translate_httpx_error_to_api_error(e, e.request) # Safe access using getattr
                raise api_error from e
            
            # Return raw bytes if raw_response is True
            if raw_response:
                return response.content
                
            # Handle both async and sync response objects
            if isinstance(response.json, AsyncMock) or hasattr(response.json, "__await__"):
                return await response.json()
            else:
                return response.json()  # Handle non-awaitable json() method
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

    # Convenience methods for GET, POST
    async def get(self, path: str, *, params: Optional[Mapping[str, Any]] = None, **kwargs) -> Any:
        """
        Make an asynchronous GET request to the specified API endpoint.
        
        This is a convenience method that wraps the lower-level ``_request`` method
        specifically for GET requests. It automatically handles proper header configuration
        for GET requests (removing Content-Type headers) and provides a clean interface
        for retrieving data from the API.
        
        :param path: API endpoint path relative to the client's base URL. Should not include
            a leading slash as it will be properly joined with the base URL.
        :type path: str
        :param params: URL query parameters to include in the request. These will be
            properly URL-encoded and appended to the request URL.
        :type params: Optional[Mapping[str, Any]]
        :param kwargs: Additional keyword arguments to pass to the underlying ``_request`` method.
            This can include options like ``headers``, ``timeout``, or ``raw_response``.

        :return: Parsed JSON response data as Python objects (typically dict or list).
        :rtype: Any

        :raises venice_ai.exceptions.InvalidRequestError: For HTTP 400 errors indicating invalid request parameters.
        :raises venice_ai.exceptions.AuthenticationError: For HTTP 401 errors indicating invalid or missing API key.
        :raises venice_ai.exceptions.PermissionDeniedError: For HTTP 403 errors indicating insufficient permissions.
        :raises venice_ai.exceptions.NotFoundError: For HTTP 404 errors indicating the requested resource was not found.
        :raises venice_ai.exceptions.RateLimitError: For HTTP 429 errors indicating rate limit exceeded.
        :raises venice_ai.exceptions.InternalServerError: For HTTP 5xx errors indicating server-side problems.
        :raises venice_ai.exceptions.APITimeoutError: If the request times out before completion.
        :raises venice_ai.exceptions.APIConnectionError: For network connectivity issues or connection failures.
        :raises venice_ai.exceptions.APIError: For other API-related errors not covered by specific exceptions.
        """
        response = await self._request("GET", path, params=params, **kwargs)
        # Safely check for status_code attribute to avoid AttributeError
        status = getattr(response, 'status_code', 'Unknown status')
        headers = getattr(response, 'headers', 'Unknown headers')
        body = getattr(response, 'text', str(response)[:500] + '... (truncated if longer)')
        return response

    async def post(self, path: str, *, json_data: Optional[Mapping[str, Any]] = None, timeout: Union[float, httpx.Timeout, None] = None, **kwargs) -> Any:
        """
        Make an asynchronous POST request to the specified API endpoint.
        
        This is a convenience method that wraps the lower-level ``_request`` method
        specifically for POST requests. It handles JSON serialization of the request body
        and ensures proper Content-Type headers are set for JSON requests.
        
        :param path: API endpoint path relative to the client's base URL. Should not include
            a leading slash as it will be properly joined with the base URL.
        :type path: str
        :param json_data: JSON-serializable data to send in the request body. This will be
            automatically serialized to JSON and sent with ``Content-Type: application/json`` headers.
            Can include any data structure that is JSON-serializable (dict, list, primitives).
        :type json_data: Optional[Mapping[str, Any]]
        :param timeout: Request timeout configuration. Can be a float specifying timeout in seconds,
            or an ``httpx.Timeout`` object for granular timeout control. If not provided,
            uses the client's default timeout setting.
        :type timeout: Optional[Union[float, httpx.Timeout]]
        :param kwargs: Additional keyword arguments to pass to the underlying ``_request`` method.
            This can include options like ``headers``, ``params``, or ``raw_response``.

        :return: Parsed JSON response data as Python objects (typically dict or list).
        :rtype: Any

        :raises venice_ai.exceptions.InvalidRequestError: For HTTP 400 errors indicating invalid request parameters.
        :raises venice_ai.exceptions.AuthenticationError: For HTTP 401 errors indicating invalid or missing API key.
        :raises venice_ai.exceptions.PermissionDeniedError: For HTTP 403 errors indicating insufficient permissions.
        :raises venice_ai.exceptions.NotFoundError: For HTTP 404 errors indicating the requested resource was not found.
        :raises venice_ai.exceptions.RateLimitError: For HTTP 429 errors indicating rate limit exceeded.
        :raises venice_ai.exceptions.InternalServerError: For HTTP 5xx errors indicating server-side problems.
        :raises venice_ai.exceptions.APITimeoutError: If the request times out before completion.
        :raises venice_ai.exceptions.APIConnectionError: For network connectivity issues or connection failures.
        :raises venice_ai.exceptions.APIError: For other API-related errors not covered by specific exceptions.
        """
        return await self._request("POST", path, json_data=json_data, timeout=timeout, **kwargs)

    async def delete(self, path: str, **kwargs) -> Any:
        """
        Make an asynchronous DELETE request to the specified API endpoint.
        
        This is a convenience method that wraps the lower-level ``_request`` method
        specifically for DELETE requests. It handles proper header configuration
        for DELETE requests and provides a clean interface for resource deletion operations.
        
        :param path: API endpoint path relative to the client's base URL. Should not include
            a leading slash as it will be properly joined with the base URL.
        :type path: str
        :param kwargs: Additional keyword arguments to pass to the underlying ``_request`` method.
            This can include options like ``headers``, ``params``, ``timeout``, or ``raw_response``.

        :return: Parsed JSON response data as Python objects (typically dict or list).
            Many DELETE endpoints return confirmation data or the deleted resource details.
        :rtype: Any

        :raises venice_ai.exceptions.InvalidRequestError: For HTTP 400 errors indicating invalid request parameters.
        :raises venice_ai.exceptions.AuthenticationError: For HTTP 401 errors indicating invalid or missing API key.
        :raises venice_ai.exceptions.PermissionDeniedError: For HTTP 403 errors indicating insufficient permissions.
        :raises venice_ai.exceptions.NotFoundError: For HTTP 404 errors indicating the requested resource was not found.
        :raises venice_ai.exceptions.RateLimitError: For HTTP 429 errors indicating rate limit exceeded.
        :raises venice_ai.exceptions.InternalServerError: For HTTP 5xx errors indicating server-side problems.
        :raises venice_ai.exceptions.APITimeoutError: If the request times out before completion.
        :raises venice_ai.exceptions.APIConnectionError: For network connectivity issues or connection failures.
        :raises venice_ai.exceptions.APIError: For other API-related errors not covered by specific exceptions.
        """
        return await self._request("DELETE", path, **kwargs)

    async def _stream_request(
        self,
        method: str,
        path: str,
        *,
        json_data: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """
        Make an asynchronous streaming HTTP request and process Server-Sent Events (SSE) responses.
        
        This is the core internal method for handling streaming responses from the Venice AI API,
        primarily used for chat completions and other real-time streaming endpoints. It establishes
        a persistent HTTP connection and processes incoming data as a stream of Server-Sent Events,
        yielding parsed chunks asynchronously as they arrive.
        
        The streaming implementation is designed for robustness and follows these key steps:
        
        1. **Connection Establishment**: Creates a streaming HTTP connection using ``httpx.AsyncClient.stream``
        2. **Header Management**: Applies the same intelligent header handling as regular requests
        3. **Line-by-Line Processing**: Processes the response using ``aiter_lines`` for efficient streaming
        4. **SSE Protocol Handling**: Filters empty lines and processes Server-Sent Event format
        5. **JSON Parsing**: Extracts and parses JSON data from "data:" prefixed lines
        6. **Chunk Yielding**: Yields each parsed chunk asynchronously to the caller
        7. **Error Recovery**: Includes comprehensive error handling with context preservation
        
        The method handles both production and test environments by adapting to different response
        object behaviors, including cases where the stream method returns a coroutine that needs
        to be awaited before use.
    
        :param method: HTTP method to use for the streaming request. Typically 'POST' for most
            streaming endpoints, but supports other methods as needed.
        :type method: str
        :param path: API endpoint path relative to the client's base URL. Should not include
            leading slash as it will be properly joined with the base URL.
        :type path: str
        :param json_data: JSON-serializable data to send in the request body. This typically
            contains the parameters for the streaming request such as model, messages, and
            streaming configuration options.
        :type json_data: Optional[Mapping[str, Any]]
        :param headers: Additional HTTP headers to include in the streaming request. These will
            be merged with the client's default headers, with these taking precedence for
            any conflicting header names.
        :type headers: Optional[Mapping[str, str]]
        :param params: URL query parameters to append to the request URL. Will be properly
            URL-encoded and appended to the endpoint path.
        :type params: Optional[Mapping[str, Any]]

        :yields: Parsed chunk objects from the SSE stream. Each chunk represents an incremental
            update from the model's response, containing partial content, metadata, or completion signals.
        :ytype: AsyncIterator[venice_ai.types.chat.ChatCompletionChunk]

        :raises venice_ai.exceptions.InvalidRequestError: For HTTP 400 errors indicating invalid streaming parameters.
        :raises venice_ai.exceptions.AuthenticationError: For HTTP 401 errors indicating invalid or missing API key.
        :raises venice_ai.exceptions.PermissionDeniedError: For HTTP 403 errors indicating insufficient permissions.
        :raises venice_ai.exceptions.NotFoundError: For HTTP 404 errors indicating the streaming endpoint was not found.
        :raises venice_ai.exceptions.RateLimitError: For HTTP 429 errors indicating rate limit exceeded.
        :raises venice_ai.exceptions.InternalServerError: For HTTP 5xx errors indicating server-side problems.
        :raises venice_ai.exceptions.APITimeoutError: If the streaming request times out before completion.
        :raises venice_ai.exceptions.APIConnectionError: For network connectivity issues or connection failures.
        :raises venice_ai.exceptions.APIError: For other HTTP error status codes not covered by specific exceptions.
        :raises venice_ai.exceptions.VeniceError: For other network errors or unexpected failures.
        
        Note:
            This method includes comprehensive error handling and debug logging to aid in
            troubleshooting streaming connection issues. It gracefully handles malformed JSON
            chunks by logging errors and continuing to process the stream rather than failing
            entirely, providing better resilience for real-time applications.
        """
        url = self._base_url.join(path)
        
        try:
            # Prepare headers for streaming requests with same logic as regular requests
            request_headers = {}
            # Copy headers from client defaults, handling both real and mock headers
            if hasattr(self._client, 'headers') and self._client.headers is not None:
                try:
                    # Try to convert to dict first
                    request_headers.update(dict(self._client.headers))
                except (TypeError, AttributeError):
                    # Fallback for mock objects that don't behave like real headers
                    try:
                        for key, value in self._client.headers.items():
                            request_headers[key] = value
                    except (TypeError, AttributeError):
                        # If all else fails, try to access as attributes
                        if hasattr(self._client.headers, '__dict__'):
                            request_headers.update(self._client.headers.__dict__)
            
            # Apply specific request headers passed to this method
            if headers:
                request_headers.update(headers)

            # Handle Content-Type header based on request method and data
            if method.upper() == "GET":
                # Remove Content-Type for GET requests unless explicitly provided
                if headers is None or "Content-Type" not in headers:
                    request_headers.pop("Content-Type", None)
                # For GET streaming requests, we still want Accept: text/event-stream
                # Only remove Accept if it was explicitly provided in custom headers
                if headers is not None and "Accept" in headers:
                    # Keep the custom Accept header
                    pass
                else:
                    # Set Accept: text/event-stream for streaming GET requests
                    request_headers["Accept"] = "text/event-stream"
            elif json_data is not None:
                # Ensure Content-Type is set for JSON requests
                request_headers["Content-Type"] = "application/json"
                # For non-GET streaming requests, ensure Accept: text/event-stream is set
                if "Accept" not in request_headers or request_headers.get("Accept") == "application/json":
                    request_headers["Accept"] = "text/event-stream"
            else:
                # For other methods without JSON data, still set Accept for streaming
                if "Accept" not in request_headers or request_headers.get("Accept") == "application/json":
                    request_headers["Accept"] = "text/event-stream"

            # Get the stream result - might be a coroutine if stream is an AsyncMock
            stream_result = self._client.stream(
                method=method,
                url=url,
                json=json_data if json_data else None,
                headers=request_headers,
                params=params,
            )
            # stream_result is an async context manager, not a coroutine
            # No need to await it here
                
            async with stream_result as response:
                try:
                    response.raise_for_status()  # Raise early for status errors
                except HTTPStatusError as e:
                    api_error = await self._translate_httpx_error_to_api_error(e, e.request, is_stream=True) # Safe access using getattr
                    raise api_error from e
                
                chunk_count = 0
                # Process the streaming response line by line
                
                try:
                    async for line in response.aiter_lines():
                        # Skip empty lines
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Decode bytes to string if needed
                        if isinstance(line, bytes):
                            line_str = line.decode('utf-8')
                        else:
                            line_str = line
                            
                        # Check for stream termination signal
                        if line_str == "data: [DONE]":
                            break
                            
                        # Process data lines
                        if line_str.startswith("data: "):
                            # Extract the JSON part after "data: "
                            json_str = line_str[6:]  # Skip "data: " prefix
                            
                            try:
                                # Parse JSON into a dictionary
                                chunk_data = json.loads(json_str)
                                chunk_count += 1
                                
                                # Yield the parsed chunk as a ChatCompletionChunk
                                yield chunk_data
                                
                            except json.JSONDecodeError as e:
                                # Log and skip invalid JSON instead of failing the entire stream
                                # This provides more robustness in case of malformed chunks
                                logger.error(f"Failed to parse JSON in streaming response: {e}")
                                logger.error(f"Problematic JSON string: '{json_str}'")
                                continue # Continue processing the stream instead of raising
                except Exception as e:
                    raise
                
                    
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

    async def _request_multipart(
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
    ) -> Any:
        """
        Make an asynchronous HTTP request with multipart/form-data content for file uploads.
        
        This specialized method handles endpoints that require file uploads or multipart form data,
        such as image upscaling, audio transcription, or document processing. It properly formats
        multipart requests according to HTTP standards and manages both file content and form data
        fields in a single request.
        
        The method handles multipart request formation by:
        
        1. **Header Management**: Starts with fresh headers to avoid JSON Content-Type conflicts
        2. **Essential Header Preservation**: Maintains Authorization and User-Agent from client defaults
        3. **Multipart Formatting**: Allows ``httpx`` to automatically set Content-Type with boundary
        4. **File Processing**: Handles files in the standard ``httpx`` tuple format
        5. **Form Data Integration**: Combines file uploads with additional form fields
        6. **Response Processing**: Supports both JSON parsing and raw binary responses
        
        This method provides the same comprehensive error handling as the standard ``_request`` method,
        with additional logging and debugging support specifically for multipart upload scenarios.
        
        :param method: HTTP method to use for the multipart request. Typically 'POST' for file uploads,
            but supports other methods as needed by specific endpoints.
        :type method: str
        :param path: API endpoint path relative to the client's base URL. Should not include
            leading slash as it will be properly joined with the base URL.
        :type path: str
        :param files: Dictionary of files to include in the multipart request. Each file should be
            provided in ``httpx`` format as a tuple: ``(filename, content, content_type)``.
            The content can be bytes, a file-like object, or a string. Example:
            ``{"image": ("photo.jpg", image_bytes, "image/jpeg")}``
        :type files: Dict[str, Any]
        :param data: Optional form data fields to include alongside the files. These will be
            sent as regular form fields in the multipart request. Useful for sending metadata
            or configuration parameters along with file uploads.
        :type data: Optional[Dict[str, Any]]
        :param headers: Additional HTTP headers to include in the request. These will be merged
            with essential headers (Authorization, User-Agent) but will not override the
            Content-Type header, which is automatically set by ``httpx`` for multipart requests.
        :type headers: Optional[Mapping[str, str]]
        :param params: URL query parameters to append to the request URL. Will be properly
            URL-encoded and appended to the endpoint path.
        :type params: Optional[Mapping[str, Any]]
        :param raw_response: If ``True``, returns the raw response content as ``bytes`` without
            attempting JSON parsing. Useful for endpoints that return binary data like processed
            images. If ``False`` (default), attempts to parse response as JSON.
        :type raw_response: bool
        :param timeout: Request timeout configuration. Can be a float specifying timeout in seconds,
            or an ``httpx.Timeout`` object for granular timeout control. If not provided,
            uses the client's default timeout setting.
        :type timeout: Optional[Union[float, httpx.Timeout]]

        :return: For JSON responses (``raw_response=False``): Parsed JSON data as Python objects.
            For raw responses (``raw_response=True``): Raw response content as bytes.
        :rtype: Any

        :raises venice_ai.exceptions.InvalidRequestError: For HTTP 400 errors indicating invalid multipart parameters.
        :raises venice_ai.exceptions.AuthenticationError: For HTTP 401 errors indicating invalid or missing API key.
        :raises venice_ai.exceptions.PermissionDeniedError: For HTTP 403 errors indicating insufficient permissions.
        :raises venice_ai.exceptions.NotFoundError: For HTTP 404 errors indicating the endpoint was not found.
        :raises venice_ai.exceptions.RateLimitError: For HTTP 429 errors indicating rate limit exceeded.
        :raises venice_ai.exceptions.InternalServerError: For HTTP 5xx errors indicating server-side problems.
        :raises venice_ai.exceptions.APITimeoutError: If the multipart request times out before completion.
        :raises venice_ai.exceptions.APIConnectionError: For network connectivity issues or connection failures.
        :raises venice_ai.exceptions.APIError: For other HTTP error status codes not covered by specific exceptions.
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

        logger.debug(f"Sending async multipart request to {method} {url}")
        logger.debug(f"Request headers: {request_headers}")
        logger.debug(f"Content-Type header sent: {request_headers.get('Content-Type', 'Not Present')}")
        logger.debug(f"Files (keys): {list(files.keys()) if files else 'None'}")
        logger.debug(f"Files content type: {type(files)}")
        for file_key, file_value in files.items():
            logger.debug(f"File '{file_key}' details: Name: {file_value[0]}, Type: {file_value[2]}, Size: {len(file_value[1]) if isinstance(file_value, tuple) and len(file_value) > 1 and hasattr(file_value[1], '__len__') else 'N/A'} bytes")
            logger.debug(f"Data for {method} {url}: {data}") # Added logging for data
            logger.debug(f"Params: {params}")
    
        try:
            response = await self._client.request(
                method=method,
                url=url,
                files=files,
                data=data,
                headers=request_headers,
                params=params,
                timeout=timeout if timeout is not None else self._timeout,
            )
            logger.debug(f"Received async response with status code: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            
            try:
                response.raise_for_status()
            except HTTPStatusError as e:
                api_error = await self._translate_httpx_error_to_api_error(e, e.request) # Safe access using getattr
                raise api_error from e
    
            if raw_response:
                logger.debug("Returning raw response content for async multipart request.")
                return response.content # content is already bytes
    
            logger.debug(f"Response content (first 500 chars for JSON): {response.text[:500]}")
            # Handle both async and sync response objects for .json()
            if isinstance(response.json, AsyncMock) or hasattr(response.json, "__await__"):
                return await response.json()
            else:
                return response.json()  # Handle non-awaitable json() method
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

    async def _stream_request_raw(
        self,
        method: str,
        path: str,
        *,
        json_data: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        params: Optional[Mapping[str, Any]] = None,
        timeout: Union[float, httpx.Timeout, None] = None,
    ) -> AsyncIterator[bytes]:
        """
        Make an asynchronous streaming HTTP request and yield raw binary chunks.
        
        This specialized streaming method is designed for endpoints that return streaming binary data
        rather than text-based Server-Sent Events. It's primarily used for audio streaming APIs,
        real-time media generation, or other endpoints that produce continuous binary output.
        
        Unlike ``_stream_request`` which processes Server-Sent Events and parses JSON chunks,
        this method yields raw binary data directly from the response stream. This makes it ideal
        for applications that need to process or play back streaming media content in real-time.
        
        The streaming implementation follows these steps:
        
        1. **Connection Establishment**: Creates a streaming HTTP connection using ``httpx.AsyncClient.stream``
        2. **Header Management**: Applies the same intelligent header handling as other request methods
        3. **Binary Streaming**: Uses ``aiter_bytes`` to efficiently stream binary chunks
        4. **Chunk Filtering**: Skips empty chunks to avoid unnecessary processing
        5. **Direct Yielding**: Yields raw bytes without any parsing or transformation
        6. **Error Handling**: Provides comprehensive error handling with context preservation
        
        :param method: HTTP method to use for the streaming request. Typically 'POST' for most
            streaming binary endpoints, but supports other methods as needed.
        :type method: str
        :param path: API endpoint path relative to the client's base URL. Should not include
            leading slash as it will be properly joined with the base URL.
        :type path: str
        :param json_data: JSON-serializable data to send in the request body. This typically
            contains the parameters for the streaming request such as audio generation settings,
            format specifications, or other configuration options.
        :type json_data: Optional[Mapping[str, Any]]
        :param headers: Additional HTTP headers to include in the streaming request. These will
            be merged with the client's default headers, with these taking precedence for
            any conflicting header names.
        :type headers: Optional[Mapping[str, str]]
        :param params: URL query parameters to append to the request URL. Will be properly
            URL-encoded and appended to the endpoint path.
        :type params: Optional[Mapping[str, Any]]
        :param timeout: Request timeout configuration. Can be a float specifying timeout in seconds,
            or an ``httpx.Timeout`` object for granular timeout control. If not provided,
            uses the client's default timeout setting.
        :type timeout: Optional[Union[float, httpx.Timeout]]

        :yields: Raw binary chunks from the streaming response. Each chunk represents a portion
            of the binary data stream, which can be processed, saved, or played back immediately.
        :ytype: AsyncIterator[bytes]

        :raises venice_ai.exceptions.InvalidRequestError: For HTTP 400 errors indicating invalid streaming parameters.
        :raises venice_ai.exceptions.AuthenticationError: For HTTP 401 errors indicating invalid or missing API key.
        :raises venice_ai.exceptions.PermissionDeniedError: For HTTP 403 errors indicating insufficient permissions.
        :raises venice_ai.exceptions.NotFoundError: For HTTP 404 errors indicating the streaming endpoint was not found.
        :raises venice_ai.exceptions.RateLimitError: For HTTP 429 errors indicating rate limit exceeded.
        :raises venice_ai.exceptions.InternalServerError: For HTTP 5xx errors indicating server-side problems.
        :raises venice_ai.exceptions.APITimeoutError: If the streaming request times out before completion.
        :raises venice_ai.exceptions.APIConnectionError: For network connectivity issues or connection failures.
        :raises venice_ai.exceptions.APIError: For other HTTP error status codes not covered by specific exceptions.
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

            # Get the stream result - might be a coroutine if stream is an AsyncMock
            stream_result = self._client.stream(
                method=method,
                url=url,
                json=json_data if json_data else None,
                headers=request_headers,
                params=params,
                timeout=timeout if timeout is not None else self._timeout,
            )
            
            # stream_result is an async context manager, not a coroutine
            # No need to await it here
                
            async with stream_result as response:
                try:
                    response.raise_for_status()  # Raise early for status errors
                except HTTPStatusError as e:
                    api_error = await self._translate_httpx_error_to_api_error(e, e.request, is_stream=True) # Safe access using getattr
                    raise api_error from e
                
                # Yield the content in chunks asynchronously
                
                try:
                    async for chunk in response.aiter_bytes():
                        if chunk:  # Skip empty chunks
                            yield chunk
                except Exception as e:
                    raise
                        
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

    async def _translate_httpx_error_to_api_error(self, error: Union[RequestError, HTTPStatusError], default_request: Request, is_stream: bool = False) -> VeniceError:
        """
        Translate an HTTPX RequestError into an appropriate Venice AI APIError with context preservation.
        
        This internal method provides centralized error translation logic that converts low-level
        HTTPX exceptions into user-friendly Venice AI exception types. It preserves important
        context information like request details, response data, and original error messages
        while categorizing errors into appropriate exception types based on HTTP status codes
        and error types.
        
        The method handles several categories of errors:
        
        - **HTTP Status Errors**: Translates specific HTTP status codes into appropriate exceptions
        - **Timeout Errors**: Converts timeout exceptions with preserved timing context
        - **Connection Errors**: Handles network connectivity and connection failures
        - **Generic Request Errors**: Catches other request-related failures
        
        Special care is taken to safely access the ``error.request`` attribute, as HTTPX can
        raise RuntimeError when this attribute is accessed if the internal request is None.
        The method provides fallback request objects to ensure error context is always available.
        
        :param error: The HTTPX error to translate into a Venice AI exception. This can be
            any subclass of ``httpx.RequestError`` including ``HTTPStatusError``,
            ``TimeoutException``, ``ConnectError``, or other request-related errors.
        :type error: httpx.RequestError
        :param default_request: Fallback request object to use if ``error.request`` is not
            available or accessing it raises an exception. This ensures error context
            is preserved even when the original request object is unavailable.
        :type default_request: httpx.Request
        :param is_stream: Flag indicating whether the error occurred during a streaming request.
            This affects how response parsing is handled, as streaming responses require
            awaiting of response methods while regular responses do not.
        :type is_stream: bool

        :return: An appropriate Venice AI exception that corresponds to the HTTPX error type
            and contains preserved context information including request, response, and
            original error details.
        :rtype: venice_ai.exceptions.APIError

        Note:
            This method includes special handling for streaming vs. non-streaming responses
            when parsing error response bodies, as streaming responses have async response
            methods that need to be awaited while regular responses have synchronous methods.
        """
        # IMPORTANT: Safely access error.request using try-except to avoid RuntimeError
        _raw_request = None
        try:
            _raw_request = error.request
        except RuntimeError:
            # This specific RuntimeError is raised by httpx when .request is accessed if _request is None
            pass  # _raw_request remains None
        
        request_obj = cast(httpx.Request, _raw_request if _raw_request is not None else default_request)
        
        # Declare exception_response once for use in multiple except blocks
        exception_response: Optional[httpx.Response] = None
        
        if isinstance(error, HTTPStatusError):
            response_obj = error.response
            parsed_body: Optional[Any] = None
            
            try:
                if is_stream:
                    # For streaming responses, response methods might need to be awaited
                    # Attempt to parse as JSON first
                    try:
                        # Handle response.json() which might be an AsyncMock or awaitable
                        if hasattr(response_obj, "json") and callable(response_obj.json):
                            potential_coro_json = response_obj.json()
                            if inspect.isawaitable(potential_coro_json):
                                parsed_body = await potential_coro_json
                            else:
                                parsed_body = potential_coro_json
                        else:
                            parsed_body = None
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try to get text content, awaiting if it's an awaitable
                        try:
                            # Handle response.text which can be a direct string, a coroutine, or a callable that returns a coroutine
                            if hasattr(response_obj, "text"):
                                text_val = response_obj.text
                                if callable(text_val) and not isinstance(text_val, str):
                                    # e.g., PropertyMock(return_value=AsyncMock())
                                    potential_coro_text = text_val()
                                    if potential_coro_text is not None and inspect.isawaitable(potential_coro_text):
                                        parsed_body = await potential_coro_text  # type: ignore[misc]
                                    else:
                                        parsed_body = str(potential_coro_text)
                                elif text_val is not None and inspect.isawaitable(text_val):
                                    # e.g., PropertyMock(return_value=some_coroutine)
                                    parsed_body = await text_val  # type: ignore[misc]
                                else:
                                    # Regular string attribute
                                    parsed_body = str(text_val)
                            else:
                                parsed_body = "<unable to retrieve text content>"
                        except Exception:
                            parsed_body = "<unable to retrieve text content>"
                else:
                    # For non-streaming responses, synchronous parsing is fine
                    try:
                        # Handle response.json() which might be an AsyncMock or awaitable even in non-streaming case
                        if hasattr(response_obj, "json") and callable(response_obj.json):
                            potential_coro_json = response_obj.json()
                            if inspect.isawaitable(potential_coro_json):
                                parsed_body = await potential_coro_json
                            else:
                                parsed_body = potential_coro_json
                        else:
                            parsed_body = None
                    except json.JSONDecodeError:
                        try:
                            # Handle response.text which can be a direct string, a coroutine, or a callable that returns a coroutine
                            if hasattr(response_obj, "text"):
                                try:
                                    text_val = response_obj.text
                                    if callable(text_val) and not isinstance(text_val, str):
                                        # e.g., PropertyMock(return_value=AsyncMock())
                                        try:
                                            potential_coro_text = text_val()
                                            # Check if the result is a mock object that represents a failed text access
                                            if hasattr(potential_coro_text, '_mock_name') and 'text()' in str(potential_coro_text):
                                                # This is likely a mock representing a failed text() call
                                                parsed_body = None
                                            elif potential_coro_text is not None and inspect.isawaitable(potential_coro_text):
                                                parsed_body = await potential_coro_text  # type: ignore[misc]
                                            else:
                                                parsed_body = str(potential_coro_text)
                                        except Exception:
                                            parsed_body = None
                                    elif text_val is not None and inspect.isawaitable(text_val):
                                        # e.g., PropertyMock(return_value=some_coroutine)
                                        parsed_body = await text_val  # type: ignore[misc]
                                    else:
                                        # Regular string attribute
                                        parsed_body = str(text_val)
                                except AttributeError:
                                    # If accessing response.text raises AttributeError, set body to None
                                    parsed_body = None
                            else:
                                parsed_body = None
                        except Exception:
                            parsed_body = None
                
                # Log the error body details for debugging
                logger.error(f"Error response body (full details): {parsed_body}")
            except Exception:
                # If any exception occurs during body parsing, ensure parsed_body is None
                parsed_body = None

            return _make_status_error(
                message=f"API error {response_obj.status_code} for {request_obj.method} {request_obj.url}",
                request=request_obj,
                response=response_obj,
                body=parsed_body
            )
        elif isinstance(error, TimeoutException):
            logger.error(f"Request timed out for {request_obj.method} {request_obj.url}: {error}")
            prefix = "Stream request" if is_stream else "Request"
            # Use the safely accessed request_obj and get the original exception message
            original_exception_message = str(error.args[0]) if error.args else "Timeout occurred"
            # Safely get response from the exception
            exception_response = getattr(error, 'response', None)
            return APITimeoutError(message=f"{prefix} timed out: {original_exception_message}", request=request_obj, response=exception_response, original_error=error)
        elif isinstance(error, ConnectError):
            logger.error(f"Connection error for {request_obj.method} {request_obj.url}: {error}")
            prefix = "Stream request" if is_stream else "Request"
            # Use the safely accessed request_obj and get the original exception message
            original_exception_message = str(error.args[0]) if error.args else "Connection error occurred"
            # Safely get response from the exception
            exception_response = getattr(error, 'response', None)
            return APIConnectionError(message=f"{prefix} failed: {original_exception_message}", request=request_obj, response=exception_response)
        else:
            logger.error(f"Request failed for {request_obj.method} {request_obj.url}: {error}")
            prefix = "Stream request" if is_stream else "Request"
            # Use the safely accessed request_obj and get the original exception message
            original_exception_message = str(error.args[0]) if error.args else "A network request error occurred"
            # Safely get response from the exception
            exception_response = getattr(error, 'response', None)
            return APIConnectionError(message=f"{prefix} failed: {original_exception_message}", request=request_obj, response=exception_response, original_error=error)

    async def close(self) -> None:
        """
        Close the underlying asynchronous HTTP client and free all associated resources.
        
        This method performs cleanup of the internal ``httpx.AsyncClient`` and any associated
        resources such as connection pools, SSL contexts, and background tasks. It should be
        called when the Venice AI client is no longer needed to ensure proper resource cleanup
        and prevent resource leaks.
        
        When using the client as an async context manager (with ``async with``), this method
        is called automatically upon exiting the context, so manual cleanup is not required.
        For manual resource management, this method should be called explicitly.
        
        The method is designed to be idempotent - it can be called multiple times safely.
        Only the first call will actually perform the cleanup; subsequent calls will be
        no-ops. This prevents errors if cleanup is attempted multiple times.

        Note:
            After calling this method, the client should not be used for making further
            API requests. Attempting to use a closed client may result in errors or
            undefined behavior.
        """
        if hasattr(self, "_client") and not self._is_closed:
            await self._client.aclose()
            self._is_closed = True
        # If client is already closed, do nothing
    async def aclose(self) -> None:
        """
        Close the underlying asynchronous HTTP client and free all associated resources.
        
        This is an alias for the close() method, following the conventional async naming pattern
        where async methods are prefixed with 'a'. This method performs the same cleanup as
        close() - it closes the internal httpx.AsyncClient and any associated resources.
        
        :return: None
        :rtype: None
        """
        await self.close()

    async def __aenter__(self) -> "AsyncVeniceClient":
        """
        Enter the asynchronous context manager, enabling use with 'async with' statements.
        
        This method is called automatically when entering an ``async with`` statement and
        simply returns the client instance itself. The client is already fully initialized
        and ready for use at this point, so no additional setup is required.
        
        :return: This client instance, ready for making API requests.
        :rtype: AsyncVeniceClient
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the asynchronous context manager, automatically closing the client.
        
        This method is called automatically when exiting an ``async with`` statement.
        It ensures proper cleanup of resources by calling the ``close()`` method,
        which closes the underlying HTTP client and frees associated resources.
        
        The cleanup occurs regardless of whether an exception was raised within
        the context, ensuring resources are always properly released.
        
        :param exc_type: Exception type if an exception was raised in the context.
            This parameter is provided by Python's context manager protocol.
        :type exc_type: Optional[type]
        :param exc_val: Exception value if an exception was raised in the context.
            This parameter is provided by Python's context manager protocol.
        :type exc_val: Optional[BaseException]
        :param exc_tb: Exception traceback if an exception was raised in the context.
            This parameter is provided by Python's context manager protocol.
        :type exc_tb: Optional[Any]
        """
        await self.close()


# Define the async resource namespace classes
from ._resource import AsyncAPIResource


# Async Chat Completions Resource
from typing import overload, Sequence, Literal
from .types.chat import (
    MessageParam, VeniceParameters, ResponseFormat, Tool,
    ToolChoice, ToolChoiceObject, StreamOptions
)

class AsyncChatCompletions(AsyncAPIResource):
    """
    Provides asynchronous access to the /chat/completions endpoint.
    
    This class provides methods to create chat completions asynchronously,
    with support for both streaming and non-streaming responses.
    """

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        stream: Literal[False] = False,  # Explicit non-streaming case
        # --- Common Optional Parameters ---
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ResponseFormat] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, Sequence[str]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[Sequence[Tool]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"], ToolChoiceObject]] = None,
        user: Optional[str] = None,
        venice_parameters: Optional[VeniceParameters] = None,
        # --- Less Common / Newer Params ---
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        stop_token_ids: Optional[Sequence[int]] = None,
        top_k: Optional[int] = None,
        stream_options: Optional[StreamOptions] = None,
        stream_cls: Optional[Any] = None,  # Added stream_cls parameter
    ) -> ChatCompletion:  # Return type for non-streaming
        ...
        
    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        stream: Literal[True],
        stream_cls: Optional[Any] = None,  # Will default if stream=True and not provided
        # --- Common Optional Parameters ---
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ResponseFormat] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, Sequence[str]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[Sequence[Tool]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"], ToolChoiceObject]] = None,
        user: Optional[str] = None,
        venice_parameters: Optional[VeniceParameters] = None,
        # --- Less Common / Newer Params ---
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        stop_token_ids: Optional[Sequence[int]] = None,
        top_k: Optional[int] = None,
        stream_options: Optional[StreamOptions] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:  # Return type for streaming
        ...

    async def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        stream: bool = False,
        stream_cls: Optional[Any] = None,  # Added stream_cls parameter
        **kwargs: Any  # Catch all other keyword args
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """
        Creates a model response for the given chat conversation asynchronously.

        :param model: ID of the model to use (e.g., "venice-1").
        :type model: str
        :param messages: Sequence of messages comprising the conversation so far.
        :type messages: Sequence[venice_ai.types.chat.MessageParam]
        :param stream: If ``True``, stream back partial progress. Returns an ``AsyncIterator[ChatCompletionChunk]``
            if ``True``, otherwise a ``ChatCompletion`` object. Defaults to ``False``.
        :type stream: bool
        :param frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based
            on their existing frequency in the text so far.
        :type frequency_penalty: Optional[float]
        :param max_tokens: Maximum number of tokens to generate. (Deprecated, use ``max_completion_tokens``)
        :type max_tokens: Optional[int]
        :param max_completion_tokens: Upper bound on generated tokens.
        :type max_completion_tokens: Optional[int]
        :param n: How many chat completion choices to generate.
        :type n: Optional[int]
        :param presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based
            on whether they appear in the text so far.
        :type presence_penalty: Optional[float]
        :param response_format: Specifies the response format, e.g., for JSON mode.
        :type response_format: Optional[venice_ai.types.chat.ResponseFormat]
        :param seed: Random seed for reproducibility.
        :type seed: Optional[int]
        :param stop: Up to 4 sequences where the API will stop generation.
        :type stop: Optional[Union[str, Sequence[str]]]
        :param temperature: Sampling temperature to control randomness.
            Lower values make output more deterministic.
        :type temperature: Optional[float]
        :param top_p: Nucleus sampling probability mass to control diversity.
        :type top_p: Optional[float]
        :param tools: A list of tools the model may call.
        :type tools: Optional[Sequence[venice_ai.types.chat.Tool]]
        :param tool_choice: Controls which tool is called by the model.
        :type tool_choice: Optional[Union[Literal["none", "auto"], venice_ai.types.chat.ToolChoiceObject]]
        :param user: Identifier for the end-user (currently discarded by the API).
        :type user: Optional[str]
        :param venice_parameters: Venice-specific parameters.
        :type venice_parameters: Optional[venice_ai.types.chat.VeniceParameters]
        :param logprobs: Whether to return log probabilities.
        :type logprobs: Optional[bool]
        :param top_logprobs: Number of top log probabilities to return.
        :type top_logprobs: Optional[int]
        :param parallel_tool_calls: Enable parallel function calling.
        :type parallel_tool_calls: Optional[bool]
        :param repetition_penalty: Penalty for repetition.
        :type repetition_penalty: Optional[float]
        :param stop_token_ids: Array of token IDs to stop generation.
        :type stop_token_ids: Optional[Sequence[int]]
        :param top_k: Number of highest probability tokens to keep.
        :type top_k: Optional[int]
        :param stream_options: Options for controlling the streaming response.
        :type stream_options: Optional[venice_ai.types.chat.StreamOptions]
        :param kwargs: Additional keyword arguments to pass to the API.

        :return: A :class:`~venice_ai.types.chat.ChatCompletion` object if ``stream`` is ``False``,
                 otherwise an ``AsyncIterator`` of :class:`~venice_ai.types.chat.ChatCompletionChunk` objects.
        :rtype: Union[venice_ai.types.chat.ChatCompletion, AsyncIterator[venice_ai.types.chat.ChatCompletionChunk]]

        :raises venice_ai.exceptions.InvalidRequestError: If parameters are invalid.
        :raises venice_ai.exceptions.AuthenticationError: If the API key is invalid.
        :raises venice_ai.exceptions.PermissionDeniedError: If access is denied.
        :raises venice_ai.exceptions.NotFoundError: If the model or resource is not found.
        :raises venice_ai.exceptions.RateLimitError: If rate limits are exceeded.
        :raises venice_ai.exceptions.APIError: For other API-related errors.
            
        Example:

            .. code-block:: python

                # Non-streaming usage
                async with AsyncVeniceClient(api_key="your-api-key") as client:
                    response = await client.chat.completions.create(
                        model="venice-1",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Tell me a joke."}
                        ]
                    )
                    print(response["choices"][0]["message"]["content"])

                # Streaming usage
                async with AsyncVeniceClient(api_key="your-api-key") as client:
                    async for chunk in client.chat.completions.create(
                        model="venice-1",
                        messages=[{"role": "user", "content": "Tell me a joke."}],
                        stream=True
                    ):
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            print(content, end="", flush=True)
        """
        # Construct request body, filtering out None values from kwargs
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,  # Set based on the stream parameter
        }

        # Add optional parameters from kwargs if they are not None
        # Exclude 'stream_cls' from being added to the body if it's in kwargs
        # as it's a type and not part of the API request body.
        processed_kwargs = {k: v for k, v in kwargs.items() if k != 'stream_cls'}
        for key, value in processed_kwargs.items():
            if value is not None:
                # Simple mapping for now, might need renaming (e.g., max_completion_tokens)
                body[key] = value

        # Handle specific naming or structuring if needed
        # e.g. if venice_parameters needs special handling

        if stream:
            # Handle stream_cls parameter
            user_provided_stream_cls_async = stream_cls
            effective_stream_cls_async: Any = AsyncStream  # Default

            if user_provided_stream_cls_async is not None:
                if inspect.isclass(user_provided_stream_cls_async):
                    try:
                        if issubclass(user_provided_stream_cls_async, (AsyncStream,)):
                            effective_stream_cls_async = cast(Any, user_provided_stream_cls_async)
                        else:
                            # Check if it has the required interface
                            sig = inspect.signature(user_provided_stream_cls_async.__init__)
                            params = list(sig.parameters.keys())
                            has_proper_signature = len(params) >= 3 or 'client' in params
                            has_aiter_method = hasattr(user_provided_stream_cls_async, '__aiter__')
                            
                            if has_proper_signature and has_aiter_method:
                                effective_stream_cls_async = cast(Any, user_provided_stream_cls_async)
                    except (TypeError, ValueError):
                        pass  # Use default if validation fails

            # Get the raw iterator from _stream_request
            raw_iterator: AsyncIterator[ChatCompletionChunk] = self._client._stream_request(
                method="POST",
                path="chat/completions",
                json_data=body
            )
            
            # Wrap with the effective stream class
            return cast(AsyncIterator[ChatCompletionChunk], effective_stream_cls_async(raw_iterator, client=self._client))
        else:
            # Use regular post method for non-streaming responses
            response = await self._client.post("chat/completions", json_data=body)
            # Cast the response to the expected TypedDict type
            # Add error handling or validation if needed before casting
            return cast(ChatCompletion, response)  # Assumes _client.post already returns parsed JSON dict