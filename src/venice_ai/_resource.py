from typing import TYPE_CHECKING, Generic, TypeVar, Dict, Any, BinaryIO, Optional, Union, Mapping
import httpx

if TYPE_CHECKING:
    from ._client import VeniceClient
    from ._async_client import AsyncVeniceClient

SyncClientT = TypeVar("SyncClientT", bound="VeniceClient")
AsyncClientT = TypeVar("AsyncClientT", bound="AsyncVeniceClient")

class APIResource(Generic[SyncClientT]):
    _client: SyncClientT

    def __init__(self, client: SyncClientT) -> None:
        self._client = client
        
    def _request_multipart(
        self,
        method: str,
        path: str,
        *,
        files: Dict[str, Any],
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Any:
        """
        Makes an HTTP request with multipart/form-data content.
        
        This method is used for endpoints that require file uploads,
        such as image upscaling.
        
        Args:
            method: HTTP method (e.g., 'POST').
            path: API endpoint path.
            files: Dictionary of files to upload.
            data: Additional form data to include.
            headers: Additional HTTP headers.
            
        Returns:
            Any: Parsed JSON response body.
        """
        url = self._client._base_url.join(path)
        
        # Create headers with Authorization but without Content-Type
        # httpx will set the correct multipart Content-Type with boundary
        request_headers: Dict[str, str] = {}
        if headers:
            request_headers.update(headers)
            
        # Ensure authorization header is present
        if "Authorization" not in request_headers:
            request_headers["Authorization"] = f"Bearer {self._client._api_key}"
            
        # Accept JSON response by default
        if "Accept" not in request_headers:
            request_headers["Accept"] = "application/json"
        
        response = self._client._client.request(
            method=method,
            url=url,
            files=files,
            data=data,
            headers=request_headers,
        )
        
        response.raise_for_status()
        return response.json()

    def _request_raw_response(
        self,
        method: str,
        path: str,
        *,
        options: Dict[str, Any],
        stream_mode: bool = False,
    ) -> "httpx.Response":
        """
        Make an HTTP request and return the raw httpx.Response object.
        
        This method is used for endpoints that need access to the raw response,
        such as for streaming audio data or getting raw binary content.
        
        Args:
            method: HTTP method (e.g., 'POST') to use for the request.
            path: API endpoint path relative to the base URL.
            options: Request options including headers, body, timeout, etc.
            stream_mode: Whether to enable streaming mode for the response.
            
        Returns:
            The raw httpx.Response object.
        """
        url = self._client._base_url.join(path)
        
        # Extract options
        headers = options.get("headers", {})
        json_data = options.get("body")
        timeout = options.get("timeout")
        
        # Prepare headers by merging default headers with any provided headers
        request_headers = dict(self._client._client.headers)
        if headers:
            request_headers.update(headers)
            
        try:
            response = self._client._client.request(
                method=method,
                url=url,
                json=json_data if json_data else None,
                headers=request_headers,
                timeout=timeout if timeout is not None else self._client._timeout,
            )
            return response
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            # Let the client handle the error translation
            from httpx import Request
            default_request = Request(method=method, url=str(url))
            if hasattr(self._client, '_translate_httpx_error_to_api_error'):
                api_error = self._client._translate_httpx_error_to_api_error(e, default_request)
                raise api_error from e
            else:
                raise

class AsyncAPIResource(Generic[AsyncClientT]):
    _client: AsyncClientT

    def __init__(self, client: AsyncClientT) -> None:
        self._client = client
        
    async def _request_multipart(
        self,
        method: str,
        path: str,
        *,
        files: Dict[str, Any],
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Any:
        """
        Makes an async HTTP request with multipart/form-data content.
        
        This method is used for endpoints that require file uploads,
        such as image upscaling.
        
        Args:
            method: HTTP method (e.g., 'POST').
            path: API endpoint path.
            files: Dictionary of files to upload.
            data: Additional form data to include.
            headers: Additional HTTP headers.
            
        Returns:
            Any: Parsed JSON response body.
        """
        url = self._client._base_url.join(path)
        
        # Create headers with Authorization but without Content-Type
        # httpx will set the correct multipart Content-Type with boundary
        request_headers: Dict[str, str] = {}
        if headers:
            request_headers.update(headers)
            
        # Ensure authorization header is present
        if "Authorization" not in request_headers:
            request_headers["Authorization"] = f"Bearer {self._client._api_key}"
            
        # Accept JSON response by default
        if "Accept" not in request_headers:
            request_headers["Accept"] = "application/json"
        
        response = await self._client._client.request(
            method=method,
            url=url,
            files=files,
            data=data,
            headers=request_headers,
        )
        
        response.raise_for_status()
        return response.json()

    async def _arequest_raw_response(
        self,
        method: str,
        path: str,
        *,
        options: Dict[str, Any],
        stream_mode: bool = False,
    ) -> "httpx.Response":
        """
        Make an async HTTP request and return the raw httpx.Response object.
        
        This method is used for endpoints that need access to the raw response,
        such as for streaming audio data or getting raw binary content.
        
        Args:
            method: HTTP method (e.g., 'POST') to use for the request.
            path: API endpoint path relative to the base URL.
            options: Request options including headers, body, timeout, etc.
            stream_mode: Whether to enable streaming mode for the response.
            
        Returns:
            The raw httpx.Response object.
        """
        url = self._client._base_url.join(path)
        
        # Extract options
        headers = options.get("headers", {})
        json_data = options.get("body")
        timeout = options.get("timeout")
        
        # Prepare headers by merging default headers with any provided headers
        request_headers = dict(self._client._client.headers)
        if headers:
            request_headers.update(headers)
            
        try:
            response = await self._client._client.request(
                method=method,
                url=url,
                json=json_data if json_data else None,
                headers=request_headers,
                timeout=timeout if timeout is not None else self._client._timeout,
            )
            return response
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            # Let the client handle the error translation
            default_request = httpx.Request(method=method, url=str(url))
            if hasattr(self._client, '_translate_httpx_error_to_api_error'):
                api_error = await self._client._translate_httpx_error_to_api_error(e, default_request)
                raise api_error from e
            else:
                raise