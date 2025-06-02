from typing import TYPE_CHECKING, Generic, TypeVar, Dict, Any, BinaryIO, Optional, Union, Mapping

if TYPE_CHECKING:
    from ._client import VeniceClient
    from ._async_client import AsyncVeniceClient
    import httpx

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