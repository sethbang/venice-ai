from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
)

import aiohttp

from .utils import serialize_form_value

if TYPE_CHECKING:
    from ._client import VeniceClient


class APIResource[ClientT: "VeniceClient"]:
    """
    Base class for all API resources.

    This class provides a common interface for API resource classes to interact
    with the `VeniceClient`. It includes helper methods for making different
    types of requests, such as standard JSON requests, multipart file uploads,
    and requests where a raw response is desired.
    """

    _client: ClientT

    def __init__(self, client: ClientT) -> None:
        self._client = client

    async def _request_multipart(
        self,
        method: str,
        path: str,
        *,
        files: dict[str, Any],
        data: dict[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: "float | aiohttp.ClientTimeout | None" = None,
    ) -> dict[str, Any] | bytes:
        """
        Makes an async HTTP request with multipart/form-data content.

        This method is used for endpoints that require file uploads,
        such as image upscaling, audio transcription, or any other API
        operation that accepts multipart form data.

        The method handles multiple file formats and automatically constructs
        the appropriate multipart/form-data request. It supports both binary
        file content and file-like objects, with optional content type specification.

        Args:
            method: HTTP method (e.g., 'POST').
            path: API endpoint path relative to the base URL.
            files: Dictionary of files to upload. Each entry can be:
                - A tuple of (filename, file_content, content_type) for full control
                - A tuple of (filename, file_content) for basic file uploads
                - Raw file content (less common, filename will be auto-generated)
            data: Additional form data to include alongside the files.
                These are serialized using `serialize_form_value()`.
            headers: Additional HTTP headers. Note that 'Authorization' and
                'Accept' headers are automatically added if not provided.

        Returns:
            Union[Dict[str, Any], bytes]: Parsed JSON response (as dict) for JSON responses,
                or raw bytes for image/binary responses (when Accept: image/* is specified).

        Raises:
            APIError: If the request fails with a non-2xx status code.
            APIConnectionError: If a network connection error occurs.

        Example:
            Basic file upload with additional form data::

                # Upload an image for upscaling
                with open('image.png', 'rb') as f:
                    image_data = f.read()

                response = await resource._request_multipart(
                    method='POST',
                    path='/v1/images/upscale',
                    files={
                        'image': ('image.png', image_data, 'image/png')
                    },
                    data={
                        'scale': 2,
                        'format': 'png'
                    }
                )

            Multiple file upload::

                # Upload multiple files in a single request
                response = await resource._request_multipart(
                    method='POST',
                    path='/v1/batch/process',
                    files={
                        'input_file': ('data.json', json_bytes, 'application/json'),
                        'config_file': ('config.yaml', yaml_bytes, 'text/yaml')
                    },
                    data={
                        'batch_id': '12345',
                        'priority': 'high'
                    }
                )

            Using file-like objects::

                # Upload directly from a file-like object
                from io import BytesIO

                buffer = BytesIO(image_bytes)
                response = await resource._request_multipart(
                    method='POST',
                    path='/v1/images/analyze',
                    files={
                        'image': ('photo.jpg', buffer, 'image/jpeg')
                    }
                )

            Custom headers for image responses::

                # Request binary image data instead of JSON
                response = await resource._request_multipart(
                    method='POST',
                    path='/v1/images/convert',
                    files={
                        'source': ('input.png', image_data, 'image/png')
                    },
                    headers={
                        'Accept': 'image/*'  # Returns raw image bytes
                    }
                )
                # response is now bytes, not JSON

        Note:
            - The 'Authorization' header is automatically added using the client's API key
            - The default 'Accept' header is 'application/json' unless overridden
            - Image responses (when Accept: image/*) return raw bytes instead of JSON
            - Form data values are automatically serialized using `serialize_form_value()`
        """
        url = str(self._client._base_url / path.lstrip("/"))

        # Create aiohttp FormData
        form_data = aiohttp.FormData()

        # Add files to form data
        for field_name, file_info in files.items():
            if isinstance(file_info, tuple):
                # Format: (filename, file_content, content_type)
                if len(file_info) == 3:
                    filename, file_content, content_type = file_info
                    if isinstance(file_content, bytes):
                        form_data.add_field(
                            field_name,
                            file_content,
                            filename=filename,
                            content_type=content_type,
                        )
                    else:
                        # Assume it's a file-like object
                        form_data.add_field(
                            field_name,
                            file_content,
                            filename=filename,
                            content_type=content_type,
                        )
                elif len(file_info) == 2:
                    # Format: (filename, file_content)
                    filename, file_content = file_info
                    form_data.add_field(field_name, file_content, filename=filename)
            else:
                # Just file content
                form_data.add_field(field_name, file_info)

        # Add additional form data
        if data:
            for key, value in data.items():
                form_data.add_field(key, serialize_form_value(value))

        # Prepare headers
        request_headers: dict[str, str] = {}
        if headers:
            request_headers.update(headers)

        # Ensure authorization header is present
        if "Authorization" not in request_headers:
            from .core.auth import create_auth_headers

            request_headers.update(create_auth_headers(self._client._api_key))

        # Accept JSON response by default
        if "Accept" not in request_headers:
            request_headers["Accept"] = "application/json"

        # Make the request using aiohttp
        session = await self._client._get_session()

        from .utils.errors import wrap_aiohttp_errors

        request_kwargs: dict[str, Any] = {
            "method": method,
            "url": url,
            "data": form_data,
            "headers": request_headers,
        }
        if timeout is not None:
            request_kwargs["timeout"] = (
                timeout
                if isinstance(timeout, aiohttp.ClientTimeout)
                else aiohttp.ClientTimeout(total=timeout)
            )

        async with (
            wrap_aiohttp_errors(),
            session.request(**request_kwargs) as response,
        ):
            if not response.ok:
                from .exceptions import _make_status_error

                body = await response.text()
                raise _make_status_error(
                    message=f"API request failed with status {response.status}",
                    request=None,
                    body=body,
                    response=response,
                )

            # Check content type to determine how to handle the response
            content_type = response.headers.get("Content-Type", "")

            # If we're expecting binary image data (Accept: image/*)
            if "image/*" in request_headers.get("Accept", "") or content_type.startswith("image/"):
                # Return binary data for image responses
                return await response.read()
            elif content_type.startswith("text/"):
                # Plain-text responses (e.g. augment.parse_text(response_format="text")
                # returns text/plain). Calling .json() here raises ContentTypeError
                # which got miswrapped as APIConnectionError — return raw bytes so
                # the caller can decode without surprises.
                return await response.read()
            else:
                # Parse as JSON for other responses
                # Cast from Any to Dict[str, Any] for type safety
                json_response: dict[str, Any] = await response.json()
                return json_response
