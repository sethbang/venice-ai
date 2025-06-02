import pytest
import httpx
import json
from unittest.mock import patch, MagicMock
from venice_ai._resource import APIResource, AsyncAPIResource
from venice_ai._client import VeniceClient
from venice_ai._async_client import AsyncVeniceClient

# Synchronous APIResource Tests
class TestAPIResource:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=VeniceClient)
        client._api_key = "test-api-key"
        client._base_url = httpx.URL("https://api.venice.ai/v1/")
        client._client = MagicMock(spec=httpx.Client)
        return client

    def test_initialization(self, mock_client):
        resource = APIResource(mock_client)
        assert resource._client == mock_client

    def test_request_multipart_basic(self, mock_client):
        resource = APIResource(mock_client)
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_client._client.request.return_value = mock_response

        files = {"file": ("test.txt", b"content", "text/plain")}
        result = resource._request_multipart("POST", "upload", files=files)

        mock_client._client.request.assert_called_once()
        assert result == {"status": "success"}
        args, kwargs = mock_client._client.request.call_args
        assert kwargs["method"] == "POST"
        assert str(kwargs["url"]).endswith("upload")
        assert kwargs["files"] == files
        assert "Authorization" in kwargs["headers"]
        assert kwargs["headers"]["Authorization"] == "Bearer test-api-key"
        assert kwargs["headers"]["Accept"] == "application/json"

    def test_request_multipart_with_data_and_headers(self, mock_client):
        resource = APIResource(mock_client)
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_client._client.request.return_value = mock_response

        files = {"file": ("test.txt", b"content", "text/plain")}
        data = {"param": "value"}
        headers = {"Custom-Header": "value"}
        result = resource._request_multipart("POST", "upload", files=files, data=data, headers=headers)

        mock_client._client.request.assert_called_once()
        assert result == {"status": "success"}
        args, kwargs = mock_client._client.request.call_args
        assert kwargs["data"] == data
        assert "Custom-Header" in kwargs["headers"]
        assert kwargs["headers"]["Custom-Header"] == "value"
        assert "Authorization" in kwargs["headers"]
        assert kwargs["headers"]["Authorization"] == "Bearer test-api-key"

    def test_request_multipart_error_handling(self, mock_client):
        resource = APIResource(mock_client)
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="Error", request=MagicMock(), response=MagicMock()
        )
        mock_client._client.request.return_value = mock_response

        files = {"file": ("test.txt", b"content", "text/plain")}
        with pytest.raises(httpx.HTTPStatusError):
            resource._request_multipart("POST", "upload", files=files)

    def test_request_multipart_default_accept_header(self, mock_client):
        resource = APIResource(mock_client)
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_client._client.request.return_value = mock_response

        files = {"file": ("test.txt", b"content", "text/plain")}
        # Call without providing 'Accept' in headers
        result = resource._request_multipart("POST", "upload", files=files, headers={"Some-Other-Header": "value"})

        mock_client._client.request.assert_called_once()
        assert result == {"status": "success"}
        args, kwargs = mock_client._client.request.call_args
        assert "Accept" in kwargs["headers"]
        assert kwargs["headers"]["Accept"] == "application/json"
        assert "Some-Other-Header" in kwargs["headers"] # Ensure other headers are preserved
# Asynchronous AsyncAPIResource Tests
@pytest.mark.asyncio
class TestAsyncAPIResource:
    @pytest.fixture
    def mock_async_client(self):
        client = MagicMock(spec=AsyncVeniceClient)
        client._api_key = "test-api-key"
        client._base_url = httpx.URL("https://api.venice.ai/v1/")
        client._client = MagicMock(spec=httpx.AsyncClient)
        return client

    async def test_initialization(self, mock_async_client):
        resource = AsyncAPIResource(mock_async_client)
        assert resource._client == mock_async_client

    async def test_request_multipart_basic(self, mock_async_client):
        resource = AsyncAPIResource(mock_async_client)
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_async_client._client.request.return_value = mock_response

        files = {"file": ("test.txt", b"content", "text/plain")}
        result = await resource._request_multipart("POST", "upload", files=files)

        mock_async_client._client.request.assert_called_once()
        assert result == {"status": "success"}
        args, kwargs = mock_async_client._client.request.call_args
        assert kwargs["method"] == "POST"
        assert str(kwargs["url"]).endswith("upload")
        assert kwargs["files"] == files
        assert "Authorization" in kwargs["headers"]
        assert kwargs["headers"]["Authorization"] == "Bearer test-api-key"
        assert kwargs["headers"]["Accept"] == "application/json"

    async def test_request_multipart_with_data_and_headers(self, mock_async_client):
        resource = AsyncAPIResource(mock_async_client)
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_async_client._client.request.return_value = mock_response

        files = {"file": ("test.txt", b"content", "text/plain")}
        data = {"param": "value"}
        headers = {"Custom-Header": "value"}
        result = await resource._request_multipart("POST", "upload", files=files, data=data, headers=headers)

        mock_async_client._client.request.assert_called_once()
        assert result == {"status": "success"}
        args, kwargs = mock_async_client._client.request.call_args
        assert kwargs["data"] == data
        assert "Custom-Header" in kwargs["headers"]
        assert kwargs["headers"]["Custom-Header"] == "value"
        assert "Authorization" in kwargs["headers"]
        assert kwargs["headers"]["Authorization"] == "Bearer test-api-key"

    async def test_request_multipart_default_accept_header(self, mock_async_client):
        resource = AsyncAPIResource(mock_async_client)
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_async_client._client.request.return_value = mock_response

        files = {"file": ("test.txt", b"content", "text/plain")}
        # Call without providing 'Accept' in headers
        result = await resource._request_multipart("POST", "upload", files=files, headers={"Some-Other-Header": "value"})

        mock_async_client._client.request.assert_called_once()
        assert result == {"status": "success"}
        args, kwargs = mock_async_client._client.request.call_args
        assert "Accept" in kwargs["headers"]
        assert kwargs["headers"]["Accept"] == "application/json"
        assert "Some-Other-Header" in kwargs["headers"]

    async def test_request_multipart_error_handling(self, mock_async_client):
        resource = AsyncAPIResource(mock_async_client)
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="Error", request=MagicMock(), response=MagicMock()
        )
        mock_async_client._client.request.return_value = mock_response

        files = {"file": ("test.txt", b"content", "text/plain")}
        with pytest.raises(httpx.HTTPStatusError):
            await resource._request_multipart("POST", "upload", files=files)
class TestAPIResourceAdditional:
    """Additional tests to improve coverage for APIResource."""
    
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=VeniceClient)
        client._api_key = "test-api-key"
        client._base_url = httpx.URL("https://api.venice.ai/v1/")
        client._client = MagicMock(spec=httpx.Client)
        return client
    
    def test_request_multipart_response_error(self, mock_client):
        """Test _request_multipart when response.raise_for_status raises an exception."""
        resource = APIResource(mock_client)
        mock_response = MagicMock()
        # Set up the mock to raise an exception
        http_error = httpx.HTTPStatusError("HTTP Error", request=MagicMock(), response=MagicMock(status_code=500))
        mock_response.raise_for_status.side_effect = http_error
        mock_client._client.request.return_value = mock_response

        files = {"file": ("test.txt", b"content", "text/plain")}
        with pytest.raises(httpx.HTTPStatusError):
            resource._request_multipart("POST", "upload", files=files)
            
        mock_response.raise_for_status.assert_called_once()
        
    def test_request_multipart_json_error(self, mock_client):
        """Test _request_multipart when response.json raises a JSONDecodeError."""
        resource = APIResource(mock_client)
        mock_response = MagicMock()
        # The response passes raise_for_status but json() raises JSONDecodeError
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_client._client.request.return_value = mock_response

        files = {"file": ("test.txt", b"content", "text/plain")}
        with pytest.raises(json.JSONDecodeError):
            resource._request_multipart("POST", "upload", files=files)
            
        mock_response.json.assert_called_once()


@pytest.mark.asyncio
class TestAsyncAPIResourceAdditional:
    """Additional tests to improve coverage for AsyncAPIResource."""
    
    @pytest.fixture
    def mock_async_client(self):
        client = MagicMock(spec=AsyncVeniceClient)
        client._api_key = "test-api-key"
        client._base_url = httpx.URL("https://api.venice.ai/v1/")
        client._client = MagicMock(spec=httpx.AsyncClient)
        return client
    
    async def test_request_multipart_response_error(self, mock_async_client):
        """Test _request_multipart when response.raise_for_status raises an exception."""
        resource = AsyncAPIResource(mock_async_client)
        mock_response = MagicMock()
        # Set up the mock to raise an exception
        http_error = httpx.HTTPStatusError("HTTP Error", request=MagicMock(), response=MagicMock(status_code=500))
        mock_response.raise_for_status.side_effect = http_error
        mock_async_client._client.request.return_value = mock_response

        files = {"file": ("test.txt", b"content", "text/plain")}
        with pytest.raises(httpx.HTTPStatusError):
            await resource._request_multipart("POST", "upload", files=files)
            
        mock_response.raise_for_status.assert_called_once()
        
    async def test_request_multipart_json_error(self, mock_async_client):
        """Test _request_multipart when response.json raises a JSONDecodeError."""
        resource = AsyncAPIResource(mock_async_client)
        mock_response = MagicMock()
        # The response passes raise_for_status but json() raises JSONDecodeError
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_async_client._client.request.return_value = mock_response

        files = {"file": ("test.txt", b"content", "text/plain")}
        with pytest.raises(json.JSONDecodeError):
            await resource._request_multipart("POST", "upload", files=files)
            
        mock_response.json.assert_called_once()