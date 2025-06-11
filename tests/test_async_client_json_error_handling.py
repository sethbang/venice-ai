"""Tests for async client JSON error handling coverage."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from venice_ai._async_client import AsyncVeniceClient


class TestAsyncClientJSONErrorHandling:
    """Test async client JSON error handling for complete coverage."""

    @pytest.fixture
    def async_client(self):
        """Create an async client for testing."""
        return AsyncVeniceClient(api_key="test-key")

    @pytest.mark.asyncio
    async def test_request_json_decode_error_with_successful_status(self, async_client):
        """Test handling of JSON decode error with successful HTTP status."""
        # Mock the httpx client and response
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.text = "invalid json response"
        mock_response.request = MagicMock()
        mock_response.request.method = "POST"
        mock_response.request.url = "https://api.venice.ai/api/v1/test"
        mock_response.aread = AsyncMock()
        mock_response.aclose = AsyncMock()
        
        with patch.object(async_client, '_client') as mock_client:
            mock_client.request = AsyncMock(return_value=mock_response)
            
            # This should trigger the JSON decode error handling path
            result = await async_client._request("POST", "test", json_data={"test": "data"})
            
            # Should return None when JSON decoding fails for successful status
            assert result is None
            
            # Verify response was properly closed
            mock_response.aread.assert_called_once()
            mock_response.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_response_not_read_error_with_successful_status(self, async_client):
        """Test handling of ResponseNotRead error with successful HTTP status."""
        # Mock the httpx client and response
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.side_effect = httpx.ResponseNotRead()
        mock_response.text = "some response text"
        mock_response.request = MagicMock()
        mock_response.request.method = "GET"
        mock_response.request.url = "https://api.venice.ai/api/v1/test"
        mock_response.aread = AsyncMock()
        mock_response.aclose = AsyncMock()
        
        with patch.object(async_client, '_client') as mock_client:
            mock_client.request = AsyncMock(return_value=mock_response)
            
            # This should trigger the ResponseNotRead error handling path
            result = await async_client._request("GET", "test")
            
            # Should return None when ResponseNotRead occurs for successful status
            assert result is None
            
            # Verify response was properly closed
            mock_response.aread.assert_called_once()
            mock_response.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_empty_response_with_successful_status(self, async_client):
        """Test handling of empty response with successful HTTP status."""
        # Mock the httpx client and response
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 204  # No Content
        mock_response.is_success = True
        mock_response.content = b""
        mock_response.aclose = AsyncMock()
        
        with patch.object(async_client, '_client') as mock_client:
            mock_client.request = AsyncMock(return_value=mock_response)
            
            # This should trigger the empty content handling path
            result = await async_client._request("DELETE", "test")
            
            # Should return None for empty successful response
            assert result is None
            
            # Verify response was properly closed
            mock_response.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_with_async_mock_json_method(self, async_client):
        """Test request with AsyncMock json method (covers the isinstance check)."""
        # Mock the httpx client and response with AsyncMock json method
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.content = b'{"success": true}'
        mock_response.json = AsyncMock(return_value={"success": True})
        mock_response.aclose = AsyncMock()
        
        with patch.object(async_client, '_client') as mock_client:
            mock_client.request = AsyncMock(return_value=mock_response)
            
            # This should trigger the AsyncMock json handling path
            result = await async_client._request("GET", "test")
            
            # Should return the JSON data
            assert result == {"success": True}
            
            # Verify response was properly closed
            mock_response.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_with_regular_json_method(self, async_client):
        """Test request with regular (non-async) json method."""
        # Create a mock response with a regular json method
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.content = b'{"data": "test"}'
        
        # Create a regular (non-async) json method
        def json_method():
            return {"data": "test"}
        
        mock_response.json = json_method
        mock_response.aclose = AsyncMock()
        
        with patch.object(async_client, '_client') as mock_client:
            mock_client.request = AsyncMock(return_value=mock_response)
            
            # This should trigger the regular json handling path
            result = await async_client._request("GET", "test")
            
            # Should return the JSON data
            assert result == {"data": "test"}
            
            # Verify response was properly closed
            mock_response.aclose.assert_called_once()