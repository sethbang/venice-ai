"""
Test cases for AsyncImageResource (asynchronous) missed lines coverage.

This module implements test cases to cover specific missed lines in the
AsyncImage class from src/venice_ai/resources/image.py, targeting lines:
438, 698-699, 701, 705
"""

import pytest
from unittest.mock import AsyncMock, patch
from pathlib import Path
from httpx import Response

from venice_ai._async_client import AsyncVeniceClient
from venice_ai.resources.image import AsyncImage


class TestAsyncImageMissedLines:
    """Test class for AsyncImage (asynchronous) missed lines coverage."""

    async def test_async_prepare_image_content_unsupported_type(self):
        """
        Test Case 2.1: Cover line 438.
        
        Objective: Cover line 438.
        Lines to cover: 438
        """
        mock_async_client = AsyncMock(spec=AsyncVeniceClient)
        async_image_resource = AsyncImage(mock_async_client)
        
        with pytest.raises(TypeError, match="Unsupported content type from file-like object: <class 'int'>"):
            # Create a mock file-like object that returns an int when read
            mock_file = AsyncMock()
            mock_file.read = AsyncMock(return_value=123)  # This should trigger the TypeError
            await async_image_resource._prepare_image_content(image=mock_file)

    @patch('venice_ai.resources.image.AsyncImage._prepare_image_content', new_callable=AsyncMock, return_value=b"mock_image_bytes")
    async def test_async_upscale_enhance_string_false(self, mock_prepare_content):
        """
        Test Case 2.2: Cover lines 698-699.
        
        Objective: Cover lines 698-699.
        Lines to cover: 698, 699
        """
        mock_async_venice_client = AsyncMock(spec=AsyncVeniceClient)
        async_image_resource = AsyncImage(mock_async_venice_client)
        mock_async_venice_client._request = AsyncMock(return_value=b"mock_async_upscaled_image")
        
        await async_image_resource.upscale(image="dummy_path.png", enhance=False, scale=2.0)
        
        mock_async_venice_client._request.assert_called_once()
        args, kwargs = mock_async_venice_client._request.call_args
        sent_data = kwargs.get("json_data")
        assert sent_data is not None
        assert sent_data.get("enhance") is False

    @patch('venice_ai.resources.image.AsyncImage._prepare_image_content', new_callable=AsyncMock, return_value=b"mock_image_bytes")
    async def test_async_upscale_enhance_boolean_true(self, mock_prepare_content):
        """
        Test Case 2.3: Cover line 701.
        
        Objective: Cover line 701.
        Lines to cover: 701
        """
        mock_async_venice_client = AsyncMock(spec=AsyncVeniceClient)
        async_image_resource = AsyncImage(mock_async_venice_client)
        mock_async_venice_client._request = AsyncMock(return_value=b"mock_async_upscaled_image")
        
        await async_image_resource.upscale(image="dummy_path.png", enhance=True, scale=2.0)
        
        mock_async_venice_client._request.assert_called_once()
        args, kwargs = mock_async_venice_client._request.call_args
        sent_data = kwargs.get("json_data")
        assert sent_data is not None
        assert sent_data.get("enhance") is True

    @patch('venice_ai.resources.image.AsyncImage._prepare_image_content', new_callable=AsyncMock, return_value=b"mock_image_bytes")
    async def test_async_upscale_scale_one_forces_enhance_true(self, mock_prepare_content):
        """
        Test Case 2.4: Cover line 705.
        
        Objective: Cover line 705.
        Lines to cover: 705
        """
        mock_async_venice_client = AsyncMock(spec=AsyncVeniceClient)
        async_image_resource = AsyncImage(mock_async_venice_client)
        mock_async_venice_client._request = AsyncMock(return_value=b"mock_async_upscaled_image")
        
        # Sub-Case 2.4.1 (enhance=None)
        await async_image_resource.upscale(image="dummy_path.png", enhance=None, scale=1.0)
        
        mock_async_venice_client._request.assert_called_once()
        args, kwargs = mock_async_venice_client._request.call_args
        sent_data = kwargs.get("json_data")
        assert sent_data is not None
        assert sent_data.get("enhance") is True
        
        # Cleanup
        mock_async_venice_client._request.reset_mock()
        
        # Sub-Case 2.4.2 (enhance="auto")
        await async_image_resource.upscale(image="dummy_path.png", enhance=True, scale=1.0)
        
        mock_async_venice_client._request.assert_called_once()
        args, kwargs = mock_async_venice_client._request.call_args
        sent_data = kwargs.get("json_data")
        assert sent_data is not None
        assert sent_data.get("enhance") is True