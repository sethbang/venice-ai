"""
Test cases for ImageResource (synchronous) missed lines coverage.

This module implements test cases to cover specific missed lines in the
Image class from src/venice_ai/resources/image.py, targeting lines:
78, 337-338, 341, 345
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from httpx import Response

from venice_ai._client import VeniceClient
from venice_ai.resources.image import Image


class TestImageSyncMissedLines:
    """Test class for Image (synchronous) missed lines coverage."""

    def test_prepare_image_content_unsupported_type(self):
        """
        Test Case 1.1: Cover line 78 (handling of unsupported image content type).
        
        Objective: Cover line 78 (handling of unsupported image content type).
        Lines to cover: 78
        """
        mock_client = Mock(spec=VeniceClient)
        image_resource = Image(mock_client)
        
        with pytest.raises(TypeError, match="Unsupported content type from file-like object: <class 'int'>"):
            # Create a mock file-like object that returns an int when read
            mock_file = Mock()
            mock_file.read.return_value = 123  # This should trigger the TypeError
            image_resource._prepare_image_content(image=mock_file)

    @patch('venice_ai.resources.image.Image._prepare_image_content', return_value=b"mock_image_bytes")
    def test_upscale_enhance_string_false(self, mock_prepare_content):
        """
        Test Case 1.2: Cover lines 337-338 (enhance="false").
        
        Objective: Cover lines 337-338 (enhance="false").
        Lines to cover: 337, 338
        """
        mock_venice_client = Mock(spec=VeniceClient)
        image_resource = Image(mock_venice_client)
        mock_venice_client._request = Mock(return_value=b"mock_upscaled_image")
        
        image_resource.upscale(image="dummy_path.png", enhance=False, scale=2.0)
        
        mock_venice_client._request.assert_called_once()
        args, kwargs = mock_venice_client._request.call_args
        sent_data = kwargs.get("json_data")
        assert sent_data is not None
        assert sent_data.get("enhance") is False

    @patch('venice_ai.resources.image.Image._prepare_image_content', return_value=b"mock_image_bytes")
    def test_upscale_enhance_boolean_true(self, mock_prepare_content):
        """
        Test Case 1.3: Cover line 341 (enhance=True).
        
        Objective: Cover line 341 (enhance=True).
        Lines to cover: 341
        """
        mock_venice_client = Mock(spec=VeniceClient)
        image_resource = Image(mock_venice_client)
        mock_venice_client._request = Mock(return_value=b"mock_upscaled_image")
        
        image_resource.upscale(image="dummy_path.png", enhance=True, scale=2.0)
        
        mock_venice_client._request.assert_called_once()
        args, kwargs = mock_venice_client._request.call_args
        sent_data = kwargs.get("json_data")
        assert sent_data is not None
        assert sent_data.get("enhance") is True

    @patch('venice_ai.resources.image.Image._prepare_image_content', return_value=b"mock_image_bytes")
    def test_upscale_scale_one_forces_enhance_true(self, mock_prepare_content):
        """
        Test Case 1.4: Cover line 345 (scale=1.0 forces enhance=True when enhance is not "false" or True).
        
        Objective: Cover line 345 (scale=1.0 forces enhance=True when enhance is not "false" or True).
        Lines to cover: 345
        """
        mock_venice_client = Mock(spec=VeniceClient)
        image_resource = Image(mock_venice_client)
        mock_venice_client._request = Mock(return_value=b"mock_upscaled_image")
        
        # Sub-Case 1.4.1 (enhance=None)
        image_resource.upscale(image="dummy_path.png", enhance=None, scale=1.0)
        
        mock_venice_client._request.assert_called_once()
        args, kwargs = mock_venice_client._request.call_args
        sent_data = kwargs.get("json_data")
        assert sent_data is not None
        assert sent_data.get("enhance") is True
        
        # Cleanup
        mock_venice_client._request.reset_mock()
        
        # Sub-Case 1.4.2 (enhance="auto")
        image_resource.upscale(image="dummy_path.png", enhance=True, scale=1.0)
        
        mock_venice_client._request.assert_called_once()
        args, kwargs = mock_venice_client._request.call_args
        sent_data = kwargs.get("json_data")
        assert sent_data is not None
        assert sent_data.get("enhance") is True