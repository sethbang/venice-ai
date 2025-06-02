"""
Additional targeted tests for VeniceClient to improve code coverage.

This module focuses on edge cases and specific scenarios that might
help improve test coverage further.
"""

import pytest
import httpx
import logging
from typing import cast
from unittest.mock import patch, MagicMock

from venice_ai._client import VeniceClient
from venice_ai.exceptions import VeniceError


class TestClientCoverageAdditional:
    """Additional tests targeting specific coverage gaps in VeniceClient."""

    def test_initialization_with_external_client_attributes(self):
        """Test that client attributes are properly handled with external client."""
        external_client = httpx.Client(
            base_url="https://custom-base.api/",
            timeout=30.0,
            headers={"User-Agent": "CustomAgent/1.0"}
        )
        
        client = VeniceClient(
            api_key="test-key", 
            http_client=external_client
        )
        
        # Should use the external client's attributes
        assert client._client == external_client
        assert client._client.headers["Authorization"] == f"Bearer test-key"
        
        # Clean up
        external_client.close()

    def test_request_with_missing_headers_dict(self):
        """Test request handling when client.headers is unexpectedly modified."""
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-key")
            
            # Remove headers from client to test edge case
            delattr(client._client, 'headers')
            client._client.headers = {}
            
            # Mock response
            mock_response = MagicMock()
            mock_response.json.return_value = {"result": "success"}
            mock_request = cast(MagicMock, client._client.request)
            mock_request.return_value = mock_response
            
            # Should handle missing headers gracefully
            result = client._request("GET", "test/endpoint")
            assert result == {"result": "success"}
            
            # Request should still be made
            mock_request.assert_called_once()

    def test_request_multipart_logger_usage(self):
        """Test that _request_multipart properly uses the logger."""
        with patch('httpx.Client') as mock_client, \
             patch('venice_ai._client.logger') as mock_logger:
            
            client = VeniceClient(api_key="test-key")
            
            # Mock response
            mock_response = MagicMock()
            mock_response.json.return_value = {"status": "success"}
            mock_response.raise_for_status = MagicMock()
            mock_request = cast(MagicMock, client._client.request)
            mock_request.return_value = mock_response
            
            # Make a multipart request
            files = {"file": ("test.jpg", b"image data", "image/jpeg")}
            client._request_multipart("POST", "test/upload", files=files)
            
            # Verify logger was used properly
            assert mock_logger.debug.call_count >= 5
            
            # Check specific logger calls
            log_calls = [args[0] for args, _ in mock_logger.debug.call_args_list]
            assert any("Sending multipart request" in str(call) for call in log_calls)
            assert any("Request headers" in str(call) for call in log_calls)
            assert any("Files" in str(call) for call in log_calls)

    def test_client_resource_initialization(self):
        """Test that client properly initializes all resources."""
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-key")
            
            # Check all resources were initialized
            assert hasattr(client, 'chat')
            assert hasattr(client, 'models')
            assert hasattr(client, 'image')
            assert hasattr(client, 'audio')
            assert hasattr(client, 'billing')
            assert hasattr(client, 'embeddings')
            assert hasattr(client, 'api_keys')
            assert hasattr(client, 'characters')

    def test_close_with_missing_client(self):
        """Test close method when _client attribute is missing."""
        with patch('httpx.Client'):
            client = VeniceClient(api_key="test-key")
            
            # Remove _client attribute to test edge case
            delattr(client, '_client')
            
            # Should not raise an error
            client.close()