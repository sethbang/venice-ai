"""
Additional test cases for the synchronous parts of utils.py to improve coverage.
"""

import pytest
import warnings
from unittest.mock import MagicMock
from venice_ai.utils import get_filtered_models
from venice_ai._client import VeniceClient

class TestUtilsSyncCoverage:
    """Test cases for synchronous functions in utils.py."""

    def test_get_filtered_models_sync(self):
        """Test the synchronous get_filtered_models function."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.models = MagicMock()
        mock_response = {
            "data": [
                {"type": "text", "model_spec": {"capabilities": {"streaming": True}}},
                {"type": "image", "model_spec": {"capabilities": {"streaming": False}}},
            ]
        }
        mock_client.models.list.return_value = mock_response

        # Test filtering by type
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            text_models = get_filtered_models(mock_client, model_type="text")
            assert len(text_models) == 1  # type: ignore
            assert text_models[0]["type"] == "text"  # type: ignore

        # Test filtering by capability
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            streaming_models = get_filtered_models(mock_client, supports_capabilities=["streaming"])
            assert len(streaming_models) == 1  # type: ignore
            assert streaming_models[0]["model_spec"]["capabilities"]["streaming"] is True  # type: ignore

    def test_get_filtered_models_sync_api_error(self):
        """Test that get_filtered_models handles API errors gracefully."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.models = MagicMock()
        mock_client.models.list.side_effect = Exception("API Error")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = get_filtered_models(mock_client)
            assert result == []

    def test_get_filtered_models_sync_no_data(self):
        """Test that get_filtered_models handles a response with no data key."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.models = MagicMock()
        mock_client.models.list.return_value = {}
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = get_filtered_models(mock_client)
            assert result == []

    def test_get_filtered_models_sync_data_is_none(self):
        """Test that get_filtered_models handles a response where data is None."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.models = MagicMock()
        mock_client.models.list.return_value = {"data": None}
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = get_filtered_models(mock_client)
            assert result == []

    def test_get_filtered_models_sync_no_spec_or_capabilities(self):
        """Test models with missing model_spec or capabilities."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.models = MagicMock()
        mock_response = {
            "data": [
                {"type": "text"},  # No model_spec
                {"type": "image", "model_spec": {}},  # No capabilities
            ]
        }
        mock_client.models.list.return_value = mock_response

        # Should not fail, and should not match the capability filter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = get_filtered_models(mock_client, supports_capabilities=["streaming"])
            assert len(result) == 0  # type: ignore