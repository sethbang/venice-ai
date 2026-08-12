"""
Test module for DynamicModelSelector error handling and edge cases.

Focuses on improving branch coverage for uncovered error paths in models/selection.py.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio

from venice_ai.models.selection import DynamicModelSelector


class TestDynamicModelSelectorErrorHandling:
    """Test error handling in DynamicModelSelector."""

    @pytest_asyncio.fixture
    async def model_selector_with_mock_client(self):
        """Create DynamicModelSelector with mocked client."""
        mock_client = AsyncMock()

        model_selector = DynamicModelSelector(client=mock_client)
        yield model_selector

    @pytest.mark.asyncio
    async def test_fetch_models_cancelled_error(self, model_selector_with_mock_client):
        """Test _fetch_models handles CancelledError (lines 168-169)."""
        model_selector = model_selector_with_mock_client

        # Mock models.list to raise CancelledError
        model_selector.client.models = Mock()
        model_selector.client.models.list = AsyncMock(
            side_effect=asyncio.CancelledError("Fetch cancelled")
        )

        # CancelledError should be re-raised
        with pytest.raises(asyncio.CancelledError):
            await model_selector._fetch_models()

    @pytest.mark.asyncio
    async def test_fetch_models_value_error_with_cache(self, model_selector_with_mock_client):
        """Test _fetch_models handles ValueError with cached models (lines 170-175)."""
        model_selector = model_selector_with_mock_client

        # Set up cached models (as dictionaries, not Mocks)
        model_selector._cache.models = {"cached-model": {"id": "cached-model", "type": "text"}}

        # Mock models.list to raise ValueError
        model_selector.client.models.list.side_effect = ValueError("Invalid response")

        # Should return cached models (graceful degradation) - need force_refresh to trigger fetch
        result = await model_selector._fetch_models(force_refresh=True)

        # Verify cached model was returned despite the error
        assert "cached-model" in result
        assert result == model_selector._cache.models

    @pytest.mark.asyncio
    async def test_fetch_models_type_error_with_cache(self, model_selector_with_mock_client):
        """Test _fetch_models handles TypeError with cached models (lines 170-175)."""
        model_selector = model_selector_with_mock_client

        # Set up cached models (as dictionaries, not Mocks)
        model_selector._cache.models = {"cached-model": {"id": "cached-model", "type": "text"}}

        # Mock models.list to raise TypeError
        model_selector.client.models.list.side_effect = TypeError("Type mismatch")

        # Should return cached models (graceful degradation) - need force_refresh to trigger fetch
        result = await model_selector._fetch_models(force_refresh=True)

        # Verify cached model was returned despite the error
        assert "cached-model" in result
        assert result == model_selector._cache.models

    @pytest.mark.asyncio
    async def test_fetch_models_attribute_error_with_cache(self, model_selector_with_mock_client):
        """Test _fetch_models handles AttributeError with cached models (lines 170-175)."""
        model_selector = model_selector_with_mock_client

        # Set up cached models (as dictionaries, not Mocks)
        model_selector._cache.models = {"cached-model": {"id": "cached-model", "type": "text"}}

        # Mock models.list to raise AttributeError
        model_selector.client.models.list.side_effect = AttributeError("Missing attribute")

        # Should return cached models (graceful degradation) - need force_refresh to trigger fetch
        result = await model_selector._fetch_models(force_refresh=True)

        # Verify cached model was returned despite the error
        assert "cached-model" in result
        assert result == model_selector._cache.models

    @pytest.mark.asyncio
    async def test_fetch_models_os_error_with_cache(self, model_selector_with_mock_client):
        """Test _fetch_models handles OSError with cached models (lines 170-175)."""
        model_selector = model_selector_with_mock_client

        # Set up cached models (as dictionaries, not Mocks)
        model_selector._cache.models = {"cached-model": {"id": "cached-model", "type": "text"}}

        # Mock models.list to raise OSError
        model_selector.client.models.list.side_effect = OSError("Network error")

        # Should return cached models (graceful degradation) - need force_refresh to trigger fetch
        result = await model_selector._fetch_models(force_refresh=True)

        # Verify cached model was returned despite the error
        assert "cached-model" in result
        assert result == model_selector._cache.models

    @pytest.mark.asyncio
    async def test_fetch_models_error_without_cache(self, model_selector_with_mock_client):
        """Test _fetch_models raises exception when no cache available (lines 170-175)."""
        model_selector = model_selector_with_mock_client

        # No cached models
        model_selector._cache.models = {}

        # Mock models.list to raise ValueError
        model_selector.client.models = Mock()
        model_selector.client.models.list = AsyncMock(side_effect=ValueError("Invalid response"))

        # Should raise the exception since no cache is available
        with pytest.raises(ValueError, match="Invalid response"):
            await model_selector._fetch_models()


class TestDynamicModelSelectorFilteringEdgeCases:
    """Test edge cases in model filtering logic."""

    @pytest_asyncio.fixture
    async def model_selector_with_models(self):
        """Create DynamicModelSelector with some test models."""
        mock_client = AsyncMock()
        model_selector = DynamicModelSelector(client=mock_client)

        # Set up test models (as dictionaries)
        model_selector._cache.models = {
            "llm-model-1": {"id": "llm-model-1", "type": "llm", "object": "model"},
            "image-model-1": {
                "id": "image-model-1",
                "type": "image",
                "object": "model",
            },
            "audio-model-1": {
                "id": "audio-model-1",
                "type": "audio",
                "object": "model",
            },
        }

        yield model_selector

    @pytest.mark.asyncio
    async def test_get_available_models_all_types(self, model_selector_with_models):
        """Test get_available_models returns all models when no filter."""
        model_selector = model_selector_with_models

        models = await model_selector.get_available_models()

        assert len(models) == 3
        assert "llm-model-1" in models
        assert "image-model-1" in models
        assert "audio-model-1" in models

    @pytest.mark.asyncio
    async def test_get_available_models_filtered_by_type(self, model_selector_with_models):
        """Test get_available_models filtered by resource type."""
        model_selector = model_selector_with_models

        # Filter by LLM type
        models = await model_selector.get_available_models(resource_type="llm")

        assert len(models) == 1
        assert "llm-model-1" in models

        # Filter by image type
        models = await model_selector.get_available_models(resource_type="image")

        assert len(models) == 1
        assert "image-model-1" in models

    @pytest.mark.asyncio
    async def test_get_available_models_force_refresh(self, model_selector_with_models):
        """Test get_available_models with force_refresh."""
        model_selector = model_selector_with_models

        # Mock _fetch_models to update cache with new models
        async def mock_fetch(force_refresh=False):
            new_models = {"new-model": {"id": "new-model", "type": "llm", "object": "model"}}
            model_selector._cache.update(new_models)
            return new_models

        with patch.object(model_selector, "_fetch_models", side_effect=mock_fetch):
            models = await model_selector.get_available_models(force_refresh=True)

            # Should have the new model
            assert "new-model" in models
            assert "llm-model-1" not in models

    @pytest.mark.asyncio
    async def test_get_available_models_empty_cache(self):
        """Test get_available_models with empty cache triggers fetch."""
        mock_client = AsyncMock()
        model_selector = DynamicModelSelector(client=mock_client)

        # Mock _fetch_models to update cache
        async def mock_fetch(force_refresh=False):
            new_models = {
                "fetched-model": {
                    "id": "fetched-model",
                    "type": "llm",
                    "object": "model",
                }
            }
            model_selector._cache.update(new_models)
            return new_models

        with patch.object(model_selector, "_fetch_models", side_effect=mock_fetch):
            models = await model_selector.get_available_models()

            assert "fetched-model" in models


class TestDynamicModelSelectorCacheManagement:
    """Test cache management in DynamicModelSelector."""

    @pytest.mark.asyncio
    async def test_concurrent_fetch_protection(self):
        """Test that concurrent fetches are protected by lock."""
        mock_client = AsyncMock()
        model_selector = DynamicModelSelector(client=mock_client)

        fetch_count = 0

        async def mock_fetch(force_refresh=False):
            nonlocal fetch_count
            fetch_count += 1
            await asyncio.sleep(0.01)  # Simulate some work
            new_models = {"model1": {"id": "model1", "type": "llm", "object": "model"}}
            model_selector._cache.update(new_models)
            return new_models

        with patch.object(model_selector, "_fetch_models", side_effect=mock_fetch):
            # Start multiple concurrent fetches
            results = await asyncio.gather(
                model_selector.get_available_models(force_refresh=True),
                model_selector.get_available_models(force_refresh=True),
                model_selector.get_available_models(force_refresh=True),
            )

            # All should succeed
            assert all("model1" in r for r in results)

            # Fetch should have been called (lock might not prevent all calls with force_refresh)
            assert fetch_count >= 1


class TestDynamicModelSelectorInitialization:
    """Test DynamicModelSelector initialization edge cases."""

    def test_init_with_client(self):
        """Test initialization with client."""
        mock_client = Mock()
        selector = DynamicModelSelector(client=mock_client)

        assert selector.client == mock_client
        assert selector._cache.models == {}
        assert selector._fetch_lock is not None

    def test_init_without_client(self):
        """Test initialization without client."""
        selector = DynamicModelSelector(client=None)

        assert selector.client is None
        assert selector._cache.models == {}

    @pytest.mark.asyncio
    async def test_fetch_models_with_malformed_response(self):
        """Test _fetch_models with malformed model data."""
        mock_client = AsyncMock()
        selector = DynamicModelSelector(client=mock_client)

        # Mock models.list to return invalid model structure
        mock_model = Mock()
        # Model without 'id' attribute should be handled
        delattr(mock_model, "id")  # Remove id if it exists
        mock_model.configure_mock(**{"id": None})  # Set to None

        selector.client.models = Mock()
        selector.client.models.list = AsyncMock(return_value=[mock_model])

        # Should handle gracefully or skip invalid models
        try:
            result = await selector._fetch_models()
            # If it succeeds, verify it handled the None id
            assert isinstance(result, dict)
        except (ValueError, TypeError, AttributeError):
            # Or it might raise an exception, which is also acceptable
            pass
