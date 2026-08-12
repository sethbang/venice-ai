"""Tests for AdaptiveSchedulerAdapter in venice_ai.factory."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.factory import AdaptiveSchedulerAdapter


class TestAdaptiveSchedulerAdapter:
    """Test AdaptiveSchedulerAdapter methods."""

    def test_is_running(self):
        mock_scheduler = MagicMock()
        mock_scheduler.is_running.return_value = True
        adapter = AdaptiveSchedulerAdapter(mock_scheduler)
        assert adapter.is_running() is True
        mock_scheduler.is_running.assert_called_once()

    @pytest.mark.asyncio
    async def test_start(self):
        mock_scheduler = AsyncMock()
        adapter = AdaptiveSchedulerAdapter(mock_scheduler)
        await adapter.start()
        mock_scheduler.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop(self):
        mock_scheduler = AsyncMock()
        adapter = AdaptiveSchedulerAdapter(mock_scheduler)
        await adapter.stop()
        mock_scheduler.stop.assert_awaited_once()

    def test_classifier_property_present(self):
        mock_scheduler = MagicMock()
        mock_scheduler.classifier = "my_classifier"
        adapter = AdaptiveSchedulerAdapter(mock_scheduler)
        assert adapter.classifier == "my_classifier"

    def test_classifier_property_absent(self):
        mock_scheduler = MagicMock(spec=[])  # no attributes
        adapter = AdaptiveSchedulerAdapter(mock_scheduler)
        assert adapter.classifier is None

    def test_circuit_breaker_property_present(self):
        mock_scheduler = MagicMock()
        mock_scheduler.circuit_breaker = "my_cb"
        adapter = AdaptiveSchedulerAdapter(mock_scheduler)
        assert adapter.circuit_breaker == "my_cb"

    def test_circuit_breaker_property_absent(self):
        mock_scheduler = MagicMock(spec=[])
        adapter = AdaptiveSchedulerAdapter(mock_scheduler)
        assert adapter.circuit_breaker is None

    @pytest.mark.asyncio
    async def test_submit_request(self):
        mock_scheduler = AsyncMock()
        mock_scheduler.submit_request.return_value = "result"
        adapter = AdaptiveSchedulerAdapter(mock_scheduler)

        metadata = MagicMock()
        request_func = AsyncMock(return_value="response")

        result = await adapter.submit_request(metadata, request_func, error_factory=None)
        assert result == "result"
        # error_factory should not be passed to the underlying scheduler
        mock_scheduler.submit_request.assert_awaited_once_with(metadata, request_func)
