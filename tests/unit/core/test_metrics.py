"""
Tests for the metrics collection system.

Covers:
- PROMETHEUS_AVAILABLE flag
- DummyMetric placeholder
- MetricsCollector (init, validation recording, reset)
- get_metrics_collector singleton
- reset_global_metrics_collector
"""

from unittest.mock import Mock, patch

import pytest

from venice_ai.core.metrics import (
    DummyMetric,
    MetricsCollector,
    get_metrics_collector,
    reset_global_metrics_collector,
)


class TestDummyMetric:
    """Tests for DummyMetric placeholder."""

    def test_labels_returns_self(self):
        dummy = DummyMetric()
        assert dummy.labels(model="test") is dummy

    def test_inc_dec_set_observe_noop(self):
        dummy = DummyMetric()
        dummy.inc(1)
        dummy.dec(1)
        dummy.set(5)
        dummy.observe(0.5)

    def test_time_context_manager(self):
        dummy = DummyMetric()
        with dummy.time():
            pass

    def test_context_manager(self):
        dummy = DummyMetric()
        with dummy:
            pass


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def setup_method(self):
        self.collector = MetricsCollector(enable_prometheus=False)

    def test_init_without_prometheus(self):
        collector = MetricsCollector(enable_prometheus=False)
        assert collector.enable_prometheus is False
        assert isinstance(collector.validation_successful_total, DummyMetric)
        assert isinstance(collector.validation_failed_total, DummyMetric)

    @patch("venice_ai.core.metrics.PROMETHEUS_AVAILABLE", False)
    def test_init_with_prometheus_unavailable(self):
        """When PROMETHEUS_AVAILABLE is False, enable_prometheus is forced off."""
        collector = MetricsCollector(enable_prometheus=True)
        assert collector.enable_prometheus is False
        assert isinstance(collector.validation_successful_total, DummyMetric)

    @patch("venice_ai.core.metrics.PROMETHEUS_AVAILABLE", True)
    @patch("venice_ai.core.metrics.CollectorRegistry")
    @patch("venice_ai.core.metrics.Counter")
    def test_init_with_prometheus(self, mock_counter, mock_registry):
        mock_registry.return_value = Mock()
        mock_counter.return_value = Mock()

        collector = MetricsCollector(enable_prometheus=True)
        assert collector.enable_prometheus is True
        mock_registry.assert_called_once()

    @patch("venice_ai.core.metrics.PROMETHEUS_AVAILABLE", True)
    @patch("venice_ai.core.metrics.CollectorRegistry", None)
    def test_init_prometheus_collector_registry_none(self):
        """Falls back to dummy when CollectorRegistry is None."""
        collector = MetricsCollector(enable_prometheus=True)
        assert collector.enable_prometheus is False
        assert isinstance(collector.validation_successful_total, DummyMetric)

    @patch("venice_ai.core.metrics.PROMETHEUS_AVAILABLE", True)
    @patch("venice_ai.core.metrics.CollectorRegistry")
    @patch("venice_ai.core.metrics.Counter")
    def test_init_prometheus_value_error(self, mock_counter, mock_registry):
        """Falls back to dummy when Counter raises ValueError."""
        mock_registry.return_value = Mock()
        mock_counter.side_effect = ValueError("bad config")

        collector = MetricsCollector(enable_prometheus=True)
        assert isinstance(collector.validation_successful_total, DummyMetric)

    @patch("venice_ai.core.metrics.PROMETHEUS_AVAILABLE", True)
    @patch("venice_ai.core.metrics.CollectorRegistry")
    @patch("venice_ai.core.metrics.Counter")
    def test_init_prometheus_attribute_error(self, mock_counter, mock_registry):
        """Falls back to dummy when Counter raises AttributeError."""
        mock_registry.return_value = Mock()
        mock_counter.side_effect = AttributeError("missing")

        collector = MetricsCollector(enable_prometheus=True)
        assert isinstance(collector.validation_successful_total, DummyMetric)

    def test_record_validation_success_dummy(self):
        """record_validation_success works with dummy metrics (no-op)."""
        self.collector.record_validation_success("TestModel")
        # No error raised; dummy .inc() is a no-op

    def test_record_validation_failure_dummy(self):
        """record_validation_failure works with dummy metrics (no-op)."""
        self.collector.record_validation_failure("TestModel")

    @patch("venice_ai.core.metrics.PROMETHEUS_AVAILABLE", True)
    @patch("venice_ai.core.metrics.CollectorRegistry")
    @patch("venice_ai.core.metrics.Counter")
    def test_record_validation_success_with_prometheus(self, mock_counter, mock_registry):
        mock_registry.return_value = Mock()
        mock_counter_instance = Mock()
        mock_counter.return_value = mock_counter_instance

        collector = MetricsCollector(enable_prometheus=True)
        collector.record_validation_success("TestModel")

        mock_counter_instance.labels.assert_called()

    @patch("venice_ai.core.metrics.PROMETHEUS_AVAILABLE", True)
    @patch("venice_ai.core.metrics.CollectorRegistry")
    @patch("venice_ai.core.metrics.Counter")
    def test_record_validation_failure_with_prometheus(self, mock_counter, mock_registry):
        mock_registry.return_value = Mock()
        mock_counter_instance = Mock()
        mock_counter.return_value = mock_counter_instance

        collector = MetricsCollector(enable_prometheus=True)
        collector.record_validation_failure("TestModel")

        mock_counter_instance.labels.assert_called()

    def test_reset_reinitialises_metrics(self):
        """reset() re-runs _init_prometheus_metrics."""
        self.collector.reset()
        # After reset, metrics are still dummy since prometheus=False
        assert isinstance(self.collector.validation_successful_total, DummyMetric)

    def test_registry_attribute_exists(self):
        """_registry attribute exists (observability/metrics.py accesses it)."""
        assert hasattr(self.collector, "_registry")

    def test_extra_kwargs_accepted(self):
        """Unknown kwargs are accepted (backward compat)."""
        collector = MetricsCollector(enable_prometheus=False, some_legacy_kwarg=42)
        assert isinstance(collector.validation_successful_total, DummyMetric)


class TestGlobalMetricsCollector:
    """Tests for global singleton functions."""

    def setup_method(self):
        reset_global_metrics_collector()

    def teardown_method(self):
        reset_global_metrics_collector()

    def test_singleton_returns_same_instance(self):
        c1 = get_metrics_collector()
        c2 = get_metrics_collector()
        assert c1 is c2

    def test_reset_creates_new_instance(self):
        c1 = get_metrics_collector()
        reset_global_metrics_collector()
        c2 = get_metrics_collector()
        assert c1 is not c2

    def test_get_metrics_collector_passes_kwargs(self):
        collector = get_metrics_collector(enable_prometheus=False)
        assert collector.enable_prometheus is False


class TestPrometheusImportFallback:
    """Test PROMETHEUS_AVAILABLE flag."""

    def test_prometheus_available_is_bool(self):
        from venice_ai.core.metrics import PROMETHEUS_AVAILABLE

        assert isinstance(PROMETHEUS_AVAILABLE, bool)


if __name__ == "__main__":
    pytest.main([__file__])
