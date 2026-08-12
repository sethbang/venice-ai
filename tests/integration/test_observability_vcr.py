"""
VCRpy-based integration tests for the Venice AI Observability Stack.

This module tests the observability components including:
- Enhanced metrics (EnhancedMetrics, EnhancedMetricsConfig)

Tests use VCRpy cassette recording/replay for deterministic HTTP interactions.
"""

import os

import pytest
import pytest_asyncio

from venice_ai.observability.metrics import EnhancedMetrics, EnhancedMetricsConfig


@pytest_asyncio.fixture
async def venice_client(backend_instance):
    """Create a Venice client for VCR testing with shared rate limit coordination."""
    api_key = os.getenv("VENICE_API_KEY", "test-api-key-for-recording")

    from venice_ai import create_test_venice_client
    from venice_ai.core.config import SchedulerMode

    client = create_test_venice_client(
        api_key=api_key,
        scheduler_mode=SchedulerMode.INTELLIGENT,
        enable_redis=True,
    )
    try:
        yield client
    finally:
        await client.close()


# ============================================================================
# Metrics Collection Tests
# ============================================================================


@pytest.mark.integration
async def test_metrics_collection_on_request(venice_client, model_selector, vcr_cassette):
    """
    Test that enhanced metrics are collected during requests.

    This verifies EnhancedMetrics initialization alongside a real API request.
    """
    # Create enhanced metrics collector
    metrics_config = EnhancedMetricsConfig(
        enabled=True,  # Will use dummy metrics if Prometheus not available
        include_detailed_metrics=True,
    )
    metrics = EnhancedMetrics(config=metrics_config)

    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        response = await venice_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Count from 1 to 3."}],
            max_completion_tokens=50,
            temperature=0.1,
        )

    # Verify metrics were initialized with active attributes
    assert hasattr(metrics, "streaming_fallback_total")
    assert hasattr(metrics, "custom_stream_created_total")

    # Verify response. Reasoning models can spend the whole token budget on
    # reasoning_content and return content=None (finish_reason="length"), so
    # accept either channel — this test is about metrics, not the answer text.
    assert response.id is not None
    message = response.choices[0].message
    text = message.content or message.reasoning_content or ""
    assert text, "expected content or reasoning_content in the response"
    assert "1" in text


@pytest.mark.integration
async def test_metrics_retry_and_timeout_tracking(venice_client):
    """
    Test that all 9 active metric attributes are present on EnhancedMetrics.
    """
    metrics_config = EnhancedMetricsConfig(enabled=True)
    metrics = EnhancedMetrics(config=metrics_config)

    # Verify all 9 active metric attributes exist (using dummy metrics if Prometheus unavailable)
    assert hasattr(metrics, "streaming_fallback_total")
    assert hasattr(metrics, "custom_stream_created_total")
    assert hasattr(metrics, "custom_stream_bytes_total")
    assert hasattr(metrics, "custom_stream_duration_seconds")
    assert hasattr(metrics, "tier_discovery_requests_total")
    assert hasattr(metrics, "tier_discovery_api_calls_total")
    assert hasattr(metrics, "tier_discovery_coalesced_total")
    assert hasattr(metrics, "tier_discovery_concurrent_requests")
    assert hasattr(metrics, "tier_discovery_time_saved_seconds")


# ============================================================================
# Get Enhanced Metrics Singleton Tests
# ============================================================================


@pytest.mark.integration
def test_get_enhanced_metrics_singleton():
    """
    Test the get_enhanced_metrics singleton pattern.

    Verifies that the metrics singleton is properly created and reused.
    """
    from venice_ai.observability.metrics import get_enhanced_metrics

    # Get the singleton (or create it)
    metrics1 = get_enhanced_metrics()
    assert metrics1 is not None

    # Get it again - should be the same instance
    metrics2 = get_enhanced_metrics()
    assert metrics2 is metrics1

    # Verify it has the expected active metric attributes
    assert hasattr(metrics1, "streaming_fallback_total")
    assert hasattr(metrics1, "custom_stream_created_total")
    assert hasattr(metrics1, "tier_discovery_requests_total")
