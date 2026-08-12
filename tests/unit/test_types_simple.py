"""
Simple test module for Venice AI type definitions.

This module provides basic tests for the type classes to ensure
they can be instantiated and their basic functionality works.

Updated for v2.0.0: Uses generated types from venice_ai.types package.
"""

# Import actual types that exist (updated for v2.0.0)
from venice_ai.types import ApiKeyUsage, ConsumptionLimit, TrailingSevenDaysUsage


class TestApiKeyTypes:
    """Test API key type definitions (v2.0.0)."""

    def test_consumption_limit_creation(self):
        """Test ConsumptionLimit model creation."""
        limit = ConsumptionLimit(usd=100.0, diem=50.0)

        assert limit.usd == 100.0
        assert limit.diem == 50.0

    def test_consumption_limit_partial_fields(self):
        """Test ConsumptionLimit with partial fields (optional)."""
        limit = ConsumptionLimit(usd=50.0, diem=None)

        assert limit.usd == 50.0
        assert limit.diem is None

    def test_consumption_limit_none_value(self):
        """Test ConsumptionLimit with None values."""
        limit = ConsumptionLimit(usd=None, diem=None)

        assert limit.usd is None
        assert limit.diem is None

    def test_api_key_usage(self):
        """Test ApiKeyUsage model (v2.0.0 with TrailingSevenDaysUsage)."""
        trailing = TrailingSevenDaysUsage(usd="50.00", diem="25.00")
        usage = ApiKeyUsage(trailingSevenDays=trailing)

        assert usage.trailingSevenDays is not None
        assert usage.trailingSevenDays.usd == "50.00"
        assert usage.trailingSevenDays.diem == "25.00"

    def test_api_key_usage_with_model_validate(self):
        """Test ApiKeyUsage with dict validation."""
        usage = ApiKeyUsage.model_validate({"trailingSevenDays": {"usd": "50.00", "diem": "25.00"}})

        assert usage.trailingSevenDays is not None
        assert usage.trailingSevenDays.usd == "50.00"
        assert usage.trailingSevenDays.diem == "25.00"

    def test_consumption_limit_serialization(self):
        """Test ConsumptionLimit serialization."""
        limit = ConsumptionLimit(usd=75.0, diem=30.0)

        # Test to dict
        limit_dict = limit.model_dump()
        assert limit_dict["usd"] == 75.0
        assert limit_dict["diem"] == 30.0

        # Test from dict
        limit2 = ConsumptionLimit.model_validate(limit_dict)
        assert limit2.usd == limit.usd
        assert limit2.diem == limit.diem

    def test_consumption_limit_large_values(self):
        """Test ConsumptionLimit with large values."""
        # Very large values
        large_limit = ConsumptionLimit(usd=1_000_000.99, diem=500_000.50)
        assert large_limit.usd == 1_000_000.99
        assert large_limit.diem == 500_000.50

        # Very small values
        small_limit = ConsumptionLimit(usd=0.01, diem=0.005)
        assert small_limit.usd == 0.01
        assert small_limit.diem == 0.005

    def test_consumption_limit_equality(self):
        """Test ConsumptionLimit equality."""
        limit1 = ConsumptionLimit(usd=100.0, diem=50.0)
        limit2 = ConsumptionLimit(usd=100.0, diem=50.0)

        assert limit1.usd == limit2.usd
        assert limit1.diem == limit2.diem


class TestTopLevelTypeReExports:
    """Verify that response/request types are re-exported from ``venice_ai.types``.

    Other video types (queue, quote, complete, retrieve) are already re-exported
    at the top level; transcription types should match for consistency.
    """

    def test_video_transcription_types_top_level_importable(self):
        from venice_ai.types import (
            VideoTranscriptionRequest,
            VideoTranscriptionResponse,
        )

        # Smoke check: types are usable Pydantic models with the expected fields.
        request = VideoTranscriptionRequest(
            url="https://www.youtube.com/watch?v=abc123",
            response_format="json",
        )
        assert request.url == "https://www.youtube.com/watch?v=abc123"
        assert request.response_format == "json"

        response = VideoTranscriptionResponse(transcript="hello world", lang="en")
        assert response.transcript == "hello world"
        assert response.lang == "en"
