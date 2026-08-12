"""
Unit tests for Venice AI identifier types.

Tests the ModelId and QueueId types, focusing on normalization behavior.
"""

import pytest
from pydantic import BaseModel

from venice_ai.types.identifiers import (
    ModelId,
    QueueId,
    normalize_model_id,
    normalize_queue_id,
)


class TestModelIdNormalization:
    """Test ModelId normalization (not strict validation)."""

    def test_preserves_case(self):
        # Model IDs are case-sensitive — inference rejects a mis-cased id, so the
        # SDK must NOT alter case (regression guard for the wai-Illustrious bug).
        assert normalize_model_id("wai-Illustrious") == "wai-Illustrious"
        assert normalize_model_id("Venice-SD35") == "Venice-SD35"

    def test_strips_whitespace(self):
        assert normalize_model_id("  llama-3.3-70b  ") == "llama-3.3-70b"

    def test_preserves_valid_characters(self):
        # Dots, hyphens should be preserved
        assert normalize_model_id("zai-org-glm-4.6") == "zai-org-glm-4.6"
        assert normalize_model_id("veo3.1-fast-text-to-video") == "veo3.1-fast-text-to-video"

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_model_id("")
        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_model_id("   ")

    def test_accepts_real_model_ids_from_api(self):
        """Test with actual model IDs from API validation."""
        api_model_ids = [
            "hidream",  # Shortest
            "kling-2.5-turbo-pro-image-to-video",  # Longest
            "zai-org-glm-4.6",  # Has dot
            "qwen3-235b-a22b-thinking-2507",  # Complex
        ]
        for model_id in api_model_ids:
            result = normalize_model_id(model_id)
            assert result == model_id


class TestQueueIdNormalization:
    """Test QueueId normalization."""

    def test_preserves_case(self):
        assert normalize_queue_id("ABC-123-DEF") == "ABC-123-DEF"

    def test_strips_whitespace(self):
        assert normalize_queue_id("  abc-123  ") == "abc-123"

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_queue_id("")
        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_queue_id("   ")

    def test_accepts_uuid_format(self):
        queue_id = "550e8400-e29b-41d4-a716-446655440000"
        result = normalize_queue_id(queue_id)
        assert result == queue_id


class TestModelIdInPydanticModel:
    """Test ModelId usage in Pydantic models."""

    def test_model_id_annotation_preserves_case(self):
        """ModelId annotation must NOT alter case (Venice ids are case-sensitive)."""

        class TestRequest(BaseModel):
            model: ModelId

        request = TestRequest(model="wai-Illustrious")
        assert request.model == "wai-Illustrious"

    def test_model_id_annotation_strips(self):
        """Test that ModelId annotation strips whitespace."""

        class TestRequest(BaseModel):
            model: ModelId

        request = TestRequest(model="  llama-3.3-70b  ")
        assert request.model == "llama-3.3-70b"

    def test_model_id_annotation_rejects_empty(self):
        """Test that ModelId annotation rejects empty values."""

        class TestRequest(BaseModel):
            model: ModelId

        with pytest.raises(Exception, match=".*"):  # Pydantic ValidationError
            TestRequest(model="")


class TestQueueIdInPydanticModel:
    """Test QueueId usage in Pydantic models."""

    def test_queue_id_annotation_preserves_case(self):
        """QueueId annotation must NOT alter case (API is authoritative)."""

        class TestRequest(BaseModel):
            queue_id: QueueId

        request = TestRequest(queue_id="ABC-123-DEF")
        assert request.queue_id == "ABC-123-DEF"

    def test_queue_id_annotation_rejects_empty(self):
        """Test that QueueId annotation rejects empty values."""

        class TestRequest(BaseModel):
            queue_id: QueueId

        with pytest.raises(Exception, match=".*"):  # Pydantic ValidationError
            TestRequest(queue_id="")
