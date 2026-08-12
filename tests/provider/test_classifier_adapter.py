"""
Comprehensive tests for venice_ai/provider/classifier_adapter.py

This module provides thorough test coverage for VeniceClassifierAdapter,
targeting 100% coverage including:
- __init__: Classifier storage (line 45)
- classify: Async classification with enum resource_type (lines 63-70)
- classify: Async classification with non-enum resource_type (lines 66-70 fallback)
- CoreRequestMetadata creation (lines 73-83)
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from adaptive_rate_limiter.protocols import RequestMetadata as CoreRequestMetadata

from venice_ai._queue_types import (
    RequestMetadata as VeniceRequestMetadata,
)
from venice_ai._queue_types import (
    ResourceType,
)
from venice_ai.provider.classifier_adapter import VeniceClassifierAdapter


class TestVeniceClassifierAdapterInit:
    """Tests for VeniceClassifierAdapter.__init__ method (line 45)."""

    def test_stores_classifier_instance(self):
        """__init__ stores the provided classifier in _classifier attribute."""
        mock_classifier = MagicMock()
        adapter = VeniceClassifierAdapter(mock_classifier)
        assert adapter._classifier is mock_classifier

    def test_accepts_request_classifier(self):
        """Adapter can be initialized with any object (duck typing)."""
        # Create a mock that simulates RequestClassifier
        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock()
        adapter = VeniceClassifierAdapter(mock_classifier)
        assert hasattr(adapter._classifier, "classify")


class TestVeniceClassifierAdapterClassify:
    """Tests for VeniceClassifierAdapter.classify method (lines 63-83)."""

    @pytest.mark.asyncio
    async def test_classify_with_enum_resource_type(self):
        """classify converts ResourceType enum to string using .value (lines 66-68)."""
        # Create mock Venice metadata with enum ResourceType
        venice_meta = VeniceRequestMetadata(
            request_id="req_123",
            model_id="llama-3.3-70b",
            resource_type=ResourceType.LLM,
            estimated_tokens=150,
            priority=5,
            timeout=30.0,
            client_id="client_abc",
            endpoint="/chat/completions",
            requires_model=True,
        )

        # Create mock classifier that returns Venice metadata
        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(return_value=venice_meta)

        adapter = VeniceClassifierAdapter(mock_classifier)
        request = {"model": "llama-3.3-70b", "endpoint": "/chat/completions"}

        result = await adapter.classify(request)

        # Verify classifier was called with the request
        mock_classifier.classify.assert_called_once_with(request)

        # Verify result type and attributes
        assert isinstance(result, CoreRequestMetadata)
        assert result.request_id == "req_123"
        assert result.model_id == "llama-3.3-70b"
        assert result.resource_type == "llm"  # Enum converted to string via .value
        assert result.estimated_tokens == 150
        assert result.priority == 5
        assert result.timeout == 30.0
        assert result.client_id == "client_abc"
        assert result.endpoint == "/chat/completions"
        assert result.requires_model is True

    @pytest.mark.asyncio
    async def test_classify_with_image_resource_type(self):
        """classify handles IMAGE resource type correctly."""
        venice_meta = VeniceRequestMetadata(
            request_id="req_456",
            model_id="flux-pro",
            resource_type=ResourceType.IMAGE,
            estimated_tokens=None,
            priority=0,
            timeout=60.0,
            client_id=None,
            endpoint="/images/generate",
            requires_model=True,
        )

        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(return_value=venice_meta)

        adapter = VeniceClassifierAdapter(mock_classifier)
        result = await adapter.classify({"model": "flux-pro", "endpoint": "/images/generate"})

        assert result.resource_type == "image"
        assert result.estimated_tokens == 0  # None converted to 0

    @pytest.mark.asyncio
    async def test_classify_with_audio_resource_type(self):
        """classify handles AUDIO resource type correctly."""
        venice_meta = VeniceRequestMetadata(
            request_id="req_789",
            model_id="tts-kokoro",
            resource_type=ResourceType.AUDIO,
            estimated_tokens=None,
            priority=1,
            timeout=45.0,
            client_id="voice_client",
            endpoint="/audio/speech",
            requires_model=True,
        )

        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(return_value=venice_meta)

        adapter = VeniceClassifierAdapter(mock_classifier)
        result = await adapter.classify({"model": "tts-kokoro"})

        assert result.resource_type == "audio"
        assert result.priority == 1
        assert result.client_id == "voice_client"

    @pytest.mark.asyncio
    async def test_classify_with_embedding_resource_type(self):
        """classify handles EMBEDDING resource type correctly."""
        venice_meta = VeniceRequestMetadata(
            request_id="req_emb",
            model_id="bge-m3",
            resource_type=ResourceType.EMBEDDING,
            estimated_tokens=50,
            priority=0,
            timeout=30.0,
            client_id=None,
            endpoint="/embeddings",
            requires_model=True,
        )

        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(return_value=venice_meta)

        adapter = VeniceClassifierAdapter(mock_classifier)
        result = await adapter.classify({"model": "bge-m3"})

        assert result.resource_type == "embedding"

    @pytest.mark.asyncio
    async def test_classify_with_api_management_resource_type(self):
        """classify handles API_MANAGEMENT resource type correctly."""
        venice_meta = VeniceRequestMetadata(
            request_id="req_api",
            model_id="unknown",
            resource_type=ResourceType.API_MANAGEMENT,
            estimated_tokens=None,
            priority=0,
            timeout=15.0,
            client_id=None,
            endpoint="/api_keys",
            requires_model=False,
        )

        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(return_value=venice_meta)

        adapter = VeniceClassifierAdapter(mock_classifier)
        result = await adapter.classify({"endpoint": "/api_keys"})

        assert result.resource_type == "api_management"
        assert result.requires_model is False

    @pytest.mark.asyncio
    async def test_classify_with_billing_resource_type(self):
        """classify handles BILLING resource type correctly."""
        venice_meta = VeniceRequestMetadata(
            request_id="req_bill",
            model_id="unknown",
            resource_type=ResourceType.BILLING,
            estimated_tokens=None,
            priority=0,
            timeout=10.0,
            client_id="billing_client",
            endpoint="/billing/usage",
            requires_model=False,
        )

        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(return_value=venice_meta)

        adapter = VeniceClassifierAdapter(mock_classifier)
        result = await adapter.classify({})

        assert result.resource_type == "billing"

    @pytest.mark.asyncio
    async def test_classify_with_characters_resource_type(self):
        """classify handles CHARACTERS resource type correctly."""
        venice_meta = VeniceRequestMetadata(
            request_id="req_char",
            model_id="character-model",
            resource_type=ResourceType.CHARACTERS,
            estimated_tokens=100,
            priority=2,
            timeout=60.0,
            client_id=None,
            endpoint="/characters",
            requires_model=True,
        )

        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(return_value=venice_meta)

        adapter = VeniceClassifierAdapter(mock_classifier)
        result = await adapter.classify({"model": "character-model"})

        assert result.resource_type == "characters"

    @pytest.mark.asyncio
    async def test_classify_with_non_enum_resource_type(self):
        """classify falls back to str() when resource_type lacks .value (lines 68-70)."""
        # Create mock metadata with a non-enum resource_type (string)
        venice_meta = MagicMock()
        venice_meta.request_id = "req_custom"
        venice_meta.model_id = "custom-model"
        venice_meta.resource_type = "custom_type"  # String, not enum - no .value attribute
        venice_meta.estimated_tokens = 200
        venice_meta.priority = 3
        venice_meta.timeout = 45.0
        venice_meta.client_id = "custom_client"
        venice_meta.endpoint = "/custom/endpoint"
        venice_meta.requires_model = True

        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(return_value=venice_meta)

        adapter = VeniceClassifierAdapter(mock_classifier)
        result = await adapter.classify({"model": "custom-model"})

        # Verify str() fallback was used since resource_type is already a string
        assert result.resource_type == "custom_type"
        assert result.request_id == "req_custom"
        assert result.estimated_tokens == 200

    @pytest.mark.asyncio
    async def test_classify_with_object_having_value_attribute(self):
        """classify uses .value when resource_type has value attribute."""

        # Create a custom object with .value attribute (like an enum)
        class CustomResourceType:
            value = "custom_enum_value"

        venice_meta = MagicMock()
        venice_meta.request_id = "req_obj"
        venice_meta.model_id = "obj-model"
        venice_meta.resource_type = CustomResourceType()
        venice_meta.estimated_tokens = 100
        venice_meta.priority = 0
        venice_meta.timeout = 60.0
        venice_meta.client_id = None
        venice_meta.endpoint = "/obj/endpoint"
        venice_meta.requires_model = True

        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(return_value=venice_meta)

        adapter = VeniceClassifierAdapter(mock_classifier)
        result = await adapter.classify({})

        # Verify .value was used
        assert result.resource_type == "custom_enum_value"

    @pytest.mark.asyncio
    async def test_classify_handles_none_estimated_tokens(self):
        """classify converts None estimated_tokens to 0."""
        venice_meta = VeniceRequestMetadata(
            request_id="req_none",
            model_id="test-model",
            resource_type=ResourceType.LLM,
            estimated_tokens=None,  # Explicitly None
            priority=0,
            timeout=60.0,
        )

        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(return_value=venice_meta)

        adapter = VeniceClassifierAdapter(mock_classifier)
        result = await adapter.classify({})

        assert result.estimated_tokens == 0

    @pytest.mark.asyncio
    async def test_classify_handles_none_priority(self):
        """classify converts None priority to 0."""
        venice_meta = MagicMock()
        venice_meta.request_id = "req_no_priority"
        venice_meta.model_id = "test-model"
        venice_meta.resource_type = ResourceType.LLM
        venice_meta.estimated_tokens = 100
        venice_meta.priority = None  # Explicitly None
        venice_meta.timeout = 60.0
        venice_meta.client_id = None
        venice_meta.endpoint = "/test"
        venice_meta.requires_model = True

        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(return_value=venice_meta)

        adapter = VeniceClassifierAdapter(mock_classifier)
        result = await adapter.classify({})

        assert result.priority == 0

    @pytest.mark.asyncio
    async def test_classify_preserves_timeout(self):
        """classify preserves timeout value from Venice metadata."""
        venice_meta = VeniceRequestMetadata(
            request_id="req_timeout",
            model_id="test-model",
            resource_type=ResourceType.LLM,
            estimated_tokens=50,
            priority=0,
            timeout=120.0,  # Custom timeout
        )

        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(return_value=venice_meta)

        adapter = VeniceClassifierAdapter(mock_classifier)
        result = await adapter.classify({})

        assert result.timeout == 120.0

    @pytest.mark.asyncio
    async def test_classify_preserves_endpoint(self):
        """classify preserves endpoint value from Venice metadata."""
        venice_meta = VeniceRequestMetadata(
            request_id="req_endpoint",
            model_id="test-model",
            resource_type=ResourceType.AUDIO,
            endpoint="/audio/transcriptions",
        )

        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(return_value=venice_meta)

        adapter = VeniceClassifierAdapter(mock_classifier)
        result = await adapter.classify({})

        assert result.endpoint == "/audio/transcriptions"


class TestVeniceClassifierAdapterIntegration:
    """Integration tests for VeniceClassifierAdapter."""

    @pytest.mark.asyncio
    async def test_multiple_classify_calls(self):
        """Adapter can handle multiple classify calls sequentially."""
        mock_classifier = MagicMock()

        # Set up different return values for each call
        mock_classifier.classify = AsyncMock(
            side_effect=[
                VeniceRequestMetadata(
                    request_id="req_1",
                    model_id="model-1",
                    resource_type=ResourceType.LLM,
                ),
                VeniceRequestMetadata(
                    request_id="req_2",
                    model_id="model-2",
                    resource_type=ResourceType.IMAGE,
                ),
            ]
        )

        adapter = VeniceClassifierAdapter(mock_classifier)

        result1 = await adapter.classify({"model": "model-1"})
        result2 = await adapter.classify({"model": "model-2"})

        assert result1.request_id == "req_1"
        assert result1.resource_type == "llm"
        assert result2.request_id == "req_2"
        assert result2.resource_type == "image"

    @pytest.mark.asyncio
    async def test_classify_passes_request_unchanged(self):
        """classify passes the request dictionary to Venice classifier unchanged."""
        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(
            return_value=VeniceRequestMetadata(
                request_id="req",
                model_id="model",
                resource_type=ResourceType.LLM,
            )
        )

        adapter = VeniceClassifierAdapter(mock_classifier)

        complex_request = {
            "model": "llama-3.3-70b",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": True,
            "endpoint": "/chat/completions",
        }

        await adapter.classify(complex_request)

        # Verify the exact request was passed through
        mock_classifier.classify.assert_called_once_with(complex_request)
