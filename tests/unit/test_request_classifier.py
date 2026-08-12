"""Core functional tests for RequestClassifier."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai._queue_types import ResourceType
from venice_ai._request_classifier import RequestClassifier
from venice_ai.core.rate_limit_discovery import RateLimitDiscovery


@pytest.fixture
def rate_limit_discovery():
    return MagicMock()


@pytest.fixture
def classifier(rate_limit_discovery):
    return RequestClassifier(rate_limit_discovery)


class TestClassifyBasic:
    """Basic classification tests for common request types."""

    @pytest.mark.asyncio
    async def test_classify_chat_completion(self, classifier):
        request = {
            "endpoint": "chat/completions",
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        metadata = await classifier.classify(request)
        assert metadata.resource_type == ResourceType.LLM
        assert metadata.model_id == "gpt-4"
        assert metadata.requires_model is True
        assert metadata.estimated_tokens > 0

    @pytest.mark.asyncio
    async def test_classify_image_generation(self, classifier):
        request = {"endpoint": "image/generate", "model": "stable-diffusion"}
        metadata = await classifier.classify(request)
        assert metadata.resource_type == ResourceType.IMAGE
        assert metadata.model_id == "stable-diffusion"
        assert metadata.requires_model is True

    @pytest.mark.asyncio
    async def test_classify_audio_speech(self, classifier):
        request = {"endpoint": "audio/speech", "model": "tts-kokoro"}
        metadata = await classifier.classify(request)
        assert metadata.resource_type == ResourceType.AUDIO
        assert metadata.model_id == "tts-kokoro"

    @pytest.mark.asyncio
    async def test_classify_embedding(self, classifier):
        request = {"endpoint": "embeddings", "model": "embedding-model"}
        metadata = await classifier.classify(request)
        assert metadata.resource_type == ResourceType.EMBEDDING
        assert metadata.model_id == "embedding-model"

    @pytest.mark.asyncio
    async def test_classify_api_key_management(self, classifier):
        request = {"endpoint": "api_keys"}
        metadata = await classifier.classify(request)
        assert metadata.resource_type == ResourceType.API_MANAGEMENT
        assert metadata.requires_model is False

    @pytest.mark.asyncio
    async def test_token_estimation(self, classifier):
        request = {
            "endpoint": "chat/completions",
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "This is a test sentence."}],
            "max_completion_tokens": 100,
        }
        metadata = await classifier.classify(request)
        # Rough estimation: len("This is a test sentence.") // 4 + 100
        expected_min_tokens = 24 // 4 + 100
        assert metadata.estimated_tokens >= expected_min_tokens

    @pytest.mark.asyncio
    async def test_token_estimation_with_list_content(self):
        """Test token estimation with list content in messages."""
        rate_limit_discovery = AsyncMock()
        classifier = RequestClassifier(rate_limit_discovery)

        request = {
            "endpoint": "chat/completions",
            "model": "llama-3.1-70b",
            "messages": [
                {
                    "content": [
                        {"type": "text", "text": "Hello world"},
                        {"type": "image", "url": "http://example.com/image.jpg"},
                    ]
                }
            ],
            "max_completion_tokens": 100,
        }

        metadata = await classifier.classify(request)
        assert metadata.estimated_tokens is not None
        assert metadata.estimated_tokens > 100  # Should include max_tokens


class TestModelFallbackClassification:
    """Model-name pattern fallback (Tier 2) — endpoint is empty so the
    classifier must choose a ResourceType from the model ID alone.

    Regression guard: ``seedream*`` must route to IMAGE, and ``gpt-*`` must
    route to LLM.  Plain Qwen LLM IDs must NOT route to IMAGE.
    """

    @pytest.mark.asyncio
    async def test_seedream_routes_to_image(self, classifier):
        metadata = await classifier.classify({"model": "seedream-v5-lite", "endpoint": ""})
        assert metadata.resource_type == ResourceType.IMAGE

    @pytest.mark.asyncio
    async def test_gpt_codex_routes_to_llm(self, classifier):
        metadata = await classifier.classify({"model": "gpt-5.3-codex", "endpoint": ""})
        assert metadata.resource_type == ResourceType.LLM

    @pytest.mark.asyncio
    async def test_qwen_llm_still_routes_to_llm(self, classifier):
        """Regression: plain Qwen LLM IDs must route to LLM, not IMAGE."""
        metadata = await classifier.classify({"model": "qwen3-235b", "endpoint": ""})
        assert metadata.resource_type == ResourceType.LLM


class TestImageEndpointPaths:
    """Endpoint-tier IMAGE routing must match the *singular* paths the
    SDK actually sends (``image/...``), not the plural ``images/...`` forms.

    The real paths emitted by ``resources/image.py`` are ``image/generate``,
    ``image/edit``, ``image/upscale``, ``image/multi-edit`` and
    ``image/background-remove``.

    Each test uses a model that matches *no* image model-name pattern (so a
    pass cannot leak in through the Tier-2 model fallback); classification must
    be driven purely by the endpoint pattern.
    """

    @pytest.mark.parametrize(
        "endpoint",
        [
            "image/generate",
            "image/edit",
            "image/upscale",
            "image/multi-edit",
            "image/background-remove",
        ],
    )
    @pytest.mark.asyncio
    async def test_singular_image_endpoints_route_to_image(self, classifier, endpoint):
        # "unknown" matches no model-name pattern, so IMAGE here can only come
        # from the endpoint tier — proving the endpoint pattern is correct.
        metadata = await classifier.classify({"endpoint": endpoint, "model": "unknown"})
        assert metadata.resource_type == ResourceType.IMAGE, (
            f"{endpoint} should route to IMAGE via endpoint pattern, got {metadata.resource_type}"
        )
        # Image endpoints must not run LLM token estimation.
        assert metadata.estimated_tokens is None

    @pytest.mark.parametrize(
        "endpoint",
        [
            "image/generate",
            "image/upscale",
            "image/multi-edit",
            "image/background-remove",
        ],
    )
    def test_singular_image_endpoints_determine_resource_type(self, classifier, endpoint):
        # Direct _determine_resource_type path with a non-image model.
        assert classifier._determine_resource_type(endpoint, "any-model") == ResourceType.IMAGE


class TestImageModelNameRouting:
    """The Feb-2026 image models must route to IMAGE by model name.

    ``qwen-image`` was being swallowed by the generic ``qwen``->LLM rule and
    ``gpt-image-2`` by the ``\\bgpt-``->LLM rule. They must reach IMAGE (the
    IMAGE list is iterated before LLM, so a dedicated IMAGE pattern wins).

    Guard: the generic LLM IDs (``qwen3-235b``, ``gpt-5.3-codex``) must still
    route to LLM — the new IMAGE patterns must be specific to ``*-image*``.
    """

    @pytest.mark.parametrize(
        "model",
        [
            "qwen-image",
            "Qwen-Image",  # case-insensitive
            "gpt-image-2",
            "GPT-Image-2",  # case-insensitive
            "seedream-v5-lite",
            "seedream",
        ],
    )
    @pytest.mark.asyncio
    async def test_image_models_route_to_image(self, classifier, model):
        # Empty endpoint forces pure Tier-2 model-name classification.
        metadata = await classifier.classify({"model": model, "endpoint": ""})
        assert metadata.resource_type == ResourceType.IMAGE, (
            f"{model} should route to IMAGE by model name, got {metadata.resource_type}"
        )

    @pytest.mark.parametrize(
        "model",
        [
            "qwen-image",
            "gpt-image-2",
            "seedream-v5-lite",
        ],
    )
    def test_image_models_determine_resource_type(self, classifier, model):
        assert classifier._determine_resource_type("", model) == ResourceType.IMAGE

    @pytest.mark.parametrize(
        "model",
        [
            "qwen3-235b",  # generic Qwen LLM, no "-image"
            "gpt-5.3-codex",  # generic GPT LLM, no "-image"
        ],
    )
    def test_generic_llm_models_still_route_to_llm(self, classifier, model):
        """Guard: adding qwen-image/gpt-image IMAGE patterns must not pull
        plain Qwen/GPT LLM IDs into the IMAGE queue."""
        assert classifier._determine_resource_type("", model) == ResourceType.LLM


class TestRequestClassifierInitialization:
    """Test RequestClassifier initialization with various configurations."""

    def test_init_with_tier_discovery(self):
        """Test initialization with TierDiscovery."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        assert classifier.rate_limit_discovery == mock_tier_discovery
        assert classifier.model_less_endpoints is not None
        assert classifier.resource_patterns is not None
        assert classifier.model_type_patterns is not None


class TestClassifyMethod:
    """Test async classify method with various request types."""

    @pytest.mark.asyncio
    async def test_classify_llm_chat_request(self):
        """Test classification of LLM chat request."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        request_data = {
            "model": "gpt-4",
            "endpoint": "chat/completions",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_completion_tokens": 100,
        }

        metadata = await classifier.classify(request_data)

        assert metadata.resource_type == ResourceType.LLM
        assert metadata.model_id == "gpt-4"
        assert metadata.endpoint == "chat/completions"
        assert metadata.requires_model is True
        assert metadata.estimated_tokens is not None and metadata.estimated_tokens > 0

    @pytest.mark.asyncio
    async def test_classify_embedding_request(self):
        """Test classification of embedding request."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        request_data = {
            "model": "text-embedding-ada-002",
            "endpoint": "embeddings",
            "input": "Embed this text",
        }

        metadata = await classifier.classify(request_data)

        assert metadata.resource_type == ResourceType.EMBEDDING
        assert metadata.model_id == "text-embedding-ada-002"
        assert metadata.endpoint == "embeddings"
        assert metadata.requires_model is True
        assert metadata.estimated_tokens is None  # Embeddings don't estimate tokens

    @pytest.mark.asyncio
    async def test_classify_image_generation_request(self):
        """Test classification of image generation request."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        request_data = {
            "model": "flux-pro",
            "endpoint": "image/generate",
            "prompt": "Generate an image",
        }

        metadata = await classifier.classify(request_data)

        assert metadata.resource_type == ResourceType.IMAGE
        assert metadata.model_id == "flux-pro"
        assert metadata.endpoint == "image/generate"
        assert metadata.requires_model is True
        assert metadata.estimated_tokens is None  # Images don't estimate tokens

    @pytest.mark.asyncio
    async def test_classify_audio_transcription_request(self):
        """Test classification of audio transcription request."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        request_data = {
            "model": "whisper-1",
            "endpoint": "audio/transcriptions",
            "file": "audio.mp3",
        }

        metadata = await classifier.classify(request_data)

        assert metadata.resource_type == ResourceType.AUDIO
        assert metadata.model_id == "whisper-1"
        assert metadata.endpoint == "audio/transcriptions"
        assert metadata.requires_model is True
        assert metadata.estimated_tokens is None  # Audio doesn't estimate tokens

    @pytest.mark.asyncio
    async def test_classify_api_management_request(self):
        """Test classification of API management request."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        request_data = {"endpoint": "api_keys/list"}

        metadata = await classifier.classify(request_data)

        assert metadata.resource_type == ResourceType.API_MANAGEMENT
        assert metadata.model_id == "unknown"
        assert metadata.endpoint == "api_keys/list"
        assert metadata.requires_model is False  # API management doesn't need model

    @pytest.mark.asyncio
    async def test_classify_with_custom_parameters(self):
        """Test classification with custom priority, timeout, and client_id."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        request_data = {
            "model": "llama-3.1-8b",
            "endpoint": "chat/completions",
            "priority": 5,
            "timeout": 120.0,
            "client_id": "test-client",
            "messages": [{"role": "user", "content": "Test"}],
        }

        metadata = await classifier.classify(request_data)

        assert metadata.priority == 5
        assert metadata.timeout == 120.0
        assert metadata.client_id == "test-client"

    @pytest.mark.asyncio
    async def test_classify_with_explicit_requires_model(self):
        """Test classification with explicit requires_model flag."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        request_data = {"endpoint": "custom/endpoint", "requires_model": False}

        metadata = await classifier.classify(request_data)

        assert metadata.requires_model is False

    @pytest.mark.asyncio
    async def test_classify_unknown_endpoint_with_model_fallback(self):
        """Test classification of unknown endpoint using model name fallback."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        request_data = {"model": "llama-3.1-70b", "endpoint": "unknown/endpoint"}

        metadata = await classifier.classify(request_data)

        # Should fallback to model pattern matching
        assert metadata.resource_type == ResourceType.LLM
        assert metadata.model_id == "llama-3.1-70b"


class TestDetermineResourceType:
    """Test _determine_resource_type method."""

    def test_endpoint_pattern_matching_llm(self):
        """Test endpoint pattern matching for LLM."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        resource_type = classifier._determine_resource_type("chat/completions", "any-model")
        assert resource_type == ResourceType.LLM

        resource_type = classifier._determine_resource_type("completions", "any-model")
        assert resource_type == ResourceType.LLM

    def test_endpoint_pattern_matching_image(self):
        """Test endpoint pattern matching for IMAGE."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        resource_type = classifier._determine_resource_type("image/generate", "any-model")
        assert resource_type == ResourceType.IMAGE

        resource_type = classifier._determine_resource_type("image/edit", "any-model")
        assert resource_type == ResourceType.IMAGE

    def test_endpoint_pattern_matching_audio(self):
        """Test endpoint pattern matching for AUDIO."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        resource_type = classifier._determine_resource_type("audio/transcriptions", "any-model")
        assert resource_type == ResourceType.AUDIO

        resource_type = classifier._determine_resource_type("audio/speech", "any-model")
        assert resource_type == ResourceType.AUDIO

    def test_endpoint_pattern_matching_embedding(self):
        """Test endpoint pattern matching for EMBEDDING."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        resource_type = classifier._determine_resource_type("embeddings", "any-model")
        assert resource_type == ResourceType.EMBEDDING

    def test_model_pattern_fallback_llm(self):
        """Test model pattern fallback for LLM models."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        # Unknown endpoint, should fallback to model pattern
        resource_type = classifier._determine_resource_type("unknown", "llama-3.1-8b")
        assert resource_type == ResourceType.LLM

        resource_type = classifier._determine_resource_type("unknown", "mistral-7b")
        assert resource_type == ResourceType.LLM

        resource_type = classifier._determine_resource_type("unknown", "qwen-2-72b")
        assert resource_type == ResourceType.LLM

    def test_model_pattern_fallback_image(self):
        """Test model pattern fallback for IMAGE models."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        resource_type = classifier._determine_resource_type("unknown", "flux-pro")
        assert resource_type == ResourceType.IMAGE

        resource_type = classifier._determine_resource_type("unknown", "stable-diffusion-xl")
        assert resource_type == ResourceType.IMAGE

    def test_model_pattern_fallback_audio(self):
        """Test model pattern fallback for AUDIO models."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        resource_type = classifier._determine_resource_type("unknown", "whisper-large")
        assert resource_type == ResourceType.AUDIO

        resource_type = classifier._determine_resource_type("unknown", "tts-kokoro-v1")
        assert resource_type == ResourceType.AUDIO

    def test_model_pattern_fallback_embedding(self):
        """Test model pattern fallback for EMBEDDING models."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        resource_type = classifier._determine_resource_type("unknown", "text-embedding-ada-002")
        assert resource_type == ResourceType.EMBEDDING

        resource_type = classifier._determine_resource_type("unknown", "bge-m3-large")
        assert resource_type == ResourceType.EMBEDDING

    def test_model_less_endpoint_handling(self):
        """Test handling of model-less endpoints."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        # api_keys endpoints should be API_MANAGEMENT
        resource_type = classifier._determine_resource_type("api_keys/list", "unknown")
        assert resource_type == ResourceType.API_MANAGEMENT

    def test_default_to_llm_for_unknown(self):
        """Test default to LLM for completely unknown patterns."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        resource_type = classifier._determine_resource_type("completely/unknown", "unknown-model")
        assert resource_type == ResourceType.LLM

    def test_case_insensitive_model_matching(self):
        """Test case insensitive model pattern matching."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        resource_type = classifier._determine_resource_type("unknown", "LLAMA-3.1-8B")
        assert resource_type == ResourceType.LLM

        resource_type = classifier._determine_resource_type("unknown", "Flux-Pro")
        assert resource_type == ResourceType.IMAGE

        resource_type = classifier._determine_resource_type("unknown", "WHISPER-LARGE")
        assert resource_type == ResourceType.AUDIO


class TestEstimateTokens:
    """Test _estimate_tokens method."""

    def test_estimate_tokens_with_messages(self):
        """Test token estimation with chat messages."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        request_data = {
            "messages": [
                {"role": "user", "content": "Hello world"},  # ~10 chars = ~2 tokens
                {"role": "assistant", "content": "Hi there"},  # ~8 chars = ~2 tokens
            ],
            "max_completion_tokens": 100,
        }

        tokens = classifier._estimate_tokens(request_data)
        # (10 + 8) / 4 = 4 tokens + 100 max_tokens = 104
        assert tokens == 104

    def test_estimate_tokens_with_multimodal_content(self):
        """Test token estimation with multimodal message content."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        request_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this",
                        },  # ~13 chars = ~3 tokens
                        {"type": "image", "image_url": "http://example.com/image.jpg"},
                    ],
                }
            ],
            "max_completion_tokens": 200,
        }

        tokens = classifier._estimate_tokens(request_data)
        # 13 / 4 = 3 tokens + 200 max_tokens = 203
        assert tokens == 203

    def test_estimate_tokens_with_prompt(self):
        """Test token estimation with direct prompt."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        request_data = {
            "prompt": "This is a test prompt",  # 21 chars = ~5 tokens
            "max_completion_tokens": 50,
        }

        tokens = classifier._estimate_tokens(request_data)
        # 21 / 4 = 5 tokens + 50 max_tokens = 55
        assert tokens == 55

    def test_estimate_tokens_with_prompt_list(self):
        """Test token estimation with prompt list."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        request_data = {
            "prompt": [
                "First prompt",
                "Second prompt",
            ],  # 12 + 13 = 25 chars = ~6 tokens
            "max_completion_tokens": 75,
        }

        tokens = classifier._estimate_tokens(request_data)
        # 25 / 4 = 6 tokens + 75 max_tokens = 81
        assert tokens == 81

    def test_estimate_tokens_no_max_tokens(self):
        """Test token estimation without max_tokens specified."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        request_data = {
            "messages": [{"role": "user", "content": "Test"}]  # 4 chars = 1 token
        }

        tokens = classifier._estimate_tokens(request_data)
        # 1 token + 150 default = 151
        assert tokens == 151

    def test_estimate_tokens_invalid_max_tokens(self):
        """Test token estimation with invalid max_tokens."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        request_data = {
            "messages": [{"role": "user", "content": "Test"}],  # 4 chars = 1 token
            "max_completion_tokens": "invalid",
        }

        tokens = classifier._estimate_tokens(request_data)
        # 1 token + 150 default (invalid max_tokens) = 151
        assert tokens == 151

    def test_estimate_tokens_empty_content(self):
        """Test token estimation with empty content."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        request_data = {"messages": [], "max_completion_tokens": 100}

        tokens = classifier._estimate_tokens(request_data)
        # No content, just max_tokens
        assert tokens == 101  # min 1 token + 100 max_tokens

    def test_estimate_tokens_minimum_one(self):
        """Test that token estimation always returns at least 1."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        request_data = {}  # No content at all

        tokens = classifier._estimate_tokens(request_data)
        # Should return 1 (minimum) + 150 (default max_tokens) = 151
        assert tokens == 151


class TestGetResourceTypeForModel:
    """Test get_resource_type_for_model method."""

    def test_get_resource_type_for_llm_model(self):
        """Test getting resource type for LLM model."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        resource_type = classifier.get_resource_type_for_model("llama-3.1-8b")
        assert resource_type == ResourceType.LLM

        resource_type = classifier.get_resource_type_for_model("mistral-7b")
        assert resource_type == ResourceType.LLM

    def test_get_resource_type_for_image_model(self):
        """Test getting resource type for IMAGE model."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        resource_type = classifier.get_resource_type_for_model("flux-pro")
        assert resource_type == ResourceType.IMAGE

        resource_type = classifier.get_resource_type_for_model("stable-diffusion-xl")
        assert resource_type == ResourceType.IMAGE

    def test_get_resource_type_for_unknown_model(self):
        """Test getting resource type for unknown model."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        resource_type = classifier.get_resource_type_for_model("unknown-model-xyz")
        assert resource_type == ResourceType.LLM  # Defaults to LLM


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_classify_with_missing_endpoint(self):
        """Test classification with missing endpoint."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        request_data = {"model": "gpt-4"}

        metadata = await classifier.classify(request_data)

        assert metadata.endpoint == ""
        assert metadata.resource_type == ResourceType.LLM  # Default

    @pytest.mark.asyncio
    async def test_classify_with_missing_model(self):
        """Test classification with missing model."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        request_data = {"endpoint": "chat/completions"}

        metadata = await classifier.classify(request_data)

        assert metadata.model_id == "unknown"
        assert metadata.resource_type == ResourceType.LLM

    @pytest.mark.asyncio
    async def test_request_id_generation(self):
        """Test that request IDs are unique UUIDs."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        request_data = {"endpoint": "chat/completions"}

        metadata1 = await classifier.classify(request_data)
        metadata2 = await classifier.classify(request_data)

        # Request IDs should be unique
        assert metadata1.request_id != metadata2.request_id

        # Should be valid UUIDs
        uuid.UUID(metadata1.request_id)
        uuid.UUID(metadata2.request_id)

    @pytest.mark.asyncio
    async def test_billing_endpoint_classification(self):
        """Test classification of billing endpoints."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        request_data = {"endpoint": "billing/usage"}

        metadata = await classifier.classify(request_data)

        assert metadata.resource_type == ResourceType.BILLING

    @pytest.mark.asyncio
    async def test_characters_endpoint_classification(self):
        """Test classification of characters endpoints."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        request_data = {"endpoint": "characters"}

        metadata = await classifier.classify(request_data)

        assert metadata.resource_type == ResourceType.CHARACTERS


class TestEstimateTokensEdgeCases:
    """Additional edge cases for token estimation from coverage improvements."""

    def test_estimate_tokens_with_negative_max_completion_tokens(self):
        """Test token estimation with negative max_completion_tokens (still numeric)."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        result = classifier._estimate_tokens({"prompt": "test", "max_completion_tokens": -100})
        # "test" = 4 chars = 1 token, plus -100 = -99
        assert result == -99

    def test_estimate_tokens_with_unrecognized_max_tokens_key(self):
        """Test token estimation with old max_tokens key (uses 150 default)."""
        mock_tier_discovery = MagicMock(spec=RateLimitDiscovery)
        classifier = RequestClassifier(rate_limit_discovery=mock_tier_discovery)

        result = classifier._estimate_tokens({"prompt": "test", "max_tokens": "invalid"})
        assert result == 151  # 1 (from "test") + 150 (default)
