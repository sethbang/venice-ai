"""
Live E2E tests for DynamicModelSelector.

These tests exercise the model selection utility against the live Venice API,
targeting code paths that lack coverage in unit/integration tests.  Each test
uses a real VeniceClient to fetch live model data, exercises selection logic
including trait-based lookups, capability-based filtering, video constraints,
cheapest-video quoting, and specialised selectors (inpaint, ASR, code,
vision, reasoning).

Prerequisites:
    VENICE_API_KEY – valid API key
"""

from __future__ import annotations

import pytest

from venice_ai.models.selection import (
    CheapestVideoResult,
    DynamicModelSelector,
    create_model_selector,
    get_chat_model,
    get_cheapest_video_model,
    get_embedding_model,
    get_multiple_models,
    get_video_model,
)

# ---------------------------------------------------------------------------
# Module-level marks
# ---------------------------------------------------------------------------
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.asyncio,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def selector(e2e_client) -> DynamicModelSelector:
    """Create a DynamicModelSelector backed by a live client."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return create_model_selector(e2e_client, cache_ttl=300.0)


@pytest.fixture
async def custom_selector(e2e_client) -> DynamicModelSelector:
    """Selector with a custom strategy that always picks the first candidate."""

    def _first(candidates):
        return candidates[0]["id"]

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return create_model_selector(e2e_client, default_selector=_first)


# ===================================================================
# 1. select_by_trait  (lines 346-359 — fully uncovered)
# ===================================================================


class TestSelectByTrait:
    """Cover the async trait-based model lookup."""

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_by_trait_default_text(self, selector):
        """Fetch the canonical 'default' text model via trait."""
        result = await selector.select_by_trait("default", resource_type="text")
        # Venice always has a default text model
        assert result is not None, "Expected a model with trait 'default' for text"
        assert isinstance(result, str)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_by_trait_fastest(self, selector):
        """Fetch the 'fastest' text model via trait."""
        result = await selector.select_by_trait("fastest", resource_type="text")
        # May or may not exist, but shouldn't error
        if result is not None:
            assert isinstance(result, str)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_by_trait_nonexistent(self, selector):
        """A made-up trait should return None."""
        result = await selector.select_by_trait("nonexistent_trait_xyz")
        assert result is None

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_by_trait_without_resource_type(self, selector):
        """Search for 'default' across all resource types."""
        result = await selector.select_by_trait("default")
        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_by_trait_with_wrong_resource_type(self, selector):
        """'default' text trait should not match video resource type."""
        text_default = await selector.select_by_trait("default", resource_type="text")
        video_default = await selector.select_by_trait("default", resource_type="video")
        # The default text model should differ from the default video model
        # (if both exist)
        if text_default and video_default:
            assert text_default != video_default


# ===================================================================
# 2. select_code_model  (lines 673-691 — fully uncovered)
# ===================================================================


class TestSelectCodeModel:
    """Cover trait-based code model selection."""

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_code_model_default(self, selector):
        """Select a code-optimised model."""
        model = await selector.select_code_model()
        assert isinstance(model, str)
        assert len(model) > 0

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_code_model_with_preferred(self, selector):
        """Preferred models override trait selection when available."""
        # First get the code model to know a valid one
        fallback = await selector.select_code_model()
        model = await selector.select_code_model(preferred_models=[fallback])
        assert model == fallback

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_code_model_with_exclude(self, selector):
        """Excluding the trait model forces capability-based fallback."""
        primary = await selector.select_code_model()
        model = await selector.select_code_model(exclude_models={primary})
        assert isinstance(model, str)
        # It may or may not differ depending on how many code models exist

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_code_model_with_custom_selector(self, selector):
        """Per-call selector overrides default strategy."""
        called = []

        def _spy(candidates):
            called.append(len(candidates))
            return candidates[0]["id"]

        code_default = await selector.select_code_model()
        model = await selector.select_code_model(exclude_models={code_default}, selector=_spy)
        assert isinstance(model, str)
        assert len(called) == 1


# ===================================================================
# 3. select_vision_model  (lines 722-739 — fully uncovered)
# ===================================================================


class TestSelectVisionModel:
    """Cover trait-based vision model selection."""

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_vision_model_default(self, selector):
        model = await selector.select_vision_model()
        assert isinstance(model, str)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_vision_model_with_preferred(self, selector):
        fallback = await selector.select_vision_model()
        model = await selector.select_vision_model(preferred_models=[fallback])
        assert model == fallback

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_vision_model_with_exclude(self, selector):
        primary = await selector.select_vision_model()
        model = await selector.select_vision_model(exclude_models={primary})
        assert isinstance(model, str)


# ===================================================================
# 4. select_reasoning_model  (lines 770-787 — fully uncovered)
# ===================================================================


class TestSelectReasoningModel:
    """Cover trait-based reasoning model selection."""

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_reasoning_model_default(self, selector):
        model = await selector.select_reasoning_model()
        assert isinstance(model, str)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_reasoning_model_with_preferred(self, selector):
        fallback = await selector.select_reasoning_model()
        model = await selector.select_reasoning_model(preferred_models=[fallback])
        assert model == fallback

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_reasoning_model_with_exclude(self, selector):
        primary = await selector.select_reasoning_model()
        model = await selector.select_reasoning_model(exclude_models={primary})
        assert isinstance(model, str)


# ===================================================================
# 5. select_video_model  (lines 932-1034 — fully uncovered)
# ===================================================================


class TestSelectVideoModel:
    """Cover the entire video model selection path."""

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_video_model_default(self, selector):
        """Basic video model selection."""
        model = await selector.select_video_model()
        assert isinstance(model, str)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_video_model_text_to_video(self, selector):
        """Filter by model_type constraint."""
        model = await selector.select_video_model(model_type="text-to-video")
        assert isinstance(model, str)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_video_model_image_to_video(self, selector):
        """Filter by image-to-video constraint; may raise if none exist."""
        try:
            model = await selector.select_video_model(model_type="image-to-video")
            assert isinstance(model, str)
        except ValueError:
            pytest.skip("No image-to-video models currently available")

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_video_model_with_audio(self, selector):
        """Filter for models supporting audio."""
        try:
            model = await selector.select_video_model(require_audio=True)
            assert isinstance(model, str)
        except ValueError:
            pytest.skip("No video models with audio support available")

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_video_model_min_resolution(self, selector):
        """Filter by minimum resolution."""
        model = await selector.select_video_model(min_resolution="720p")
        assert isinstance(model, str)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_video_model_min_duration(self, selector):
        """Filter by minimum duration."""
        model = await selector.select_video_model(min_duration="5s")
        assert isinstance(model, str)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_video_model_combined_constraints(self, selector):
        """Apply multiple constraints simultaneously."""
        try:
            model = await selector.select_video_model(
                model_type="text-to-video",
                min_resolution="720p",
                min_duration="5s",
            )
            assert isinstance(model, str)
        except ValueError:
            pytest.skip("No video models match combined constraints")

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_video_model_no_match_raises(self, selector):
        """Impossible constraints should raise ValueError."""
        with pytest.raises(ValueError, match="No video models found"):
            await selector.select_video_model(min_resolution="4k", min_duration="999s")

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_video_model_with_preferred(self, selector):
        """Preferred model list is respected."""
        fallback = await selector.select_video_model()
        model = await selector.select_video_model(preferred_models=[fallback])
        assert model == fallback

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_video_model_with_exclude(self, selector):
        """Excluding a model forces a different selection."""
        primary = await selector.select_video_model()
        try:
            model = await selector.select_video_model(exclude_models={primary})
            assert model != primary
        except ValueError:
            # Only one video model available
            pass

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_video_model_with_custom_selector(self, selector):
        """Per-call custom selector overrides defaults."""
        called = []

        def _spy(candidates):
            called.append(len(candidates))
            return candidates[0]["id"]

        model = await selector.select_video_model(selector=_spy)
        assert isinstance(model, str)
        assert len(called) == 1


# ===================================================================
# 6. select_cheapest_video_model  (lines 1101-1225 — fully uncovered)
# ===================================================================


class TestSelectCheapestVideoModel:
    """Cover the quote-based cheapest video model selection."""

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_cheapest_video_basic(self, selector):
        """Basic cheapest video selection via quoting."""
        result = await selector.select_cheapest_video_model(
            duration="5s",
        )
        assert isinstance(result, CheapestVideoResult)
        assert isinstance(result.model, str)
        assert result.quote_usd >= 0
        assert len(result.all_quotes) >= 1
        # The cheapest model should be in all_quotes
        assert result.model in result.all_quotes

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_cheapest_video_with_model_type(self, selector):
        """Filter by model_type before quoting."""
        result = await selector.select_cheapest_video_model(
            duration="5s",
            model_type="text-to-video",
        )
        assert isinstance(result, CheapestVideoResult)
        assert result.quote_usd >= 0

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_cheapest_video_with_resolution(self, selector):
        """Pass an explicit resolution for the quote."""
        result = await selector.select_cheapest_video_model(
            duration="5s",
            resolution="720p",
        )
        assert isinstance(result, CheapestVideoResult)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_cheapest_video_exclude_beta(self, selector):
        """Beta models should be excluded by default."""
        result = await selector.select_cheapest_video_model(
            duration="5s",
            exclude_beta=True,
        )
        assert isinstance(result, CheapestVideoResult)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_cheapest_video_include_beta(self, selector):
        """Including beta models may yield more candidates."""
        result = await selector.select_cheapest_video_model(
            duration="5s",
            exclude_beta=False,
        )
        assert isinstance(result, CheapestVideoResult)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_cheapest_video_min_constraints(self, selector):
        """Apply constraint filters before quoting."""
        try:
            result = await selector.select_cheapest_video_model(
                duration="5s",
                min_resolution="720p",
                min_duration="5s",
            )
            assert isinstance(result, CheapestVideoResult)
        except ValueError:
            pytest.skip("No video models match constraint filters for quoting")

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_cheapest_video_no_candidates_raises(self, selector):
        """Impossible constraints should raise ValueError before quoting."""
        with pytest.raises(ValueError):
            await selector.select_cheapest_video_model(
                duration="999s",
                min_duration="999s",
            )

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_cheapest_video_all_quotes_sorted(self, selector):
        """Verify the returned model really has the lowest quote."""
        result = await selector.select_cheapest_video_model(
            duration="5s",
        )
        if len(result.all_quotes) > 1:
            min_price = min(result.all_quotes.values())
            assert result.quote_usd == pytest.approx(min_price, abs=1e-6)


# ===================================================================
# 7. select_inpaint_model  (lines 1302-1329 — fully uncovered)
# ===================================================================


class TestSelectInpaintModel:
    """Cover inpaint model selection."""

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_inpaint_model_default(self, selector):
        try:
            model = await selector.select_inpaint_model()
            assert isinstance(model, str)
        except ValueError:
            pytest.skip("No inpaint models available in current API")

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_inpaint_model_with_preferred(self, selector):
        try:
            fallback = await selector.select_inpaint_model()
            model = await selector.select_inpaint_model(preferred_models=[fallback])
            assert model == fallback
        except ValueError:
            pytest.skip("No inpaint models available")

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_inpaint_model_with_custom_selector(self, selector):
        called = []

        def _spy(candidates):
            called.append(len(candidates))
            return candidates[0]["id"]

        try:
            model = await selector.select_inpaint_model(selector=_spy)
            assert isinstance(model, str)
            assert len(called) == 1
        except ValueError:
            pytest.skip("No inpaint models available")


# ===================================================================
# 8. select_asr_model  (lines 1351-1378 — fully uncovered)
# ===================================================================


class TestSelectASRModel:
    """Cover ASR (speech recognition) model selection."""

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_asr_model_default(self, selector):
        try:
            model = await selector.select_asr_model()
            assert isinstance(model, str)
        except ValueError:
            pytest.skip("No ASR models available in current API")

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_asr_model_with_preferred(self, selector):
        try:
            fallback = await selector.select_asr_model()
            model = await selector.select_asr_model(preferred_models=[fallback])
            assert model == fallback
        except ValueError:
            pytest.skip("No ASR models available")

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_asr_model_with_custom_selector(self, selector):
        called = []

        def _spy(candidates):
            called.append(len(candidates))
            return candidates[0]["id"]

        try:
            model = await selector.select_asr_model(selector=_spy)
            assert isinstance(model, str)
            assert len(called) == 1
        except ValueError:
            pytest.skip("No ASR models available")


# ===================================================================
# 9. select_chat_model — capability-based filtering
#    (lines 471-519, 522-535 — partially/fully uncovered)
# ===================================================================


class TestSelectChatModelCapabilities:
    """Cover capability-based filtering branches in select_chat_model."""

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_require_vision(self, selector):
        """Vision-capable models only."""
        model = await selector.select_chat_model(require_vision=True)
        assert isinstance(model, str)
        # Verify the model really has vision capability
        info = await selector.get_model_info(model)
        caps = info.get("model_spec", {}).get("capabilities", {})
        assert caps.get("supportsVision", False)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_require_reasoning(self, selector):
        """Reasoning-capable models only."""
        model = await selector.select_chat_model(require_reasoning=True)
        assert isinstance(model, str)
        info = await selector.get_model_info(model)
        caps = info.get("model_spec", {}).get("capabilities", {})
        assert caps.get("supportsReasoning", False)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_require_code_optimization(self, selector):
        """Code-optimised models only."""
        model = await selector.select_chat_model(require_code_optimization=True)
        assert isinstance(model, str)
        info = await selector.get_model_info(model)
        caps = info.get("model_spec", {}).get("capabilities", {})
        assert caps.get("optimizedForCode", False)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_require_response_schema(self, selector):
        """Models supporting response schema only."""
        try:
            model = await selector.select_chat_model(require_response_schema=True)
            assert isinstance(model, str)
            info = await selector.get_model_info(model)
            caps = info.get("model_spec", {}).get("capabilities", {})
            assert caps.get("supportsResponseSchema", False)
        except ValueError:
            pytest.skip("No models with response schema support available")

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_min_context_tokens(self, selector):
        """Models meeting a minimum context window."""
        model = await selector.select_chat_model(min_context_tokens=4096)
        assert isinstance(model, str)
        info = await selector.get_model_info(model)
        assert info.get("availableContextTokens", 0) >= 4096

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_min_context_tokens_large(self, selector):
        """Filter for large-context models."""
        try:
            model = await selector.select_chat_model(min_context_tokens=128000)
            assert isinstance(model, str)
            info = await selector.get_model_info(model)
            assert info.get("availableContextTokens", 0) >= 128000
        except ValueError:
            pytest.skip("No models with >= 128K context available")

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_exclude_beta(self, selector):
        """Beta models should be excluded."""
        model = await selector.select_chat_model(exclude_beta=True)
        assert isinstance(model, str)
        info = await selector.get_model_info(model)
        assert not info.get("beta", False)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_require_private(self, selector):
        """Only private-data models."""
        try:
            model = await selector.select_chat_model(require_private=True)
            assert isinstance(model, str)
            info = await selector.get_model_info(model)
            assert info.get("privacy") == "private"
        except ValueError:
            pytest.skip("No private models available")

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_combined_capability_filters(self, selector):
        """Stack multiple capability filters."""
        try:
            model = await selector.select_chat_model(
                require_vision=True,
                exclude_beta=True,
                min_context_tokens=4096,
            )
            assert isinstance(model, str)
        except ValueError:
            pytest.skip("No models match combined capability filters")

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_impossible_capability_raises(self, selector):
        """Impossible filter combination raises ValueError."""
        with pytest.raises(ValueError, match="No chat models found"):
            await selector.select_chat_model(
                require_vision=True,
                require_reasoning=True,
                require_code_optimization=True,
                require_response_schema=True,
                min_context_tokens=999_999_999,
                require_private=True,
                exclude_beta=True,
            )

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_prefer_recommended(self, selector):
        """Venice-recommended models should be preferred when flag is set."""
        model = await selector.select_chat_model(prefer_recommended=True)
        assert isinstance(model, str)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_prefer_recommended_with_exclude_beta(self, selector):
        """Combine prefer_recommended with capability filter."""
        model = await selector.select_chat_model(prefer_recommended=True, exclude_beta=True)
        assert isinstance(model, str)


# ===================================================================
# 10. Custom selector integration  (lines 540-546)
# ===================================================================


class TestCustomSelectorIntegration:
    """Verify custom selectors receive full model objects."""

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_default_selector_receives_model_dicts(self, custom_selector):
        """The default_selector strategy gets full model dicts with pricing."""
        model = await custom_selector.select_chat_model()
        assert isinstance(model, str)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_per_call_selector_overrides_default(self, custom_selector):
        """A per-call selector takes priority over the default one."""
        per_call_ids = []

        def _collect(candidates):
            per_call_ids.extend([c["id"] for c in candidates])
            return candidates[-1]["id"]

        model = await custom_selector.select_chat_model(selector=_collect)
        assert isinstance(model, str)
        assert model in per_call_ids

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_custom_selector_on_function_calling(self, custom_selector):
        """Custom selector on function calling model selection."""
        model = await custom_selector.select_function_calling_model()
        assert isinstance(model, str)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_custom_selector_on_embedding(self, custom_selector):
        """Custom selector on embedding model selection."""
        model = await custom_selector.select_embedding_model()
        assert isinstance(model, str)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_custom_selector_on_image(self, custom_selector):
        """Custom selector on image model selection."""
        model = await custom_selector.select_image_model()
        assert isinstance(model, str)


# ===================================================================
# 11. Helper functions  (lines 1500-1540 — partially uncovered)
# ===================================================================


class TestHelperFunctions:
    """Cover the module-level helper functions."""

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_get_chat_model(self, e2e_client):
        model = await get_chat_model(e2e_client)
        assert isinstance(model, str)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_get_embedding_model(self, e2e_client):
        model = await get_embedding_model(e2e_client)
        assert isinstance(model, str)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_get_multiple_models(self, e2e_client):
        models = await get_multiple_models(e2e_client, count=2)
        assert isinstance(models, list)
        assert len(models) == 2
        # All entries should be model ID strings
        for m in models:
            assert isinstance(m, str)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_get_video_model(self, e2e_client):
        """Cover the get_video_model helper (lines 1506-1507)."""
        model = await get_video_model(e2e_client)
        assert isinstance(model, str)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_get_video_model_with_type(self, e2e_client):
        model = await get_video_model(e2e_client, model_type="text-to-video")
        assert isinstance(model, str)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_get_cheapest_video_model(self, e2e_client):
        """Cover the get_cheapest_video_model helper (lines 1539-1540)."""
        result = await get_cheapest_video_model(e2e_client, duration="5s")
        assert isinstance(result, CheapestVideoResult)
        assert isinstance(result.model, str)
        assert result.quote_usd >= 0

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_get_cheapest_video_model_with_type(self, e2e_client):
        result = await get_cheapest_video_model(
            e2e_client, model_type="text-to-video", duration="5s"
        )
        assert isinstance(result, CheapestVideoResult)


# ===================================================================
# 12. get_available_models & cache behaviour
# ===================================================================


class TestGetAvailableModels:
    """Cover get_available_models including offline-model filtering."""

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_get_available_text_models(self, selector):
        models = await selector.get_available_models(resource_type="text")
        assert isinstance(models, list)
        assert len(models) > 0
        for m in models:
            assert isinstance(m, str)

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_get_available_image_models(self, selector):
        models = await selector.get_available_models(resource_type="image")
        assert isinstance(models, list)
        assert len(models) > 0

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_get_available_video_models(self, selector):
        models = await selector.get_available_models(resource_type="video")
        assert isinstance(models, list)
        assert len(models) > 0

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_get_available_all_types(self, selector):
        """No resource_type filter returns all models."""
        models = await selector.get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_force_refresh(self, selector):
        """Force-refresh should re-fetch from API."""
        models1 = await selector.get_available_models(resource_type="text")
        models2 = await selector.get_available_models(resource_type="text", force_refresh=True)
        # Both should be non-empty; order may differ
        assert len(models1) > 0
        assert len(models2) > 0

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_cache_info(self, selector):
        """Verify cache state after fetching models."""
        await selector.get_available_models()
        info = selector.get_cache_info()
        assert info["model_count"] > 0
        assert info["is_expired"] is False
        assert info["ttl_seconds"] == 300.0


# ===================================================================
# 13. get_model_info
# ===================================================================


class TestGetModelInfo:
    """Cover get_model_info for known and unknown models."""

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_get_model_info_known(self, selector):
        """Info for a model that exists."""
        models = await selector.get_available_models(resource_type="text")
        info = await selector.get_model_info(models[0])
        assert info is not None
        assert "id" in info
        assert info["id"] == models[0]

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_get_model_info_unknown(self, selector):
        """Info for a model that doesn't exist returns None."""
        info = await selector.get_model_info("nonexistent-model-xyz-12345")
        assert info is None

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_model_info_contains_spec(self, selector):
        """Model info should contain model_spec with capabilities."""
        models = await selector.get_available_models(resource_type="text")
        info = await selector.get_model_info(models[0])
        assert info is not None
        assert "model_spec" in info
        spec = info["model_spec"]
        assert "capabilities" in spec


# ===================================================================
# 14. _get_trait_model sync helper  (line 369 uncovered)
# ===================================================================


class TestGetTraitModelSync:
    """Cover the synchronous _get_trait_model helper."""

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_get_trait_model_empty_cache(self, e2e_client):
        """Before any fetch, cache is empty → returns None."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            fresh_selector = create_model_selector(e2e_client)
        result = fresh_selector._get_trait_model("default")
        assert result is None

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_get_trait_model_after_fetch(self, selector):
        """After fetching, trait lookup should succeed."""
        await selector.get_available_models()
        result = selector._get_trait_model("default", resource_type="text")
        # May or may not find a default trait, but should not error
        if result is not None:
            assert isinstance(result, str)


# ===================================================================
# 15. select_models_for_concurrency_test  (already covered but
#     exercises diversity traits — enriches branch coverage)
# ===================================================================


class TestSelectModelsForConcurrency:
    """Verify concurrency model selection with trait diversity."""

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_two_text_models(self, selector):
        models = await selector.select_models_for_concurrency_test(count=2)
        assert len(models) == 2
        assert models[0] != models[1]

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_three_text_models(self, selector):
        """Three models exercises all diversity trait slots."""
        try:
            models = await selector.select_models_for_concurrency_test(count=3)
            assert len(models) == 3
            assert len(set(models)) == 3  # All unique
        except ValueError:
            pytest.skip("Fewer than 3 text models available")

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_select_image_models_for_concurrency(self, selector):
        """Concurrency selection for image models."""
        try:
            models = await selector.select_models_for_concurrency_test(
                count=2, resource_type="image"
            )
            assert len(models) == 2
        except ValueError:
            pytest.skip("Fewer than 2 image models available")

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_concurrency_with_exclude(self, selector):
        """Excluding a model still returns the requested count."""
        models_all = await selector.get_available_models(resource_type="text")
        if len(models_all) < 3:
            pytest.skip("Not enough text models to test exclude + concurrency")
        models = await selector.select_models_for_concurrency_test(
            count=2, exclude_models={models_all[0]}
        )
        assert len(models) == 2
        assert models_all[0] not in models

    @pytest.mark.flaky(reruns=2, reruns_delay=3)
    async def test_concurrency_raises_on_insufficient(self, selector):
        """Requesting more models than available raises ValueError."""
        with pytest.raises(ValueError, match="Need .* models"):
            await selector.select_models_for_concurrency_test(count=999)
