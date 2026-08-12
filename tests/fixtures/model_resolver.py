"""
Dynamic model resolution for test fixtures.

Resolves semantic model aliases (e.g., DEFAULT_TEXT_MODEL, VISION_MODEL) to
current Venice model IDs using the DynamicModelSelector. Results are cached
to a JSON file so that normal test runs don't require API calls.

Usage:
    # Refresh the cache (requires VENICE_API_KEY):
    poetry run pytest tests/ --refresh-models

    # Normal runs load from cache, falling back to hardcoded defaults:
    poetry run pytest tests/
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_FILE = Path(__file__).parent.parent / ".model_cache.json"

# Caches older than this are treated as stale and ignored. A Venice model that
# was available at refresh time can be deprecated or go offline within a week
# (TEE / beta models especially), so a stale alias can pin tests to a broken
# model and burn a whole test run on retries before failing.
CACHE_MAX_AGE = timedelta(hours=48)

# ---------------------------------------------------------------------------
# Alias resolution strategies
#
# Each entry maps a ModelConfig field name to a resolution strategy tuple:
#   ("trait", "<trait_name>", "<resource_type>")  — select_by_trait()
#   ("capability", "<cap_field>", "<resource_type>") — first model with capability
#   ("type", "<resource_type>")                     — first available of type
# ---------------------------------------------------------------------------
ALIAS_STRATEGIES: dict[str, tuple[str, ...]] = {
    # --- Text: trait-based ---
    "DEFAULT_TEXT_MODEL": ("trait", "default", "text"),
    "FASTEST_TEXT_MODEL": ("trait", "fastest", "text"),
    "MOST_INTELLIGENT_MODEL": ("trait", "most_intelligent", "text"),
    "MOST_UNCENSORED_MODEL": ("trait", "most_uncensored", "text"),
    "DEFAULT_REASONING_MODEL": ("trait", "default_reasoning", "text"),
    "DEFAULT_CODE_MODEL": ("trait", "default_code", "text"),
    "DEFAULT_VISION_MODEL": ("trait", "default_vision", "text"),
    "FUNCTION_CALLING_MODEL": ("trait", "function_calling_default", "text"),
    # --- Text: capability-based ---
    "REASONING_MODEL": ("capability", "supportsReasoning", "text"),
    "CODE_MODEL": ("capability", "optimizedForCode", "text"),
    "VISION_MODEL": ("capability", "supportsVision", "text"),
    "WEB_SEARCH_MODEL": ("capability", "supportsWebSearch", "text"),
    "RESPONSE_SCHEMA_MODEL": ("capability", "supportsResponseSchema", "text"),
    "LOG_PROBS_MODEL": ("capability", "supportsLogProbs", "text"),
    # --- Image: trait-based ---
    "DEFAULT_IMAGE_MODEL": ("trait", "default", "image"),
    "HIGHEST_QUALITY_IMAGE_MODEL": ("trait", "highest_quality", "image"),
    "UNCENSORED_IMAGE_MODEL": ("trait", "most_uncensored", "image"),
    # --- Simple type-based ---
    "EMBEDDING_MODEL": ("type", "embedding"),
    "TTS_MODEL": ("type", "tts"),
    "UPSCALER_MODEL": ("type", "upscale"),
    "INPAINT_MODEL": ("type", "inpaint"),
}

# Aliases that are just copies of other aliases (resolved after primaries)
DERIVED_ALIASES: dict[str, str] = {
    "DEFAULT_TEXT_MODEL_ID": "DEFAULT_TEXT_MODEL",
    "SMALL_TEXT_MODEL_ID": "SMALL_TEXT_MODEL",
    "FASTEST_TEXT_MODEL_ID": "FASTEST_TEXT_MODEL",
    "MEDIUM_TEXT_MODEL_ID": "MEDIUM_TEXT_MODEL",
    "LARGE_TEXT_MODEL_ID": "LARGE_TEXT_MODEL",
    "UNCENSORED_MODEL_ID": "UNCENSORED_MODEL",
    "REASONING_MODEL_ID": "REASONING_MODEL",
    "CODE_MODEL_ID": "CODE_MODEL",
    "VISION_MODEL_ID": "VISION_MODEL",
    "DEFAULT_IMAGE_MODEL_ID": "DEFAULT_IMAGE_MODEL",
    "EMBEDDING_MODEL_ID": "EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_MODEL": "EMBEDDING_MODEL",
    "TTS_MODEL_ID": "TTS_MODEL",
    "DEFAULT_TTS_MODEL": "TTS_MODEL",
    "UPSCALER_MODEL_ID": "UPSCALER_MODEL",
    "INPAINT_MODEL_ID": "INPAINT_MODEL",
}


# ---------------------------------------------------------------------------
# Resolution helpers (operate on the fetched model dict from _fetch_models)
# ---------------------------------------------------------------------------


def _resolve_by_trait(
    models: dict[str, Any],
    trait: str,
    resource_type: str,
) -> str | None:
    """Find the model assigned to a Venice trait."""
    for model_id, data in models.items():
        if data.get("type") != resource_type:
            continue
        if data.get("offline", False):
            continue
        if trait in data.get("traits", []):
            return model_id
    return None


def _resolve_by_capability(
    models: dict[str, Any],
    capability: str,
    resource_type: str,
) -> str | None:
    """Find the first online model with a specific capability flag."""
    for model_id, data in models.items():
        if data.get("type") != resource_type:
            continue
        if data.get("offline", False):
            continue
        caps = data.get("model_spec", {}).get("capabilities", {})
        if caps.get(capability, False):
            return model_id
    return None


def _resolve_by_type(models: dict[str, Any], resource_type: str) -> str | None:
    """Find the first online model of a given resource type."""
    for model_id, data in models.items():
        if data.get("type") != resource_type:
            continue
        if data.get("offline", False):
            continue
        return model_id
    return None


def _resolve_alias(models: dict[str, Any], strategy: tuple[str, ...]) -> str | None:
    """Resolve a single alias using its strategy tuple."""
    kind = strategy[0]
    match kind:
        case "trait":
            return _resolve_by_trait(models, strategy[1], strategy[2])
        case "capability":
            return _resolve_by_capability(models, strategy[1], strategy[2])
        case "type":
            return _resolve_by_type(models, strategy[1])
        case _:
            return None


# ---------------------------------------------------------------------------
# Model list builders
# ---------------------------------------------------------------------------


def _get_text_models_by_price(models: dict[str, Any]) -> dict[str, list[str]]:
    """Categorize text models into cheap/medium/expensive by pricing."""
    priced: list[tuple[str, float]] = []
    for model_id, data in models.items():
        if data.get("type") != "text" or data.get("offline", False):
            continue
        pricing = data.get("model_spec", {}).get("pricing")
        if pricing and isinstance(pricing, dict):
            # Use output token price as the primary sort key
            output = pricing.get("output")
            if isinstance(output, dict):
                usd = output.get("usd", 0)
                if usd and usd > 0:
                    priced.append((model_id, usd))

    priced.sort(key=lambda x: x[1])
    if not priced:
        return {"cheap": [], "medium": [], "expensive": []}

    n = len(priced)
    third = max(1, n // 3)
    return {
        "cheap": [m for m, _ in priced[:third]],
        "medium": [m for m, _ in priced[third : 2 * third]],
        "expensive": [m for m, _ in priced[2 * third :]],
    }


def _build_model_lists(models: dict[str, Any]) -> dict[str, list[str]]:
    """Build categorized model lists from live API data."""
    lists: dict[str, list[str]] = {
        "FUNCTION_CALLING_MODELS": [],
        "VISION_CAPABLE_MODELS": [],
        "ALL_IMAGE_MODELS": [],
        "BETA_MODELS": [],
    }

    for model_id, data in models.items():
        if data.get("offline", False):
            continue
        caps = data.get("model_spec", {}).get("capabilities", {})
        mtype = data.get("type", "")

        if caps.get("supportsFunctionCalling"):
            lists["FUNCTION_CALLING_MODELS"].append(model_id)
        if caps.get("supportsVision"):
            lists["VISION_CAPABLE_MODELS"].append(model_id)
        if mtype == "image":
            lists["ALL_IMAGE_MODELS"].append(model_id)
        if data.get("beta", False):
            lists["BETA_MODELS"].append(model_id)

    price_tiers = _get_text_models_by_price(models)
    lists["CHEAP_TEXT_MODELS"] = price_tiers.get("cheap", [])[:3]
    lists["EXPENSIVE_TEXT_MODELS"] = price_tiers.get("expensive", [])[:3]

    return lists


def _pick_from_tier(
    models: dict[str, Any],
    tier: str,
) -> str | None:
    """Pick a representative model from a pricing tier."""
    tiers = _get_text_models_by_price(models)
    candidates = tiers.get(tier, [])
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Model metadata builder
# ---------------------------------------------------------------------------


def _build_metadata(
    models: dict[str, Any],
    alias_ids: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Build MODEL_METADATA from live API data for key models."""
    # Include models that are referenced by aliases
    target_ids = set(alias_ids.values())
    metadata: dict[str, dict[str, Any]] = {}

    for model_id in target_ids:
        data = models.get(model_id)
        if not data:
            continue

        mtype = data.get("type", "text")
        entry: dict[str, Any] = {"type": mtype}

        if mtype == "text":
            caps = data.get("model_spec", {}).get("capabilities", {})
            entry["context_tokens"] = data.get("availableContextTokens")
            entry["supports_function_calling"] = caps.get("supportsFunctionCalling", False)
            entry["supports_vision"] = caps.get("supportsVision", False)
            entry["supports_web_search"] = caps.get("supportsWebSearch", False)
            entry["quantization"] = caps.get("quantization", "not-available")
            # Derive pricing tier from position
            tiers = _get_text_models_by_price(models)
            if model_id in tiers.get("cheap", []):
                entry["pricing_tier"] = "cheap"
            elif model_id in tiers.get("expensive", []):
                entry["pricing_tier"] = "expensive"
            else:
                entry["pricing_tier"] = "medium"
        elif mtype == "image":
            constraints = data.get("model_spec", {}).get("constraints", {})
            steps = constraints.get("steps", {})
            entry["steps_default"] = steps.get("default")
            entry["steps_max"] = steps.get("max")
            entry["prompt_char_limit"] = constraints.get("promptCharacterLimit")
            entry["pricing_tier"] = "standard"
        elif mtype == "tts":
            entry["pricing_tier"] = "standard"

        metadata[model_id] = entry

    return metadata


# ---------------------------------------------------------------------------
# TTS voices extraction
# ---------------------------------------------------------------------------


def _extract_tts_voices(models: dict[str, Any], tts_model_id: str | None) -> list[str]:
    """Extract voice list from the TTS model's spec."""
    if not tts_model_id:
        return []
    data = models.get(tts_model_id)
    if not data:
        return []
    spec = data.get("model_spec", {})
    # voices may be on the spec directly or under constraints
    voices = spec.get("voices")
    if isinstance(voices, list) and voices:
        return voices
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def resolve_and_cache(
    api_key: str,
    base_url: str = "https://api.venice.ai/api/v1",
) -> dict[str, Any]:
    """
    Resolve all model aliases from the Venice API and write the cache file.

    Makes a single API call via DynamicModelSelector._fetch_models(), then
    resolves every alias using the cached model data.

    Returns the cache dict that was written.
    """
    from venice_ai import VeniceClient
    from venice_ai.models.selection import DynamicModelSelector

    async with VeniceClient(api_key=api_key, base_url=base_url) as client:
        selector = DynamicModelSelector(client, cache_ttl=600.0)
        models = await selector._fetch_models(force_refresh=True)

    # --- Resolve primary aliases ---
    aliases: dict[str, str] = {}
    for alias, strategy in ALIAS_STRATEGIES.items():
        resolved = _resolve_alias(models, strategy)
        if resolved:
            aliases[alias] = resolved

    # --- Price-tier aliases ---
    small = _pick_from_tier(models, "cheap")
    if small:
        aliases["SMALL_TEXT_MODEL"] = small
    medium = _pick_from_tier(models, "medium")
    if medium:
        aliases["MEDIUM_TEXT_MODEL"] = medium
    large = _pick_from_tier(models, "expensive")
    if large:
        aliases["LARGE_TEXT_MODEL"] = large

    # --- Uncensored aliases (trait + capability fallback) ---
    uncensored = _resolve_by_trait(models, "most_uncensored", "text")
    if not uncensored:
        uncensored = _resolve_by_capability(models, "supportsWebSearch", "text")
    if uncensored:
        aliases.setdefault("UNCENSORED_MODEL", uncensored)

    # --- Derived aliases ---
    for derived, source in DERIVED_ALIASES.items():
        if source in aliases:
            aliases[derived] = aliases[source]

    # --- Model lists ---
    lists = _build_model_lists(models)

    # --- Metadata ---
    metadata = _build_metadata(models, aliases)

    # --- TTS voices ---
    tts_id = aliases.get("TTS_MODEL")
    voices = _extract_tts_voices(models, tts_id)

    cache = {
        "resolved_at": datetime.now(UTC).isoformat(),
        "model_count": len(models),
        "aliases": aliases,
        "lists": lists,
        "metadata": metadata,
        "voices": voices,
    }

    CACHE_FILE.write_text(json.dumps(cache, indent=2, default=str))
    logger.info(f"Model cache written to {CACHE_FILE} ({len(aliases)} aliases resolved)")
    return cache


def load_cache() -> dict[str, Any] | None:
    """
    Load the model cache from disk.

    Returns None if the cache file doesn't exist, can't be parsed, or is older
    than ``CACHE_MAX_AGE``. Callers fall back to hardcoded defaults on None.
    """
    if not CACHE_FILE.exists():
        return None
    try:
        data = json.loads(CACHE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning(f"Could not parse model cache at {CACHE_FILE}")
        return None

    resolved_at_str = data.get("resolved_at")
    if resolved_at_str:
        try:
            resolved_at = datetime.fromisoformat(str(resolved_at_str).replace("Z", "+00:00"))
            if resolved_at.tzinfo is None:
                resolved_at = resolved_at.replace(tzinfo=UTC)
            if datetime.now(UTC) - resolved_at > CACHE_MAX_AGE:
                logger.warning(
                    f"Model cache at {CACHE_FILE} is older than "
                    f"{CACHE_MAX_AGE}; ignoring (refresh with --refresh-models)."
                )
                return None
        except ValueError:
            logger.warning(f"Could not parse resolved_at={resolved_at_str!r} from {CACHE_FILE}")
            return None

    return data
