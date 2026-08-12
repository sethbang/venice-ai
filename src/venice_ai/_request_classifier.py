"""Request classification for Venice AI SDK rate limiting.

Classifies requests by endpoint pattern and model name into ResourceType
categories (LLM, IMAGE, AUDIO, EMBEDDING, etc.) for queue routing.
"""

import logging
import re
import uuid
from typing import Any

from ._queue_types import RequestMetadata, ResourceType
from .core.rate_limit_discovery import RateLimitDiscovery
from .validation.validators import validate_priority, validate_timeout

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex patterns (module-level so they are compiled once at import)
# ---------------------------------------------------------------------------

# Endpoint patterns for model-less endpoints
_RE_API_KEYS = re.compile(r"api_keys.*")
_RE_IMAGE_EDIT = re.compile(r"image/edit")
# Control-plane endpoints that take no model parameter.
_RE_IMAGE_STYLES = re.compile(r"image/styles")
_RE_CRYPTO_RPC = re.compile(r"crypto/rpc")

_MODEL_LESS_ENDPOINTS: frozenset[re.Pattern[str]] = frozenset(
    {
        _RE_API_KEYS,
        _RE_IMAGE_EDIT,
        _RE_IMAGE_STYLES,
        _RE_CRYPTO_RPC,
    }
)

# Endpoint → resource type patterns (primary classification)
_RE_CHAT_COMPLETIONS = re.compile(r"chat/completions")
_RE_RESPONSES = re.compile(r"^responses$|/responses$")
# Image endpoints are *singular* (``image/...``) — these are the real paths
# emitted by ``resources/image.py`` (POST /image/generate, /image/edit,
# /image/upscale, /image/multi-edit, /image/background-remove).
_RE_IMAGE_GENERATE = re.compile(r"image/generate")
_RE_IMAGE_UPSCALE = re.compile(r"image/upscale")
_RE_IMAGE_MULTI_EDIT = re.compile(r"image/multi-edit")
_RE_IMAGE_BACKGROUND_REMOVE = re.compile(r"image/background-remove")
# OpenAI-compat plural path emitted by ``resources/image.py`` (simple_generate
# targets ``images/generations``); exists in swagger and must route to IMAGE.
_RE_IMAGES_GENERATIONS = re.compile(r"images/generations")
# OpenAI-compat ``images/variations`` path: routed to IMAGE for parity with
# ``images/generations``. The SDK emits the singular ``image/*`` paths instead,
# so this mapping is a harmless no-op today.
_RE_IMAGES_VARIATIONS = re.compile(r"images/variations")
_RE_AUDIO_TRANSCRIPTIONS = re.compile(r"audio/transcriptions")
_RE_AUDIO_SPEECH = re.compile(r"audio/speech")
_RE_AUDIO_TRANSLATIONS = re.compile(r"audio/translations")
_RE_AUDIO_VOICES = re.compile(r"audio/voices")
# Music generation shares the /audio/* queue family (March 2026 changelog).
_RE_AUDIO_QUEUE = re.compile(r"audio/queue")
_RE_AUDIO_QUOTE = re.compile(r"audio/quote")
_RE_AUDIO_RETRIEVE = re.compile(r"audio/retrieve")
_RE_AUDIO_COMPLETE = re.compile(r"audio/complete")
_RE_EMBEDDINGS = re.compile(r"embeddings")
_RE_VIDEO_TRANSCRIPTIONS = re.compile(r"video/transcriptions")
# Async video-generation lifecycle endpoints (mirror the /audio/* music family).
_RE_VIDEO_QUEUE = re.compile(r"video/queue")
_RE_VIDEO_QUOTE = re.compile(r"video/quote")
_RE_VIDEO_RETRIEVE = re.compile(r"video/retrieve")
_RE_VIDEO_COMPLETE = re.compile(r"video/complete")
_RE_BILLING_USAGE_ANALYTICS = re.compile(r"billing/usage-analytics")
_RE_BILLING_USAGE = re.compile(r"billing/usage")
_RE_BILLING_BALANCE = re.compile(r"billing/balance")
_RE_CHARACTERS = re.compile(r"characters")

_RESOURCE_PATTERNS: dict[ResourceType, list[re.Pattern[str]]] = {
    ResourceType.LLM: [_RE_CHAT_COMPLETIONS, _RE_RESPONSES],
    ResourceType.IMAGE: [
        _RE_IMAGE_GENERATE,
        _RE_IMAGE_MULTI_EDIT,
        _RE_IMAGE_EDIT,
        _RE_IMAGE_UPSCALE,
        _RE_IMAGE_BACKGROUND_REMOVE,
        _RE_IMAGES_GENERATIONS,
        _RE_IMAGES_VARIATIONS,
    ],
    ResourceType.AUDIO: [
        _RE_AUDIO_TRANSCRIPTIONS,
        _RE_AUDIO_SPEECH,
        _RE_AUDIO_TRANSLATIONS,
        _RE_AUDIO_VOICES,
    ],
    ResourceType.MUSIC: [
        _RE_AUDIO_QUEUE,
        _RE_AUDIO_QUOTE,
        _RE_AUDIO_RETRIEVE,
        _RE_AUDIO_COMPLETE,
    ],
    ResourceType.EMBEDDING: [_RE_EMBEDDINGS],
    ResourceType.API_MANAGEMENT: [_RE_API_KEYS, _RE_IMAGE_STYLES, _RE_CRYPTO_RPC],
    ResourceType.BILLING: [_RE_BILLING_USAGE_ANALYTICS, _RE_BILLING_USAGE, _RE_BILLING_BALANCE],
    ResourceType.CHARACTERS: [_RE_CHARACTERS],
    ResourceType.VIDEO: [
        _RE_VIDEO_TRANSCRIPTIONS,
        _RE_VIDEO_QUEUE,
        _RE_VIDEO_QUOTE,
        _RE_VIDEO_RETRIEVE,
        _RE_VIDEO_COMPLETE,
    ],
}

# Model name → resource type patterns (fallback, case-insensitive)
_RE_MODEL_LLAMA = re.compile(r"llama", re.IGNORECASE)
_RE_MODEL_QWEN = re.compile(r"qwen", re.IGNORECASE)
_RE_MODEL_DEEPSEEK = re.compile(r"deepseek", re.IGNORECASE)
_RE_MODEL_DOLPHIN = re.compile(r"dolphin", re.IGNORECASE)
_RE_MODEL_MISTRAL = re.compile(r"mistral", re.IGNORECASE)
_RE_MODEL_VENICE_UNCENSORED = re.compile(r"venice-uncensored", re.IGNORECASE)
_RE_MODEL_CLAUDE = re.compile(r"claude", re.IGNORECASE)
_RE_MODEL_GLM = re.compile(r"glm", re.IGNORECASE)
_RE_MODEL_FLUX = re.compile(r"flux", re.IGNORECASE)
_RE_MODEL_STABLE_DIFFUSION = re.compile(r"stable-diffusion", re.IGNORECASE)
_RE_MODEL_LUSTIFY = re.compile(r"lustify", re.IGNORECASE)
_RE_MODEL_PONY = re.compile(r"pony", re.IGNORECASE)
_RE_MODEL_FLUENTLY = re.compile(r"fluently", re.IGNORECASE)
_RE_MODEL_UPSCALER = re.compile(r"upscaler", re.IGNORECASE)
_RE_MODEL_EDIT_IMAGE = re.compile(r"edit-image", re.IGNORECASE)
_RE_MODEL_TTS_KOKORO = re.compile(r"tts-kokoro", re.IGNORECASE)
_RE_MODEL_WHISPER = re.compile(r"whisper", re.IGNORECASE)
_RE_MODEL_EMBEDDING = re.compile(r"embedding", re.IGNORECASE)
_RE_MODEL_BGE_M3 = re.compile(r"bge-m3", re.IGNORECASE)

_RE_MODEL_SEEDREAM = re.compile(r"seedream", re.IGNORECASE)
# Image models whose names would otherwise be swallowed by the generic
# ``qwen``->LLM / ``\bgpt-``->LLM rules. These are anchored on the ``-image``
# suffix so plain Qwen/GPT LLM IDs (``qwen3-235b``, ``gpt-5.3-codex``) are not
# captured. The IMAGE list is iterated before LLM, so these win the match.
_RE_MODEL_QWEN_IMAGE = re.compile(r"qwen-image", re.IGNORECASE)
_RE_MODEL_GPT_IMAGE = re.compile(r"gpt-image", re.IGNORECASE)

# OpenAI GPT-family LLMs (e.g. gpt-5.3-codex). Anchored with a word-boundary so
# unrelated identifiers containing the substring don't accidentally match.
_RE_MODEL_GPT = re.compile(r"\bgpt-", re.IGNORECASE)

# Music models (March 2026 changelog). Kept anchored / specific so the more
# generic vendor names (e.g. a hypothetical bare ``elevenlabs`` LLM) would not
# pre-empt the music routing.
_RE_MODEL_ELEVENLABS_MUSIC = re.compile(r"elevenlabs-music", re.IGNORECASE)
_RE_MODEL_ELEVENLABS_SOUND_EFFECTS = re.compile(r"elevenlabs-sound-effects", re.IGNORECASE)
_RE_MODEL_ACE_STEP = re.compile(r"ace-step", re.IGNORECASE)
_RE_MODEL_MINIMAX_MUSIC = re.compile(r"minimax-music", re.IGNORECASE)
_RE_MODEL_STABLE_AUDIO = re.compile(r"stable-audio", re.IGNORECASE)
_RE_MODEL_MMAUDIO = re.compile(r"mmaudio", re.IGNORECASE)

# Dict iteration order determines match precedence. IMAGE / AUDIO / EMBEDDING
# are checked before LLM so specialised variants route to their correct queue
# instead of falling into the generic LLM bucket.
_MODEL_TYPE_PATTERNS: dict[ResourceType, list[re.Pattern[str]]] = {
    ResourceType.IMAGE: [
        _RE_MODEL_QWEN_IMAGE,
        _RE_MODEL_GPT_IMAGE,
        _RE_MODEL_SEEDREAM,
        _RE_MODEL_FLUX,
        _RE_MODEL_STABLE_DIFFUSION,
        _RE_MODEL_LUSTIFY,
        _RE_MODEL_PONY,
        _RE_MODEL_FLUENTLY,
        _RE_MODEL_UPSCALER,
        _RE_MODEL_EDIT_IMAGE,
    ],
    ResourceType.MUSIC: [
        _RE_MODEL_ELEVENLABS_MUSIC,
        _RE_MODEL_ELEVENLABS_SOUND_EFFECTS,
        _RE_MODEL_ACE_STEP,
        _RE_MODEL_MINIMAX_MUSIC,
        _RE_MODEL_STABLE_AUDIO,
        _RE_MODEL_MMAUDIO,
    ],
    ResourceType.AUDIO: [_RE_MODEL_TTS_KOKORO, _RE_MODEL_WHISPER],
    ResourceType.EMBEDDING: [_RE_MODEL_EMBEDDING, _RE_MODEL_BGE_M3],
    ResourceType.LLM: [
        _RE_MODEL_LLAMA,
        _RE_MODEL_QWEN,
        _RE_MODEL_DEEPSEEK,
        _RE_MODEL_DOLPHIN,
        _RE_MODEL_MISTRAL,
        _RE_MODEL_VENICE_UNCENSORED,
        _RE_MODEL_CLAUDE,
        _RE_MODEL_GLM,
        _RE_MODEL_GPT,
    ],
}


class RequestClassifier:
    """Classifies requests into ResourceType categories for queue routing.

    Uses a two-tier strategy: (1) endpoint regex matching, then (2) model name
    pattern matching as fallback. Defaults to LLM for unknown patterns.
    Also estimates token usage for LLM requests.
    """

    def __init__(self, rate_limit_discovery: RateLimitDiscovery):
        """Initialize with endpoint and model pattern registries."""
        self.rate_limit_discovery = rate_limit_discovery

        # Reference module-level pre-compiled pattern registries
        self.model_less_endpoints = _MODEL_LESS_ENDPOINTS
        self.resource_patterns = _RESOURCE_PATTERNS
        self.model_type_patterns = _MODEL_TYPE_PATTERNS

    async def classify(self, request: dict[str, Any]) -> RequestMetadata:
        """Classify a request dict into a RequestMetadata for queue routing.

        Extracts model/endpoint, determines ResourceType, estimates tokens
        (for LLM requests), and validates priority/timeout fields.

        Raises:
            TypeError: If request is not a dict or fields have wrong types.
            ValueError: If request is empty or priority/timeout are invalid.
        """
        if not isinstance(request, dict):
            raise TypeError(f"Request must be a dictionary, got {type(request).__name__}")

        if not request:
            raise ValueError("Request dictionary cannot be empty")

        model_id = request.get("model", "unknown")
        if not isinstance(model_id, str):
            raise TypeError(
                f"Request 'model' field must be a string, got {type(model_id).__name__}"
            )

        endpoint = request.get("endpoint", "")
        if not isinstance(endpoint, str):
            raise TypeError(
                f"Request 'endpoint' field must be a string, got {type(endpoint).__name__}"
            )

        resource_type = self._determine_resource_type(endpoint, model_id)

        estimated_tokens = None
        if resource_type == ResourceType.LLM:
            estimated_tokens = self._estimate_tokens(request)

        # Normalize priority to int up front so the value flowing into
        # RequestMetadata.priority (typed `int`) matches the dataclass field.
        # `request.get("priority", 0)` is `Any | None`, which pyright (rightly)
        # rejects as not assignable to int; coerce + validate in one shot.
        priority_raw = request.get("priority", 0)
        try:
            priority = int(priority_raw) if priority_raw is not None else 0
            validate_priority(priority, min_val=0, max_val=10)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid request priority: {e}") from e

        timeout = request.get("timeout", 60.0)
        if timeout is not None:
            try:
                validate_timeout(float(timeout), min_val=0.1, max_val=300.0, param_name="timeout")
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid request timeout: {e}") from e

        client_id = request.get("client_id")
        if client_id is not None and not isinstance(client_id, str):
            raise TypeError(
                f"Request 'client_id' field must be a string, got {type(client_id).__name__}"
            )

        if "requires_model" in request:
            requires_model = request["requires_model"]
        else:
            requires_model = not any(
                pattern.search(endpoint) for pattern in self.model_less_endpoints
            )

        return RequestMetadata(
            request_id=str(uuid.uuid4()),
            model_id=model_id,
            resource_type=resource_type,
            estimated_tokens=estimated_tokens,
            priority=priority,
            timeout=timeout,
            client_id=client_id,
            endpoint=endpoint,
            requires_model=requires_model,
        )

    def _determine_resource_type(self, endpoint: str, model_id: str) -> ResourceType:
        """Determine ResourceType via endpoint patterns, then model name fallback.

        Tiers: endpoint match → model name match → model-less check → default LLM.
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Determining resource type for endpoint='{endpoint}', model_id='{model_id}'"
            )

        # Tier 1: Endpoint pattern matching
        for resource_type, patterns in self.resource_patterns.items():
            for pattern in patterns:
                if pattern.search(endpoint):
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"Matched endpoint pattern '{pattern.pattern}' to resource type {resource_type}"
                        )
                    return resource_type

        # Tier 2: Model name pattern matching
        if model_id and model_id != "unknown":
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("No endpoint pattern matched, falling back to model patterns.")
            for resource_type, patterns in self.model_type_patterns.items():
                for pattern in patterns:
                    if pattern.search(model_id):
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                f"Matched model pattern '{pattern.pattern}' to resource type {resource_type}"
                            )
                        return resource_type

        # Tier 3: Model-less endpoints default to API_MANAGEMENT
        is_model_less = any(pattern.search(endpoint) for pattern in self.model_less_endpoints)

        if is_model_less:
            logger.debug(
                f"No specific resource pattern matched for endpoint '{endpoint}', "
                "but it is in the model-less list. Defaulting to API_MANAGEMENT."
            )
            return ResourceType.API_MANAGEMENT

        # Tier 4: Default to LLM
        logger.debug(
            f"No patterns matched for endpoint '{endpoint}' and it's not in the "
            "model-less list. Defaulting to LLM."
        )
        return ResourceType.LLM

    def _estimate_tokens(self, request: dict[str, Any]) -> int:
        """Estimate total token count for an LLM request.

        Uses ~4 chars/token heuristic on message content + prompt text,
        then adds max_completion_tokens (default 150) for expected output.
        Returns at least 1.
        """
        total_chars = 0

        # Sum character lengths from chat messages
        messages = request.get("messages", [])
        for message in messages:
            if isinstance(message, dict):
                content = message.get("content", "")
                if isinstance(content, str):
                    total_chars += len(content)
                elif isinstance(content, list):
                    # Multimodal: extract text portions only
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            total_chars += len(item.get("text", ""))

        # Also count direct prompt text (completion endpoint format)
        prompt = request.get("prompt", "")
        if isinstance(prompt, str):
            total_chars += len(prompt)
        elif isinstance(prompt, list):
            for p in prompt:
                if isinstance(p, str):
                    total_chars += len(p)

        # ~4 chars per token, minimum 1
        estimated_tokens = max(1, total_chars // 4)

        # Add expected output tokens
        max_tokens = request.get("max_completion_tokens", 150)
        if isinstance(max_tokens, (int, float)):
            estimated_tokens += int(max_tokens)
        else:
            estimated_tokens += 150

        return int(estimated_tokens)

    def get_resource_type_for_model(self, model_id: str) -> ResourceType:
        """Classify a model by name only (no endpoint context). Defaults to LLM."""
        return self._determine_resource_type("", model_id)
