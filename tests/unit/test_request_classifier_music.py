"""Request-classifier routing for the March 2026 music additions.

Both the endpoint patterns (``audio/queue|quote|retrieve|complete``) and the
launch-day model names should route to ``ResourceType.MUSIC`` — *not* to
``AUDIO`` (which is reserved for TTS/ASR and has separate rate limits).
"""

from unittest.mock import MagicMock

import pytest

from venice_ai._queue_types import ResourceType
from venice_ai._request_classifier import RequestClassifier


@pytest.fixture
def classifier() -> RequestClassifier:
    return RequestClassifier(MagicMock())


MUSIC_ENDPOINTS = [
    "audio/queue",
    "audio/quote",
    "audio/retrieve",
    "audio/complete",
]

MUSIC_MODEL_IDS = [
    "elevenlabs-music",
    "elevenlabs-sound-effects",
    "ace-step-1.5",
    "minimax-music-2.0",
    "stable-audio-2.5",
    "mmaudio-v2",
]


@pytest.mark.parametrize("endpoint", MUSIC_ENDPOINTS)
@pytest.mark.asyncio
async def test_music_endpoints_route_to_music(classifier: RequestClassifier, endpoint: str) -> None:
    req = {"endpoint": endpoint, "model": "elevenlabs-music"}
    metadata = await classifier.classify(req)
    assert metadata.resource_type is ResourceType.MUSIC


@pytest.mark.parametrize("model_id", MUSIC_MODEL_IDS)
@pytest.mark.asyncio
async def test_music_model_ids_route_to_music(classifier: RequestClassifier, model_id: str) -> None:
    # No endpoint hint — forces the classifier onto the model-name fallback.
    req = {"endpoint": "", "model": model_id}
    metadata = await classifier.classify(req)
    assert metadata.resource_type is ResourceType.MUSIC


@pytest.mark.asyncio
async def test_audio_speech_still_routes_to_audio(classifier: RequestClassifier) -> None:
    """TTS on /audio/speech must not get siphoned into MUSIC."""
    metadata = await classifier.classify({"endpoint": "audio/speech", "model": "tts-kokoro"})
    assert metadata.resource_type is ResourceType.AUDIO


@pytest.mark.asyncio
async def test_transcriptions_still_routes_to_audio(
    classifier: RequestClassifier,
) -> None:
    """Whisper on /audio/transcriptions stays in the AUDIO queue."""
    metadata = await classifier.classify(
        {"endpoint": "audio/transcriptions", "model": "whisper-large-v3"}
    )
    assert metadata.resource_type is ResourceType.AUDIO
