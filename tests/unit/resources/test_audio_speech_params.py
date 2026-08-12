"""
Unit tests for ``create_speech(prompt=..., temperature=..., top_p=...)``.

Venice documents these as optional body fields on POST /audio/speech
(see ``api-reference/endpoint/audio/speech.md``):

* ``prompt`` — style prompt (Qwen 3 TTS)
* ``temperature`` — sampling temperature
* ``top_p`` — nucleus sampling

The SDK must accept and forward them verbatim in the JSON body when set, and
omit them when unset.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.resources.audio import Audio


@pytest.fixture
def audio_resource() -> Audio:
    client = MagicMock()
    client._request = AsyncMock(return_value=b"\x00\x01fakeaudio")
    return Audio(client)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_create_speech_forwards_prompt(audio_resource: Audio) -> None:
    await audio_resource.create_speech(
        input="Hello world",
        model="tts-qwen3",
        voice="Vivian",
        prompt="Excited and energetic.",
    )

    kwargs = audio_resource._client._request.call_args.kwargs  # type: ignore[attr-defined]
    body = kwargs["json_data"]
    assert body["prompt"] == "Excited and energetic."


@pytest.mark.asyncio
async def test_create_speech_forwards_temperature_and_top_p(audio_resource: Audio) -> None:
    await audio_resource.create_speech(
        input="Sampled speech",
        model="tts-qwen3",
        voice="Vivian",
        temperature=0.9,
        top_p=0.85,
    )

    body = audio_resource._client._request.call_args.kwargs["json_data"]  # type: ignore[attr-defined]
    assert body["temperature"] == 0.9
    assert body["top_p"] == 0.85


@pytest.mark.asyncio
async def test_create_speech_omits_new_params_when_not_set(audio_resource: Audio) -> None:
    await audio_resource.create_speech(
        input="Hello",
        model="tts-kokoro",
        voice="af_sky",
    )

    body = audio_resource._client._request.call_args.kwargs["json_data"]  # type: ignore[attr-defined]
    assert "prompt" not in body
    assert "temperature" not in body
    assert "top_p" not in body
