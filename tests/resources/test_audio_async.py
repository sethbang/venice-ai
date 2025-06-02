"""
Tests for the asynchronous AsyncAudio resource.
"""

import pytest
import httpx
from io import BytesIO
from typing import AsyncIterator

from venice_ai import AsyncVeniceClient
from venice_ai.types.audio import Voice, ResponseFormat
from venice_ai.exceptions import APIError, AuthenticationError


@pytest.mark.asyncio
async def test_create_speech_success_async(httpx_mock):
    """Tests successful asynchronous speech generation (non-streaming)."""
    # Mock binary audio data
    mock_audio_data = b"mock audio binary data"

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/audio/speech",
        content=mock_audio_data,
        headers={"Content-Type": "audio/mp3"},
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        response = await client.audio.create_speech(
            model="venice-speech-1",
            input="Hello, world!",
            voice=Voice.NOVA
        )

    assert isinstance(response, bytes)
    assert response == mock_audio_data


@pytest.mark.asyncio
async def test_create_speech_with_options_async(httpx_mock):
    """Tests asynchronous speech generation with additional options."""
    mock_audio_data = b"mock audio with options binary data"

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/audio/speech",
        content=mock_audio_data,
        headers={"Content-Type": "audio/flac"},
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        response = await client.audio.create_speech(
            model="venice-speech-1",
            input="This is a test with options.",
            voice=Voice.ALLOY,
            response_format=ResponseFormat.FLAC,
            speed=1.5
        )

    assert isinstance(response, bytes)
    assert response == mock_audio_data


@pytest.mark.asyncio
async def test_create_speech_streaming_async(httpx_mock):
    """Tests asynchronous streaming speech generation."""
    # pytest.skip("Skipping due to httpx 0.28.1 streaming compatibility issues") # Re-enabled test
    # Create multiple chunks of mock audio data for streaming
    mock_chunks = [
        b"chunk1",
        b"chunk2",
        b"chunk3"
    ]

    # Configure the mock to stream chunks
    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/audio/speech",
        content=b"".join(mock_chunks),
        headers={"Content-Type": "audio/mp3"},
        status_code=200
        # Configure streaming behavior
        # stream=True # Removed based on hypothesis
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        # Await the method to get the async iterator
        response_iterator = await client.audio.create_speech(
            model="venice-speech-1",
            input="This is streaming audio content.",
            voice=Voice.SHIMMER,
            stream=True
        )

        assert isinstance(response_iterator, AsyncIterator)
        # Consume the async iterator to ensure the mock is marked as requested
        _ = [chunk async for chunk in response_iterator]


@pytest.mark.asyncio
async def test_create_speech_error_async(httpx_mock):
    """Tests asynchronous error handling for speech generation."""
    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/audio/speech",
        status_code=401,
        json={"error": {"message": "Invalid API key", "type": "authentication_error"}},
    )

    async with AsyncVeniceClient(api_key="invalid-key") as client:
        with pytest.raises(AuthenticationError) as excinfo:
            await client.audio.create_speech(
                model="venice-speech-1",
                input="This should fail.",
                voice=Voice.NOVA
            )

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 401
    assert "Invalid API key" in str(excinfo.value)


@pytest.mark.asyncio
async def test_create_speech_streaming_with_options_async(httpx_mock):
    """Tests asynchronous streaming speech generation with additional options."""
    # Mock streaming response
    mock_chunks = [
        b"stream_chunk1",
        b"stream_chunk2",
        b"stream_chunk3"
    ]

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/audio/speech",
        content=b"".join(mock_chunks),
        headers={"Content-Type": "audio/wav"},
        status_code=200
        # stream=True # Removed based on hypothesis
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        # Await the method to get the async iterator
        response_iterator = await client.audio.create_speech(
            model="venice-speech-1",
            input="This is streaming audio content with options.",
            voice=Voice.ONYX,
            response_format=ResponseFormat.WAV,
            speed=0.8,
            stream=True
        )

        assert isinstance(response_iterator, AsyncIterator)
        # Consume the async iterator to ensure the mock is marked as requested
        _ = [chunk async for chunk in response_iterator]