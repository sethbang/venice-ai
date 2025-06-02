"""
Tests for the Audio and AsyncAudio resources.
"""

import pytest
import httpx
from io import BytesIO
from typing import Iterator, AsyncIterator

from venice_ai import VeniceClient, AsyncVeniceClient
from venice_ai.types.audio import Voice, ResponseFormat
from venice_ai.exceptions import (
    APIError, AuthenticationError, InvalidRequestError,
    PermissionDeniedError, NotFoundError, RateLimitError
)


# Synchronous Tests
def test_create_speech_success(httpx_mock):
    """Tests successful speech generation (non-streaming)."""
    mock_audio_data = b"mock audio binary data"

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/audio/speech",
        content=mock_audio_data,
        headers={"Content-Type": "audio/mp3"},
        status_code=200,
    )

    client = VeniceClient(api_key="test-key")
    response = client.audio.create_speech(
        model="venice-speech-1",
        input="Hello, world!",
        voice=Voice.NOVA
    )

    assert isinstance(response, bytes)
    assert response == mock_audio_data


def test_create_speech_with_options(httpx_mock):
    """Tests speech generation with additional options."""
    mock_audio_data = b"mock audio with options binary data"

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/audio/speech",
        content=mock_audio_data,
        headers={"Content-Type": "audio/flac"},
        status_code=200,
    )

    client = VeniceClient(api_key="test-key")
    response = client.audio.create_speech(
        model="venice-speech-1",
        input="This is a test with options.",
        voice=Voice.ALLOY,
        response_format=ResponseFormat.FLAC,
        speed=1.5
    )

    assert isinstance(response, bytes)
    assert response == mock_audio_data


def test_create_speech_streaming(httpx_mock):
    """Tests streaming speech generation."""
    # pytest.skip("Skipping due to httpx 0.28.1 streaming compatibility issues") # Re-enabled test
    mock_chunks = [b"chunk1", b"chunk2", b"chunk3"]

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/audio/speech",
        content=b"".join(mock_chunks),
        headers={"Content-Type": "audio/mp3"},
        status_code=200
        # stream=True  # Removed due to httpx 0.28.1 compatibility issues
    )

    client = VeniceClient(api_key="test-key")
    response_iterator = client.audio.create_speech(
        model="venice-speech-1",
        input="This is streaming audio content.",
        voice=Voice.SHIMMER,
        stream=True
    )

    assert isinstance(response_iterator, Iterator)
    # Consume the iterator to ensure the mock is marked as requested
    list(response_iterator)


def test_create_speech_invalid_speed(httpx_mock):
    """Tests parameter validation for invalid speed value."""
    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/audio/speech",
        status_code=400,
        json={"error": {"message": "Speed must be between 0.25 and 4.0", "type": "invalid_request_error"}},
    )

    client = VeniceClient(api_key="test-key")
    with pytest.raises(InvalidRequestError) as excinfo:
        client.audio.create_speech(
            model="venice-speech-1",
            input="This should fail due to invalid speed.",
            voice=Voice.NOVA,
            speed=5.0
        )

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 400
    assert "Speed must be between 0.25 and 4.0" in str(excinfo.value)


def test_create_speech_errors(httpx_mock):
    """Tests various error responses for speech generation."""
    error_cases = [
        (401, {"error": {"message": "Invalid API key", "type": "authentication_error"}}, AuthenticationError),
        (403, {"error": {"message": "Permission denied", "type": "permission_error"}}, PermissionDeniedError),
        (404, {"error": {"message": "Model not found", "type": "not_found_error"}}, NotFoundError),
        (429, {"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}, RateLimitError),
        (400, {"error": {"message": "Invalid input", "type": "invalid_request_error"}}, InvalidRequestError),
    ]

    for status_code, error_body, error_class in error_cases:
        httpx_mock.add_response(
            method="POST",
            url="https://api.venice.ai/api/v1/audio/speech",
            status_code=status_code,
            json=error_body,
        )

        client = VeniceClient(api_key="test-key")
        with pytest.raises(error_class) as excinfo:
            client.audio.create_speech(
                model="venice-speech-1",
                input="This should fail.",
                voice=Voice.NOVA
            )

        assert excinfo.value.response is not None
        assert excinfo.value.response.status_code == status_code


# Asynchronous Tests
@pytest.mark.asyncio
async def test_create_speech_success_async(httpx_mock):
    """Tests successful asynchronous speech generation (non-streaming)."""
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
    mock_chunks = [b"chunk1", b"chunk2", b"chunk3"]

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/audio/speech",
        content=b"".join(mock_chunks),
        headers={"Content-Type": "audio/mp3"},
        status_code=200
        # stream=True  # Removed due to httpx 0.28.1 compatibility issues
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
async def test_create_speech_invalid_speed_async(httpx_mock):
    """Tests asynchronous parameter validation for invalid speed value."""
    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/audio/speech",
        status_code=400,
        json={"error": {"message": "Speed must be between 0.25 and 4.0", "type": "invalid_request_error"}},
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        with pytest.raises(InvalidRequestError) as excinfo:
            await client.audio.create_speech(
                model="venice-speech-1",
                input="This should fail due to invalid speed.",
                voice=Voice.NOVA,
                speed=5.0
            )

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 400
    assert "Speed must be between 0.25 and 4.0" in str(excinfo.value)


@pytest.mark.asyncio
async def test_create_speech_errors_async(httpx_mock):
    """Tests various asynchronous error responses for speech generation."""
    error_cases = [
        (401, {"error": {"message": "Invalid API key", "type": "authentication_error"}}, AuthenticationError),
        (403, {"error": {"message": "Permission denied", "type": "permission_error"}}, PermissionDeniedError),
        (404, {"error": {"message": "Model not found", "type": "not_found_error"}}, NotFoundError),
        (429, {"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}, RateLimitError),
        (400, {"error": {"message": "Invalid input", "type": "invalid_request_error"}}, InvalidRequestError),
    ]

    for status_code, error_body, error_class in error_cases:
        httpx_mock.add_response(
            method="POST",
            url="https://api.venice.ai/api/v1/audio/speech",
            status_code=status_code,
            json=error_body,
        )

        async with AsyncVeniceClient(api_key="test-key") as client:
            with pytest.raises(error_class) as excinfo:
                await client.audio.create_speech(
                    model="venice-speech-1",
                    input="This should fail.",
                    voice=Voice.NOVA
                )

        assert excinfo.value.response is not None
        assert excinfo.value.response.status_code == status_code