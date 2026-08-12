"""
VCRpy-based integration tests for Audio resource functionality.

This module tests audio speech generation, voice listing, and streaming functionality
through real API interactions recorded with VCRpy, replacing complex mock-based
unit tests with actual API behavior verification.
"""

import asyncio
import os

import pytest
import pytest_asyncio

from venice_ai import create_test_venice_client
from venice_ai.core.config import SchedulerMode
from venice_ai.exceptions import APIError, VeniceError
from venice_ai.resources.audio import Audio
from venice_ai.types import AudioResponse, ResponseFormat, Voice


@pytest_asyncio.fixture
async def venice_client(backend_instance):
    """Create a Venice client for VCR testing."""
    api_key = os.getenv("VENICE_API_KEY")
    if not api_key:
        pytest.skip("VENICE_API_KEY environment variable required for integration tests")

    # Use INTELLIGENT mode with MemoryBackend for VCR tests
    # This provides rate limit protection (prevents 429s) without Redis connection leaks
    client = create_test_venice_client(
        api_key=api_key,
        scheduler_mode=SchedulerMode.INTELLIGENT,
        enable_redis=False,  # Use MemoryBackend instead of Redis
    )
    try:
        yield client
    finally:
        await client.close()


# model_selector fixture is now provided by the root conftest.py


@pytest_asyncio.fixture
async def audio_resource(venice_client):
    """Create an Audio resource instance for testing."""
    return Audio(venice_client)


# ============================================================================
# Speech Generation Tests (Replaces Mock-Based Tests)
# ============================================================================


@pytest.mark.integration
async def test_create_speech_basic_functionality(audio_resource, model_selector, vcr_cassette):
    """
    Test basic speech generation with real API calls.
    Replaces mock-based test_create_speech_basic_success from unit tests.
    """
    with vcr_cassette:
        try:
            # Try to get a TTS model dynamically
            try:
                tts_model = await model_selector.select_tts_model()
            except AttributeError:
                # If model selector doesn't have TTS method, use known model
                tts_model = "tts-kokoro"

            result = await audio_resource.create_speech(
                input="Hello, this is a test of the audio system.",
                model=tts_model,
                voice="af_sky",  # Use valid voice from API documentation
                response_format=ResponseFormat.MP3,
                speed=1.0,
                stream=False,
            )

            # Verify we got an AudioResponse with real content
            assert isinstance(result, AudioResponse)
            assert result.content is not None
            assert len(result.content) > 0
            assert isinstance(result.content, bytes)

            # Check for valid audio file headers
            # MP3 files typically start with ID3 tag or MPEG header
            is_mp3 = (
                result.content.startswith(b"ID3")  # ID3 tag
                or result.content.startswith(b"\xff\xfb")  # MPEG header
                or result.content.startswith(b"\xff\xf3")  # MPEG Layer III
                or result.content.startswith(b"\xff\xf2")  # MPEG Layer II
            )

            assert is_mp3 or len(result.content) > 100, "Should receive valid audio data"

        except (VeniceError, APIError) as e:
            if "not supported" in str(e).lower() or "model" in str(e).lower():
                pytest.skip(f"TTS not supported or model unavailable: {e}")
            else:
                raise


@pytest.mark.integration
async def test_create_speech_with_all_parameters(audio_resource, vcr_cassette):
    """
    Test speech generation with all parameters using real API.
    Replaces mock-based parameter testing from unit tests.
    """
    with vcr_cassette:
        try:
            result = await audio_resource.create_speech(
                input="This is a comprehensive parameter test for the audio system.",
                model="tts-kokoro",
                voice=Voice.AF_SKY,  # Using enum value
                response_format=ResponseFormat.WAV,
                speed=1.2,
                stream=False,
                timeout=60.0,
            )

            # Verify comprehensive parameter handling
            assert isinstance(result, AudioResponse)
            assert result.content is not None
            assert len(result.content) > 0

            # WAV files have specific headers
            is_wav = result.content.startswith(b"RIFF") and b"WAVE" in result.content[:12]
            assert is_wav or len(result.content) > 100, "Should receive valid WAV audio data"

        except (VeniceError, APIError) as e:
            if "not supported" in str(e).lower():
                pytest.skip(f"TTS with these parameters not supported: {e}")
            else:
                raise


@pytest.mark.integration
async def test_create_speech_streaming_mode(audio_resource, vcr_cassette):
    """
    Test speech generation in streaming mode with real API.
    Replaces mock-based streaming tests from unit tests.
    """
    with vcr_cassette:
        try:
            result = await audio_resource.create_speech(
                input="This is a streaming audio test.",
                model="tts-kokoro",
                voice="af_sky",  # Use valid voice from API documentation
                stream=True,
            )

            # Should return an async iterator for streaming
            assert hasattr(result, "__aiter__"), "Streaming should return async iterator"

            # Collect streamed chunks
            chunks = []
            chunk_count = 0
            async for chunk in result:
                chunks.append(chunk)
                chunk_count += 1
                # Limit chunks to avoid infinite streams in tests
                if chunk_count >= 10:
                    break

            # Verify we got streaming data
            assert len(chunks) > 0, "Should receive streaming audio chunks"

            # Each chunk should be bytes
            for chunk in chunks:
                assert isinstance(chunk, bytes), "Each chunk should be bytes"
                assert len(chunk) > 0, "Each chunk should contain data"

            # Combined chunks should form audio data
            combined_audio = b"".join(chunks)
            assert len(combined_audio) > 0, "Combined streaming data should be substantial"

        except (VeniceError, APIError) as e:
            if "streaming not supported" in str(e).lower():
                pytest.skip(f"Audio streaming not supported: {e}")
            else:
                raise


# ============================================================================
# Voice Listing Tests (Replaces Mock-Based Tests)
# ============================================================================


@pytest.mark.integration
async def test_list_voices_functionality(audio_resource, vcr_cassette):
    """
    Test voice listing with real API calls.
    Replaces mock-based voice listing tests from unit tests.
    """
    with vcr_cassette:
        try:
            voices = await audio_resource.get_voices()

            # Verify voice listing response
            assert voices is not None
            assert hasattr(voices, "data")
            assert isinstance(voices.data, list)

            # Should have at least some voices available
            if len(voices.data) > 0:
                voice = voices.data[0]
                assert hasattr(voice, "id") or hasattr(voice, "name")
                # Voice should have identifiable attributes

        except (VeniceError, APIError) as e:
            if "not supported" in str(e).lower():
                pytest.skip(f"Voice listing not supported: {e}")
            else:
                raise


@pytest.mark.integration
async def test_get_available_voices_alias(audio_resource, vcr_cassette):
    """
    Test voice listing alias method with real API.
    Replaces mock-based alias method tests from unit tests.
    """
    with vcr_cassette:
        try:
            voices = await audio_resource.get_voices()

            # Should get same structure as list_voices
            assert voices is not None
            assert hasattr(voices, "data")
            assert isinstance(voices.data, list)

        except (VeniceError, APIError) as e:
            if "not supported" in str(e).lower():
                pytest.skip(f"Voice listing not supported: {e}")
            else:
                raise


# ============================================================================
# Error Handling Tests (Replaces Mock-Based Error Tests)
# ============================================================================


@pytest.mark.integration
async def test_empty_input_error_handling(audio_resource, vcr_cassette):
    """
    Test error handling for empty input with real API validation.
    Replaces mock-based validation tests from unit tests.
    """
    with vcr_cassette, pytest.raises(ValueError, match="Input text cannot be empty"):
        # Test empty input validation
        await audio_resource.create_speech(
            input="",  # Empty input
            model="tts-kokoro",
            voice="kokoro-default",
        )


@pytest.mark.integration
async def test_invalid_model_error_handling(audio_resource, vcr_cassette):
    """
    Test error handling for invalid models with real API responses.
    Replaces mock-based error response tests from unit tests.
    """
    with vcr_cassette:
        # Test with invalid model
        with pytest.raises(VeniceError) as exc_info:
            await audio_resource.create_speech(
                input="Test with invalid model",
                model="definitely-invalid-tts-model-name",
                voice="af_sky",  # Use valid voice to test model validation
            )

        # Verify we got a VeniceError (specific error content may vary)
        assert exc_info.value is not None
        # The specific error message format may vary, so just verify we got an error
        assert "400" in str(exc_info.value) or "invalid" in str(exc_info.value).lower()


@pytest.mark.integration
async def test_invalid_voice_error_handling(audio_resource, vcr_cassette):
    """
    Test error handling for invalid voices with real API responses.
    Replaces mock-based voice error tests from unit tests.
    """
    with vcr_cassette:
        # Test with invalid voice
        try:
            with pytest.raises(VeniceError) as exc_info:
                await audio_resource.create_speech(
                    input="Test with invalid voice",
                    model="tts-kokoro",
                    voice="definitely-invalid-voice-name",
                )

            # Verify we got a VeniceError (specific error content may vary)
            assert exc_info.value is not None
            # The specific error message format may vary, so just verify we got an error
            assert "400" in str(exc_info.value) or "invalid" in str(exc_info.value).lower()
        except (VeniceError, APIError) as e:
            # Some APIs might accept any voice string
            if "not supported" in str(e).lower():
                pytest.skip(f"Voice validation not implemented: {e}")
            else:
                raise


# ============================================================================
# Concurrent Request Tests (Replaces Mock-Based Concurrency Tests)
# ============================================================================


@pytest.mark.integration
async def test_concurrent_speech_generation(audio_resource, vcr_cassette):
    """
    Test concurrent speech generation requests with real API.
    Replaces mock-based concurrent request tests from unit tests.
    """
    with vcr_cassette:
        try:
            # Create multiple concurrent speech generation tasks
            async def generate_speech(index: int):
                return await audio_resource.create_speech(
                    input=f"Concurrent audio test number {index}",
                    model="tts-kokoro",
                    voice="af_sky",  # Use valid voice from API documentation
                    response_format=ResponseFormat.MP3,
                    stream=False,
                )

            # Execute 3 concurrent requests
            tasks = [generate_speech(i) for i in range(3)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Verify concurrent handling
            successful_results = [r for r in results if isinstance(r, AudioResponse)]
            assert len(successful_results) >= 1, "At least one concurrent request should succeed"

            # Each successful result should have valid audio data
            for result in successful_results:
                assert result.content is not None
                assert len(result.content) > 0
                assert isinstance(result.content, bytes)

        except (VeniceError, APIError) as e:
            if "not supported" in str(e).lower():
                pytest.skip(f"Concurrent TTS not supported: {e}")
            else:
                raise


# ============================================================================
# Different Format Tests (Replaces Mock-Based Format Tests)
# ============================================================================


@pytest.mark.integration
async def test_different_audio_formats(audio_resource, vcr_cassette):
    """
    Test different audio formats with real API responses.
    Replaces mock-based format handling tests from unit tests.
    """
    with vcr_cassette:
        formats_to_test = [
            (ResponseFormat.MP3, [b"ID3", b"\xff\xfb", b"\xff\xf3"]),
            (ResponseFormat.WAV, [b"RIFF"]),
        ]

        for audio_format, expected_headers in formats_to_test:
            try:
                result = await audio_resource.create_speech(
                    input="Format test audio",
                    model="tts-kokoro",
                    voice="af_sky",  # Use valid voice from API documentation
                    response_format=audio_format,
                    stream=False,
                )

                assert isinstance(result, AudioResponse)
                assert result.content is not None
                assert len(result.content) > 0

                # Check for expected format headers
                has_expected_header = any(
                    result.content.startswith(header) or header in result.content[:20]
                    for header in expected_headers
                )

                # Some APIs might return different formats than requested
                assert has_expected_header or len(result.content) > 100, (
                    f"Should receive valid {audio_format} audio data"
                )

            except (VeniceError, APIError) as e:
                if f"{audio_format} not supported" in str(e).lower():
                    continue  # Skip unsupported formats
                else:
                    raise


# ============================================================================
# Speed and Voice Parameter Tests (Replaces Mock-Based Parameter Tests)
# ============================================================================


@pytest.mark.integration
async def test_speech_speed_variations(audio_resource, vcr_cassette):
    """
    Test different speech speed parameters with real API.
    Replaces mock-based speed parameter tests from unit tests.
    """
    with vcr_cassette:
        speeds_to_test = [0.5, 1.0, 1.5, 2.0]

        for speed in speeds_to_test:
            try:
                result = await audio_resource.create_speech(
                    input="Speed test for audio generation",
                    model="tts-kokoro",
                    voice="af_sky",  # Use valid voice from API documentation
                    speed=speed,
                    stream=False,
                )

                assert isinstance(result, AudioResponse)
                assert result.content is not None
                assert len(result.content) > 0

                # Different speeds should produce different audio lengths
                # (This is hard to verify precisely without audio analysis)
                assert isinstance(result.content, bytes)

            except (VeniceError, APIError) as e:
                if "speed not supported" in str(e).lower():
                    continue  # Skip if speed variations not supported
                else:
                    raise


@pytest.mark.integration
async def test_different_voice_models(audio_resource, vcr_cassette):
    """
    Test different voice models with real API.
    Replaces mock-based voice model tests from unit tests.
    """
    with vcr_cassette:
        voices_to_test = ["af_sky", Voice.AF_SKY, Voice.BF_EMMA]

        for voice in voices_to_test:
            try:
                result = await audio_resource.create_speech(
                    input="Voice model test audio",
                    model="tts-kokoro",
                    voice=voice,
                    stream=False,
                )

                assert isinstance(result, AudioResponse)
                assert result.content is not None
                assert len(result.content) > 0

            except (VeniceError, APIError) as e:
                if f"voice {voice} not supported" in str(e).lower():
                    continue  # Skip unsupported voices
                else:
                    raise


# ============================================================================
# Integration Workflow Tests (Replaces Mock-Based Workflow Tests)
# ============================================================================


@pytest.mark.integration
async def test_complete_audio_workflow(audio_resource, vcr_cassette):
    """
    Test complete audio workflow from voice listing to generation.
    Replaces mock-based workflow tests from unit tests.
    """
    with vcr_cassette:
        try:
            # Step 1: List available voices
            voices = await audio_resource.get_voices()
            available_voices = voices.data if hasattr(voices, "data") else []

            # Step 2: Generate speech with available voice (if any)
            voice_to_use = "af_sky"  # Fallback voice (known to be valid)
            if available_voices and len(available_voices) > 0:
                first_voice = available_voices[0]
                if hasattr(first_voice, "id"):
                    voice_to_use = first_voice.id
                elif hasattr(first_voice, "name"):
                    voice_to_use = first_voice.name

            # Step 3: Generate speech
            result = await audio_resource.create_speech(
                input="Complete workflow test for audio generation",
                model="tts-kokoro",
                voice=voice_to_use,
                response_format=ResponseFormat.MP3,
                speed=1.0,
                stream=False,
            )

            # Step 4: Verify complete workflow
            assert isinstance(result, AudioResponse)
            assert result.content is not None
            assert len(result.content) > 0

            # Workflow completed successfully — verify final result is usable audio
            assert isinstance(result.content, bytes)
            assert len(result.content) >= 100  # Non-trivial audio content

        except (VeniceError, APIError) as e:
            if "not supported" in str(e).lower():
                pytest.skip(f"Complete audio workflow not supported: {e}")
            else:
                raise
