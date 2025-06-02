import pytest
import pytest_asyncio
import os
from typing import AsyncIterator, cast
from venice_ai import VeniceClient, AsyncVeniceClient
from venice_ai.types.audio import Voice, ResponseFormat
from venice_ai.exceptions import VeniceError, InvalidRequestError

# Define default TTS model for testing
DEFAULT_TTS_MODEL = "tts-kokoro"

# Path to save generated audio for testing
TEST_AUDIO_STREAM_OUTPUT_PATH = "e2e_tests/output/test_audio_stream.mp3"
TEST_AUDIO_VOICE_OUTPUT_PATH = "e2e_tests/output/test_audio_voice_{}.mp3"
TEST_AUDIO_FORMAT_OUTPUT_PATH = "e2e_tests/output/test_audio_format_{}.{}"

# Ensure the output directory exists
os.makedirs(os.path.dirname(TEST_AUDIO_STREAM_OUTPUT_PATH), exist_ok=True)

# Advanced Audio Tests focusing on streaming, voice variations, formats, and edge cases

def test_speech_generation_streaming_sync(venice_client: VeniceClient):
    """Tests synchronous text-to-speech generation with streaming."""
    text = "This is a test of streaming audio with Venice AI text to speech."
    
    # Generate speech with streaming
    audio_chunks = venice_client.audio.create_speech(
        model=DEFAULT_TTS_MODEL,
        voice=Voice.AM_MICHAEL,
        input=text,
        stream=True
    )
    
    # Collect chunks and verify data
    full_audio = b""
    chunk_count = 0
    for chunk in audio_chunks:
        assert isinstance(chunk, bytes)
        assert len(chunk) > 0
        full_audio += chunk
        chunk_count += 1
    
    assert chunk_count > 1, "Streaming should return multiple chunks"
    assert len(full_audio) > 0
    
    # Save the combined audio for manual verification
    with open(TEST_AUDIO_STREAM_OUTPUT_PATH, "wb") as f:
        f.write(full_audio)
    
    assert os.path.exists(TEST_AUDIO_STREAM_OUTPUT_PATH)
    assert os.path.getsize(TEST_AUDIO_STREAM_OUTPUT_PATH) > 0

@pytest.mark.asyncio
async def test_speech_generation_streaming_async(async_venice_client: AsyncVeniceClient):
    """Tests asynchronous text-to-speech generation with streaming."""
    text = "This is an async test of streaming audio with Venice AI text to speech."
    
    # Generate speech with streaming
    audio_chunks = await async_venice_client.audio.create_speech(
        model=DEFAULT_TTS_MODEL,
        voice=Voice.AM_MICHAEL,
        input=text,
        stream=True,
        timeout=120.0
    )
    # Type annotation to help Pylance understand this is an AsyncIterator when stream=True
    audio_chunks = cast(AsyncIterator[bytes], audio_chunks)
    
    # Collect chunks and verify data
    full_audio = b""
    chunk_count = 0
    async for chunk in audio_chunks:
        assert isinstance(chunk, bytes)
        assert len(chunk) > 0
        full_audio += chunk
        chunk_count += 1
    
    assert chunk_count > 1, "Streaming should return multiple chunks"
    assert len(full_audio) > 0
    
    # Save the combined audio for manual verification
    async_stream_output_path = "e2e_tests/output/test_audio_stream_async.mp3"
    with open(async_stream_output_path, "wb") as f:
        f.write(full_audio)
    
    assert os.path.exists(async_stream_output_path)
    assert os.path.getsize(async_stream_output_path) > 0

def test_speech_generation_different_voices_sync(venice_client: VeniceClient):
    """Tests synchronous TTS with different voice options."""
    text = "Testing different voices with Venice AI."
    voices = [Voice.AM_MICHAEL, Voice.AM_SANTA, Voice.AM_LIAM]
    
    for voice in voices:
        audio_data = venice_client.audio.create_speech(
            model=DEFAULT_TTS_MODEL,
            voice=voice,
            input=text
        )
        
        assert isinstance(audio_data, bytes)
        assert len(audio_data) > 0
        
        # Save with voice name in filename for manual verification
        voice_name = voice.value.replace(":", "_")
        output_path = TEST_AUDIO_VOICE_OUTPUT_PATH.format(voice_name)
        with open(output_path, "wb") as f:
            f.write(audio_data)
        
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

def test_speech_generation_different_formats_sync(venice_client: VeniceClient):
    """Tests synchronous TTS with different response formats."""
    text = "Testing different audio formats with Venice AI."
    formats = {
        ResponseFormat.MP3: "mp3",
        ResponseFormat.AAC: "aac",
        ResponseFormat.FLAC: "flac"
    }
    
    for format_enum, extension in formats.items():
        audio_data = venice_client.audio.create_speech(
            model=DEFAULT_TTS_MODEL,
            voice=Voice.AM_MICHAEL,
            input=text,
            response_format=format_enum
        )
        
        assert isinstance(audio_data, bytes)
        assert len(audio_data) > 0
        
        # Save with format in filename for manual verification
        output_path = TEST_AUDIO_FORMAT_OUTPUT_PATH.format(format_enum.value, extension)
        with open(output_path, "wb") as f:
            f.write(audio_data)
        
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

def test_speech_generation_speed_edge_cases_sync(venice_client: VeniceClient):
    """Tests synchronous TTS with edge case speed values."""
    text = "Testing speed edge cases with Venice AI."
    speeds = [0.25, 4.0]  # Minimum and maximum allowed speeds
    
    for speed in speeds:
        audio_data = venice_client.audio.create_speech(
            model=DEFAULT_TTS_MODEL,
            voice=Voice.AM_MICHAEL,
            input=text,
            speed=speed
        )
        
        assert isinstance(audio_data, bytes)
        assert len(audio_data) > 0

def test_speech_generation_speed_invalid_sync(venice_client: VeniceClient):
    """Tests error handling for invalid speed values."""
    text = "This test should fail due to invalid speed."
    invalid_speeds = [0.1, 5.0]  # Outside allowed range
    
    for speed in invalid_speeds:
        with pytest.raises(InvalidRequestError) as excinfo:
            venice_client.audio.create_speech(
                model=DEFAULT_TTS_MODEL,
                voice=Voice.AM_MICHAEL,
                input=text,
                speed=speed
            )
        assert excinfo.value is not None