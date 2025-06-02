import pytest
import pytest_asyncio
import os
from venice_ai import VeniceClient, AsyncVeniceClient
from venice_ai.exceptions import VeniceError, InvalidRequestError

# Define default TTS model and voice for testing
DEFAULT_TTS_MODEL = "tts-kokoro"  # Using the specified TTS model
DEFAULT_TTS_VOICE = "am_michael"   # Using the specified voice

# Path to save generated audio for testing
TEST_AUDIO_OUTPUT_PATH = "e2e_tests/output/test_audio.mp3"

# Ensure the output directory exists
os.makedirs(os.path.dirname(TEST_AUDIO_OUTPUT_PATH), exist_ok=True)

# Functional Tests for Audio API

def test_speech_generation_sync(venice_client: VeniceClient):
    """Tests synchronous text-to-speech generation."""
    text = "This is a test of the Venice AI text to speech synthesis system."
    
    # Generate speech
    audio_data = venice_client.audio.create_speech(
        model=DEFAULT_TTS_MODEL,
        voice=DEFAULT_TTS_VOICE,
        input=text
    )
    
    # Check that we got binary data back
    assert isinstance(audio_data, bytes)
    assert len(audio_data) > 0
    
    # Optionally save the audio for manual verification
    with open(TEST_AUDIO_OUTPUT_PATH, "wb") as f:
        f.write(audio_data)
    
    # Check that the file was created and has content
    assert os.path.exists(TEST_AUDIO_OUTPUT_PATH)
    assert os.path.getsize(TEST_AUDIO_OUTPUT_PATH) > 0

@pytest.mark.asyncio
async def test_speech_generation_async(async_venice_client: AsyncVeniceClient):
    """Tests asynchronous text-to-speech generation."""
    text = "This is an async test of the Venice AI text to speech synthesis system."
    
    # Generate speech
    audio_data = await async_venice_client.audio.create_speech(
        model=DEFAULT_TTS_MODEL,
        voice=DEFAULT_TTS_VOICE,
        input=text
    )
    
    # Check that we got binary data back
    assert isinstance(audio_data, bytes)
    assert len(audio_data) > 0
    
    # Optionally save the audio for manual verification
    async_output_path = "e2e_tests/output/test_audio_async.mp3"
    with open(async_output_path, "wb") as f:
        f.write(audio_data)
    
    # Check that the file was created and has content
    assert os.path.exists(async_output_path)
    assert os.path.getsize(async_output_path) > 0

def test_speech_generation_with_parameters_sync(venice_client: VeniceClient):
    """Tests synchronous TTS with various parameters."""
    text = "This test includes additional parameters for speech synthesis."
    
    # Generate speech with parameters
    audio_data = venice_client.audio.create_speech(
        model=DEFAULT_TTS_MODEL,
        voice=DEFAULT_TTS_VOICE,
        input=text,
        speed=1.2,        # Slightly faster than normal
        response_format="mp3"  # Specify format explicitly
    )
    
    # Check that we got binary data back
    assert isinstance(audio_data, bytes)
    assert len(audio_data) > 0
    
    # Save with parameters in filename for manual verification
    params_output_path = "e2e_tests/output/test_audio_params.mp3"
    with open(params_output_path, "wb") as f:
        f.write(audio_data)
    
    assert os.path.exists(params_output_path)
    assert os.path.getsize(params_output_path) > 0

@pytest.mark.asyncio
async def test_speech_generation_with_parameters_async(async_venice_client: AsyncVeniceClient):
    """Tests asynchronous TTS with various parameters."""
    text = "This async test includes additional parameters for speech synthesis."
    
    # Generate speech with parameters
    audio_data = await async_venice_client.audio.create_speech(
        model=DEFAULT_TTS_MODEL,
        voice=DEFAULT_TTS_VOICE,
        input=text,
        speed=0.8,        # Slower than normal
        response_format="mp3"
    )
    
    # Check that we got binary data back
    assert isinstance(audio_data, bytes)
    assert len(audio_data) > 0
    
    # Save for manual verification
    params_async_output_path = "e2e_tests/output/test_audio_params_async.mp3"
    with open(params_async_output_path, "wb") as f:
        f.write(audio_data)
    
    assert os.path.exists(params_async_output_path)
    assert os.path.getsize(params_async_output_path) > 0

def test_speech_generation_long_text_sync(venice_client: VeniceClient):
    """Tests synchronous TTS with longer text."""
    # Create a longer text to test TTS limits
    long_text = " ".join(["This is paragraph number " + str(i) + " in our test." for i in range(10)])
    long_text += " We are testing the capability to handle longer texts in the TTS system."
    
    # Generate speech
    audio_data = venice_client.audio.create_speech(
        model=DEFAULT_TTS_MODEL,
        voice=DEFAULT_TTS_VOICE,
        input=long_text
    )
    
    # Check that we got binary data back
    assert isinstance(audio_data, bytes)
    assert len(audio_data) > 0
    
    # Save for manual verification
    long_output_path = "e2e_tests/output/test_audio_long.mp3"
    with open(long_output_path, "wb") as f:
        f.write(audio_data)
    
    assert os.path.exists(long_output_path)
    assert os.path.getsize(long_output_path) > 0
    # The longer text should produce a larger file
    assert os.path.getsize(long_output_path) > os.path.getsize(TEST_AUDIO_OUTPUT_PATH)

def test_speech_generation_error_empty_input_sync(venice_client: VeniceClient):
    """Tests error handling for empty input text."""
    with pytest.raises(ValueError) as excinfo:
        venice_client.audio.create_speech(
            model=DEFAULT_TTS_MODEL,
            voice=DEFAULT_TTS_VOICE,
            input=""  # Empty input should raise an error
        )
    
    assert excinfo.value is not None
    # Check error message if API provides specific messages
    # assert "input cannot be empty" in str(excinfo.value).lower()

def test_speech_generation_unsupported_format_sync(venice_client: VeniceClient):
    """Tests error handling for unsupported format."""
    with pytest.raises(Exception) as excinfo:
        venice_client.audio.create_speech(
            model=DEFAULT_TTS_MODEL,
            voice=DEFAULT_TTS_VOICE,
            input="This is a test",
            response_format="invalid_format"  # Invalid format should raise an error
        )
    
    assert excinfo.value is not None