"""
Comprehensive unit tests for Audio API resources.

This module provides complete test coverage for the Audio API resources,
including both synchronous and asynchronous versions of speech generation
functionality. Tests cover various scenarios including successful operations,
error handling, parameter validation, and streaming responses.
"""

import pytest
import httpx
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Iterator, AsyncIterator

from venice_ai._client import VeniceClient
from venice_ai._async_client import AsyncVeniceClient
from venice_ai.resources.audio import Audio, AsyncAudio
from venice_ai.types.audio import Voice, ResponseFormat, VoiceList
from venice_ai.exceptions import APIError


class TestAudioSpeech:
    """Test cases for synchronous Audio.create_speech method."""

    def test_create_speech_success_non_streaming(self):
        """Test successful speech generation without streaming."""
        # Create mock client
        mock_client = MagicMock(spec=VeniceClient)
        
        # Mock the _request method to return bytes directly
        mock_client._request.return_value = b"audio_content_bytes"
        
        # Create Audio resource
        audio_resource = Audio(mock_client)
        
        # Call create_speech
        result = audio_resource.create_speech(
            model="tts-kokoro",
            input="Hello, world!",
            voice=Voice.ALLOY,
            response_format=ResponseFormat.MP3,
            speed=1.0
        )
        
        # Verify result
        assert result == b"audio_content_bytes"
        
        # Verify the request was made correctly
        mock_client._request.assert_called_once_with(
            method="POST",
            path="audio/speech",
            json_data={
                "input": "Hello, world!",
                "model": "tts-kokoro",
                "voice": Voice.ALLOY,
                "response_format": ResponseFormat.MP3,
                "speed": 1.0,
            },
            headers={"Accept": "audio/*"},
            raw_response=True,
            timeout=None,
        )

    def test_create_speech_success_streaming(self):
        """Test successful speech generation with streaming."""
        # Create mock client
        mock_client = MagicMock(spec=VeniceClient)
        
        # Mock the _stream_request_raw method to return an iterator
        mock_client._stream_request_raw.return_value = iter([b"chunk1", b"chunk2", b"chunk3"])
        
        # Create Audio resource
        audio_resource = Audio(mock_client)
        
        # Call create_speech with streaming
        result = audio_resource.create_speech(
            model="tts-kokoro",
            input="Hello, streaming world!",
            voice="af_alloy",
            response_format="wav",
            speed=1.5,
            stream=True
        )
        
        # Verify result is an iterator
        assert isinstance(result, Iterator)
        
        # Consume the iterator and verify chunks
        chunks = list(result)
        assert chunks == [b"chunk1", b"chunk2", b"chunk3"]
        
        # Verify the request was made correctly
        mock_client._stream_request_raw.assert_called_once_with(
            method="POST",
            path="audio/speech",
            json_data={
                "input": "Hello, streaming world!",
                "model": "tts-kokoro",
                "voice": "af_alloy",
                "response_format": "wav",
                "speed": 1.5,
            },
            headers={"Accept": "audio/*"},
            timeout=None,
        )

    def test_create_speech_with_different_voices(self):
        """Test speech generation with different voice options."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client._request.return_value = b"audio_content"
        
        audio_resource = Audio(mock_client)
        
        # Test with Voice enum
        audio_resource.create_speech(
            model="tts-kokoro",
            input="Test with enum voice",
            voice=Voice.NOVA
        )
        
        # Test with string voice
        audio_resource.create_speech(
            model="tts-kokoro",
            input="Test with string voice",
            voice="am_onyx"
        )
        
        # Verify both calls were made
        assert mock_client._request.call_count == 2

    def test_create_speech_with_different_formats(self):
        """Test speech generation with different response formats."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client._request.return_value = b"audio_content"
        
        audio_resource = Audio(mock_client)
        
        # Test different formats
        formats = [ResponseFormat.MP3, ResponseFormat.WAV, ResponseFormat.FLAC, "aac", "opus"]
        
        for fmt in formats:
            audio_resource.create_speech(
                model="tts-kokoro",
                input="Test format",
                voice=Voice.ALLOY,
                response_format=fmt
            )
        
        assert mock_client._request.call_count == len(formats)

    def test_create_speech_with_speed_variations(self):
        """Test speech generation with different speed values."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client._request.return_value = b"audio_content"
        
        audio_resource = Audio(mock_client)
        
        # Test different speed values
        speeds = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0]
        
        for speed in speeds:
            audio_resource.create_speech(
                model="tts-kokoro",
                input="Test speed",
                voice=Voice.ALLOY,
                speed=speed
            )
        
        assert mock_client._request.call_count == len(speeds)

    def test_create_speech_with_timeout(self):
        """Test speech generation with custom timeout."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client._request.return_value = b"audio_content"
        
        audio_resource = Audio(mock_client)
        
        # Test with timeout
        timeout = httpx.Timeout(30.0)
        audio_resource.create_speech(
            model="tts-kokoro",
            input="Test with timeout",
            voice=Voice.ALLOY,
            timeout=timeout
        )
        
        # Verify timeout was passed correctly
        call_args = mock_client._request.call_args
        assert call_args[1]["timeout"] == timeout

    def test_create_speech_empty_input_validation(self):
        """Test that empty input raises ValueError."""
        mock_client = MagicMock(spec=VeniceClient)
        audio_resource = Audio(mock_client)
        
        with pytest.raises(ValueError, match="Input text cannot be empty for speech generation"):
            audio_resource.create_speech(
                model="tts-kokoro",
                input="",
                voice=Voice.ALLOY
            )

    def test_create_speech_error_handling_non_streaming(self):
        """Test error handling for non-streaming requests."""
        mock_client = MagicMock(spec=VeniceClient)
        
        # Mock the _request method to raise an APIError
        mock_response = MagicMock()
        mock_client._request.side_effect = APIError("API Error", response=mock_response)
        
        audio_resource = Audio(mock_client)
        
        with pytest.raises(APIError):
            audio_resource.create_speech(
                model="tts-kokoro",
                input="Test error",
                voice=Voice.ALLOY
            )

    def test_create_speech_error_handling_streaming(self):
        """Test error handling for streaming requests."""
        mock_client = MagicMock(spec=VeniceClient)
        
        # Mock the _stream_request_raw method to raise an HTTPStatusError
        mock_client._stream_request_raw.side_effect = httpx.HTTPStatusError("Unauthorized", request=MagicMock(), response=MagicMock())
        
        audio_resource = Audio(mock_client)
        
        with pytest.raises(httpx.HTTPStatusError):
            audio_resource.create_speech(
                model="tts-kokoro",
                input="Test streaming error",
                voice=Voice.ALLOY,
                stream=True
            )


class TestAudioGetVoices:
    """Test cases for synchronous Audio.get_voices method."""

    def test_get_voices_success_no_filters(self):
        """Test successful retrieval of voices without filters."""
        mock_client = MagicMock(spec=VeniceClient)
        
        # Mock the models.list response
        mock_models_response = {
            "data": [
                {
                    "id": "tts-kokoro",
                    "model_spec": {
                        "voices": ["af_alloy", "am_onyx", "bf_alice", "zm_yunjian"]
                    }
                }
            ]
        }
        mock_models = MagicMock()
        mock_models.list.return_value = mock_models_response
        mock_client.models = mock_models
        
        audio_resource = Audio(mock_client)
        result = audio_resource.get_voices()
        
        # Verify result structure
        assert result["object"] == "list"
        assert len(result["data"]) == 4
        assert result["model_id_filter"] is None
        assert result["gender_filter"] is None
        assert result["region_code_filter"] is None
        
        # Verify voice details
        voice_ids = [voice["id"] for voice in result["data"]]
        assert "af_alloy" in voice_ids
        assert "am_onyx" in voice_ids
        assert "bf_alice" in voice_ids
        assert "zm_yunjian" in voice_ids

    def test_get_voices_with_model_filter(self):
        """Test voice retrieval with model ID filter."""
        mock_client = MagicMock(spec=VeniceClient)
        
        mock_models_response = {
            "data": [
                {
                    "id": "tts-kokoro",
                    "model_spec": {
                        "voices": ["af_alloy", "am_onyx"]
                    }
                },
                {
                    "id": "tts-other",
                    "model_spec": {
                        "voices": ["bf_alice"]
                    }
                }
            ]
        }
        mock_models = MagicMock()
        mock_models.list.return_value = mock_models_response
        mock_client.models = mock_models
        
        audio_resource = Audio(mock_client)
        result = audio_resource.get_voices(model_id="tts-kokoro")
        
        # Should only return voices from tts-kokoro model
        assert len(result["data"]) == 2
        assert result["model_id_filter"] == "tts-kokoro"
        
        voice_ids = [voice["id"] for voice in result["data"]]
        assert "af_alloy" in voice_ids
        assert "am_onyx" in voice_ids
        assert "bf_alice" not in voice_ids

    def test_get_voices_with_gender_filter(self):
        """Test voice retrieval with gender filter."""
        mock_client = MagicMock(spec=VeniceClient)
        
        mock_models_response = {
            "data": [
                {
                    "id": "tts-kokoro",
                    "model_spec": {
                        "voices": ["af_alloy", "am_onyx", "bf_alice", "bm_daniel"]
                    }
                }
            ]
        }
        mock_models = MagicMock()
        mock_models.list.return_value = mock_models_response
        mock_client.models = mock_models
        
        audio_resource = Audio(mock_client)
        
        # Test female filter
        result_female = audio_resource.get_voices(gender="female")
        assert len(result_female["data"]) == 2
        assert result_female["gender_filter"] == "female"
        
        female_voices = [voice["id"] for voice in result_female["data"]]
        assert "af_alloy" in female_voices
        assert "bf_alice" in female_voices
        
        # Test male filter
        result_male = audio_resource.get_voices(gender="male")
        assert len(result_male["data"]) == 2
        assert result_male["gender_filter"] == "male"
        
        male_voices = [voice["id"] for voice in result_male["data"]]
        assert "am_onyx" in male_voices
        assert "bm_daniel" in male_voices

    def test_get_voices_with_region_filter(self):
        """Test voice retrieval with region code filter."""
        mock_client = MagicMock(spec=VeniceClient)
        
        mock_models_response = {
            "data": [
                {
                    "id": "tts-kokoro",
                    "model_spec": {
                        "voices": ["af_alloy", "am_onyx", "bf_alice", "zm_yunjian"]
                    }
                }
            ]
        }
        mock_models = MagicMock()
        mock_models.list.return_value = mock_models_response
        mock_client.models = mock_models
        
        audio_resource = Audio(mock_client)
        
        # Test American voices (prefix 'af')
        result_american = audio_resource.get_voices(region_code="af")
        assert len(result_american["data"]) == 1
        assert result_american["region_code_filter"] == "af"
        assert result_american["data"][0]["id"] == "af_alloy"

    def test_get_voices_voice_detail_parsing(self):
        """Test that voice details are parsed correctly."""
        mock_client = MagicMock(spec=VeniceClient)
        
        mock_models_response = {
            "data": [
                {
                    "id": "tts-kokoro",
                    "model_spec": {
                        "voices": ["af_alloy", "zm_yunjian"]
                    }
                }
            ]
        }
        mock_models = MagicMock()
        mock_models.list.return_value = mock_models_response
        mock_client.models = mock_models
        
        audio_resource = Audio(mock_client)
        result = audio_resource.get_voices()
        
        # Find the American female voice
        af_voice = next(voice for voice in result["data"] if voice["id"] == "af_alloy")
        assert af_voice["model_id"] == "tts-kokoro"
        assert af_voice["gender"] == "female"
        assert af_voice["region_code"] == "af"
        assert af_voice["language"] == "English"
        assert af_voice["accent"] == "American"
        
        # Find the Chinese male voice
        zm_voice = next(voice for voice in result["data"] if voice["id"] == "zm_yunjian")
        assert zm_voice["model_id"] == "tts-kokoro"
        assert zm_voice["gender"] == "male"
        assert zm_voice["region_code"] == "zm"
        assert zm_voice["language"] == "Mandarin Chinese"
        assert zm_voice["accent"] == "Standard"


class TestAsyncAudioSpeech:
    """Test cases for asynchronous AsyncAudio.create_speech method."""

    @pytest.mark.asyncio
    @patch.object(AsyncAudio, '_arequest_raw_response')
    async def test_async_create_speech_success_non_streaming(self, mock_arequest_raw_response):
        """Test successful async speech generation without streaming."""
        # Create mock client
        mock_client = MagicMock(spec=AsyncVeniceClient)
        
        # Create mock response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.content = b"async_audio_content_bytes"
        
        # Configure the mock
        mock_arequest_raw_response.return_value = mock_response
        
        # Create AsyncAudio resource
        audio_resource = AsyncAudio(mock_client)
        
        # Call create_speech
        result = await audio_resource.create_speech(
            model="tts-kokoro",
            input="Hello, async world!",
            voice=Voice.NOVA,
            response_format=ResponseFormat.WAV,
            speed=1.2
        )
        
        # Verify result
        assert result == b"async_audio_content_bytes"
        
        # Verify the request was made correctly
        mock_arequest_raw_response.assert_called_once_with(
            "POST",
            "audio/speech",
            options={
                "headers": {"Accept": "audio/*"},
                "body": {
                    "input": "Hello, async world!",
                    "model": "tts-kokoro",
                    "voice": Voice.NOVA,
                    "response_format": ResponseFormat.WAV,
                    "speed": 1.2,
                },
                "timeout": None,
            },
            stream_mode=False
        )

    @pytest.mark.asyncio
    @patch.object(AsyncAudio, '_arequest_raw_response')
    async def test_async_create_speech_success_streaming(self, mock_arequest_raw_response):
        """Test successful async speech generation with streaming."""
        # Create mock client
        mock_client = MagicMock(spec=AsyncVeniceClient)
        
        # Create mock response with aiter_bytes method
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        
        async def mock_aiter_bytes(chunk_size):
            for chunk in [b"async_chunk1", b"async_chunk2", b"async_chunk3"]:
                yield chunk
        
        mock_response.aiter_bytes = mock_aiter_bytes
        
        # Configure the mock
        mock_arequest_raw_response.return_value = mock_response
        
        # Create AsyncAudio resource
        audio_resource = AsyncAudio(mock_client)
        
        # Call create_speech with streaming
        result = await audio_resource.create_speech(
            model="tts-kokoro",
            input="Hello, async streaming world!",
            voice="bm_daniel",
            response_format="flac",
            speed=0.8,
            stream=True
        )
        
        # Verify result is an async iterator
        assert hasattr(result, '__aiter__')
        
        # Consume the async iterator and verify chunks
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        
        assert chunks == [b"async_chunk1", b"async_chunk2", b"async_chunk3"]
        
        # Verify the request was made correctly
        mock_arequest_raw_response.assert_called_once_with(
            "POST",
            "audio/speech",
            options={
                "headers": {"Accept": "audio/*"},
                "body": {
                    "input": "Hello, async streaming world!",
                    "model": "tts-kokoro",
                    "voice": "bm_daniel",
                    "response_format": "flac",
                    "speed": 0.8,
                },
                "timeout": None,
            },
            stream_mode=True
        )

    @pytest.mark.asyncio
    async def test_async_create_speech_empty_input_validation(self):
        """Test that empty input raises ValueError in async version."""
        mock_client = MagicMock(spec=AsyncVeniceClient)
        audio_resource = AsyncAudio(mock_client)
        
        with pytest.raises(ValueError, match="Input text cannot be empty for speech generation"):
            await audio_resource.create_speech(
                model="tts-kokoro",
                input="",
                voice=Voice.ALLOY
            )

    @pytest.mark.asyncio
    @patch.object(AsyncAudio, '_arequest_raw_response')
    async def test_async_create_speech_error_handling_non_streaming(self, mock_arequest_raw_response):
        """Test error handling for async non-streaming requests."""
        mock_client = MagicMock(spec=AsyncVeniceClient)
        
        # Create mock response with error status
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.request = MagicMock()
        mock_response.aread = AsyncMock()
        
        # Mock the error translation method
        mock_client._translate_httpx_error_to_api_error = AsyncMock(side_effect=APIError("Async API Error", response=mock_response))
        mock_arequest_raw_response.return_value = mock_response
        
        audio_resource = AsyncAudio(mock_client)
        
        with pytest.raises(APIError):
            await audio_resource.create_speech(
                model="tts-kokoro",
                input="Test async error",
                voice=Voice.ALLOY
            )
        
        # Verify response was read before error translation
        mock_response.aread.assert_called_once()

    @pytest.mark.asyncio
    @patch.object(AsyncAudio, '_arequest_raw_response')
    async def test_async_create_speech_error_handling_streaming(self, mock_arequest_raw_response):
        """Test error handling for async streaming requests."""
        mock_client = MagicMock(spec=AsyncVeniceClient)
        
        # Create mock response with error status
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 403
        mock_response.aread = AsyncMock()
        mock_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError("Forbidden", request=MagicMock(), response=mock_response))
        
        mock_arequest_raw_response.return_value = mock_response
        
        audio_resource = AsyncAudio(mock_client)
        
        with pytest.raises(httpx.HTTPStatusError):
            await audio_resource.create_speech(
                model="tts-kokoro",
                input="Test async streaming error",
                voice=Voice.ALLOY,
                stream=True
            )
        
        # Verify response was read before raising
        mock_response.aread.assert_called_once()


class TestAsyncAudioGetVoices:
    """Test cases for asynchronous AsyncAudio.get_voices method."""

    @pytest.mark.asyncio
    async def test_async_get_voices_success(self):
        """Test successful async retrieval of voices."""
        mock_client = MagicMock(spec=AsyncVeniceClient)
        
        # Mock the models.list response
        mock_models_response = {
            "data": [
                {
                    "id": "tts-kokoro",
                    "model_spec": {
                        "voices": ["af_alloy", "am_onyx"]
                    }
                }
            ]
        }
        mock_models = MagicMock()
        mock_models.list = AsyncMock(return_value=mock_models_response)
        mock_client.models = mock_models
        
        audio_resource = AsyncAudio(mock_client)
        result = await audio_resource.get_voices()
        
        # Verify result structure
        assert result["object"] == "list"
        assert len(result["data"]) == 2
        assert result["model_id_filter"] is None
        assert result["gender_filter"] is None
        assert result["region_code_filter"] is None
        
        # Verify voice details
        voice_ids = [voice["id"] for voice in result["data"]]
        assert "af_alloy" in voice_ids
        assert "am_onyx" in voice_ids

    @pytest.mark.asyncio
    async def test_async_get_voices_with_filters(self):
        """Test async voice retrieval with multiple filters."""
        mock_client = MagicMock(spec=AsyncVeniceClient)
        
        mock_models_response = {
            "data": [
                {
                    "id": "tts-kokoro",
                    "model_spec": {
                        "voices": ["af_alloy", "am_onyx", "bf_alice"]
                    }
                }
            ]
        }
        mock_models = MagicMock()
        mock_models.list = AsyncMock(return_value=mock_models_response)
        mock_client.models = mock_models
        
        audio_resource = AsyncAudio(mock_client)
        result = await audio_resource.get_voices(
            model_id="tts-kokoro",
            gender="female",
            region_code="af"
        )
        
        # Should only return af_alloy (American female from tts-kokoro)
        assert len(result["data"]) == 1
        assert result["data"][0]["id"] == "af_alloy"
        assert result["model_id_filter"] == "tts-kokoro"
        assert result["gender_filter"] == "female"
        assert result["region_code_filter"] == "af"


class TestAudioEdgeCases:
    """Test edge cases and boundary conditions for Audio resources."""

    def test_get_voices_with_invalid_voice_format(self):
        """Test handling of voices with invalid format."""
        mock_client = MagicMock(spec=VeniceClient)
        
        mock_models_response = {
            "data": [
                {
                    "id": "tts-kokoro",
                    "model_spec": {
                        "voices": ["invalid_voice", "a_short", "af_alloy"]  # Invalid formats
                    }
                }
            ]
        }
        mock_models = MagicMock()
        mock_models.list.return_value = mock_models_response
        mock_client.models = mock_models
        
        audio_resource = Audio(mock_client)
        result = audio_resource.get_voices()
        
        # Should handle invalid voices gracefully
        assert len(result["data"]) == 3
        
        # Find the properly formatted voice
        valid_voice = next(voice for voice in result["data"] if voice["id"] == "af_alloy")
        assert valid_voice["gender"] == "female"
        
        # Invalid voices should have unknown gender and region_code set to the prefix
        invalid_voice = next(voice for voice in result["data"] if voice["id"] == "invalid_voice")
        assert invalid_voice["gender"] == "unknown"  # Second char 'n' is not 'm' or 'f'
        assert invalid_voice["region_code"] == "invalid"  # The prefix is still extracted
        assert invalid_voice["language"] == "Italian"  # First char 'i' maps to Italian
        assert invalid_voice["accent"] == "Standard"  # From the mapping

    def test_get_voices_with_no_models(self):
        """Test handling when no models are returned."""
        mock_client = MagicMock(spec=VeniceClient)
        
        mock_models_response = {"data": []}
        mock_models = MagicMock()
        mock_models.list.return_value = mock_models_response
        mock_client.models = mock_models
        
        audio_resource = Audio(mock_client)
        result = audio_resource.get_voices()
        
        # Should return empty list
        assert result["object"] == "list"
        assert len(result["data"]) == 0

    def test_get_voices_with_model_without_id(self):
        """Test handling of models without ID."""
        mock_client = MagicMock(spec=VeniceClient)
        
        mock_models_response = {
            "data": [
                {
                    # No "id" field
                    "model_spec": {
                        "voices": ["af_alloy"]
                    }
                },
                {
                    "id": "tts-kokoro",
                    "model_spec": {
                        "voices": ["am_onyx"]
                    }
                }
            ]
        }
        mock_models = MagicMock()
        mock_models.list.return_value = mock_models_response
        mock_client.models = mock_models
        
        audio_resource = Audio(mock_client)
        result = audio_resource.get_voices()
        
        # Should only include voices from models with valid IDs
        assert len(result["data"]) == 1
        assert result["data"][0]["id"] == "am_onyx"

    def test_get_voices_with_model_without_voices(self):
        """Test handling of models without voices."""
        mock_client = MagicMock(spec=VeniceClient)
        
        mock_models_response = {
            "data": [
                {
                    "id": "tts-kokoro",
                    "model_spec": {}  # No "voices" field
                },
                {
                    "id": "tts-other",
                    "model_spec": {
                        "voices": ["af_alloy"]
                    }
                }
            ]
        }
        mock_models = MagicMock()
        mock_models.list.return_value = mock_models_response
        mock_client.models = mock_models
        
        audio_resource = Audio(mock_client)
        result = audio_resource.get_voices()
        
        # Should only include voices from models that have voices
        assert len(result["data"]) == 1
        assert result["data"][0]["id"] == "af_alloy"