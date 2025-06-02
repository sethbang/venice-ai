"""
Venice AI Audio API resources.

This module provides classes for interacting with the Venice AI Audio API,
supporting speech synthesis operations. The module includes both synchronous
and asynchronous interfaces for audio generation with various voice options
and output formats.

The audio API allows for:
- Converting text to natural-sounding speech (text-to-speech)
- Selecting from multiple voice options for speech synthesis
- Controlling speech speed and output format
- Both full and streaming response modes
"""

import httpx
from typing import Dict, Any, Optional, Union, Iterator, AsyncIterator, TYPE_CHECKING, cast

from .._resource import APIResource, AsyncAPIResource
from venice_ai.types.audio import CreateSpeechRequest, Voice, ResponseFormat

if TYPE_CHECKING:
    from .._client import VeniceClient
    from .._async_client import AsyncVeniceClient


class Audio(APIResource):
    """
    Provides access to text-to-speech (TTS) audio generation operations.
    
    This class handles synchronous audio generation requests, supporting both
    streaming and non-streaming modes. It allows conversion of text to natural-sounding
    speech using various voice models and output formats.
    
    :param client: The Venice AI client instance used for making API requests.
    :type client: VeniceClient
    
    .. note::
        This class is typically accessed through the ``VeniceClient.audio`` property
        rather than being instantiated directly.
    """
    
    def create_speech(
        self,
        *,
        model: str,
        input: str,
        voice: Union[str, Voice],
        response_format: Optional[Union[str, ResponseFormat]] = None,
        speed: Optional[float] = None,
        stream: Optional[bool] = False,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Union[bytes, Iterator[bytes]]:
        """
        Generates audio from input text.
        
        Converts the provided text to speech using the specified model and voice.
        The audio can be returned either as complete binary data or as a stream
        of audio chunks for real-time processing.
        
        :param model: ID of the model to use for speech generation (e.g., "tts-kokoro").
        :type model: str
        :param input: The text to convert to speech. Maximum length varies by model.
        :type input: str
        :param voice: The voice to use for the generated audio. Can be a string literal
            or a :class:`~venice_ai.types.audio.Voice` enum value (e.g., Voice.KOKORO_DEFAULT
            or "kokoro-default").
        :type voice: Union[str, venice_ai.types.audio.Voice]
        :param response_format: The format to return the audio in. Can be a string literal or a
            :class:`~venice_ai.types.audio.ResponseFormat` enum value. Defaults to "mp3".
        :type response_format: Optional[Union[str, venice_ai.types.audio.ResponseFormat]]
        :param speed: The speed of the generated audio. Select a value from 0.25 to 4.0.
            Defaults to 1.0.
        :type speed: Optional[float]
        :param stream: Whether to stream the audio data. If True, returns an Iterator
            of audio chunks. If False, returns the complete audio data. Defaults to False.
        :type stream: Optional[bool]
        :param timeout: Request timeout in seconds or an httpx.Timeout object.
            If not provided, uses the client's default timeout.
        :type timeout: Optional[Union[float, httpx.Timeout]]
        
        :return: If stream is False, returns the audio data as bytes. If stream is True,
            returns an Iterator yielding chunks of audio data as bytes.
        :rtype: Union[bytes, Iterator[bytes]]
        
        :raises venice_ai.exceptions.APIError: If the API request fails.
        :raises ValueError: If the input text is empty or invalid parameters are provided.
        
        Example:
            Basic non-streaming text-to-speech:
            
            .. code-block:: python
            
                from venice_ai import VeniceClient
                from venice_ai.types.audio import Voice, ResponseFormat
                
                client = VeniceClient()
                
                # Generate speech with enum values
                audio_bytes = client.audio.create_speech(
                    model="tts-kokoro",
                    input="Hello, this is a test.",
                    voice=Voice.KOKORO_DEFAULT
                )
                
                # Save to file
                with open("speech.mp3", "wb") as f:
                    f.write(audio_bytes)
                
                # Using string literals and different format
                audio_bytes = client.audio.create_speech(
                    model="tts-kokoro",
                    input="Hello with different settings.",
                    voice="kokoro-default",
                    response_format="wav",
                    speed=1.2
                )
            
            Streaming text-to-speech:
            
            .. code-block:: python
            
                # Stream audio data
                stream = client.audio.create_speech(
                    model="tts-kokoro",
                    input="This is a streamed audio example.",
                    voice="kokoro-default",
                    stream=True
                )
                
                # Write streamed chunks to file
                with open("streamed_speech.mp3", "wb") as f:
                    for chunk in stream:
                        f.write(chunk)
        """
        # Validate input
        if not input:
            raise ValueError("Input text cannot be empty for speech generation")

        # Build the request body
        body: Dict[str, Any] = {
            "model": model,
            "input": input,
            "voice": voice,
        }
        
        # Add optional parameters if they're not None
        if response_format is not None:
            body["response_format"] = response_format
            
        if speed is not None:
            body["speed"] = speed
            
        # Set headers to accept audio
        headers = {"Accept": "audio/*"}
        
        if stream:
            # For streaming, use a hypothetical _stream_request_raw method
            # This would need to be implemented in the client or resource base class
            return cast(Union[bytes, Iterator[bytes]], self._client._stream_request_raw(
                "POST",
                "audio/speech",
                json_data=body,
                headers=headers,
                timeout=timeout
            ))
        else:
            # For non-streaming, use the existing _request method with raw_response=True
            return cast(Union[bytes, Iterator[bytes]], self._client._request(
                "POST",
                "audio/speech",
                json_data=body,
                headers=headers,
                raw_response=True,
                timeout=timeout
            ))


class AsyncAudio(AsyncAPIResource):
    """
    Provides access to text-to-speech (TTS) audio generation operations asynchronously.
    
    This class handles asynchronous audio generation requests, supporting both
    streaming and non-streaming modes. It allows conversion of text to natural-sounding
    speech using various voice models and output formats in async applications.
    
    :param client: The async Venice AI client instance used for making API requests.
    :type client: AsyncVeniceClient
    
    .. note::
        This class is typically accessed through the ``AsyncVeniceClient.audio`` property
        rather than being instantiated directly.
    """
    
    async def create_speech(
        self,
        *,
        model: str,
        input: str,
        voice: Union[str, Voice],
        response_format: Optional[Union[str, ResponseFormat]] = None,
        speed: Optional[float] = None,
        stream: Optional[bool] = False,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Union[bytes, AsyncIterator[bytes]]:
        """
        Generates audio from input text asynchronously.
        
        Converts the provided text to speech using the specified model and voice
        using asynchronous requests. The audio can be returned either as complete
        binary data or as an async stream of audio chunks for real-time processing.
        
        :param model: ID of the model to use for speech generation (e.g., "tts-kokoro").
        :type model: str
        :param input: The text to convert to speech. Maximum length varies by model.
        :type input: str
        :param voice: The voice to use for the generated audio. Can be a string literal
            or a :class:`~venice_ai.types.audio.Voice` enum value (e.g., Voice.KOKORO_DEFAULT
            or "kokoro-default").
        :type voice: Union[str, venice_ai.types.audio.Voice]
        :param response_format: The format to return the audio in. Can be a string literal or a
            :class:`~venice_ai.types.audio.ResponseFormat` enum value. Defaults to "mp3".
        :type response_format: Optional[Union[str, venice_ai.types.audio.ResponseFormat]]
        :param speed: The speed of the generated audio. Select a value from 0.25 to 4.0.
            Defaults to 1.0.
        :type speed: Optional[float]
        :param stream: Whether to stream the audio data. If True, returns an AsyncIterator
            of audio chunks. If False, returns the complete audio data. Defaults to False.
        :type stream: Optional[bool]
        :param timeout: Request timeout in seconds or an httpx.Timeout object.
            If not provided, uses the client's default timeout.
        :type timeout: Optional[Union[float, httpx.Timeout]]
        
        :return: If stream is False, returns the audio data as bytes (awaitable). If stream is True,
            returns an AsyncIterator yielding chunks of audio data as bytes.
        :rtype: Union[bytes, AsyncIterator[bytes]]
        
        :raises venice_ai.exceptions.APIError: If the API request fails.
        :raises ValueError: If the input text is empty or invalid parameters are provided.
        
        Example:
            Basic non-streaming text-to-speech:
            
            .. code-block:: python
            
                import asyncio
                from venice_ai import AsyncVeniceClient
                from venice_ai.types.audio import Voice, ResponseFormat
                
                async def generate_speech():
                    client = AsyncVeniceClient()
                    
                    # Generate speech with enum values
                    audio_bytes = await client.audio.create_speech(
                        model="tts-kokoro",
                        input="Hello, this is a test.",
                        voice=Voice.KOKORO_DEFAULT
                    )
                    
                    # Save to file
                    with open("speech.mp3", "wb") as f:
                        f.write(audio_bytes)
                    
                    # Using string literals and different format
                    audio_bytes = await client.audio.create_speech(
                        model="tts-kokoro",
                        input="Hello with different settings.",
                        voice="kokoro-default",
                        response_format="wav",
                        speed=1.2
                    )
                
                asyncio.run(generate_speech())
            
            Streaming text-to-speech:
            
            .. code-block:: python
            
                async def stream_speech():
                    client = AsyncVeniceClient()
                    
                    # Stream audio data
                    stream = client.audio.create_speech(
                        model="tts-kokoro",
                        input="This is a streamed audio example.",
                        voice="kokoro-default",
                        stream=True
                    )
                    
                    # Write streamed chunks to file
                    with open("streamed_speech.mp3", "wb") as f:
                        async for chunk in stream:
                            f.write(chunk)
                
                asyncio.run(stream_speech())
        """
        # Validate input
        if not input:
            raise ValueError("Input text cannot be empty for speech generation")

        # Build the request body
        body: Dict[str, Any] = {
            "model": model,
            "input": input,
            "voice": voice,
        }
        
        # Add optional parameters if they're not None
        if response_format is not None:
            body["response_format"] = response_format
            
        if speed is not None:
            body["speed"] = speed
            
        # Set headers to accept audio
        headers = {"Accept": "audio/*"}
        
        if stream:
            # For streaming, return the async generator directly (no await)
            # This method returns an AsyncIterator for streaming audio chunks
            return cast(Union[bytes, AsyncIterator[bytes]], self._client._stream_request_raw(
                "POST",
                "audio/speech",
                json_data=body,
                headers=headers,
                timeout=timeout
            ))
        else:
            # For non-streaming, use the existing _request method with raw_response=True
            return cast(Union[bytes, AsyncIterator[bytes]], await self._client._request(
                "POST",
                "audio/speech",
                json_data=body,
                headers=headers,
                raw_response=True,
                timeout=timeout
            ))