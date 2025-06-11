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
from typing import List, Literal, Optional, Dict, Any, Union, Iterator, AsyncIterator, TYPE_CHECKING, cast, overload
from httpx import Response as HttpxResponse

from .._resource import APIResource, AsyncAPIResource
from ..types.audio import Voice, ResponseFormat, VoiceDetail, VoiceList
from ..types.models import ModelList as SDKModelList
from ..exceptions import _make_status_error

if TYPE_CHECKING:
    from .._client import VeniceClient
    from .._async_client import AsyncVeniceClient


REGION_LANGUAGE_MAPPING: Dict[str, Dict[str, str]] = {
    "a": {"language": "English", "accent": "American"},
    "b": {"language": "English", "accent": "British"},
    "c": {"language": "English", "accent": "Canadian"},
    "d": {"language": "German", "accent": "Standard"},
    "e": {"language": "Spanish", "accent": "European Standard"},
    "f": {"language": "French", "accent": "Standard"},
    "g": {"language": "English", "accent": "General"},
    "h": {"language": "English", "accent": "General"}, # Placeholder, can be refined
    "i": {"language": "Italian", "accent": "Standard"},
    "j": {"language": "Japanese", "accent": "Standard"},
    "k": {"language": "Korean", "accent": "Standard"},
    "p": {"language": "Portuguese", "accent": "Standard"},
    "r": {"language": "Russian", "accent": "Standard"},
    "s": {"language": "English", "accent": "Scottish"},
    "u": {"language": "English", "accent": "US"}, # Alternative/Specific US
    "w": {"language": "English", "accent": "Welsh"},
    "x": {"language": "English", "accent": "Australian"},
    "y": {"language": "English", "accent": "Indian"},
    "z": {"language": "Mandarin Chinese", "accent": "Standard"},
}


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
    
    @overload
    def create_speech(
        self,
        *,
        input: str,
        model: str,
        voice: Union[str, Voice],
        response_format: Optional[Union[str, ResponseFormat]] = "mp3",
        speed: Optional[float] = 1.0,
        stream: Literal[False] = False,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> bytes: ...

    @overload
    def create_speech(
        self,
        *,
        input: str,
        model: str,
        voice: Union[str, Voice],
        response_format: Optional[Union[str, ResponseFormat]] = "mp3",
        speed: Optional[float] = 1.0,
        stream: Literal[True],
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Iterator[bytes]: ...

    def create_speech(
        self,
        *,
        input: str,
        model: str,
        voice: Union[str, Voice],
        response_format: Optional[Union[str, ResponseFormat]] = "mp3",
        speed: Optional[float] = 1.0,
        stream: bool = False,
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

        # Build request options
        options = {
            "headers": {"Accept": "audio/*"},
            "body": {
                "input": input,
                "model": model,
                "voice": voice,
                "response_format": response_format,
                "speed": speed,
            },
            "timeout": timeout,
        }

        

        if stream:
            # Use the client's streaming method for raw bytes
            return self._client._stream_request_raw(
                method="POST",
                path="audio/speech",
                json_data=options.get("body"),
                headers=options.get("headers"),
                timeout=options.get("timeout"),
            )
        else:
            # Use the client's regular request method with raw_response=True
            return self._client._request(
                method="POST",
                path="audio/speech",
                json_data=options.get("body"),
                headers=options.get("headers"),
                raw_response=True,
                timeout=options.get("timeout"),
            )

    def get_voices(
        self,
        *,
        model_id: Optional[str] = None,
        gender: Optional[Literal["male", "female", "unknown"]] = None,
        region_code: Optional[str] = None, # e.g., "af", "zm"
    ) -> VoiceList:
        """
        Lists available text-to-speech (TTS) voices, with optional filtering.

        This method retrieves information about available voices for TTS models,
        allowing filtering by model ID, gender, and region code.

        Args:
            model_id: Optional. If provided, only voices for this specific TTS model ID
                will be returned.
            gender: Optional. Filter voices by gender ("male", "female", "unknown").
                Gender is inferred from the voice ID prefix.
            region_code: Optional. Filter voices by the raw two-letter region/language
                prefix from the voice ID (e.g., "af" for American Female-sounding,
                "zm" for Chinese Male-sounding).

        Returns:
            A VoiceList object containing a list of VoiceDetail objects that match
            the filter criteria, along with information about the applied filters.

        Raises:
            venice_ai.exceptions.APIError: If an API error occurs during the request
                to the underlying models endpoint.
        """
        all_voice_details: List[VoiceDetail] = []
        
        # Type hint for clarity, self._client.models is Models resource instance
        sdk_models_list_response: SDKModelList = self._client.models.list(type="tts")

        for model_data in sdk_models_list_response.get("data", []):
            current_model_id = cast(Optional[str], model_data.get("id"))

            if not current_model_id: # Skip if model has no ID
                continue

            # Apply model_id filter if provided
            if model_id is not None and current_model_id != model_id:
                continue

            model_spec = cast(Dict[str, Any], model_data.get("model_spec", {}))
            voice_ids_from_api = cast(List[str], model_spec.get("voices", []))

            for raw_voice_id in voice_ids_from_api:
                parsed_gender: Optional[Literal["male", "female", "unknown"]] = "unknown"
                parsed_region_code: Optional[str] = None
                parsed_language: Optional[str] = None
                parsed_accent: Optional[str] = None

                if "_" in raw_voice_id and len(raw_voice_id.split('_')[0]) >= 2:
                    prefix = raw_voice_id.split('_')[0]
                    parsed_region_code = prefix
                    
                    # Infer gender from the second character of the prefix
                    gender_char = prefix[1:2].lower() # ensure lowercase for comparison
                    if gender_char == 'm':
                        parsed_gender = "male"
                    elif gender_char == 'f':
                        parsed_gender = "female"
                    
                    # Infer language and accent from the first character of the prefix
                    lang_char = prefix[0:1].lower() # ensure lowercase for mapping
                    lang_info = REGION_LANGUAGE_MAPPING.get(lang_char)
                    if lang_info:
                        parsed_language = lang_info["language"]
                        parsed_accent = lang_info["accent"]
                
                # Apply gender filter
                if gender is not None and parsed_gender != gender:
                    continue
                
                # Apply region_code filter
                if region_code is not None and parsed_region_code != region_code:
                    continue

                voice_detail_obj: VoiceDetail = {
                    "id": raw_voice_id,
                    "model_id": current_model_id,
                    "gender": parsed_gender,
                    "region_code": parsed_region_code,
                    "language": parsed_language,
                    "accent": parsed_accent,
                }
                all_voice_details.append(voice_detail_obj)

        return {
            "object": "list",
            "data": all_voice_details,
            "model_id_filter": model_id,
            "gender_filter": gender,
            "region_code_filter": region_code,
        }


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
    
    @overload
    async def create_speech(
        self,
        *,
        input: str,
        model: str,
        voice: Union[str, Voice],
        response_format: Optional[Union[str, ResponseFormat]] = "mp3",
        speed: Optional[float] = 1.0,
        stream: Literal[False] = False,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> bytes: ...

    @overload
    async def create_speech(
        self,
        *,
        input: str,
        model: str,
        voice: Union[str, Voice],
        response_format: Optional[Union[str, ResponseFormat]] = "mp3",
        speed: Optional[float] = 1.0,
        stream: Literal[True],
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> AsyncIterator[bytes]: ...

    async def create_speech(
        self,
        *,
        input: str,
        model: str,
        voice: Union[str, Voice],
        response_format: Optional[Union[str, ResponseFormat]] = "mp3",
        speed: Optional[float] = 1.0,
        stream: bool = False,
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

        # Build request options
        options = {
            "headers": {"Accept": "audio/*"},
            "body": {
                "input": input,
                "model": model,
                "voice": voice,
                "response_format": response_format,
                "speed": speed,
            },
            "timeout": timeout,
        }
        

        if stream:
            # Make a request that returns the raw httpx.Response for streaming
            raw_response: HttpxResponse = await self._arequest_raw_response("POST", "audio/speech", options=options, stream_mode=True)
            
            # Check for errors before attempting to stream
            if raw_response.status_code >= 400:
                await raw_response.aread()  # Consume body to release connection before raising
                raw_response.raise_for_status()

            return raw_response.aiter_bytes(chunk_size=4096)
        else:
            # For non-streaming, get the raw response and return content
            raw_response_non_stream: HttpxResponse = await self._arequest_raw_response("POST", "audio/speech", options=options, stream_mode=False)

            if raw_response_non_stream.status_code >= 400:
                await raw_response_non_stream.aread()  # Ensure the response is read before raising/translating
                # Create an HTTPStatusError to leverage the client's main translation logic
                http_error = httpx.HTTPStatusError(
                    message=f"HTTP {raw_response_non_stream.status_code} error while making API request to {raw_response_non_stream.request.url}",
                    request=raw_response_non_stream.request,
                    response=raw_response_non_stream
                )
                # Use the client's translator.
                # default_request should be the request that led to this error.
                # is_stream is False for this non-streaming path.
                raise await self._client._translate_httpx_error_to_api_error(http_error, default_request=http_error.request, is_stream=False)
                
            # If not an error, it means the request was successful.
            return raw_response_non_stream.content

    async def get_voices(
        self,
        *,
        model_id: Optional[str] = None,
        gender: Optional[Literal["male", "female", "unknown"]] = None,
        region_code: Optional[str] = None, # e.g., "af", "zm"
    ) -> VoiceList:
        """
        Lists available text-to-speech (TTS) voices asynchronously, with optional filtering.

        This method retrieves information about available voices for TTS models,
        allowing filtering by model ID, gender, and region code.

        Args:
            model_id: Optional. If provided, only voices for this specific TTS model ID
                will be returned.
            gender: Optional. Filter voices by gender ("male", "female", "unknown").
                Gender is inferred from the voice ID prefix.
            region_code: Optional. Filter voices by the raw two-letter region/language
                prefix from the voice ID (e.g., "af" for American Female-sounding,
                "zm" for Chinese Male-sounding).

        Returns:
            A VoiceList object containing a list of VoiceDetail objects that match
            the filter criteria, along with information about the applied filters.

        Raises:
            venice_ai.exceptions.APIError: If an API error occurs during the request
                to the underlying models endpoint.
        """
        all_voice_details: List[VoiceDetail] = []

        # Type hint for clarity, self._client.models is AsyncModels resource instance
        sdk_models_list_response: SDKModelList = await self._client.models.list(type="tts")

        for model_data in sdk_models_list_response.get("data", []):
            current_model_id = cast(Optional[str], model_data.get("id"))

            if not current_model_id: # Skip if model has no ID
                continue

            # Apply model_id filter if provided
            if model_id is not None and current_model_id != model_id:
                continue
            
            model_spec = cast(Dict[str, Any], model_data.get("model_spec", {}))
            voice_ids_from_api = cast(List[str], model_spec.get("voices", []))

            for raw_voice_id in voice_ids_from_api:
                parsed_gender: Optional[Literal["male", "female", "unknown"]] = "unknown"
                parsed_region_code: Optional[str] = None
                parsed_language: Optional[str] = None
                parsed_accent: Optional[str] = None

                if "_" in raw_voice_id and len(raw_voice_id.split('_')[0]) >= 2:
                    prefix = raw_voice_id.split('_')[0]
                    parsed_region_code = prefix
                    
                    gender_char = prefix[1:2].lower()
                    if gender_char == 'm':
                        parsed_gender = "male"
                    elif gender_char == 'f':
                        parsed_gender = "female"
                    
                    lang_char = prefix[0:1].lower()
                    lang_info = REGION_LANGUAGE_MAPPING.get(lang_char)
                    if lang_info:
                        parsed_language = lang_info["language"]
                        parsed_accent = lang_info["accent"]
                
                if gender is not None and parsed_gender != gender:
                    continue
                
                if region_code is not None and parsed_region_code != region_code:
                    continue

                voice_detail_obj: VoiceDetail = {
                    "id": raw_voice_id,
                    "model_id": current_model_id,
                    "gender": parsed_gender,
                    "region_code": parsed_region_code,
                    "language": parsed_language,
                    "accent": parsed_accent,
                }
                all_voice_details.append(voice_detail_obj)

        return {
            "object": "list",
            "data": all_voice_details,
            "model_id_filter": model_id,
            "gender_filter": gender,
            "region_code_filter": region_code,
        }