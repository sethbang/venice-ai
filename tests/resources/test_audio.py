"""
Tests for Audio resources covering edge cases and branch conditions.
This module specifically focuses on test cases that ensure proper validation
and error handling in the Audio resource classes.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from venice_ai._async_client import AsyncVeniceClient
from venice_ai.resources.audio import AsyncAudio
from venice_ai.types.audio import Voice


@pytest.mark.asyncio
async def test_async_audio_create_speech_empty_input():
    """
    Test that AsyncAudio.create_speech raises a ValueError when given an empty string as input.
    This test verifies the 'if not input:' branch in the create_speech method.
    """
    # Create a mock AsyncVeniceClient
    mock_async_client = MagicMock(spec=AsyncVeniceClient)
    # Set up the post method as an AsyncMock
    mock_async_client._request = AsyncMock()
    
    # Create an AsyncAudio instance with the mock client
    async_audio_resource = AsyncAudio(mock_async_client)
    
    # Test with empty string
    with pytest.raises(ValueError, match="Input text cannot be empty for speech generation"):
        await async_audio_resource.create_speech(
            model="tts-1", 
            input="", 
            voice=Voice.ALLOY
        )
    
    # Verify that the _request method was not called
    mock_async_client._request.assert_not_called()
    
    # Also verify that the _stream_request_raw method was not called (for streaming case)
    if hasattr(mock_async_client, "_stream_request_raw"):
        mock_async_client._stream_request_raw.assert_not_called()