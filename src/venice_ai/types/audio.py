"""
Audio type re-exports for Venice AI SDK.

This module provides convenient imports for audio-related types, delegating to
the canonical locations in the types package hierarchy.

The types are exported from:
- ``venice_ai.types.enums``: Voice and ResponseFormat enums
- ``venice_ai.types.api.audio``: VoiceDetail, VoiceList, AudioResponse

Example usage:
    >>> from venice_ai.types.audio import Voice, ResponseFormat
    >>> voice = Voice.AF_ALLOY
    >>> format = ResponseFormat.MP3
"""

# Re-export audio-related enums from canonical locations
# Re-export audio response types from api module
from .api.audio import AudioResponse, VoiceDetail, VoiceList
from .enums import ResponseFormat, Voice

__all__ = [
    # Enums
    "Voice",
    "ResponseFormat",
    # Response types
    "VoiceDetail",
    "VoiceList",
    "AudioResponse",
]
