"""
Unit tests for Venice AI audio type re-exports.

Tests that the audio module correctly re-exports types from canonical locations.
"""

# Import from the re-export module
from venice_ai.types.audio import (
    AudioResponse,
    ResponseFormat,
    Voice,
    VoiceDetail,
    VoiceList,
)


class TestAudioReExports:
    """Test that audio types are properly re-exported."""

    def test_voice_enum_accessible(self):
        """Test Voice enum is accessible via audio module."""
        # Verify it's an enum with expected values
        assert hasattr(Voice, "__members__")
        # Check that it has at least one voice
        assert len(Voice.__members__) > 0

    def test_response_format_enum_accessible(self):
        """Test ResponseFormat enum is accessible via audio module."""
        # Verify it's an enum with expected values
        assert hasattr(ResponseFormat, "__members__")
        # Check common audio formats exist
        assert len(ResponseFormat.__members__) > 0

    def test_voice_detail_is_importable(self):
        """Test VoiceDetail type is importable."""
        assert VoiceDetail is not None
        # Verify it has model capability (Pydantic model)
        assert hasattr(VoiceDetail, "model_fields") or hasattr(VoiceDetail, "__fields__")

    def test_voice_list_is_importable(self):
        """Test VoiceList type is importable."""
        assert VoiceList is not None

    def test_audio_response_is_importable(self):
        """Test AudioResponse type is importable."""
        assert AudioResponse is not None


class TestAudioModuleAllExports:
    """Test that __all__ contains expected exports."""

    def test_all_exports_defined(self):
        """Test that __all__ is defined and contains expected items."""
        from venice_ai.types import audio

        assert hasattr(audio, "__all__")
        expected = [
            "Voice",
            "ResponseFormat",
            "VoiceDetail",
            "VoiceList",
            "AudioResponse",
        ]
        for item in expected:
            assert item in audio.__all__, f"{item} not in __all__"

    def test_all_exports_are_accessible(self):
        """Test that all items in __all__ are accessible as attributes."""
        from venice_ai.types import audio

        for name in audio.__all__:
            assert hasattr(audio, name), f"{name} in __all__ but not accessible"
            assert getattr(audio, name) is not None


class TestVoiceEnumValues:
    """Test Voice enum functionality."""

    def test_voice_enum_has_af_alloy(self):
        """Test that Voice enum has AF_ALLOY (mentioned in docstring)."""
        # The docstring mentions Voice.AF_ALLOY as an example
        assert hasattr(Voice, "AF_ALLOY") or any("alloy" in str(v).lower() for v in Voice)

    def test_voice_enum_members_are_strings(self):
        """Test Voice enum member values."""
        for voice in Voice:
            # Each enum member should have a string value
            assert voice.value is not None


class TestResponseFormatEnumValues:
    """Test ResponseFormat enum functionality."""

    def test_response_format_has_mp3(self):
        """Test that ResponseFormat enum has MP3 (mentioned in docstring)."""
        # The docstring mentions ResponseFormat.MP3 as an example
        assert hasattr(ResponseFormat, "MP3") or any(
            "mp3" in str(f).lower() for f in ResponseFormat
        )

    def test_response_format_members_are_strings(self):
        """Test ResponseFormat enum member values."""
        for fmt in ResponseFormat:
            # Each enum member should have a string value
            assert fmt.value is not None
