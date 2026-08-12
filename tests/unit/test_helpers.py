"""Unit tests for venice_ai.helpers module."""

import io
from typing import Literal, Optional  # noqa: UP035, UP045

import pytest
from pydantic import BaseModel

from venice_ai.helpers import (
    Conversation,
    _python_type_to_json_schema,
    cosine_similarity,
    detect_image_format,
    fit_image_bytes,
    normalize_duration_seconds,
    tool_from_function,
    tool_from_model,
)

# ============================================================================
# _python_type_to_json_schema tests
# ============================================================================


class TestPythonTypeToJsonSchema:
    """Tests for the type-hint → JSON Schema converter."""

    def test_str(self):
        assert _python_type_to_json_schema(str) == {"type": "string"}

    def test_int(self):
        assert _python_type_to_json_schema(int) == {"type": "integer"}

    def test_float(self):
        assert _python_type_to_json_schema(float) == {"type": "number"}

    def test_bool(self):
        assert _python_type_to_json_schema(bool) == {"type": "boolean"}

    def test_bare_list_rejected(self):
        # Bare ``list`` (no item type) cannot describe an items schema.
        with pytest.raises(TypeError, match="Bare list"):
            _python_type_to_json_schema(list)

    def test_bare_dict_rejected(self):
        # Bare ``dict`` (no value type) cannot describe an additionalProperties schema.
        with pytest.raises(TypeError, match="Bare dict"):
            _python_type_to_json_schema(dict)

    def test_list_of_str(self):
        assert _python_type_to_json_schema(list[str]) == {
            "type": "array",
            "items": {"type": "string"},
        }

    def test_list_of_int(self):
        assert _python_type_to_json_schema(list[int]) == {
            "type": "array",
            "items": {"type": "integer"},
        }

    def test_dict_str_int(self):
        assert _python_type_to_json_schema(dict[str, int]) == {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        }

    def test_optional_str(self):
        # typing.Optional form
        assert _python_type_to_json_schema(Optional[str]) == {"type": "string"}  # noqa: UP045  # type: ignore[arg-type]

    def test_optional_int(self):
        assert _python_type_to_json_schema(Optional[int]) == {"type": "integer"}  # noqa: UP045  # type: ignore[arg-type]

    def test_pep604_str_or_none(self):
        # PEP 604 ``str | None`` resolves to types.UnionType, treated identically to Optional[str]
        assert _python_type_to_json_schema(str | None) == {"type": "string"}

    def test_pep604_int_or_none(self):
        assert _python_type_to_json_schema(int | None) == {"type": "integer"}

    def test_pep604_list_of_optional(self):
        # Nested PEP 604 inside a generic
        assert _python_type_to_json_schema(list[int | None]) == {
            "type": "array",
            "items": {"type": "integer"},
        }

    def test_pep604_multi_member_union_rejected(self):
        # Multi-member non-Optional unions are rejected for both typing.Union and PEP 604 forms
        with pytest.raises(TypeError, match="Unions with multiple non-None members"):
            _python_type_to_json_schema(int | str)

    def test_typing_union_multi_member_rejected(self):
        from typing import Union as _Union  # noqa: UP035

        with pytest.raises(TypeError, match="Unions with multiple non-None members"):
            _python_type_to_json_schema(_Union[int, str])  # type: ignore[arg-type]  # noqa: UP007

    def test_literal(self):
        schema = _python_type_to_json_schema(Literal["a", "b", "c"])  # type: ignore[arg-type]
        assert schema == {"type": "string", "enum": ["a", "b", "c"]}

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported type"):
            _python_type_to_json_schema(bytes)

    def test_nested_list_of_list(self):
        schema = _python_type_to_json_schema(list[list[str]])
        assert schema == {
            "type": "array",
            "items": {"type": "array", "items": {"type": "string"}},
        }


# ============================================================================
# tool_from_model tests
# ============================================================================


class WeatherQuery(BaseModel):
    """Get the current weather for a location."""

    city: str
    units: str = "celsius"


class TestToolFromModel:
    """Tests for tool_from_model."""

    def test_basic(self):
        tool = tool_from_model(WeatherQuery)

        assert tool.type == "function"
        assert tool.function is not None
        assert tool.function.name == "WeatherQuery"
        assert tool.function.description == "Get the current weather for a location."
        assert "properties" in tool.function.parameters
        assert "city" in tool.function.parameters["properties"]

    def test_custom_name(self):
        tool = tool_from_model(WeatherQuery, name="get_weather")
        assert tool.function is not None
        assert tool.function.name == "get_weather"

    def test_custom_description(self):
        tool = tool_from_model(WeatherQuery, description="Custom desc")
        assert tool.function is not None
        assert tool.function.description == "Custom desc"

    def test_no_docstring_fallback(self):
        class NoDocs(BaseModel):
            x: int

        tool = tool_from_model(NoDocs)
        assert tool.function is not None
        assert tool.function.description == ""

    def test_schema_uses_pydantic(self):
        tool = tool_from_model(WeatherQuery)
        assert tool.function is not None
        schema = tool.function.parameters
        # Pydantic's model_json_schema produces a proper JSON Schema
        assert schema.get("type") == "object"
        assert "city" in schema["properties"]


# ============================================================================
# tool_from_function tests
# ============================================================================


def greet(name: str, excited: bool = False) -> str:
    """Greet someone by name."""
    return f"Hello, {name}{'!' if excited else '.'}"


def no_hints(x, y):
    """No type hints."""
    pass


class TestToolFromFunction:
    """Tests for tool_from_function."""

    def test_basic(self):
        tool = tool_from_function(greet)

        assert tool.type == "function"
        assert tool.function is not None
        assert tool.function.name == "greet"
        assert tool.function.description == "Greet someone by name."

        params = tool.function.parameters
        assert params["type"] == "object"
        assert "name" in params["properties"]
        assert params["properties"]["name"] == {"type": "string"}
        assert "excited" in params["properties"]
        assert params["properties"]["excited"] == {"type": "boolean"}

    def test_required_params(self):
        tool = tool_from_function(greet)
        # 'name' has no default → required; 'excited' has default → not required
        assert tool.function is not None
        assert "name" in tool.function.parameters["required"]
        assert "excited" not in tool.function.parameters["required"]

    def test_custom_name_and_desc(self):
        tool = tool_from_function(greet, name="say_hello", description="Say hi")
        assert tool.function is not None
        assert tool.function.name == "say_hello"
        assert tool.function.description == "Say hi"

    def test_no_type_hints_defaults_to_str(self):
        tool = tool_from_function(no_hints)
        assert tool.function is not None
        params = tool.function.parameters
        # Parameters without hints default to str
        assert params["properties"]["x"] == {"type": "string"}
        assert params["properties"]["y"] == {"type": "string"}

    def test_complex_hints(self):
        def search(query: str, tags: list[str], limit: int = 10) -> list[dict]:
            """Search things."""
            return []

        tool = tool_from_function(search)
        assert tool.function is not None
        params = tool.function.parameters
        assert params["properties"]["query"] == {"type": "string"}
        assert params["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}
        assert params["properties"]["limit"] == {"type": "integer"}
        assert "query" in params["required"]
        assert "tags" in params["required"]
        assert "limit" not in params["required"]

    def test_no_docstring(self):
        def bare(x: int):
            pass

        tool = tool_from_function(bare)
        assert tool.function is not None
        assert tool.function.description == ""

    def test_self_param_skipped(self):
        """Ensure 'self' parameter is excluded from schema."""

        class Foo:
            def method(self, x: int) -> str:
                return str(x)

        tool = tool_from_function(Foo().method)
        assert tool.function is not None
        assert "self" not in tool.function.parameters["properties"]

    def test_pep604_optional_param(self):
        """Function args using ``T | None`` (PEP 604) build a tool successfully."""

        def find_rhyme(word: str, max_results: int | None = None) -> list[str]:
            """Suggest rhyming words."""
            return []

        tool = tool_from_function(find_rhyme)
        assert tool.function is not None
        params = tool.function.parameters
        assert params["properties"]["word"] == {"type": "string"}
        assert params["properties"]["max_results"] == {"type": "integer"}
        assert "word" in params["required"]
        assert "max_results" not in params["required"]


# ============================================================================
# Conversation tests
# ============================================================================


class TestConversation:
    """Tests for the Conversation helper."""

    def test_empty_conversation(self):
        conv = Conversation()
        assert conv.messages == []

    def test_system_message(self):
        conv = Conversation(system="You are helpful.")
        msgs = conv.messages
        assert len(msgs) == 1
        assert msgs[0].content == "You are helpful."

    def test_add_user(self):
        conv = Conversation()
        result = conv.add_user("Hello")
        # Returns self for chaining
        assert result is conv
        assert len(conv.messages) == 1
        assert conv.messages[0].content == "Hello"

    def test_chaining(self):
        conv = Conversation(system="System prompt")
        conv.add_user("First").add_user("Second")
        assert len(conv.messages) == 3  # system + 2 user

    def test_add_tool_result(self):
        from venice_ai.types.api.requests.chat import ToolMessage

        conv = Conversation()
        result = conv.add_tool_result("call_123", '{"temp": 72}')
        assert result is conv
        msgs = conv.messages
        assert len(msgs) == 1
        msg = msgs[0]
        assert isinstance(msg, ToolMessage)
        assert msg.tool_call_id == "call_123"
        assert msg.content == '{"temp": 72}'

    def test_messages_returns_copy(self):
        conv = Conversation()
        conv.add_user("Hi")
        msgs1 = conv.messages
        msgs2 = conv.messages
        assert msgs1 == msgs2
        assert msgs1 is not msgs2  # Shallow copy

    def test_add_response(self):
        """Test add_response with a mock ChatCompletionResponse."""
        from venice_ai.types.api.chat import (
            ChatChoice,
            ChatCompletionResponse,
            ChatMessage,
            ChatUsage,
        )

        response = ChatCompletionResponse(
            id="resp-1",
            object="chat.completion",
            created=1000000,
            model="test-model",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="I'm here to help!"),
                    finish_reason="stop",
                    stop_reason=None,
                )
            ],
            usage=ChatUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15, prompt_tokens_details=None
            ),
            prompt_logprobs=None,
            venice_parameters=None,
            service_tier=None,
            system_fingerprint=None,
            kv_transfer_params=None,
        )

        conv = Conversation(system="System")
        conv.add_user("Hi")
        conv.add_response(response)

        msgs = conv.messages
        assert len(msgs) == 3
        assert msgs[2].content == "I'm here to help!"


# ============================================================================
# cosine_similarity tests
# ============================================================================


class TestCosineSimilarity:
    """Tests for the cosine_similarity helper."""

    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_known_value(self):
        # Hand-verified: dot=11, |a|=sqrt(5), |b|=sqrt(25)=5 → 11/(sqrt(5)*5)
        a = [1.0, 2.0]
        b = [3.0, 4.0]
        expected = 11 / (5**0.5 * 5)
        assert cosine_similarity(a, b) == pytest.approx(expected)

    def test_accepts_tuples(self):
        # Sequence[float] should accept tuples just as well as lists
        assert cosine_similarity((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_empty_vectors_raise(self):
        with pytest.raises(ValueError, match="non-empty"):
            cosine_similarity([], [])

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError, match="zero vector"):
            cosine_similarity([0.0, 0.0], [1.0, 1.0])
        with pytest.raises(ValueError, match="zero vector"):
            cosine_similarity([1.0, 1.0], [0.0, 0.0])


# ============================================================================
# detect_image_format tests
# ============================================================================


class TestDetectImageFormat:
    """Tests for magic-byte image format detection."""

    def test_png(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
        assert detect_image_format(png) == ("png", "image/png")

    def test_jpeg(self):
        jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        assert detect_image_format(jpeg) == ("jpg", "image/jpeg")

    def test_webp(self):
        webp = b"RIFF\x00\x00\x00\x00WEBPVP8 \x00\x00"
        assert detect_image_format(webp) == ("webp", "image/webp")

    def test_gif(self):
        gif = b"GIF89a" + b"\x00" * 10
        assert detect_image_format(gif) == ("gif", "image/gif")

    def test_unknown(self):
        assert detect_image_format(b"\x00\x01\x02\x03") == ("bin", "application/octet-stream")

    def test_empty_is_unknown(self):
        assert detect_image_format(b"") == ("bin", "application/octet-stream")

    def test_riff_without_webp_marker_is_unknown(self):
        # WAV files also start with RIFF — must require WEBP marker
        wav = b"RIFF\x00\x00\x00\x00WAVEfmt "
        assert detect_image_format(wav) == ("bin", "application/octet-stream")


# ============================================================================
# fit_image_bytes tests
# ============================================================================


def _make_image_bytes(width: int, height: int, fmt: str = "JPEG") -> bytes:
    """Synthesize a flat-color image in memory for tests."""
    import io as _io

    from PIL import Image

    img = Image.new("RGB", (width, height), color=(120, 80, 200))
    buf = _io.BytesIO()
    img.save(buf, format=fmt, quality=90)
    return buf.getvalue()


class TestFitImageBytes:
    """Tests for fit_image_bytes — client-side image downscaling."""

    def test_small_image_returned_unchanged(self):
        # Image below max_dim should be returned as-is (same identity)
        data = _make_image_bytes(400, 300)
        out = fit_image_bytes(data, max_dim=1024)
        assert out is data

    def test_large_image_resized_within_max_dim(self):
        from PIL import Image

        data = _make_image_bytes(2000, 3000)
        out = fit_image_bytes(data, max_dim=1024)
        assert out is not data
        assert len(out) < len(data)

        with Image.open(io.BytesIO(out)) as img:
            assert max(img.size) <= 1024
            # Aspect ratio preserved (within rounding)
            ratio_in = 2000 / 3000
            ratio_out = img.size[0] / img.size[1]
            assert abs(ratio_in - ratio_out) < 0.01

    def test_resized_emits_jpeg(self):
        # Even a PNG input should come back as JPEG when resized
        data = _make_image_bytes(2000, 2000, fmt="PNG")
        out = fit_image_bytes(data, max_dim=1024)
        assert out.startswith(b"\xff\xd8\xff")  # JPEG magic

    def test_png_with_alpha_converted_to_rgb(self):
        # RGBA PNG must convert without error
        import io as _io

        from PIL import Image

        img = Image.new("RGBA", (2000, 2000), color=(120, 80, 200, 128))
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()

        out = fit_image_bytes(data, max_dim=512)
        with Image.open(_io.BytesIO(out)) as resized:
            assert resized.mode == "RGB"
            assert max(resized.size) <= 512

    def test_quality_kwarg_affects_size(self):
        data = _make_image_bytes(1500, 1500)
        small = fit_image_bytes(data, max_dim=1024, quality=20)
        big = fit_image_bytes(data, max_dim=1024, quality=95)
        assert len(small) < len(big)

    def test_max_dim_kwarg_respected(self):
        from PIL import Image

        data = _make_image_bytes(2000, 2000)
        out = fit_image_bytes(data, max_dim=256)
        with Image.open(io.BytesIO(out)) as img:
            assert max(img.size) <= 256

    def test_image_at_exactly_max_dim_returned_unchanged(self):
        # Boundary: max(width, height) == max_dim is the "no resize" case
        data = _make_image_bytes(1024, 768)
        out = fit_image_bytes(data, max_dim=1024)
        assert out is data


# ============================================================================
# normalize_duration_seconds tests
# ============================================================================


class TestNormalizeDurationSeconds:
    """``5`` / ``"5"`` / ``"5s"`` / ``"5 seconds"`` all coerce to ``5``."""

    def test_int_passthrough(self):
        assert normalize_duration_seconds(5) == 5
        assert normalize_duration_seconds(60) == 60
        assert normalize_duration_seconds(210) == 210

    def test_string_int(self):
        assert normalize_duration_seconds("5") == 5
        assert normalize_duration_seconds("210") == 210

    def test_string_with_s_suffix(self):
        assert normalize_duration_seconds("5s") == 5
        assert normalize_duration_seconds("10S") == 10  # case-insensitive

    def test_string_with_seconds_suffix(self):
        assert normalize_duration_seconds("5 seconds") == 5
        assert normalize_duration_seconds("60 SECONDS") == 60
        assert normalize_duration_seconds("90sec") == 90

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            normalize_duration_seconds(-1)

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            normalize_duration_seconds(0)

    def test_unparseable_string_raises(self):
        with pytest.raises(ValueError, match="Could not parse"):
            normalize_duration_seconds("forever")
        with pytest.raises(ValueError, match="Could not parse"):
            normalize_duration_seconds("5.5")  # floats not supported

    def test_bool_rejected(self):
        with pytest.raises(ValueError, match="cannot be a bool"):
            normalize_duration_seconds(True)  # type: ignore[arg-type]

    def test_wrong_type_raises(self):
        with pytest.raises(ValueError, match="must be int or str"):
            normalize_duration_seconds(5.0)  # type: ignore[arg-type]
