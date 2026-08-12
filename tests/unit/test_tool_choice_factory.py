"""Unit tests for venice_ai.ToolChoice factory helpers."""

from venice_ai import ToolChoice
from venice_ai.core.models.common import SpecificToolChoice, ToolChoiceFunction


class TestToolChoiceAuto:
    def test_returns_literal_string(self):
        assert ToolChoice.auto() == "auto"

    def test_assignable_to_chat_signature(self):
        # The chat completions signature accepts Literal["none","auto"] | SpecificToolChoice;
        # the factory output should be a plain "auto" string that satisfies it.
        value = ToolChoice.auto()
        assert isinstance(value, str)


class TestToolChoiceNone:
    def test_returns_literal_string(self):
        assert ToolChoice.none() == "none"


class TestToolChoiceFunctionFactory:
    def test_produces_specific_tool_choice(self):
        choice = ToolChoice.function("get_weather")
        assert isinstance(choice, SpecificToolChoice)
        assert choice.type == "function"
        assert isinstance(choice.function, ToolChoiceFunction)
        assert choice.function.name == "get_weather"

    def test_serializes_to_expected_dict_shape(self):
        # The wire format is {"type": "function", "function": {"name": "..."}}.
        # That's the dict users currently hand-build; the factory should
        # round-trip to the exact same shape via model_dump.
        choice = ToolChoice.function("calculate")
        assert choice.model_dump() == {
            "type": "function",
            "function": {"name": "calculate"},
        }

    def test_preserves_arbitrary_function_name(self):
        choice = ToolChoice.function("user.defined_fn-1")
        assert choice.function.name == "user.defined_fn-1"


class TestToolChoiceTopLevelExport:
    def test_importable_from_top_level(self):
        from venice_ai import ToolChoice as TC

        assert TC.auto() == "auto"
        assert TC.none() == "none"
        assert TC.function("x").function.name == "x"
