"""Tests for ChatChoice.stop_reason int-to-str coercion."""

from venice_ai.types.api.chat import ChatChoice, ChatMessage


def test_stop_reason_int_coerced_to_str():
    choice = ChatChoice(
        index=0,
        message=ChatMessage(role="assistant", content="hello"),
        finish_reason="stop",
        stop_reason=128008,  # type: ignore[arg-type]  # intentionally passing int to test coercion
    )
    assert choice.stop_reason == "128008"
    assert isinstance(choice.stop_reason, str)


def test_stop_reason_none_preserved():
    choice = ChatChoice(
        index=0,
        message=ChatMessage(role="assistant", content="hello"),
        finish_reason="stop",
        stop_reason=None,
    )
    assert choice.stop_reason is None


def test_stop_reason_str_preserved():
    choice = ChatChoice(
        index=0,
        message=ChatMessage(role="assistant", content="hello"),
        finish_reason="stop",
        stop_reason="stop",
    )
    assert choice.stop_reason == "stop"
