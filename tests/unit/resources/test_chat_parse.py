"""Unit tests for client.chat.completions.parse() — auto-validating create()."""

from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, ValidationError

from venice_ai.resources.chat.completions import ChatCompletions
from venice_ai.types.api import UserMessage
from venice_ai.types.api.chat import (
    ChatChoice,
    ChatCompletionResponse,
    ChatMessage,
    ChatUsage,
    ParsedChatCompletion,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Person(BaseModel):
    name: str
    age: int


def _make_completion(content: str | None) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="resp-test",
        object="chat.completion",
        created=1000000,
        model="fake-test-model",
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
                stop_reason=None,
            )
        ],
        usage=ChatUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_tokens_details=None,
        ),
        prompt_logprobs=None,
        venice_parameters=None,
        service_tier=None,
        system_fingerprint=None,
        kv_transfer_params=None,
    )


def _build_chat(returning: ChatCompletionResponse) -> tuple[ChatCompletions, AsyncMock]:
    """Build a ChatCompletions whose create() returns *returning*.

    Returns the resource plus the create-mock so tests can inspect call args.
    """
    chat = ChatCompletions.__new__(ChatCompletions)
    chat._client = MagicMock()  # type: ignore[attr-defined]
    create_mock = AsyncMock(return_value=returning)
    chat.create = create_mock  # type: ignore[method-assign]
    return chat, create_mock


# ---------------------------------------------------------------------------
# parse() — happy paths
# ---------------------------------------------------------------------------


class TestParseHappyPath:
    @pytest.mark.asyncio
    async def test_returns_parsed_chat_completion(self):
        chat, _ = _build_chat(_make_completion('{"name": "Alice", "age": 30}'))

        result = await chat.parse(
            model="fake-test-model",
            messages=[UserMessage(content="Tell me about Alice.")],
            response_format=Person,
        )
        assert isinstance(result, ParsedChatCompletion)
        assert isinstance(result.parsed, Person)
        assert result.parsed.name == "Alice"
        assert result.parsed.age == 30

    @pytest.mark.asyncio
    async def test_response_passthrough(self):
        completion = _make_completion('{"name": "Bob", "age": 22}')
        chat, _ = _build_chat(completion)

        result = await chat.parse(
            model="fake-test-model",
            messages=[UserMessage(content="...")],
            response_format=Person,
        )
        assert result.response is completion
        assert result.usage is not None
        assert result.usage.total_tokens == 15
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_builds_json_schema_format(self):
        chat, create_mock = _build_chat(_make_completion('{"name": "X", "age": 1}'))

        await chat.parse(
            model="fake-test-model",
            messages=[UserMessage(content="...")],
            response_format=Person,
        )
        # Inspect what create() was called with.
        assert create_mock.await_args is not None
        kwargs = create_mock.await_args.kwargs
        rf = kwargs["response_format"]
        assert rf.type == "json_schema"
        assert rf.json_schema["name"] == "Person"  # default to class name
        assert rf.json_schema["strict"] is True
        # The schema body must be the Pydantic model's JSON schema.
        assert rf.json_schema["schema"] == Person.model_json_schema()

    @pytest.mark.asyncio
    async def test_custom_schema_name_and_strict_false(self):
        chat, create_mock = _build_chat(_make_completion('{"name": "Y", "age": 2}'))

        await chat.parse(
            model="fake-test-model",
            messages=[UserMessage(content="...")],
            response_format=Person,
            schema_name="my_schema",
            strict=False,
        )
        assert create_mock.await_args is not None
        rf = create_mock.await_args.kwargs["response_format"]
        assert rf.json_schema["name"] == "my_schema"
        assert rf.json_schema["strict"] is False

    @pytest.mark.asyncio
    async def test_forwards_extra_kwargs_to_create(self):
        chat, create_mock = _build_chat(_make_completion('{"name": "Z", "age": 3}'))

        await chat.parse(
            model="fake-test-model",
            messages=[UserMessage(content="...")],
            response_format=Person,
            temperature=0.5,
            max_completion_tokens=200,
        )
        assert create_mock.await_args is not None
        kwargs = create_mock.await_args.kwargs
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_completion_tokens"] == 200


# ---------------------------------------------------------------------------
# parse() — error paths
# ---------------------------------------------------------------------------


class TestParseErrors:
    @pytest.mark.asyncio
    async def test_validation_error_on_schema_mismatch(self):
        # Server returns valid JSON but doesn't satisfy Person.
        chat, _ = _build_chat(_make_completion('{"name": "OnlyName"}'))

        with pytest.raises(ValidationError):
            await chat.parse(
                model="fake-test-model",
                messages=[UserMessage(content="...")],
                response_format=Person,
            )

    @pytest.mark.asyncio
    async def test_value_error_on_none_content(self):
        # Tool-call-only turns have content=None — parse_as raises ValueError.
        chat, _ = _build_chat(_make_completion(None))

        with pytest.raises(ValueError, match="no content"):
            await chat.parse(
                model="fake-test-model",
                messages=[UserMessage(content="...")],
                response_format=Person,
            )

    @pytest.mark.asyncio
    async def test_rejects_stream_kwarg(self):
        chat, _ = _build_chat(_make_completion('{"name": "S", "age": 1}'))

        with pytest.raises(ValueError, match="does not support stream=True"):
            await chat.parse(
                model="fake-test-model",
                messages=[UserMessage(content="...")],
                response_format=Person,
                stream=True,
            )


# ---------------------------------------------------------------------------
# ParsedChatCompletion convenience
# ---------------------------------------------------------------------------


class TestParsedChatCompletionWrapper:
    def test_finish_reason_with_no_choices(self):
        # Edge case — defensive accessor should return None.
        empty = ChatCompletionResponse(
            id="r",
            object="chat.completion",
            created=0,
            model="fake-test-model",
            choices=[],
            usage=None,
            prompt_logprobs=None,
            venice_parameters=None,
            service_tier=None,
            system_fingerprint=None,
            kv_transfer_params=None,
        )
        wrapped = ParsedChatCompletion(response=empty, parsed=Person(name="x", age=1))
        assert wrapped.finish_reason is None

    def test_top_level_export(self):
        from venice_ai import ParsedChatCompletion as TopLevel

        assert TopLevel is ParsedChatCompletion

    def test_is_frozen(self):
        wrapped = ParsedChatCompletion(
            response=_make_completion('{"name": "F", "age": 1}'),
            parsed=Person(name="F", age=1),
        )
        with pytest.raises(FrozenInstanceError):
            wrapped.parsed = Person(name="other", age=2)  # type: ignore[misc]
