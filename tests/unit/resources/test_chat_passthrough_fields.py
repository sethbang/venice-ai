"""
Unit tests for newly-documented chat-completion passthrough fields.

Per the API docs (``api-reference/endpoint/chat/completions.md``) POST
``/chat/completions`` accepts these additional body fields that were not being
forwarded by the SDK:

- ``store`` (boolean) — OpenAI-compat storage flag
- ``text`` (object with ``verbosity``) — OpenAI-compat text configuration
- ``include`` (string array) — OpenAI-compat include specifier
- ``metadata`` (object) — OpenAI-compat metadata dict
- ``prompt_cache_retention`` (enum ``default`` / ``extended`` / ``24h``) —
  Venice cache-retention control

The SDK must also accept ``prompt_logprobs`` on the response (this field was
already present; a test is added as a regression guard).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.resources.chat.completions import ChatCompletions
from venice_ai.types.api import ChatCompletionResponse


def _fake_response() -> ChatCompletionResponse:
    return ChatCompletionResponse.model_validate(
        {
            "id": "cmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "hi",
                    },
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "prompt_logprobs": None,
        }
    )


@pytest.fixture
def chat_resource() -> ChatCompletions:
    client = MagicMock()
    client.post = AsyncMock(return_value=_fake_response())
    return ChatCompletions(client)  # type: ignore[arg-type]


def _body_sent(chat_resource: ChatCompletions) -> dict:
    return chat_resource._client.post.call_args.kwargs["json_data"]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_create_forwards_store(chat_resource: ChatCompletions) -> None:
    await chat_resource.create(
        model="some-text-model",
        messages=[{"role": "user", "content": "hi"}],  # type: ignore[list-item]
        store=True,
    )
    assert _body_sent(chat_resource)["store"] is True


@pytest.mark.asyncio
async def test_create_forwards_text_object(chat_resource: ChatCompletions) -> None:
    await chat_resource.create(
        model="some-text-model",
        messages=[{"role": "user", "content": "hi"}],  # type: ignore[list-item]
        text={"verbosity": "low"},
    )
    assert _body_sent(chat_resource)["text"] == {"verbosity": "low"}


@pytest.mark.asyncio
async def test_create_forwards_include(chat_resource: ChatCompletions) -> None:
    await chat_resource.create(
        model="some-text-model",
        messages=[{"role": "user", "content": "hi"}],  # type: ignore[list-item]
        include=["reasoning.encrypted_content"],
    )
    assert _body_sent(chat_resource)["include"] == ["reasoning.encrypted_content"]


@pytest.mark.asyncio
async def test_create_forwards_metadata(chat_resource: ChatCompletions) -> None:
    await chat_resource.create(
        model="some-text-model",
        messages=[{"role": "user", "content": "hi"}],  # type: ignore[list-item]
        metadata={"trace_id": "abc-123", "experiment": "bar"},
    )
    assert _body_sent(chat_resource)["metadata"] == {
        "trace_id": "abc-123",
        "experiment": "bar",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["default", "extended", "24h"])
async def test_create_forwards_prompt_cache_retention(
    chat_resource: ChatCompletions, value: str
) -> None:
    await chat_resource.create(
        model="some-text-model",
        messages=[{"role": "user", "content": "hi"}],  # type: ignore[list-item]
        prompt_cache_retention=value,  # type: ignore[arg-type]
    )
    assert _body_sent(chat_resource)["prompt_cache_retention"] == value


@pytest.mark.asyncio
async def test_create_omits_new_fields_when_unset(chat_resource: ChatCompletions) -> None:
    await chat_resource.create(
        model="some-text-model",
        messages=[{"role": "user", "content": "hi"}],  # type: ignore[list-item]
    )
    body = _body_sent(chat_resource)
    for field in ("store", "text", "include", "metadata", "prompt_cache_retention"):
        assert field not in body


def test_response_accepts_prompt_logprobs() -> None:
    """Regression guard: ChatCompletionResponse must model ``prompt_logprobs``."""
    response = _fake_response()
    assert hasattr(response, "prompt_logprobs")
    assert response.prompt_logprobs is None


@pytest.mark.asyncio
async def test_create_forwards_dynamic_temp_params(
    chat_resource: ChatCompletions,
) -> None:
    """``max_temp`` / ``min_temp`` / ``min_p`` are documented first-class
    params and must surface on the typed signature, not only via ``**kwargs``."""
    await chat_resource.create(
        model="some-text-model",
        messages=[{"role": "user", "content": "hi"}],  # type: ignore[list-item]
        max_temp=1.5,
        min_temp=0.1,
        min_p=0.05,
    )
    body = _body_sent(chat_resource)
    assert body["max_temp"] == 1.5
    assert body["min_temp"] == 0.1
    assert body["min_p"] == 0.05
