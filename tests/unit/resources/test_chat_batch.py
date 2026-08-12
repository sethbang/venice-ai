"""Unit tests for client.chat.completions.batch()."""

import asyncio

import pytest

from venice_ai.resources.chat.completions import ChatCompletions
from venice_ai.types.api import UserMessage
from venice_ai.types.api.chat import (
    ChatChoice,
    ChatCompletionResponse,
    ChatMessage,
    ChatUsage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_completion(content: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=f"resp-{content}",
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


def _build_chat_with_create(create_impl):
    """Build a ChatCompletions whose .create coroutine is *create_impl*."""
    chat = ChatCompletions.__new__(ChatCompletions)
    chat.create = create_impl  # type: ignore[method-assign]
    return chat


# ---------------------------------------------------------------------------
# batch() — happy paths
# ---------------------------------------------------------------------------


class TestBatchHappyPath:
    @pytest.mark.asyncio
    async def test_preserves_order(self):
        async def fake_create(**kwargs):
            # Echo back the prompt content so we can verify slot mapping.
            content = kwargs["messages"][0].content
            return _make_completion(str(content))

        chat = _build_chat_with_create(fake_create)
        requests = [{"model": "m", "messages": [UserMessage(content=str(i))]} for i in range(5)]
        results = await chat.batch(requests)
        assert len(results) == 5
        for i, r in enumerate(results):
            assert isinstance(r, ChatCompletionResponse)
            assert r.choices[0].message.content == str(i)

    @pytest.mark.asyncio
    async def test_empty_requests_returns_empty_list(self):
        async def fake_create(**kwargs):  # pragma: no cover — should not be called
            raise AssertionError("create() should not be called for empty batch")

        chat = _build_chat_with_create(fake_create)
        assert await chat.batch([]) == []


# ---------------------------------------------------------------------------
# batch() — partial failures
# ---------------------------------------------------------------------------


class TestBatchPartialFailures:
    @pytest.mark.asyncio
    async def test_return_exceptions_default_true(self):
        async def fake_create(**kwargs):
            content = kwargs["messages"][0].content
            if content == "boom":
                raise RuntimeError("synthetic failure")
            return _make_completion(str(content))

        chat = _build_chat_with_create(fake_create)
        requests = [
            {"model": "m", "messages": [UserMessage(content="ok-1")]},
            {"model": "m", "messages": [UserMessage(content="boom")]},
            {"model": "m", "messages": [UserMessage(content="ok-2")]},
        ]
        results = await chat.batch(requests)
        assert len(results) == 3
        assert isinstance(results[0], ChatCompletionResponse)
        assert isinstance(results[1], RuntimeError)
        assert isinstance(results[2], ChatCompletionResponse)
        assert "synthetic failure" in str(results[1])

    @pytest.mark.asyncio
    async def test_return_exceptions_false_raises_first(self):
        async def fake_create(**kwargs):
            content = kwargs["messages"][0].content
            if content == "boom":
                raise RuntimeError("first failure")
            return _make_completion(str(content))

        chat = _build_chat_with_create(fake_create)
        requests = [
            {"model": "m", "messages": [UserMessage(content="ok")]},
            {"model": "m", "messages": [UserMessage(content="boom")]},
        ]
        with pytest.raises(RuntimeError, match="first failure"):
            await chat.batch(requests, return_exceptions=False)


# ---------------------------------------------------------------------------
# batch() — concurrency control
# ---------------------------------------------------------------------------


class TestBatchConcurrency:
    @pytest.mark.asyncio
    async def test_max_concurrency_limit(self):
        in_flight = 0
        peak = 0
        gate = asyncio.Event()

        async def fake_create(**kwargs):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                # Yield and let other coroutines pile up to test the cap.
                # A few zero-second sleeps gives the scheduler chances to
                # multiplex; max_concurrency must keep peak at the cap.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                content = kwargs["messages"][0].content
                return _make_completion(str(content))
            finally:
                in_flight -= 1
                gate.set()

        chat = _build_chat_with_create(fake_create)
        requests = [{"model": "m", "messages": [UserMessage(content=f"q{i}")]} for i in range(20)]
        results = await chat.batch(requests, max_concurrency=3)
        assert len(results) == 20
        assert all(isinstance(r, ChatCompletionResponse) for r in results)
        # Concurrency cap was honored.
        assert peak <= 3, f"peak={peak} exceeded max_concurrency=3"
        # And we actually ran multiple in parallel (sanity — not 1).
        assert peak >= 2

    @pytest.mark.asyncio
    async def test_invalid_max_concurrency_raises(self):
        async def fake_create(**kwargs):  # pragma: no cover
            raise AssertionError("should not be called")

        chat = _build_chat_with_create(fake_create)
        with pytest.raises(ValueError, match=">= 1"):
            await chat.batch(
                [{"model": "m", "messages": [UserMessage(content="x")]}],
                max_concurrency=0,
            )


# ---------------------------------------------------------------------------
# batch() — stream rejection
# ---------------------------------------------------------------------------


class TestBatchStreamRejection:
    @pytest.mark.asyncio
    async def test_stream_true_request_collected_as_value_error(self):
        async def fake_create(**kwargs):  # pragma: no cover — never called for stream=True slot
            content = kwargs["messages"][0].content
            return _make_completion(str(content))

        chat = _build_chat_with_create(fake_create)
        requests = [
            {"model": "m", "messages": [UserMessage(content="a")]},
            {"model": "m", "messages": [UserMessage(content="b")], "stream": True},
        ]
        results = await chat.batch(requests)
        assert isinstance(results[0], ChatCompletionResponse)
        assert isinstance(results[1], ValueError)
        assert "stream=True" in str(results[1])

    @pytest.mark.asyncio
    async def test_stream_true_request_raises_when_return_exceptions_false(self):
        async def fake_create(**kwargs):
            return _make_completion(str(kwargs["messages"][0].content))

        chat = _build_chat_with_create(fake_create)
        requests = [
            {"model": "m", "messages": [UserMessage(content="a")], "stream": True},
        ]
        with pytest.raises(ValueError, match="stream=True"):
            await chat.batch(requests, return_exceptions=False)
