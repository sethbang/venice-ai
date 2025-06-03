"""
Tests to improve coverage for chat completions functionality.

This module focuses on testing the specific lines and branches that are missing
coverage in src/venice_ai/resources/chat/completions.py, including:
- Deprecation warnings for max_tokens parameter
- Custom streaming class validation logic
- Parameter variation in request body construction
"""

import pytest
import warnings
import inspect
from typing import Iterator, AsyncIterator, Any, Dict, List, cast
from unittest.mock import MagicMock, patch, AsyncMock

from venice_ai import VeniceClient, AsyncVeniceClient
from venice_ai.resources.chat.completions import ChatCompletions, AsyncChatCompletions
from venice_ai.streaming import Stream, AsyncStream
from venice_ai.types.chat import MessageParam, ChatCompletion, ChatCompletionChunk


class TestChatCompletionsDeprecationWarnings:
    """Test deprecation warnings for max_tokens parameter."""

    def test_max_tokens_deprecation_warning_sync(self):
        """Test deprecation warning when only max_tokens is provided (sync)."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.post.return_value = {"id": "test", "object": "chat.completion", "choices": []}
        
        completions = ChatCompletions(mock_client)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=100  # This should trigger deprecation warning
            )
            
            # Verify deprecation warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "max_tokens" in str(w[0].message)
            assert "max_completion_tokens" in str(w[0].message)
        
        # Verify the request was made with max_completion_tokens
        call_args = mock_client.post.call_args
        assert call_args[1]["json_data"]["max_completion_tokens"] == 100
        assert "max_tokens" not in call_args[1]["json_data"]

    def test_both_max_tokens_and_max_completion_tokens_warning_sync(self):
        """Test warning when both max_tokens and max_completion_tokens are provided (sync)."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.post.return_value = {"id": "test", "object": "chat.completion", "choices": []}
        
        completions = ChatCompletions(mock_client)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=100,  # This should be ignored
                max_completion_tokens=200  # This should take precedence
            )
            
            # Verify deprecation warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Both" in str(w[0].message)
            assert "ignored" in str(w[0].message)
        
        # Verify max_completion_tokens takes precedence
        call_args = mock_client.post.call_args
        assert call_args[1]["json_data"]["max_completion_tokens"] == 200
        assert "max_tokens" not in call_args[1]["json_data"]

    @pytest.mark.asyncio
    async def test_max_tokens_deprecation_warning_async(self):
        """Test deprecation warning when only max_tokens is provided (async)."""
        mock_client = AsyncMock(spec=AsyncVeniceClient)
        mock_client.post.return_value = {"id": "test", "object": "chat.completion", "choices": []}
        
        completions = AsyncChatCompletions(mock_client)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            await completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=100  # This should trigger deprecation warning
            )
            
            # Verify deprecation warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "max_tokens" in str(w[0].message)
            assert "max_completion_tokens" in str(w[0].message)
        
        # Verify the request was made with max_completion_tokens
        call_args = mock_client.post.call_args
        assert call_args[1]["json_data"]["max_completion_tokens"] == 100
        assert "max_tokens" not in call_args[1]["json_data"]

    @pytest.mark.asyncio
    async def test_both_max_tokens_and_max_completion_tokens_warning_async(self):
        """Test warning when both max_tokens and max_completion_tokens are provided (async)."""
        mock_client = AsyncMock(spec=AsyncVeniceClient)
        mock_client.post.return_value = {"id": "test", "object": "chat.completion", "choices": []}
        
        completions = AsyncChatCompletions(mock_client)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            await completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=100,  # This should be ignored
                max_completion_tokens=200  # This should take precedence
            )
            
            # Verify deprecation warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Both" in str(w[0].message)
            assert "ignored" in str(w[0].message)
        
        # Verify max_completion_tokens takes precedence
        call_args = mock_client.post.call_args
        assert call_args[1]["json_data"]["max_completion_tokens"] == 200
        assert "max_tokens" not in call_args[1]["json_data"]


class CustomValidStreamClass:
    """A custom stream class that should pass validation."""
    
    def __init__(self, iterator, client=None):
        self.iterator = iterator
        self.client = client
    
    def __iter__(self):
        return iter(self.iterator)


class CustomInvalidStreamClass:
    """A custom stream class that should fail validation (missing __iter__)."""
    
    def __init__(self, iterator, client=None):
        self.iterator = iterator
        self.client = client


class CustomAsyncValidStreamClass:
    """A custom async stream class that should pass validation."""
    
    def __init__(self, iterator, client=None):
        self.iterator = iterator
        self.client = client
    
    def __aiter__(self):
        return self.iterator


class CustomAsyncInvalidStreamClass:
    """A custom async stream class that should fail validation (missing __aiter__)."""
    
    def __init__(self, iterator, client=None):
        self.iterator = iterator
        self.client = client


class TestChatCompletionsStreamingClassValidation:
    """Test custom streaming class validation logic."""

    def test_custom_valid_stream_class_sync(self):
        """Test streaming with a valid custom stream class (sync)."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_iterator = iter([{"id": "test", "object": "chat.completion.chunk", "choices": []}])
        mock_client._stream_request.return_value = mock_iterator
        
        completions = ChatCompletions(mock_client)
        
        # Use cast to bypass type checking for testing purposes
        result = completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            stream=True,
            stream_cls=cast(Any, CustomValidStreamClass)
        )
        
        # Verify the custom stream class was used
        assert isinstance(result, CustomValidStreamClass)
        mock_client._stream_request.assert_called_once()

    def test_custom_invalid_stream_class_sync(self):
        """Test streaming with an invalid custom stream class falls back to default (sync)."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_iterator = iter([{"id": "test", "object": "chat.completion.chunk", "choices": []}])
        mock_client._stream_request.return_value = mock_iterator
        
        completions = ChatCompletions(mock_client)
        
        with patch('venice_ai.resources.chat.completions.Stream') as MockStream:
            MockStream.return_value = "mocked_stream"
            
            result = completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
                stream=True,
                stream_cls=cast(Any, CustomInvalidStreamClass)  # Missing __iter__ method
            )
            
            # Verify fallback to default Stream class
            MockStream.assert_called_once_with(mock_iterator, client=mock_client)
            assert result == "mocked_stream"

    def test_non_class_stream_cls_sync(self):
        """Test streaming with a non-class stream_cls falls back to default (sync)."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_iterator = iter([{"id": "test", "object": "chat.completion.chunk", "choices": []}])
        mock_client._stream_request.return_value = mock_iterator
        
        completions = ChatCompletions(mock_client)
        
        with patch('venice_ai.resources.chat.completions.Stream') as MockStream:
            MockStream.return_value = "mocked_stream"
            
            result = completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
                stream=True,
                stream_cls=cast(Any, "not_a_class")  # Not a class
            )
            
            # Verify fallback to default Stream class
            MockStream.assert_called_once_with(mock_iterator, client=mock_client)
            assert result == "mocked_stream"

    def test_stream_subclass_sync(self):
        """Test streaming with a Stream subclass (sync)."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_iterator = iter([{"id": "test", "object": "chat.completion.chunk", "choices": []}])
        mock_client._stream_request.return_value = mock_iterator
        
        class CustomStreamSubclass(Stream):
            pass
        
        completions = ChatCompletions(mock_client)
        
        result = completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            stream=True,
            stream_cls=cast(Any, CustomStreamSubclass)
        )
        
        # Verify the custom subclass was used
        assert isinstance(result, CustomStreamSubclass)

    @pytest.mark.asyncio
    async def test_custom_valid_stream_class_async(self):
        """Test streaming with a valid custom stream class (async)."""
        mock_client = AsyncMock(spec=AsyncVeniceClient)
        
        async def mock_async_iterator():
            yield {"id": "test", "object": "chat.completion.chunk", "choices": []}
        
        mock_client._stream_request.return_value = mock_async_iterator()
        
        completions = AsyncChatCompletions(mock_client)
        
        result = await completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            stream=True,
            stream_cls=cast(Any, CustomAsyncValidStreamClass)
        )
        
        # Verify the custom stream class was used
        assert isinstance(result, CustomAsyncValidStreamClass)
        mock_client._stream_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_custom_invalid_stream_class_async(self):
        """Test streaming with an invalid custom stream class falls back to default (async)."""
        mock_client = AsyncMock(spec=AsyncVeniceClient)
        
        async def mock_async_iterator():
            yield {"id": "test", "object": "chat.completion.chunk", "choices": []}
        
        mock_client._stream_request.return_value = mock_async_iterator()
        
        completions = AsyncChatCompletions(mock_client)
        
        with patch('venice_ai.resources.chat.completions.AsyncStream') as MockAsyncStream:
            MockAsyncStream.return_value = "mocked_async_stream"
            
            result = await completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
                stream=True,
                stream_cls=cast(Any, CustomAsyncInvalidStreamClass)  # Missing __aiter__ method
            )
            
            # Verify fallback to default AsyncStream class
            MockAsyncStream.assert_called_once()
            assert result == "mocked_async_stream"

    @pytest.mark.asyncio
    async def test_async_stream_subclass_async(self):
        """Test streaming with an AsyncStream subclass (async)."""
        mock_client = AsyncMock(spec=AsyncVeniceClient)
        
        async def mock_async_iterator():
            yield {"id": "test", "object": "chat.completion.chunk", "choices": []}
        
        mock_client._stream_request.return_value = mock_async_iterator()
        
        class CustomAsyncStreamSubclass(AsyncStream):
            pass
        
        completions = AsyncChatCompletions(mock_client)
        
        result = await completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            stream=True,
            stream_cls=cast(Any, CustomAsyncStreamSubclass)
        )
        
        # Verify the custom subclass was used
        assert isinstance(result, CustomAsyncStreamSubclass)


class TestChatCompletionsParameterVariation:
    """Test parameter variation in request body construction."""

    def test_none_parameter_filtering_sync(self):
        """Test that None parameters are filtered out of request body (sync)."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.post.return_value = {"id": "test", "object": "chat.completion", "choices": []}
        
        completions = ChatCompletions(mock_client)
        
        completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.7,
            top_p=None,  # This should be filtered out
            frequency_penalty=None,  # This should be filtered out
            presence_penalty=0.5,
            max_completion_tokens=None  # This should be filtered out
        )
        
        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        
        # Verify None values are filtered out
        assert "top_p" not in request_body
        assert "frequency_penalty" not in request_body
        assert "max_completion_tokens" not in request_body
        
        # Verify non-None values are included
        assert request_body["temperature"] == 0.7
        assert request_body["presence_penalty"] == 0.5

    @pytest.mark.asyncio
    async def test_none_parameter_filtering_async(self):
        """Test that None parameters are filtered out of request body (async)."""
        mock_client = AsyncMock(spec=AsyncVeniceClient)
        mock_client.post.return_value = {"id": "test", "object": "chat.completion", "choices": []}
        
        completions = AsyncChatCompletions(mock_client)
        
        await completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.7,
            top_p=None,  # This should be filtered out
            frequency_penalty=None,  # This should be filtered out
            presence_penalty=0.5,
            max_completion_tokens=None  # This should be filtered out
        )
        
        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        
        # Verify None values are filtered out
        assert "top_p" not in request_body
        assert "frequency_penalty" not in request_body
        assert "max_completion_tokens" not in request_body
        
        # Verify non-None values are included
        assert request_body["temperature"] == 0.7
        assert request_body["presence_penalty"] == 0.5

    @pytest.mark.asyncio
    async def test_stream_cls_filtering_async(self):
        """Test that stream_cls is filtered out of request body in async context."""
        mock_client = AsyncMock(spec=AsyncVeniceClient)
        
        async def mock_async_iterator():
            yield {"id": "test", "object": "chat.completion.chunk", "choices": []}
        
        mock_client._stream_request.return_value = mock_async_iterator()
        
        completions = AsyncChatCompletions(mock_client)
        
        # This should not raise an error and stream_cls should be filtered out
        result = await completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            stream=True,
            stream_cls=cast(Any, CustomAsyncValidStreamClass),
            temperature=0.7
        )
        
        # Verify stream_cls was not included in the request body
        call_args = mock_client._stream_request.call_args
        if call_args and len(call_args) > 1 and call_args[1]:
            request_body = call_args[1]["json_data"]
            
            assert "stream_cls" not in request_body
            assert request_body["temperature"] == 0.7
            assert request_body["stream"] is True