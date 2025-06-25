"""
Unit tests for non-streaming code paths in chat completions functionality.

These tests specifically target the non-streaming paths (stream=False) in both
ChatCompletions.create and AsyncChatCompletions.create methods using unittest.mock
to directly test the classes without HTTP layer dependencies.
"""

import unittest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, Iterator, AsyncIterator, cast
import asyncio

from venice_ai.resources.chat.completions import ChatCompletions, AsyncChatCompletions
from venice_ai.streaming import Stream, AsyncStream
from venice_ai.types.chat import ChatCompletion, ChatCompletionChunk


# Dummy class for testing incompatible stream_cls
class IncompatibleStreamWrapper:
    def __init__(self, iterator: Any, client: Any):
        pass

async def dummy_async_iterator():
    """A dummy async iterator yielding mock ChatCompletionChunk objects."""
    # Example structure for a ChatCompletionChunk. Adjust if different.
    from venice_ai.types.chat import ChatCompletionChunkChoice, ChatCompletionChunkChoiceDelta
    choice = ChatCompletionChunkChoice(index=0, delta=ChatCompletionChunkChoiceDelta(content="test"))
    yield ChatCompletionChunk(id="chunk1", choices=[choice], model="test-model", object="chat.completion.chunk", created=123)

# New dummy class for patching 'venice_ai.resources.chat.completions.AsyncStream'.
# This class will be used as a type in the issubclass check, resolving the TypeError.
class DummyAsyncStreamForTest(AsyncStream): # Inheriting from AsyncStream
    pass
    
class TestChatCompletionsNonStreaming(unittest.TestCase):
    """Unit tests for ChatCompletions non-streaming functionality."""
    
    def test_chat_completions_create_non_streaming(self):
        """Test ChatCompletions.create with stream=False (non-streaming)."""
        mock_client = Mock()
        dummy_response: ChatCompletion = cast(ChatCompletion, {
            "id": "chatcmpl-123", "object": "chat.completion", "created": 1234567890,
            "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop", "logprobs": None}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })
        mock_client.post = Mock(return_value=dummy_response) 
        completions_resource = ChatCompletions(client=mock_client) 
        response = completions_resource.create(
            model="test-model", messages=[{"role": "user", "content": "Hi"}], stream=False
        )
        expected_payload = {"model": "test-model", "messages": [{"role": "user", "content": "Hi"}], "stream": False}
        mock_client.post.assert_called_once_with("chat/completions", json_data=expected_payload, cast_to=ChatCompletion)
        self.assertEqual(response, dummy_response)

    def test_chat_completions_create_non_streaming_default_stream(self):
        """Test ChatCompletions.create with default stream parameter (should be False)."""
        mock_client = Mock()
        dummy_response: ChatCompletion = cast(ChatCompletion, {
            "id": "chatcmpl-456", "object": "chat.completion", "created": 1234567890,
            "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Default response!"}, "finish_reason": "stop", "logprobs": None}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}
        })
        mock_client.post = Mock(return_value=dummy_response)
        completions_resource = ChatCompletions(client=mock_client)
        response = completions_resource.create(model="test-model", messages=[{"role": "user", "content": "Hello"}])
        expected_payload = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}], "stream": False}
        mock_client.post.assert_called_once_with("chat/completions", json_data=expected_payload, cast_to=ChatCompletion)
        self.assertEqual(response, dummy_response)

    @patch('venice_ai.resources.chat.completions.Stream', spec=Stream)
    def test_chat_completions_create_streaming_with_incompatible_class_stream_cls(self, MockSDKStream):
        """Test ChatCompletions.create with stream=True and an incompatible class for stream_cls."""
        mock_client = Mock()
        dummy_iterator: Iterator[ChatCompletionChunk] = iter([
            ChatCompletionChunk(id="chunk1", choices=[{"index": 0, "delta": {"content": "test"}}], model="test-model", object="chat.completion.chunk", created=123) # type: ignore
        ])
        mock_client._stream_request = Mock(return_value=dummy_iterator)
        
        mock_sdk_stream_instance = Mock(spec=Stream)
        MockSDKStream.return_value = mock_sdk_stream_instance

        completions_resource = ChatCompletions(client=mock_client)
        
        result_stream = completions_resource.create(
            model="test-model",
            messages=[{"role": "user", "content": "Stream test"}],
            stream=True,
            stream_cls=IncompatibleStreamWrapper  # type: ignore[arg-type]
        )
        
        mock_client._stream_request.assert_called_once()
        MockSDKStream.assert_called_once_with(dummy_iterator, client=mock_client)
        self.assertIs(result_stream, mock_sdk_stream_instance)
        self.assertNotIsInstance(result_stream, IncompatibleStreamWrapper)


class TestAsyncChatCompletionsNonStreaming(unittest.IsolatedAsyncioTestCase):
    """Unit tests for AsyncChatCompletions non-streaming functionality."""
    
    async def test_async_chat_completions_create_non_streaming(self):
        """Test AsyncChatCompletions.create with stream=False (non-streaming)."""
        mock_async_client = AsyncMock() 
        dummy_response: ChatCompletion = cast(ChatCompletion, {
            "id": "chatcmpl-789", "object": "chat.completion", "created": 1234567890,
            "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Async Hello!"}, "finish_reason": "stop", "logprobs": None}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18}
        })
        mock_async_client.post = AsyncMock(return_value=dummy_response)
        async_completions_resource = AsyncChatCompletions(client=mock_async_client)
        response = await async_completions_resource.create(
            model="test-model", messages=[{"role": "user", "content": "Hi"}], stream=False
        )
        expected_payload = {"model": "test-model", "messages": [{"role": "user", "content": "Hi"}], "stream": False}
        mock_async_client.post.assert_called_once_with("chat/completions", json_data=expected_payload, cast_to=ChatCompletion)
        self.assertEqual(response, dummy_response)

    async def test_async_chat_completions_create_non_streaming_default_stream(self):
        """Test AsyncChatCompletions.create with default stream parameter (should be False)."""
        mock_async_client = AsyncMock()
        dummy_response: ChatCompletion = cast(ChatCompletion, {
            "id": "chatcmpl-101112", "object": "chat.completion", "created": 1234567890,
            "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Async default response!"}, "finish_reason": "stop", "logprobs": None}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23}
        })
        mock_async_client.post = AsyncMock(return_value=dummy_response)
        async_completions_resource = AsyncChatCompletions(client=mock_async_client)
        response = await async_completions_resource.create(
            model="test-model", messages=[{"role": "user", "content": "Hello async"}]
        )
        expected_payload = {"model": "test-model", "messages": [{"role": "user", "content": "Hello async"}], "stream": False}
        mock_async_client.post.assert_called_once_with("chat/completions", json_data=expected_payload, cast_to=ChatCompletion)
        self.assertEqual(response, dummy_response)

    @patch('venice_ai.resources.chat.completions.AsyncStream', new_callable=lambda: DummyAsyncStreamForTest)
    @patch.object(DummyAsyncStreamForTest, '__new__')
    async def test_async_chat_completions_create_streaming_with_incompatible_class_stream_cls(
        self,
        mock_dummy_class_new: Mock, # Mock for DummyAsyncStreamForTest.__new__
        MockPatchedAsyncStreamType: Mock # This will be DummyAsyncStreamForTest class itself
    ):
        """Test AsyncChatCompletions.create with stream=True and an incompatible class for stream_cls."""
        mock_async_client = AsyncMock()
        
        dummy_iterator_instance: AsyncIterator[ChatCompletionChunk] = dummy_async_iterator()
        mock_async_client._stream_request = AsyncMock(return_value=dummy_iterator_instance)
        
        expected_returned_stream_instance = AsyncMock(spec=AsyncStream) # The instance we expect __new__ to return
        mock_dummy_class_new.return_value = expected_returned_stream_instance

        async_completions_resource = AsyncChatCompletions(client=mock_async_client)
        
        result_stream = await async_completions_resource.create(
            model="test-model",
            messages=[{"role": "user", "content": "Async stream test"}],
            stream=True,
            stream_cls=IncompatibleStreamWrapper  # type: ignore[arg-type]
        )
        
        mock_async_client._stream_request.assert_called_once()
        
        mock_dummy_class_new.assert_called_once_with(
            DummyAsyncStreamForTest,
            dummy_iterator_instance,
            client=mock_async_client
        )
        
        self.assertIs(result_stream, expected_returned_stream_instance)
        self.assertNotIsInstance(result_stream, IncompatibleStreamWrapper)

if __name__ == '__main__':
    unittest.main()