"""
Tests for ChatCompletions.create method to cover specific missed lines of code.
Targeting missed lines: 95, 129, 272->271, 286->310, 305
"""

import pytest
import inspect
from unittest.mock import Mock, MagicMock
from typing import List, Sequence, cast, Iterator
from venice_ai.types.chat import MessageParam # Import MessageParam
from venice_ai.resources.chat.completions import ChatCompletions
from venice_ai._client import VeniceClient
from venice_ai.streaming import Stream
from venice_ai.types.chat import ChatCompletion, ChatCompletionChunk


class TestChatCompletionsSyncMissedLines:
    """Test cases to cover specific missed lines in ChatCompletions (synchronous)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_venice_client = Mock(spec=VeniceClient)
        self.completions = ChatCompletions(self.mock_venice_client)
        self.test_messages = [{"role": "user", "content": "test message"}]

    def test_chat_completions_create_with_optional_arg(self):
        """
        Test Case 1.1: Cover line 272 (processing of kwargs).
        Lines to cover: 272
        """
        # Setup mock
        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "response"}}]
        }
        self.mock_venice_client.post.return_value = mock_response

        # Action: Call create with optional temperature parameter
        result = self.completions.create(
            model="test-model",
            messages=cast(List[MessageParam], self.test_messages),
            stream=False,
            temperature=0.5
        )

        # Assertions
        self.mock_venice_client.post.assert_called_once()
        call_args = self.mock_venice_client.post.call_args
        json_data = call_args[1]["json_data"]
        
        # Verify that temperature was processed and added to the request body
        assert "temperature" in json_data
        assert json_data["temperature"] == 0.5
        assert result == mock_response

    def test_chat_completions_stream_true_custom_valid_stream_cls(self):
        """
        Test Case 1.2: Cover lines 286, 295, 296, 310 (path where stream_cls is a valid subclass of Stream).
        Lines to cover: 286, 295, 296, 310
        """
        # Setup: Define a dummy class that subclasses Stream
        class MyCustomSyncStream(Stream):
            pass

        # Mock _stream_request to return a dummy iterator
        dummy_iterator = iter([{"data": "chunk1"}])
        self.mock_venice_client._stream_request.return_value = dummy_iterator

        # Action: Call create with stream=True and custom valid stream_cls
        result = self.completions.create(
            model="test-model",
            messages=cast(List[MessageParam], self.test_messages),
            stream=True,
            stream_cls=MyCustomSyncStream  # type: ignore[arg-type]
        )

        # Assertions
        assert isinstance(result, MyCustomSyncStream)
        self.mock_venice_client._stream_request.assert_called_once()
        
        # Verify the call arguments
        call_args = self.mock_venice_client._stream_request.call_args
        assert call_args[1]["method"] == "POST"
        assert call_args[1]["path"] == "chat/completions"
        assert call_args[1]["json_data"]["stream"] is True

    def test_chat_completions_stream_true_custom_invalid_class_stream_cls(self):
        """
        Test Case 1.3: Cover lines 286, 295 (false), 300, 303, 310 (path where stream_cls is a class but not a Stream subclass).
        Lines to cover: 286, 300, 303, 310
        """
        # Setup: Use a simple class that's not a Stream subclass
        class NotAStream:
            def __init__(self, **data):
                pass
        stream_cls_arg = NotAStream

        # Mock _stream_request to return a dummy iterator
        dummy_iterator = iter([{"data": "chunk2"}])
        self.mock_venice_client._stream_request.return_value = dummy_iterator

        # Action: Call create with stream=True and invalid stream_cls
        result = self.completions.create(
            model="test-model",
            messages=cast(List[MessageParam], self.test_messages),
            stream=True,
            stream_cls=stream_cls_arg
        )

        # Assertions: Should default to Stream since dict is not a Stream subclass
        assert isinstance(result, Stream)
        self.mock_venice_client._stream_request.assert_called_once()
        
        # Verify the call arguments
        call_args = self.mock_venice_client._stream_request.call_args
        assert call_args[1]["method"] == "POST"
        assert call_args[1]["path"] == "chat/completions"
        assert call_args[1]["json_data"]["stream"] is True

    def test_chat_completions_stream_true_non_class_stream_cls(self):
        """
        Test Case 1.4: Cover lines 286 (false), 308 (implicit else), 310 (path where stream_cls is not a class).
        Lines to cover: 286 (evaluates to false), 310
        """
        # Setup: Use an integer (not a class) as stream_cls
        stream_cls_arg = 123

        # Mock _stream_request to return a dummy iterator
        dummy_iterator = iter([{"data": "chunk3"}])
        self.mock_venice_client._stream_request.return_value = dummy_iterator

        # Action: Call create with stream=True and non-class stream_cls
        result = self.completions.create(
            model="test-model",
            messages=cast(List[MessageParam], self.test_messages),
            stream=True,
            stream_cls=stream_cls_arg  # type: ignore[arg-type]
        )

        # Assertions: Should default to Stream since 123 is not a class
        assert isinstance(result, Stream)
        self.mock_venice_client._stream_request.assert_called_once()
        
        # Verify the call arguments
        call_args = self.mock_venice_client._stream_request.call_args
        assert call_args[1]["method"] == "POST"
        assert call_args[1]["path"] == "chat/completions"
        assert call_args[1]["json_data"]["stream"] is True

    def test_chat_completions_stream_true_default_stream_cls(self):
        """
        Test Case 1.5: Cover line 310 when stream_cls is None (default behavior).
        Lines to cover: 310 (and implicitly lines 280-281, 283 (false path))
        """
        # Mock _stream_request to return a dummy iterator
        dummy_iterator = iter([{"data": "chunk4"}])
        self.mock_venice_client._stream_request.return_value = dummy_iterator

        # Action: Call create with stream=True and omit stream_cls (defaults to None)
        result = self.completions.create(
            model="test-model",
            messages=cast(List[MessageParam], self.test_messages),
            stream=True
            # stream_cls is omitted, should default to None and use Stream
        )

        # Assertions: Should use default Stream
        assert isinstance(result, Stream)
        self.mock_venice_client._stream_request.assert_called_once()
        
        # Verify the call arguments
        call_args = self.mock_venice_client._stream_request.call_args
        assert call_args[1]["method"] == "POST"
        assert call_args[1]["path"] == "chat/completions"
        assert call_args[1]["json_data"]["stream"] is True