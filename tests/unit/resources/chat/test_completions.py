"""
Unit tests for the chat completions module.

This module contains unit tests for both the synchronous and asynchronous
versions of the ChatCompletions.create() method, focusing on parameter handling
and body construction for both streaming and non-streaming calls.
"""

import pytest
import pytest_asyncio
from unittest.mock import MagicMock
from typing import Dict, Any, Sequence, Iterator, AsyncIterator, cast

from venice_ai._client import VeniceClient
from venice_ai._async_client import AsyncVeniceClient
from venice_ai.resources.chat.completions import ChatCompletions, AsyncChatCompletions
from venice_ai.types.chat import (
    ChatCompletion, ChatCompletionChunk, MessageParam,
    ToolChoiceObject, Tool, ResponseFormat, StreamOptions, VeniceParameters
)

# Test fixtures and sample data
SAMPLE_MESSAGES: Sequence[MessageParam] = [
    {"role": "user", "content": "Hello, how are you?"}
]

SAMPLE_RESPONSE: ChatCompletion = cast(ChatCompletion, {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "test-model",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "Hello there!"},
        "finish_reason": "stop"
    }],
    "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}
})

SAMPLE_STREAM_CHUNK: ChatCompletionChunk = {
    "id": "chatcmpl-123",
    "object": "chat.completion.chunk",
    "created": 1677652288,
    "model": "test-model",
    "choices": [{
        "index": 0,
        "delta": {"content": "Hello"},
        "finish_reason": None
    }]
}


# Synchronous Tests
class TestChatCompletions:
    """Unit tests for the synchronous ChatCompletions class."""

    def test_sync_create_basic_non_streaming(self):
        """
        Test create() with minimal required parameters in non-streaming mode.
        
        This test verifies that:
        1. The post method is called with correct arguments.
        2. The returned value matches the mocked response.
        """
        # Setup client mock
        client_mock = MagicMock(spec=VeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        
        # Initialize resource with mock client
        completions = ChatCompletions(client_mock)
        
        # Call the create method
        response = completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES
        )
        
        # Verify post was called with correct parameters
        client_mock.post.assert_called_once_with(
            "chat/completions",
            json_data={
                "model": "test-model",
                "messages": SAMPLE_MESSAGES,
                "stream": False
            }
        )
        
        # Verify response is passed through correctly
        assert response == SAMPLE_RESPONSE

    def test_sync_create_basic_streaming(self):
        """
        Test create() with minimal required parameters in streaming mode.
        
        This test verifies that:
        1. The _stream_request method is called with correct arguments.
        2. The returned value is an iterator yielding the mocked chunks.
        """
        # Setup client mock and stream result
        client_mock = MagicMock(spec=VeniceClient)
        mock_iterator = iter([SAMPLE_STREAM_CHUNK])
        client_mock._stream_request.return_value = mock_iterator
        
        # Initialize resource with mock client
        completions = ChatCompletions(client_mock)
        
        # Call the create method with stream=True
        response = completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            stream=True
        )
        
        # Verify _stream_request was called with correct parameters
        client_mock._stream_request.assert_called_once_with(
            "POST",
            "chat/completions",
            json_data={
                "model": "test-model",
                "messages": SAMPLE_MESSAGES,
                "stream": True
            }
        )
        
        # Verify response is an iterator and yields the expected chunk
        assert isinstance(response, Iterator)
        chunks = list(response)
        assert len(chunks) == 1
        assert chunks[0] == SAMPLE_STREAM_CHUNK

    def test_sync_create_with_temperature(self):
        """Test parameter passthrough for temperature."""
        client_mock = MagicMock(spec=VeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        completions = ChatCompletions(client_mock)
        
        completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            temperature=0.7
        )
        
        # Verify temperature was included in the request
        client_mock.post.assert_called_once()
        call_args = client_mock.post.call_args
        assert call_args[1]['json_data']['temperature'] == 0.7

    def test_sync_create_with_top_p(self):
        """Test parameter passthrough for top_p."""
        client_mock = MagicMock(spec=VeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        completions = ChatCompletions(client_mock)
        
        completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            top_p=0.9
        )
        
        # Verify top_p was included in the request
        client_mock.post.assert_called_once()
        call_args = client_mock.post.call_args
        assert call_args[1]['json_data']['top_p'] == 0.9

    def test_sync_create_with_max_tokens(self):
        """Test parameter passthrough for max_tokens."""
        client_mock = MagicMock(spec=VeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        completions = ChatCompletions(client_mock)
        
        completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            max_tokens=100
        )
        
        # Verify max_tokens was included in the request
        client_mock.post.assert_called_once()
        call_args = client_mock.post.call_args
        assert call_args[1]['json_data']['max_tokens'] == 100

    def test_sync_create_with_max_completion_tokens(self):
        """Test parameter passthrough for max_completion_tokens."""
        client_mock = MagicMock(spec=VeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        completions = ChatCompletions(client_mock)
        
        completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            max_completion_tokens=150
        )
        
        # Verify max_completion_tokens was included in the request
        client_mock.post.assert_called_once()
        call_args = client_mock.post.call_args
        assert call_args[1]['json_data']['max_completion_tokens'] == 150

    def test_sync_create_with_frequency_penalty(self):
        """Test parameter passthrough for frequency_penalty."""
        client_mock = MagicMock(spec=VeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        completions = ChatCompletions(client_mock)
        
        completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            frequency_penalty=0.5
        )
        
        # Verify frequency_penalty was included in the request
        client_mock.post.assert_called_once()
        call_args = client_mock.post.call_args
        assert call_args[1]['json_data']['frequency_penalty'] == 0.5

    def test_sync_create_with_presence_penalty(self):
        """Test parameter passthrough for presence_penalty."""
        client_mock = MagicMock(spec=VeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        completions = ChatCompletions(client_mock)
        
        completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            presence_penalty=0.2
        )
        
        # Verify presence_penalty was included in the request
        client_mock.post.assert_called_once()
        call_args = client_mock.post.call_args
        assert call_args[1]['json_data']['presence_penalty'] == 0.2

    def test_sync_create_with_tools(self):
        """Test parameter passthrough for tools."""
        client_mock = MagicMock(spec=VeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        completions = ChatCompletions(client_mock)
        
        tools: Sequence[Tool] = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        
        completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            tools=tools
        )
        
        # Verify tools was included in the request
        client_mock.post.assert_called_once()
        call_args = client_mock.post.call_args
        assert call_args[1]['json_data']['tools'] == tools

    def test_sync_create_with_tool_choice_auto(self):
        """Test parameter passthrough for tool_choice='auto'."""
        client_mock = MagicMock(spec=VeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        completions = ChatCompletions(client_mock)
        
        completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            tool_choice="auto"
        )
        
        # Verify tool_choice was included in the request
        client_mock.post.assert_called_once()
        call_args = client_mock.post.call_args
        assert call_args[1]['json_data']['tool_choice'] == "auto"

    def test_sync_create_with_tool_choice_none(self):
        """Test parameter passthrough for tool_choice='none'."""
        client_mock = MagicMock(spec=VeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        completions = ChatCompletions(client_mock)
        
        completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            tool_choice="none"
        )
        
        # Verify tool_choice was included in the request
        client_mock.post.assert_called_once()
        call_args = client_mock.post.call_args
        assert call_args[1]['json_data']['tool_choice'] == "none"

    def test_sync_create_with_tool_choice_object(self):
        """Test parameter passthrough for tool_choice as an object."""
        client_mock = MagicMock(spec=VeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        completions = ChatCompletions(client_mock)
        
        tool_choice: ToolChoiceObject = {
            "type": "function",
            "function": {"name": "get_weather"}
        }
        
        completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            tool_choice=tool_choice
        )
        
        # Verify tool_choice was included in the request
        client_mock.post.assert_called_once()
        call_args = client_mock.post.call_args
        assert call_args[1]['json_data']['tool_choice'] == tool_choice

    def test_sync_create_with_response_format(self):
        """Test parameter passthrough for response_format."""
        client_mock = MagicMock(spec=VeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        completions = ChatCompletions(client_mock)
        
        response_format: ResponseFormat = {"type": "json_object"}
        
        completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            response_format=response_format
        )
        
        # Verify response_format was included in the request
        client_mock.post.assert_called_once()
        call_args = client_mock.post.call_args
        assert call_args[1]['json_data']['response_format'] == response_format

    def test_sync_create_with_seed(self):
        """Test parameter passthrough for seed."""
        client_mock = MagicMock(spec=VeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        completions = ChatCompletions(client_mock)
        
        completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            seed=42
        )
        
        # Verify seed was included in the request
        client_mock.post.assert_called_once()
        call_args = client_mock.post.call_args
        assert call_args[1]['json_data']['seed'] == 42

    def test_sync_create_with_logprobs(self):
        """Test parameter passthrough for logprobs."""
        client_mock = MagicMock(spec=VeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        completions = ChatCompletions(client_mock)
        
        completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            logprobs=True
        )
        
        # Verify logprobs was included in the request
        client_mock.post.assert_called_once()
        call_args = client_mock.post.call_args
        assert call_args[1]['json_data']['logprobs'] is True

    def test_sync_create_streaming_with_stream_options(self):
        """Test parameter passthrough for stream_options in streaming mode."""
        client_mock = MagicMock(spec=VeniceClient)
        mock_iterator = iter([SAMPLE_STREAM_CHUNK])
        client_mock._stream_request.return_value = mock_iterator
        completions = ChatCompletions(client_mock)
        
        stream_options: StreamOptions = {"include_usage": True}
        
        completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            stream=True,
            stream_options=stream_options
        )
        
        # Verify stream_options was included in the request
        client_mock._stream_request.assert_called_once()
        call_args = client_mock._stream_request.call_args
        assert call_args[1]['json_data']['stream_options'] == stream_options

    def test_sync_create_with_multiple_parameters(self):
        """Test parameter passthrough for multiple parameters."""
        client_mock = MagicMock(spec=VeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        completions = ChatCompletions(client_mock)
        
        # Create with multiple parameters
        response_format: ResponseFormat = {"type": "json_object"}
        venice_parameters = cast(VeniceParameters, {"include_venice_system_prompt": True})
        
        completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            temperature=0.8,
            max_completion_tokens=200,
            top_p=0.95,
            response_format=response_format,
            venice_parameters=venice_parameters
        )
        
        # Verify all parameters were included in the request
        client_mock.post.assert_called_once()
        call_args = client_mock.post.call_args
        assert call_args[1]['json_data']['temperature'] == 0.8
        assert call_args[1]['json_data']['max_completion_tokens'] == 200
        assert call_args[1]['json_data']['top_p'] == 0.95
        assert call_args[1]['json_data']['response_format'] == response_format
        assert call_args[1]['json_data']['venice_parameters'] == venice_parameters


    def test_sync_create_non_streaming_direct_return_coverage(self):
        """
        Test coverage for the direct return statement in non-streaming mode.
        
        This test specifically targets line 290 (approx) in completions.py:
        `return response` in the non-streaming path of ChatCompletions.create
        """
        # Setup client mock
        client_mock = MagicMock(spec=VeniceClient)
        mock_chat_completion_response: ChatCompletion = cast(ChatCompletion, {
            "id": "chatcmpl-testcoverage",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "covered"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })
        client_mock.post.return_value = mock_chat_completion_response

        # Initialize resource with mock client
        completions = ChatCompletions(client_mock)
        messages_payload: Sequence[MessageParam] = [{"role": "user", "content": "test"}]
        
        # Call create method with stream=False
        actual_response = completions.create(
            model="test-model",
            messages=messages_payload,
            stream=False
        )

        # Verify post was called
        client_mock.post.assert_called_once()
        
        # Specifically verify the object identity to ensure the return statement is executed
        assert actual_response is mock_chat_completion_response


# Asynchronous Tests
class TestAsyncChatCompletions:
    """Unit tests for the asynchronous AsyncChatCompletions class."""

    @pytest.mark.asyncio
    async def test_async_create_basic_non_streaming(self):
        """
        Test async create() with minimal required parameters in non-streaming mode.
        
        This test verifies that:
        1. The post method is called with correct arguments.
        2. The returned value matches the mocked response.
        """
        # Setup client mock
        client_mock = MagicMock(spec=AsyncVeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        
        # Initialize resource with mock client
        completions = AsyncChatCompletions(client_mock)
        
        # Call the create method
        response = await completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES
        )
        
        # Verify post was called with correct parameters
        client_mock.post.assert_called_once_with(
            "chat/completions",
            json_data={
                "model": "test-model",
                "messages": SAMPLE_MESSAGES,
                "stream": False
            }
        )
        
        # Verify response is passed through correctly
        assert response == SAMPLE_RESPONSE

    @pytest.mark.asyncio
    async def test_async_create_basic_streaming(self):
        """
        Test async create() with minimal required parameters in streaming mode.
        
        This test verifies that:
        1. The _stream_request method is called with correct arguments.
        2. The returned value is an async iterator yielding the mocked chunks.
        """
        # Setup client mock
        client_mock = MagicMock(spec=AsyncVeniceClient)
        
        # Create a mock async iterator
        class MockAsyncIterator:
            def __init__(self, items):
                self.items = items
                self.index = 0
            
            def __aiter__(self):
                return self
                
            async def __anext__(self):
                if self.index < len(self.items):
                    item = self.items[self.index]
                    self.index += 1
                    return item
                else:
                    raise StopAsyncIteration
        
        # Set up the mock to return our async iterator
        mock_async_iterator = MockAsyncIterator([SAMPLE_STREAM_CHUNK])
        client_mock._stream_request.return_value = mock_async_iterator
        
        # Initialize resource with mock client
        completions = AsyncChatCompletions(client_mock)
        
        # Call the create method with stream=True
        response = await completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            stream=True
        )
        
        # Verify _stream_request was called with correct parameters
        client_mock._stream_request.assert_called_once_with(
            "POST",
            "chat/completions",
            json_data={
                "model": "test-model",
                "messages": SAMPLE_MESSAGES,
                "stream": True
            }
        )
        
        # Verify response is an async iterator
        assert hasattr(response, '__aiter__')
        
        # Collect and check chunks
        chunks = []
        async for chunk in response:
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert chunks[0] == SAMPLE_STREAM_CHUNK

    @pytest.mark.asyncio
    async def test_async_create_with_temperature(self):
        """Test parameter passthrough for temperature in async mode."""
        client_mock = MagicMock(spec=AsyncVeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        completions = AsyncChatCompletions(client_mock)
        
        await completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            temperature=0.7
        )
        
        # Verify temperature was included in the request
        client_mock.post.assert_called_once()
        call_args = client_mock.post.call_args
        assert call_args[1]['json_data']['temperature'] == 0.7

    @pytest.mark.asyncio
    async def test_async_create_with_top_p(self):
        """Test parameter passthrough for top_p in async mode."""
        client_mock = MagicMock(spec=AsyncVeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        completions = AsyncChatCompletions(client_mock)
        
        await completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            top_p=0.9
        )
        
        # Verify top_p was included in the request
        client_mock.post.assert_called_once()
        call_args = client_mock.post.call_args
        assert call_args[1]['json_data']['top_p'] == 0.9

    @pytest.mark.asyncio
    async def test_async_create_with_tools(self):
        """Test parameter passthrough for tools in async mode."""
        client_mock = MagicMock(spec=AsyncVeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        completions = AsyncChatCompletions(client_mock)
        
        tools: Sequence[Tool] = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        
        await completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            tools=tools
        )
        
        # Verify tools was included in the request
        client_mock.post.assert_called_once()
        call_args = client_mock.post.call_args
        assert call_args[1]['json_data']['tools'] == tools

    @pytest.mark.asyncio
    async def test_async_create_with_tool_choice_auto(self):
        """Test parameter passthrough for tool_choice='auto' in async mode."""
        client_mock = MagicMock(spec=AsyncVeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        completions = AsyncChatCompletions(client_mock)
        
        await completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            tool_choice="auto"
        )
        
        # Verify tool_choice was included in the request
        client_mock.post.assert_called_once()
        call_args = client_mock.post.call_args
        assert call_args[1]['json_data']['tool_choice'] == "auto"

    @pytest.mark.asyncio
    async def test_async_create_with_tool_choice_none(self):
        """Test parameter passthrough for tool_choice='none' in async mode."""
        client_mock = MagicMock(spec=AsyncVeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        completions = AsyncChatCompletions(client_mock)
        
        await completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            tool_choice="none"
        )
        
        # Verify tool_choice was included in the request
        client_mock.post.assert_called_once()
        call_args = client_mock.post.call_args
        assert call_args[1]['json_data']['tool_choice'] == "none"

    @pytest.mark.asyncio
    async def test_async_create_with_tool_choice_object(self):
        """Test parameter passthrough for tool_choice as an object in async mode."""
        client_mock = MagicMock(spec=AsyncVeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        completions = AsyncChatCompletions(client_mock)
        
        tool_choice: ToolChoiceObject = {
            "type": "function",
            "function": {"name": "get_weather"}
        }
        
        await completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            tool_choice=tool_choice
        )
        
        # Verify tool_choice was included in the request
        client_mock.post.assert_called_once()
        call_args = client_mock.post.call_args
        assert call_args[1]['json_data']['tool_choice'] == tool_choice

    @pytest.mark.asyncio
    async def test_async_create_with_response_format(self):
        """Test parameter passthrough for response_format in async mode."""
        client_mock = MagicMock(spec=AsyncVeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        completions = AsyncChatCompletions(client_mock)
        
        response_format: ResponseFormat = {"type": "json_object"}
        
        await completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            response_format=response_format
        )
        
        # Verify response_format was included in the request
        client_mock.post.assert_called_once()
        call_args = client_mock.post.call_args
        assert call_args[1]['json_data']['response_format'] == response_format

    @pytest.mark.asyncio
    async def test_async_create_streaming_with_stream_options(self):
        """Test parameter passthrough for stream_options in async streaming mode."""
        # Setup client mock
        client_mock = MagicMock(spec=AsyncVeniceClient)
        
        # Create a mock async iterator
        class MockAsyncIterator:
            def __init__(self, items):
                self.items = items
                self.index = 0
            
            def __aiter__(self):
                return self
                
            async def __anext__(self):
                if self.index < len(self.items):
                    item = self.items[self.index]
                    self.index += 1
                    return item
                else:
                    raise StopAsyncIteration
        
        # Set up the mock to return our async iterator
        mock_async_iterator = MockAsyncIterator([SAMPLE_STREAM_CHUNK])
        client_mock._stream_request.return_value = mock_async_iterator
        
        completions = AsyncChatCompletions(client_mock)
        
        stream_options: StreamOptions = {"include_usage": True}
        
        await completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            stream=True,
            stream_options=stream_options
        )
        
        # Verify stream_options was included in the request
        client_mock._stream_request.assert_called_once()
        call_args = client_mock._stream_request.call_args
        assert call_args[1]['json_data']['stream_options'] == stream_options

    @pytest.mark.asyncio
    async def test_async_create_with_multiple_parameters(self):
        """Test parameter passthrough for multiple parameters in async mode."""
        client_mock = MagicMock(spec=AsyncVeniceClient)
        client_mock.post.return_value = SAMPLE_RESPONSE
        completions = AsyncChatCompletions(client_mock)
        
        # Create with multiple parameters
        response_format: ResponseFormat = {"type": "json_object"}
        venice_parameters = cast(VeniceParameters, {"include_venice_system_prompt": True})
        
        await completions.create(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            temperature=0.8,
            max_completion_tokens=200,
            top_p=0.95,
            response_format=response_format,
            venice_parameters=venice_parameters,
            seed=42,
            logprobs=True
        )
        
        # Verify all parameters were included in the request
        client_mock.post.assert_called_once()
        call_args = client_mock.post.call_args
        assert call_args[1]['json_data']['temperature'] == 0.8
        assert call_args[1]['json_data']['max_completion_tokens'] == 200
        assert call_args[1]['json_data']['top_p'] == 0.95
        assert call_args[1]['json_data']['response_format'] == response_format
        assert call_args[1]['json_data']['venice_parameters'] == venice_parameters
        assert call_args[1]['json_data']['seed'] == 42
        assert call_args[1]['json_data']['logprobs'] is True

    @pytest.mark.asyncio
    async def test_async_create_non_streaming_direct_return_coverage(self):
        """
        Test coverage for the direct return statement in async non-streaming mode.
        
        This test specifically targets line 520 (approx) in completions.py:
        `return response` in the non-streaming path of AsyncChatCompletions.create
        """
        # Setup client mock
        client_mock = MagicMock(spec=AsyncVeniceClient)
        mock_chat_completion_response: ChatCompletion = cast(ChatCompletion, {
            "id": "chatcmpl-asynctestcoverage",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "covered"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })
        client_mock.post.return_value = mock_chat_completion_response

        # Initialize resource with mock client
        completions = AsyncChatCompletions(client_mock)
        messages_payload: Sequence[MessageParam] = [{"role": "user", "content": "test"}]
        
        # Call create method with stream=False
        actual_response = await completions.create(
            model="test-model",
            messages=messages_payload,
            stream=False
        )

        # Verify post was called
        client_mock.post.assert_called_once()
        
        # Specifically verify the object identity to ensure the return statement is executed
        assert actual_response is mock_chat_completion_response