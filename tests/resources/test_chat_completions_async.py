"""
Tests for the AsyncChatCompletions.create method to ensure code coverage.
"""

import pytest
from typing import cast
from unittest.mock import AsyncMock, MagicMock
import httpx

from venice_ai.resources.chat.completions import AsyncChatCompletions
from venice_ai.exceptions import VeniceError, MissingStreamClassError # Import the actual exception
from venice_ai.types.chat import ChatCompletion, ChatCompletionChunk, MessageParam
from venice_ai.streaming import AsyncStream # Import for type checking default

# Mock AsyncStream class for testing
class MockAsyncStream:
    def __init__(self, iterator, *args, **kwargs):
        self.iterator = iterator

class TestAsyncChatCompletionsCreateMethod:
    
    @pytest.mark.asyncio
    async def test_create_with_stream_true_and_stream_cls_none_uses_default(self):
        """
        Test that when stream=True and stream_cls is None, it defaults correctly
        and _client._stream_request is called.
        """
        # Create a mock async client
        mock_client = AsyncMock()
    
        # Mock _stream_request to return a dummy async iterator
        async def dummy_iterator():
            yield {"data": "chunk1"}
    
        mock_client._stream_request = MagicMock(return_value=dummy_iterator())
    
        # Create the AsyncChatCompletions instance with the mock client
        completions = AsyncChatCompletions(mock_client)

        
        # Define test messages
        raw_messages = [{"role": "user", "content": "hello"}]
        messages = [cast(MessageParam, msg) for msg in raw_messages]
        
        # Call create with stream=True and no explicit stream_cls
        response = await completions.create(
            messages=messages,
            model="gpt-4",
            stream=True
            # stream_cls is omitted, should default to AsyncStream in completions.py
        )
        
        # Verify that _client._stream_request was called
        mock_client._stream_request.assert_called_once()
        _args_req_call, kwargs_req_call = mock_client._stream_request.call_args
        assert kwargs_req_call["method"] == "POST"
        assert kwargs_req_call["path"] == "chat/completions"
        assert kwargs_req_call["json_data"]["model"] == "gpt-4"
        assert kwargs_req_call["json_data"]["messages"] == messages
        assert kwargs_req_call["json_data"]["stream"] is True

        # Verify the response is an instance of AsyncStream (the default)
        # and it wraps the iterator from _stream_request
        assert isinstance(response, AsyncStream)
        # To check it wraps the correct iterator, we'd need to inspect response.iterator
        # or iterate through it, which might be too complex for this unit test's scope.
        # For now, type check and ensuring _stream_request was called is sufficient.
        
        # Ensure it's an async iterator
        assert hasattr(response, '__aiter__')
        # Consume the iterator to avoid RuntimeWarning for unawaited generator
        async for _ in response:
            pass
    
    @pytest.mark.asyncio
    async def test_create_with_stream_true_and_stream_cls_provided(self):
        """
        Test that _client._stream_request is called and the result is wrapped by stream_cls
        when stream=True and stream_cls is provided.
        """
        # Create a mock async client
        mock_client = AsyncMock()
        
        # Mock _stream_request to return a dummy async iterator
        async def dummy_iterator_for_provided_cls():
            yield {"data": "chunk_for_provided_cls"}
            
        dummy_async_iter = dummy_iterator_for_provided_cls()
        mock_client._stream_request = MagicMock(return_value=dummy_async_iter)
        
        # Create the AsyncChatCompletions instance with the mock client
        completions = AsyncChatCompletions(mock_client)

        from venice_ai.streaming import AsyncStream as VeniceAsyncBaseStream # Alias
        from unittest.mock import patch # Ensure patch is imported

        init_calls = [] # To store call arguments to __init__

        class InspectableAsyncStream(VeniceAsyncBaseStream):
            # This __init__ will be patched.
            def __init__(self, iterator, *, client):
                super().__init__(iterator, client=client)

        # With autospec=True, side_effect for __init__ receives the instance as the first argument.
        def init_wrapper(actual_instance_self, iterator_param, *, client): # Reverted signature
            # actual_instance_self is now directly passed by the mock framework.
            init_calls.append({
                'self': actual_instance_self,
                'iterator': iterator_param,
                'client': client
            })
            # Call the original __init__ of the *superclass* (VeniceAsyncBaseStream)
            # because InspectableAsyncStream.__init__ is the one being mocked.
            VeniceAsyncBaseStream.__init__(actual_instance_self, iterator_param, client=client)
            return None # __init__ must return None
        
        # Define test messages
        raw_messages = [{"role": "user", "content": "hello"}]
        messages = [cast(MessageParam, msg) for msg in raw_messages]

        with patch.object(InspectableAsyncStream, '__init__', side_effect=init_wrapper, autospec=True) as mock_init_on_inspectable:
            response = await completions.create(
                messages=messages,
                model="gpt-4",
                stream=True,
                stream_cls=InspectableAsyncStream,  # type: ignore[arg-type]
            )
    
            # Verify that _client._stream_request was called with the expected arguments
            mock_client._stream_request.assert_called_once()
            _args_req_call_stream, kwargs_req_call_stream = mock_client._stream_request.call_args
        
            assert kwargs_req_call_stream["method"] == "POST"
            assert kwargs_req_call_stream["path"] == "chat/completions"
            assert kwargs_req_call_stream["json_data"]["model"] == "gpt-4"
            assert kwargs_req_call_stream["json_data"]["messages"] == messages
            assert kwargs_req_call_stream["json_data"]["stream"] is True
        
            # Verify that InspectableAsyncStream.__init__ was called correctly via our wrapper
            assert len(init_calls) == 1
            call_info = init_calls[0]
            assert call_info['iterator'] is dummy_async_iter
            assert call_info['client'] is mock_client
            assert isinstance(call_info['self'], InspectableAsyncStream)
            
            # Verify that the response is an instance of InspectableAsyncStream
            assert isinstance(response, InspectableAsyncStream)
    
            # Ensure it's an async iterator and consume it
            assert hasattr(response, '__aiter__')
            async for _ in response:
                pass

    @pytest.mark.asyncio
    async def test_create_with_stream_false_returns_chat_completion_object(self):
        """
        Test that AsyncChatCompletions.create with stream=False returns a ChatCompletion object.
        Verifies the proper overload is called based on the stream parameter.
        """
        # Create a mock async client
        mock_client = AsyncMock()
        
        # Setup mock response
        mock_response_json = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288, 
            "model": "gpt-3.5-turbo-0125",
            "choices": [{
                "index": 0, 
                "message": {"role": "assistant", "content": "Hello there!"}, 
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}
        }
        
        # Setup the post method to be an AsyncMock that returns our mock response
        mock_client.post = AsyncMock(return_value=mock_response_json)
        
        # Create the AsyncChatCompletions instance with the mock client
        completions = AsyncChatCompletions(mock_client)
        
        # Define test messages
        raw_messages = [{"role": "user", "content": "hello"}]
        messages = [cast(MessageParam, msg) for msg in raw_messages]
        
        # Call create with stream=False
        response = await completions.create(
            messages=messages,
            model="gpt-4",
            stream=False
        )
        
        # Verify that _client.post was called with the expected arguments
        mock_client.post.assert_called_once()
        
        # Get the call arguments
        _args_post_call, kwargs_post_call = mock_client.post.call_args
        
        # Verify method and path
        assert _args_post_call[0] == "chat/completions" # .post path is the first positional arg
        
        # Verify json_data contains expected values
        assert kwargs_post_call["json_data"]["model"] == "gpt-4"
        assert kwargs_post_call["json_data"]["messages"] == messages
        assert kwargs_post_call["json_data"]["stream"] is False
        
        # Verify the response is a ChatCompletion object
        assert isinstance(response, dict)
        assert response["object"] == "chat.completion"
        assert "choices" in response
        assert "message" in response["choices"][0]

    @pytest.mark.asyncio
    async def test_create_with_stream_true_returns_async_stream_object(self):
        """
        Test that AsyncChatCompletions.create with stream=True returns an AsyncStream object.
        Verifies the proper overload is called based on the stream parameter.
        """
        # Create a mock async client
        mock_client = AsyncMock()
        
        # Setup mock async generator for streaming responses
        async def mock_async_chunk_generator():
            mock_chunk_response = {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677652288,
                "model": "gpt-3.5-turbo-0125",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello"},
                    "finish_reason": None
                }]
            }
            yield mock_chunk_response
        
        # Create an actual async generator
        async_generator = mock_async_chunk_generator()
        
        # Setup the _stream_request method to return our async generator
        mock_client._stream_request = MagicMock(return_value=async_generator)
        
        # Create the AsyncChatCompletions instance with the mock client
        completions = AsyncChatCompletions(mock_client)
        
        # Define test messages
        raw_messages = [{"role": "user", "content": "hello"}]
        messages = [cast(MessageParam, msg) for msg in raw_messages]

        from venice_ai.streaming import AsyncStream as VeniceAsyncBaseStream # Alias for clarity
        from unittest.mock import patch # Ensure patch is imported

        init_calls_test_returns_object = [] # Separate list for this test's __init__ calls

        class LocalTestAsyncStream(VeniceAsyncBaseStream):
            # This __init__ will be patched.
            def __init__(self, iterator, *, client):
                super().__init__(iterator, client=client)

        # With autospec=True, side_effect for __init__ receives the instance as the first argument.
        def init_wrapper_returns_object(actual_instance_self, iterator_param, *, client):
            init_calls_test_returns_object.append({
                'self': actual_instance_self,
                'iterator': iterator_param,
                'client': client
            })
            VeniceAsyncBaseStream.__init__(actual_instance_self, iterator_param, client=client)
            return None # __init__ must return None
            
        with patch.object(LocalTestAsyncStream, '__init__', side_effect=init_wrapper_returns_object, autospec=True) as mock_init_local_test_stream:
            response = await completions.create(
                messages=messages,
                model="gpt-4",
                stream=True,
                stream_cls=LocalTestAsyncStream,  # type: ignore[arg-type]
            )
    
            # Verify that _client._stream_request was called with the expected arguments
            mock_client._stream_request.assert_called_once()
            _args_stream_call_obj, kwargs_stream_call_obj = mock_client._stream_request.call_args
        
            assert kwargs_stream_call_obj["method"] == "POST"
            assert kwargs_stream_call_obj["path"] == "chat/completions"
            assert kwargs_stream_call_obj["json_data"]["model"] == "gpt-4"
            assert kwargs_stream_call_obj["json_data"]["messages"] == messages
            assert kwargs_stream_call_obj["json_data"]["stream"] is True
        
            # Verify that LocalTestAsyncStream.__init__ was called (via mock_init_local_test_stream)
            mock_init_local_test_stream.assert_called_once()
            
            # Check arguments recorded by our wrapper
            assert len(init_calls_test_returns_object) == 1
            call_info = init_calls_test_returns_object[0]
            assert call_info['iterator'] is async_generator
            assert call_info['client'] is mock_client
            assert isinstance(call_info['self'], LocalTestAsyncStream)

            # Verify that the response is an instance of LocalTestAsyncStream
            assert isinstance(response, LocalTestAsyncStream)
    
            # If the returned object is an async iterator, consume it
            if hasattr(response, '__aiter__'):
                 async for _ in response: # Consume to prevent RuntimeWarning
                    pass


class TestAsyncChatCompletionsAsyncMissedLines:
   """Test cases to cover specific missed lines in AsyncChatCompletions (asynchronous)."""

   def setup_method(self):
       """Set up test fixtures."""
       from venice_ai._async_client import AsyncVeniceClient
       from unittest.mock import AsyncMock
       self.mock_async_venice_client = AsyncMock(spec=AsyncVeniceClient)
       self.async_completions = AsyncChatCompletions(self.mock_async_venice_client)
       raw_test_messages = [{"role": "user", "content": "test message"}]
       self.test_messages = [cast(MessageParam, msg) for msg in raw_test_messages]

   @pytest.mark.asyncio
   async def test_async_chat_completions_create_with_optional_arg(self):
       """
       Test Case 2.1: Cover line 534 (processing of kwargs).
       Lines to cover: 534
       """
       # Setup mock
       mock_response = {
           "id": "chatcmpl-123",
           "object": "chat.completion",
           "choices": [{"message": {"role": "assistant", "content": "response"}}]
       }
       self.mock_async_venice_client.post.return_value = mock_response

       # Action: Call create with optional temperature parameter
       result = await self.async_completions.create(
           model="test-model",
           messages=self.test_messages,
           stream=False,
           temperature=0.5
       )

       # Assertions
       self.mock_async_venice_client.post.assert_called_once()
       call_args = self.mock_async_venice_client.post.call_args
       json_data = call_args[1]["json_data"]
       
       # Verify that temperature was processed and added to the request body
       assert "temperature" in json_data
       assert json_data["temperature"] == 0.5
       assert result == mock_response

   @pytest.mark.asyncio
   async def test_async_chat_completions_stream_true_custom_valid_stream_cls(self):
       """
       Test Case 2.2: Cover lines 545, 546, 547, 554 (path where stream_cls is valid AsyncStream subclass).
       Lines to cover: 545, 546, 547, 554
       """
       # Setup: Define a dummy class that subclasses AsyncStream
       class MyCustomAsyncStream(AsyncStream):
           pass

       # Mock _stream_request to return a dummy async iterator
       async def dummy_async_iterator():
           yield {"data": "async_chunk1"}

       self.mock_async_venice_client._stream_request.return_value = dummy_async_iterator()

       # Action: Call create with stream=True and custom valid stream_cls
       result = await self.async_completions.create(
           model="test-model",
           messages=self.test_messages,
           stream=True,
           stream_cls=MyCustomAsyncStream  # type: ignore[arg-type]
       )

       # Assertions
       assert isinstance(result, MyCustomAsyncStream)
       self.mock_async_venice_client._stream_request.assert_called_once()
       
       # Verify the call arguments
       call_args = self.mock_async_venice_client._stream_request.call_args
       assert call_args[1]["method"] == "POST"
       assert call_args[1]["path"] == "chat/completions"
       assert call_args[1]["json_data"]["stream"] is True

       # Consume the async iterator to avoid warnings
       async for _ in result:
           pass

   @pytest.mark.asyncio
   async def test_async_chat_completions_stream_true_custom_invalid_class_stream_cls(self):
       """
       Test Case 2.3: Cover lines 545, 546 (false), 548, 550, 554 (path where stream_cls is a class but not AsyncStream subclass).
       Lines to cover: 545, 548, 550, 554
       """
       # Setup: Use dict as a class that's not an AsyncStream subclass
       stream_cls_arg = dict

       # Mock _stream_request to return a dummy async iterator
       async def dummy_async_iterator():
           yield {"data": "async_chunk2"}

       self.mock_async_venice_client._stream_request.return_value = dummy_async_iterator()

       # Action: Call create with stream=True and invalid stream_cls
       result = await self.async_completions.create(
           model="test-model",
           messages=self.test_messages,
           stream=True,
           stream_cls=stream_cls_arg
       )

       # Assertions: Should default to AsyncStream since dict is not an AsyncStream subclass
       assert isinstance(result, AsyncStream)
       self.mock_async_venice_client._stream_request.assert_called_once()
       
       # Verify the call arguments
       call_args = self.mock_async_venice_client._stream_request.call_args
       assert call_args[1]["method"] == "POST"
       assert call_args[1]["path"] == "chat/completions"
       assert call_args[1]["json_data"]["stream"] is True

       # Consume the async iterator to avoid warnings
       async for _ in result:
           pass

   @pytest.mark.asyncio
   async def test_async_chat_completions_stream_true_non_class_stream_cls(self):
       """
       Test Case 2.4: Cover lines 545 (false), 551 (implicit else), 554 (path where stream_cls is not a class).
       Lines to cover: 545 (evaluates to false), 554
       """
       # Setup: Use an integer (not a class) as stream_cls
       stream_cls_arg = 123

       # Mock _stream_request to return a dummy async iterator
       async def dummy_async_iterator():
           yield {"data": "async_chunk3"}

       self.mock_async_venice_client._stream_request.return_value = dummy_async_iterator()

       # Action: Call create with stream=True and non-class stream_cls
       result = await self.async_completions.create(
           model="test-model",
           messages=self.test_messages,
           stream=True,
           stream_cls=stream_cls_arg  # type: ignore[arg-type]
       )

       # Assertions: Should default to AsyncStream since 123 is not a class
       assert isinstance(result, AsyncStream)
       self.mock_async_venice_client._stream_request.assert_called_once()
       
       # Verify the call arguments
       call_args = self.mock_async_venice_client._stream_request.call_args
       assert call_args[1]["method"] == "POST"
       assert call_args[1]["path"] == "chat/completions"
       assert call_args[1]["json_data"]["stream"] is True

       # Consume the async iterator to avoid warnings
       async for _ in result:
           pass

   @pytest.mark.asyncio
   async def test_async_chat_completions_stream_true_default_stream_cls(self):
       """
       Test Case 2.5: Cover line 554 when stream_cls is None.
       Lines to cover: 554 (and implicitly lines 541-542, 544 (false path))
       """
       # Mock _stream_request to return a dummy async iterator
       async def dummy_async_iterator():
           yield {"data": "async_chunk4"}

       self.mock_async_venice_client._stream_request.return_value = dummy_async_iterator()

       # Action: Call create with stream=True and omit stream_cls (defaults to None)
       result = await self.async_completions.create(
           model="test-model",
           messages=self.test_messages,
           stream=True
           # stream_cls is omitted, should default to None and use AsyncStream
       )

       # Assertions: Should use default AsyncStream
       assert isinstance(result, AsyncStream)
       self.mock_async_venice_client._stream_request.assert_called_once()
       
       # Verify the call arguments
       call_args = self.mock_async_venice_client._stream_request.call_args
       assert call_args[1]["method"] == "POST"
       assert call_args[1]["path"] == "chat/completions"
       assert call_args[1]["json_data"]["stream"] is True

       # Consume the async iterator to avoid warnings
       async for _ in result:
           pass