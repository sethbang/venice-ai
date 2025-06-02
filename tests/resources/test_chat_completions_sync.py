"""
Tests for the ChatCompletions.create method to ensure code coverage.
"""

import pytest
from typing import cast
from unittest.mock import MagicMock
import httpx

from venice_ai.resources.chat.completions import ChatCompletions
from venice_ai.exceptions import VeniceError
from venice_ai.types.chat import ChatCompletion, ChatCompletionChunk, MessageParam

# Define a custom exception class that matches what the code expects
class MissingStreamClassError(VeniceError):
    """Raised when stream=True but no stream_cls is provided."""
    pass

# Mock Stream class for testing, now inheriting from venice_ai.streaming.Stream
from venice_ai.streaming import Stream as VeniceBaseStream # Use an alias for clarity
class MockStream(VeniceBaseStream):
    def __init__(self, iterator, *args, **kwargs):
        # If VeniceBaseStream.__init__ needs specific args (like 'client'),
        # we might need to call super().__init__ appropriately.
        # For this test, the main goal is that MockStream is *used*.
        # The original MockStream just stored the iterator.
        # If Stream.__init__ must be called, it expects a 'client' kwarg.
        # The call from completions.create is: effective_stream_cls(raw_iterator, client=self._client)
        # So, client will be in kwargs if *args doesn't grab it.
        # Let's call super for completeness, assuming client is in kwargs.
        client_arg = kwargs.pop('client', None) # Extract client if present
        if client_arg:
            super().__init__(iterator, client=client_arg)
        else:
            # Fallback if client wasn't passed as expected, or handle error
            # For this mock, just setting iterator might be enough if super isn't strictly needed
            # for the test's assertions. However, to be a "good" subclass:
            # raise TypeError("MockStream expects 'client' keyword argument if used as a Stream subclass.")
            # For now, let's assume the test doesn't rely on Stream's full init.
            self._iterator = iterator # Mimic Stream's internal attribute
        
        # Store the original iterator for assertion, if different from Stream's internal
        self.raw_iterator_for_test = iterator

class TestChatCompletionsCreateMethod:
    
    def test_create_with_stream_true_and_stream_cls_none(self):
        """
        Test that when stream=True and stream_cls is None, it defaults to Stream
        and _client._stream_request is called.
        """
        # Create a mock client
        mock_client = MagicMock()
        mock_client._stream_request.return_value = iter([{"data": "chunk1"}]) # Dummy iterator
        
        # Create the ChatCompletions instance with the mock client
        completions = ChatCompletions(mock_client)
        
        # Define test messages
        raw_messages = [{"role": "user", "content": "hello"}]
        messages = [cast(MessageParam, msg) for msg in raw_messages]
        
        # Call create with stream=True but without stream_cls
        response = completions.create(
            messages=messages,
            model="gpt-4",
            stream=True
            # stream_cls is omitted, should default to Stream
        )
        
        # Verify that _client._stream_request was called
        mock_client._stream_request.assert_called_once()
        # Verify the response is an instance of Stream (the default)
        from venice_ai.streaming import Stream # Import for type checking
        assert isinstance(response, Stream)
        # Consume the iterator
        for _ in response:
            pass
    
    def test_create_with_stream_true_and_stream_cls_provided(self):
        """
        Test that _client._stream_request is called and the result is wrapped by the provided stream_cls
        when stream=True and a valid stream_cls is provided.
        """
        from unittest.mock import patch, ANY # Import patch and ANY
        from venice_ai.streaming import Stream # Import Stream for subclassing

        # Create a mock client
        mock_client = MagicMock()
        dummy_iterator = iter([{"data": "chunk_for_provided_cls"}])
        mock_client._stream_request.return_value = dummy_iterator
        
        # Create the ChatCompletions instance with the mock client
        completions = ChatCompletions(mock_client)
        
        # Define a custom stream class that inherits from Stream
        # This will pass the issubclass check in completions.py
        class MyCustomSyncStream(Stream):
            pass

        # Define test messages
        raw_messages = [{"role": "user", "content": "hello"}]
        messages = [cast(MessageParam, msg) for msg in raw_messages]
        
        # We want to assert that MyCustomSyncStream's __init__ is called,
        # and that the original __init__ logic still runs.
        # We will patch MyCustomSyncStream.__init__ and use a side_effect
        # to capture arguments and call the original Stream.__init__.
        init_calls = []

        # Patch __init__ on MyCustomSyncStream.
        # __init__ should return None. The mock's default return_value is a new mock,
        # so explicitly set it to None or ensure side_effect returns None.
        with patch.object(MyCustomSyncStream, '__init__', return_value=None, autospec=True) as mock_init_method:

            # With autospec=True, the side_effect receives the instance ('self') as the first argument.
            def init_wrapper_closure(actual_instance_self, iterator_param, *, client):
                init_calls.append({
                    'self': actual_instance_self,
                    'iterator': iterator_param,
                    'client': client
                })
                # Call the original Stream.__init__ method directly on the actual instance.
                Stream.__init__(actual_instance_self, iterator_param, client=client)

                # __init__ methods must return None.
                return None

            mock_init_method.side_effect = init_wrapper_closure

            # Call create with stream=True and our custom stream_cls
            response = completions.create(  # type: ignore[call-overload]
                messages=messages,
                model="gpt-4",
                stream=True,
                stream_cls=MyCustomSyncStream  # type: ignore[arg-type]
            )
        
            # Verify that _client._stream_request was called with the expected arguments
            mock_client._stream_request.assert_called_once()
            _args_req, kwargs_req = mock_client._stream_request.call_args
            
            assert kwargs_req["method"] == "POST"
            assert kwargs_req["path"] == "chat/completions"
            
            # Verify json_data contains expected values
            assert kwargs_req["json_data"]["model"] == "gpt-4"
            assert kwargs_req["json_data"]["messages"] == messages
            assert kwargs_req["json_data"]["stream"] is True
            
            # Verify that our init_wrapper (and thus MyCustomSyncStream.__init__) was called
            assert len(init_calls) == 1
            call_info = init_calls[0]

            # Assert that the mock __init__ (mock_init_method) was called once.
            # The arguments to __init__ are (self_instance, iterator_arg, *, client_kwarg)
            mock_init_method.assert_called_once_with(ANY, dummy_iterator, client=mock_client)

            # Check the arguments captured by our init_wrapper_closure
            assert call_info['iterator'] is dummy_iterator
            assert call_info['client'] is mock_client
            assert isinstance(call_info['self'], MyCustomSyncStream)

            # The response will be an instance of MyCustomSyncStream.
            assert isinstance(response, MyCustomSyncStream)

            # Consume if iterable (MyCustomSyncStream should be iterable if Stream is)
            # This will now work because the original __init__ of Stream (via MyCustomSyncStream) runs.
            if hasattr(response, '__iter__') and not isinstance(response, (dict, str)):
                list(response) # Consume the iterator

    def test_create_with_stream_false_returns_chat_completion_object(self):
        """
        Test that ChatCompletions.create with stream=False returns a ChatCompletion object.
        Verifies the proper overload is called based on the stream parameter.
        """
        # Create a mock client
        mock_client = MagicMock()
        
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
        mock_httpx_response = MagicMock(spec=httpx.Response)
        mock_httpx_response.json.return_value = mock_response_json
        mock_client.post.return_value = mock_response_json  # Directly return the dict
        
        # Create the ChatCompletions instance with the mock client
        completions = ChatCompletions(mock_client)
        
        # Define test messages
        raw_messages = [{"role": "user", "content": "hello"}]
        messages = [cast(MessageParam, msg) for msg in raw_messages]
        
        # Call create with stream=False (default)
        response = completions.create(
            messages=messages,
            model="gpt-4",
            stream=False
        )
        
        # Verify that _client.post was called with the expected arguments
        mock_client.post.assert_called_once()
        
        # Get the call arguments
        _args_post, kwargs_post = mock_client.post.call_args
        
        # Verify method and path
        assert _args_post[0] == "chat/completions" # .post path is the first positional arg
        
        # Verify json_data contains expected values
        assert kwargs_post["json_data"]["model"] == "gpt-4"
        assert kwargs_post["json_data"]["messages"] == messages
        assert kwargs_post["json_data"]["stream"] is False
        
        # Verify the response is a ChatCompletion object
        assert isinstance(response, dict)
        assert response["object"] == "chat.completion"
        assert "choices" in response
        assert "message" in response["choices"][0]

    def test_create_with_stream_true_returns_stream_object(self):
        """
        Test that ChatCompletions.create with stream=True returns a Stream object.
        Verifies the proper overload is called based on the stream parameter.
        """
        # Create a mock client
        mock_client = MagicMock()
        
        # Mock the _client._stream_request method
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
        mock_stream_iterator = iter([mock_chunk_response])
        mock_client._stream_request.return_value = mock_stream_iterator
        
        # Create the ChatCompletions instance with the mock client
        completions = ChatCompletions(mock_client)
        
        # Define test messages
        raw_messages = [{"role": "user", "content": "hello"}]
        messages = [cast(MessageParam, msg) for msg in raw_messages]
        
        # Call create with stream=True
        response = completions.create(
            messages=messages,
            model="gpt-4",
            stream=True,
            stream_cls=MockStream  # Provide the mock stream class
        )
        
        # Verify that _client._stream_request was called with the expected arguments
        mock_client._stream_request.assert_called_once()
        
        # Get the call arguments
        _args_stream, kwargs_stream = mock_client._stream_request.call_args
        
        # Verify method and path
        assert kwargs_stream["method"] == "POST"
        assert kwargs_stream["path"] == "chat/completions"
        
        # Verify json_data contains expected values
        assert kwargs_stream["json_data"]["model"] == "gpt-4"
        assert kwargs_stream["json_data"]["messages"] == messages
        assert kwargs_stream["json_data"]["stream"] is True
        
        # Verify that the returned object is an instance of MockStream
        # and that MockStream was initialized with the mock_stream_iterator
        assert isinstance(response, MockStream)
        assert response.raw_iterator_for_test is mock_stream_iterator # Changed from response.iterator
        
        # Verify we can iterate through the response
        # Since MockStream now inherits from Stream, it should be iterable itself.
        # The raw_iterator_for_test is the original one, Stream might wrap it.
        # For this test, iterating 'response' directly is more idiomatic for a Stream.
        first_chunk = next(iter(response)) # Iterate through the Stream-like object
        assert first_chunk == mock_chunk_response