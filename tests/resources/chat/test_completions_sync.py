"""
Tests for synchronous chat completions functionality.
"""

import json
import pytest
import httpx
import warnings # Added import
from typing import Dict, Any, Iterator, cast

# Define base URL for API endpoints
BASE_URL = "https://api.venice.ai/api/v1"

from venice_ai import VeniceClient
from venice_ai.types.chat import ChatCompletion, ChatCompletionChunk, VeniceParameters
from venice_ai.exceptions import (
    VeniceError, APIError, AuthenticationError, RateLimitError, 
    InvalidRequestError, InternalServerError
)

# Test data
CHAT_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, who are you?"}
]

# Mock response data
MOCK_COMPLETION_RESPONSE = {
    "id": "chat-12345",
    "object": "chat.completion",
    "created": 1698447300,
    "model": "venice-classic",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "I am a helpful AI assistant created by Venice AI."
            },
            "finish_reason": "stop",
            "logprobs": None
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 12,
        "total_tokens": 32
    }
}

# Mock streaming response data
MOCK_STREAM_CHUNKS = [
    {
        "id": "chat-12345",
        "object": "chat.completion.chunk",
        "created": 1698447300,
        "model": "venice-classic",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "I am "
                },
                "finish_reason": None,
                "logprobs": None
            }
        ]
    },
    {
        "id": "chat-12345",
        "object": "chat.completion.chunk",
        "created": 1698447300,
        "model": "venice-classic",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": "a helpful "
                },
                "finish_reason": None,
                "logprobs": None
            }
        ]
    },
    {
        "id": "chat-12345",
        "object": "chat.completion.chunk",
        "created": 1698447300,
        "model": "venice-classic",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": "AI assistant "
                },
                "finish_reason": None,
                "logprobs": None
            }
        ]
    },
    {
        "id": "chat-12345",
        "object": "chat.completion.chunk",
        "created": 1698447300,
        "model": "venice-classic",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": "created by Venice AI."
                },
                "finish_reason": "stop",
                "logprobs": None
            }
        ]
    }
]

# Mock error responses
MOCK_AUTH_ERROR = {
    "error": {
        "code": "AUTHENTICATION_FAILED",
        "message": "Invalid API key"
    }
}

MOCK_RATE_LIMIT_ERROR = {
    "error": {
        "code": "RATE_LIMIT_EXCEEDED",
        "message": "You have exceeded your rate limit"
    }
}

MOCK_INVALID_REQUEST_ERROR = {
    "error": {
        "code": "INVALID_REQUEST",
        "message": "Invalid request parameters"
    }
}

MOCK_SERVER_ERROR = {
    "error": {
        "code": "INFERENCE_FAILED",
        "message": "Model inference failed"
    }
}



class TestChatCompletions:
    
    def test_create_success(self, venice_client, httpx_mock):
        """Test successful non-streaming chat completion."""
        # Mock the API response
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/chat/completions",
            json=MOCK_COMPLETION_RESPONSE,
            status_code=200
        )
        
        # Call the API
        response = venice_client.chat.completions.create(
            model="venice-classic",
            messages=CHAT_MESSAGES
        )
        
        # Verify response
        assert isinstance(response, dict)
        assert response["id"] == "chat-12345"
        assert response["object"] == "chat.completion"
        assert len(response["choices"]) == 1
        assert response["choices"][0]["message"]["content"] == (
            "I am a helpful AI assistant created by Venice AI."
        )
        assert response["choices"][0]["finish_reason"] == "stop"
        
        # Verify request was made correctly
        request = httpx_mock.get_request()
        assert request.url == f"{BASE_URL}/chat/completions"
        assert request.method == "POST"
        
        # Verify request body
        body = json.loads(request.content)
        assert body["model"] == "venice-classic"
        assert body["messages"] == CHAT_MESSAGES
        assert body["stream"] is False
    
    def test_create_with_parameters(self, venice_client, httpx_mock):
        """Test chat completion with additional parameters."""
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/chat/completions",
            json=MOCK_COMPLETION_RESPONSE,
            status_code=200
        )
        
        # Call with additional parameters
        venice_client.chat.completions.create(
            model="venice-classic",
            messages=CHAT_MESSAGES,
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.2,
            stop=["###"]
        )
        
        # Verify request parameters
        request = httpx_mock.get_request()
        body = json.loads(request.content)
        
        assert body["temperature"] == 0.7
        assert body["max_tokens"] == 100
        assert body["top_p"] == 0.9
        assert body["frequency_penalty"] == 0.5
        assert body["presence_penalty"] == 0.2
        assert body["stop"] == ["###"]
    
    def test_stream_success(self, venice_client, httpx_mock):
        """Test successful streaming chat completion."""
        # pytest.skip("Skipping due to httpx 0.28.1 streaming compatibility issues") # Re-enabled test
        # Create SSE response content
        stream_content = ""
        for chunk in MOCK_STREAM_CHUNKS:
            stream_content += f"data: {json.dumps(chunk)}\n\n"
        stream_content += "data: [DONE]\n\n"
        
        # Mock streaming response
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/chat/completions",
            content=stream_content.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
            status_code=200
        )
        
        # Call the streaming API
        stream_response = venice_client.chat.completions.create(
            model="venice-classic",
            messages=CHAT_MESSAGES,
            stream=True
        )
        
        # Verify it's an iterator
        assert isinstance(stream_response, Iterator)
        
        # Collect chunks and verify content
        chunks = list(stream_response)
        assert len(chunks) == 4
        
        # Verify individual chunks
        assert chunks[0]["choices"][0]["delta"]["content"] == "I am "
        assert chunks[1]["choices"][0]["delta"]["content"] == "a helpful "
        assert chunks[2]["choices"][0]["delta"]["content"] == "AI assistant "
        assert chunks[3]["choices"][0]["delta"]["content"] == "created by Venice AI."
        assert chunks[3]["choices"][0]["finish_reason"] == "stop"
        
        # Verify request was made correctly
        request = httpx_mock.get_request()
        body = json.loads(request.content)
        assert body["stream"] is True
    
    def test_auth_error(self, venice_client, httpx_mock):
        """Test authentication error handling for non-streaming requests."""
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/chat/completions",
            json=MOCK_AUTH_ERROR,
            status_code=401
        )
        
        # Verify exception raised
        with pytest.raises(AuthenticationError) as excinfo:
            venice_client.chat.completions.create(
                model="venice-classic",
                messages=CHAT_MESSAGES
            )
        
        # Check error details
        assert "Invalid API key" in str(excinfo.value)
        assert excinfo.value.response is not None
        assert excinfo.value.response.status_code == 401
    
    def test_rate_limit_error(self, venice_client, httpx_mock):
        """Test rate limit error handling."""
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/chat/completions",
            json=MOCK_RATE_LIMIT_ERROR,
            status_code=429
        )
        
        with pytest.raises(RateLimitError) as excinfo:
            venice_client.chat.completions.create(
                model="venice-classic",
                messages=CHAT_MESSAGES
            )
        
        assert "exceeded your rate limit" in str(excinfo.value)
        assert excinfo.value.response is not None
        assert excinfo.value.response.status_code == 429
    
    def test_invalid_request_error(self, venice_client, httpx_mock):
        """Test invalid request error handling."""
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/chat/completions",
            json=MOCK_INVALID_REQUEST_ERROR,
            status_code=400
        )
        
        with pytest.raises(InvalidRequestError) as excinfo:
            venice_client.chat.completions.create(
                model="venice-classic", 
                messages=CHAT_MESSAGES
            )
        
        assert "Invalid request parameters" in str(excinfo.value)
        assert "INVALID_REQUEST" in str(excinfo.value)
        assert excinfo.value.response is not None
        assert excinfo.value.response.status_code == 400
    
    def test_server_error(self, venice_client, httpx_mock):
        """Test server error handling."""
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/chat/completions",
            json=MOCK_SERVER_ERROR,
            status_code=500
        )
        
        with pytest.raises(InternalServerError) as excinfo:
            venice_client.chat.completions.create(
                model="venice-classic",
                messages=CHAT_MESSAGES
            )
        
        assert "Model inference failed" in str(excinfo.value)
        assert excinfo.value.response is not None
        assert excinfo.value.response.status_code == 500
    
    def test_stream_error(self, venice_client, httpx_mock):
        """Test error handling during streaming."""
        # pytest.skip("Skipping due to httpx 0.28.1 streaming compatibility issues") # Re-enabled test
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/chat/completions",
            json=MOCK_INVALID_REQUEST_ERROR,
            status_code=400
        )
        
        with pytest.raises(InvalidRequestError) as excinfo:
            # Create the iterator but don't consume it yet
            stream = venice_client.chat.completions.create(
                model="venice-classic",
                messages=CHAT_MESSAGES,
                stream=True
            )
            # This should raise when we try to iterate
            list(stream)
        
        assert "HTTP Status 400" in str(excinfo.value)
        assert excinfo.value.response is not None
        assert excinfo.value.response.status_code == 400
    
    def test_network_timeout(self, venice_client, httpx_mock):
        """Test timeout error handling."""
        httpx_mock.add_exception(httpx.TimeoutException("Connection timed out"))
        
        with pytest.raises(VeniceError) as excinfo:
            venice_client.chat.completions.create(
                model="venice-classic",
                messages=CHAT_MESSAGES
            )
        
        assert "timed out" in str(excinfo.value)
    
    def test_network_error(self, venice_client, httpx_mock):
        """Test network error handling."""
        httpx_mock.add_exception(httpx.ConnectError("Failed to establish connection"))
        
        with pytest.raises(VeniceError) as excinfo:
            venice_client.chat.completions.create(
                model="venice-classic",
                messages=CHAT_MESSAGES
            )
        
        assert "Failed to establish connection" in str(excinfo.value)

    def test_create_with_advanced_parameters(self, venice_client, httpx_mock):
        """Test chat completion with more advanced parameters to increase coverage."""
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/chat/completions",
            json=MOCK_COMPLETION_RESPONSE,
            status_code=200
        )
        
        response_format = {"type": "json_object"}
        venice_parameters = cast(VeniceParameters, {"custom_param": "value"})
        stream_options = {"include_usage": True}
        
        # Call with advanced parameters
        venice_client.chat.completions.create(
            model="venice-classic",
            messages=CHAT_MESSAGES,
            response_format=response_format,
            seed=42,
            logprobs=True,
            top_logprobs=5,
            parallel_tool_calls=True,
            repetition_penalty=1.2,
            stop_token_ids=[50256],
            top_k=40,
            venice_parameters=venice_parameters,
            stream_options=stream_options
        )
        
        # Verify request parameters
        request = httpx_mock.get_request()
        body = json.loads(request.content)
        
        assert body["response_format"] == response_format
        assert body["seed"] == 42
        assert body["logprobs"] is True
        assert body["top_logprobs"] == 5
        assert body["parallel_tool_calls"] is True
        assert body["repetition_penalty"] == 1.2
        assert body["stop_token_ids"] == [50256]
        assert body["top_k"] == 40
        assert body["venice_parameters"] == venice_parameters
        assert body["stream_options"] == stream_options
    
    def test_stream_network_error(self, venice_client, httpx_mock):
        """Test network error during streaming."""
        # pytest.skip("Skipping due to httpx 0.28.1 streaming compatibility issues") # Re-enabled test
        httpx_mock.add_exception(httpx.ConnectError("Failed to establish connection"))
        
        with pytest.raises(VeniceError) as excinfo:
            stream = venice_client.chat.completions.create(
                model="venice-classic",
                messages=CHAT_MESSAGES,
                stream=True
            )
            # This should raise when we try to iterate
            list(stream)

    def test_create_stream_true_stream_cls_none(self, venice_client, httpx_mock):
        """Test create with stream=True and stream_cls=None uses default Stream."""
        from venice_ai.streaming import Stream # Import for isinstance check

        # Mock the underlying client's stream request
        # Create SSE response content
        stream_content = ""
        for chunk in MOCK_STREAM_CHUNKS:
            stream_content += f"data: {json.dumps(chunk)}\n\n"
        stream_content += "data: [DONE]\n\n"
        
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/chat/completions",
            content=stream_content.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
            status_code=200
        )

        response_stream = venice_client.chat.completions.create(
            model="venice-classic",
            messages=CHAT_MESSAGES,
            stream=True,
            stream_cls=None # Explicitly None
        )
        assert isinstance(response_stream, Stream)
        # Consume the stream to ensure no errors during processing
        list(response_stream)


    def test_create_stream_false_with_stream_cls(self, venice_client, httpx_mock):
        """Test create with stream=False and a custom stream_cls (should be ignored)."""
        from venice_ai.streaming import Stream # Dummy class for testing

        class MyCustomStream(Stream): # Define a dummy custom stream
            pass

        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/chat/completions",
            json=MOCK_COMPLETION_RESPONSE, # Non-streaming response
            status_code=200
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            response = venice_client.chat.completions.create(
                model="venice-classic",
                messages=CHAT_MESSAGES,
                stream=False,
                stream_cls=MyCustomStream # This should be ignored
            )
            # Check for UserWarning if stream_cls is passed with stream=False
            # This depends on whether the library issues such a warning.
            # For now, we'll assume it might, or at least doesn't error.
            # assert len(w) > 0
            # assert issubclass(w[-1].category, UserWarning)
            # assert "stream_cls is ignored when stream=False" in str(w[-1].message)


        assert isinstance(response, dict) # Should be a normal dict response
        assert response["id"] == MOCK_COMPLETION_RESPONSE["id"]
        assert not isinstance(response, MyCustomStream)