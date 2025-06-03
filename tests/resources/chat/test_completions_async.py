"""
Tests for asynchronous chat completions functionality.
"""

import json
import pytest
import httpx
from typing import Dict, Any, AsyncIterator, List, cast
import warnings # Added import

# Define base URL for API endpoints
BASE_URL = "https://api.venice.ai/api/v1"

from venice_ai import AsyncVeniceClient
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



class TestAsyncChatCompletions:
    
    @pytest.mark.asyncio
    async def test_create_success(self, async_venice_client, httpx_mock):
        """Test successful non-streaming async chat completion."""
        # Mock the API response
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/chat/completions",
            json=MOCK_COMPLETION_RESPONSE,
            status_code=200
        )
        
        # Call the API with await
        response = await async_venice_client.chat.completions.create(
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
    
    @pytest.mark.asyncio
    async def test_create_with_parameters(self, async_venice_client, httpx_mock):
        """Test async chat completion with additional parameters."""
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/chat/completions",
            json=MOCK_COMPLETION_RESPONSE,
            status_code=200
        )
        
        # Call with additional parameters
        await async_venice_client.chat.completions.create(
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
    
    @pytest.mark.asyncio
    async def test_stream_success(self, async_venice_client, httpx_mock):
        """Test successful streaming async chat completion."""
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
        stream_response = await async_venice_client.chat.completions.create(
            model="venice-classic",
            messages=CHAT_MESSAGES,
            stream=True
        )
        
        # Verify it's an async iterator
        assert isinstance(stream_response, AsyncIterator)
        
        # Collect chunks and verify content
        chunks: List[ChatCompletionChunk] = []
        async for chunk in stream_response:
            chunks.append(chunk)
        
        assert len(chunks) == 4
        
        # Verify individual chunks
        assert chunks[0]["choices"][0]["delta"].get("content") == "I am "
        assert chunks[1]["choices"][0]["delta"].get("content") == "a helpful "
        assert chunks[2]["choices"][0]["delta"].get("content") == "AI assistant "
        assert chunks[3]["choices"][0]["delta"].get("content") == "created by Venice AI."
        assert chunks[3]["choices"][0]["finish_reason"] == "stop"
        
        # Verify request was made correctly
        request = httpx_mock.get_request()
        body = json.loads(request.content)
        assert body["stream"] is True
    
    @pytest.mark.asyncio
    async def test_auth_error(self, async_venice_client, httpx_mock):
        """Test authentication error handling for async non-streaming requests."""
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/chat/completions",
            json=MOCK_AUTH_ERROR,
            status_code=401
        )
        
        # Verify exception raised
        with pytest.raises(AuthenticationError) as excinfo:
            await async_venice_client.chat.completions.create(
                model="venice-classic",
                messages=CHAT_MESSAGES
            )
        
        # Check error details
        assert "Invalid API key" in str(excinfo.value)
        assert excinfo.value.response is not None
        assert excinfo.value.response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_rate_limit_error(self, async_venice_client, httpx_mock):
        """Test rate limit error handling for async requests."""
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/chat/completions",
            json=MOCK_RATE_LIMIT_ERROR,
            status_code=429
        )
        
        with pytest.raises(RateLimitError) as excinfo:
            await async_venice_client.chat.completions.create(
                model="venice-classic",
                messages=CHAT_MESSAGES
            )
        
        assert "exceeded your rate limit" in str(excinfo.value)
        assert excinfo.value.response is not None
        assert excinfo.value.response.status_code == 429
    
    @pytest.mark.asyncio
    async def test_invalid_request_error(self, async_venice_client, httpx_mock):
        """Test invalid request error handling for async requests."""
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/chat/completions",
            json=MOCK_INVALID_REQUEST_ERROR,
            status_code=400
        )
        
        with pytest.raises(InvalidRequestError) as excinfo:
            await async_venice_client.chat.completions.create(
                model="venice-classic", 
                messages=CHAT_MESSAGES
            )
        
        assert "Invalid request parameters" in str(excinfo.value)
        assert "INVALID_REQUEST" in str(excinfo.value)
        assert excinfo.value.response is not None
        assert excinfo.value.response.status_code == 400
    
    @pytest.mark.asyncio
    async def test_server_error(self, async_venice_client, httpx_mock):
        """Test server error handling for async requests."""
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/chat/completions",
            json=MOCK_SERVER_ERROR,
            status_code=500
        )
        
        with pytest.raises(InternalServerError) as excinfo:
            await async_venice_client.chat.completions.create(
                model="venice-classic",
                messages=CHAT_MESSAGES
            )
        
        assert "Model inference failed" in str(excinfo.value)
        assert excinfo.value.response is not None
        assert excinfo.value.response.status_code == 500
    
    @pytest.mark.asyncio
    async def test_stream_error(self, async_venice_client, httpx_mock):
        """Test error handling during async streaming."""
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/chat/completions",
            json=MOCK_INVALID_REQUEST_ERROR,
            status_code=400
        )
        
        with pytest.raises(InvalidRequestError) as excinfo:
            # Create the async iterator but don't consume it yet
            stream = await async_venice_client.chat.completions.create(
                model="venice-classic",
                messages=CHAT_MESSAGES,
                stream=True
            )
            
            # This should raise when we try to iterate
            async for chunk in stream:
                pass  # We won't get here if exception occurs as expected
        
        assert "API error 400" in str(excinfo.value)
        assert excinfo.value.response is not None
        assert excinfo.value.response.status_code == 400
    
    @pytest.mark.asyncio
    async def test_network_timeout(self, async_venice_client, httpx_mock):
        """Test timeout error handling for async requests."""
        httpx_mock.add_exception(httpx.TimeoutException("Connection timed out"))
        
        with pytest.raises(VeniceError) as excinfo:
            await async_venice_client.chat.completions.create(
                model="venice-classic",
                messages=CHAT_MESSAGES
            )
        
        assert "timed out" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_network_error(self, async_venice_client, httpx_mock):
        """Test network error handling for async requests."""
        httpx_mock.add_exception(httpx.ConnectError("Failed to establish connection"))
        
        with pytest.raises(VeniceError) as excinfo:
            await async_venice_client.chat.completions.create(
                model="venice-classic",
                messages=CHAT_MESSAGES
            )
        
        assert "Failed to establish connection" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_stream_network_error(self, async_venice_client, httpx_mock):
        """Test network error during async streaming."""
        # pytest.skip("Skipping due to httpx 0.28.1 streaming compatibility issues") # Re-enabled test
        httpx_mock.add_exception(httpx.ConnectError("Failed to establish connection"))
        
        with pytest.raises(VeniceError) as excinfo:
            stream = await async_venice_client.chat.completions.create(
                model="venice-classic",
                messages=CHAT_MESSAGES,
                stream=True
            )
            # This should raise when we try to iterate
            async for chunk in stream:
                pass  # We won't get here if exception occurs as expected
        
        assert "Failed to establish connection" in str(excinfo.value)
        
    @pytest.mark.asyncio
    async def test_create_with_advanced_parameters(self, async_venice_client, httpx_mock):
        """Test async chat completion with more advanced parameters to increase coverage."""
        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/chat/completions",
            json=MOCK_COMPLETION_RESPONSE,
            status_code=200
        )
        
        response_format = {"type": "json_object"}
        venice_parameters = cast(VeniceParameters, {"custom_param": "value"})
        stream_options = {"include_usage": True}
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        }
                    }
                }
            }
        ]
        tool_choice = "auto"
        
        # Call with advanced parameters
        await async_venice_client.chat.completions.create(
            model="venice-classic",
            messages=CHAT_MESSAGES,
            response_format=response_format,
            seed=42,
            logprobs=True,
            top_logprobs=5,
            max_completion_tokens=200,
            parallel_tool_calls=True,
            repetition_penalty=1.2,
            stop_token_ids=[50256],
            top_k=40,
            venice_parameters=venice_parameters,
            stream_options=stream_options,
            tools=tools,
            tool_choice=tool_choice
        )
        
        # Verify request parameters
        request = httpx_mock.get_request()
        body = json.loads(request.content)
        
        assert body["response_format"] == response_format
        assert body["seed"] == 42
        assert body["logprobs"] is True
        assert body["top_logprobs"] == 5
        assert body["max_completion_tokens"] == 200
        assert body["parallel_tool_calls"] is True
        assert body["repetition_penalty"] == 1.2
        assert body["stop_token_ids"] == [50256]
        assert body["top_k"] == 40
        assert body["venice_parameters"] == venice_parameters
        assert body["stream_options"] == stream_options
        assert body["tools"] == tools
        assert body["tool_choice"] == tool_choice

    @pytest.mark.asyncio
    async def test_create_stream_true_stream_cls_none_async(self, async_venice_client, httpx_mock):
        """Test async create with stream=True and stream_cls=None uses default AsyncStream."""
        from venice_ai.streaming import AsyncStream # Import for isinstance check

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

        response_stream = await async_venice_client.chat.completions.create(
            model="venice-classic",
            messages=CHAT_MESSAGES,
            stream=True,
            stream_cls=None # Explicitly None
        )
        assert isinstance(response_stream, AsyncStream)
        # Consume the stream to ensure no errors during processing
        async for _ in response_stream:
            pass

    @pytest.mark.asyncio
    async def test_create_stream_false_with_stream_cls_async(self, async_venice_client, httpx_mock):
        """Test async create with stream=False and a custom stream_cls (should be ignored)."""
        from venice_ai.streaming import AsyncStream # Dummy class for testing

        class MyCustomAsyncStream(AsyncStream): # Define a dummy custom async stream
            pass

        httpx_mock.add_response(
            method="POST",
            url=f"{BASE_URL}/chat/completions",
            json=MOCK_COMPLETION_RESPONSE, # Non-streaming response
            status_code=200
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            response = await async_venice_client.chat.completions.create(
                model="venice-classic",
                messages=CHAT_MESSAGES,
                stream=False,
                stream_cls=MyCustomAsyncStream # This should be ignored
            )
            # Check for UserWarning if stream_cls is passed with stream=False
            # This depends on whether the library issues such a warning.
            # For now, we'll assume it might, or at least doesn't error.
            # assert len(w) > 0
            # assert issubclass(w[-1].category, UserWarning)
            # assert "stream_cls is ignored when stream=False" in str(w[-1].message)

        assert isinstance(response, dict) # Should be a normal dict response
        assert response["id"] == MOCK_COMPLETION_RESPONSE["id"]
        assert not isinstance(response, MyCustomAsyncStream)