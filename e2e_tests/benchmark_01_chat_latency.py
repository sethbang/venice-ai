import pytest
import os
import time
import asyncio
from venice_ai import VeniceClient
from typing import Optional, cast, List
from venice_ai import AsyncVeniceClient

# Initialize clients
# Clients are provided via fixtures from conftest.py

# Test configurations
SHORT_PROMPT = "Hello, how are you?"
LONG_PROMPT = (
    "Please provide a detailed explanation of the following concept: "
    "Artificial Intelligence and its impact on modern society. Include "
    "examples of AI applications, potential benefits, and ethical concerns. "
    "Your response should be comprehensive and cover various aspects of the topic."
)
MODEL = "qwen-2.5-qwq-32b"  # Using a model listed in the test runner output

def measure_time(func):
    """Decorator to measure execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
    return wrapper

def measure_time_async(func):
    """Decorator to measure execution time of an async function."""
    def wrapper(*args, **kwargs):
        async def inner():
            start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            end_time = time.perf_counter()
            return result, end_time - start_time
        return inner()
    return wrapper

# Synchronous Tests
def test_chat_latency_short_prompt_non_streaming(venice_client):
    """Test latency for a short prompt without streaming."""
    @measure_time
    def make_request():
        response = venice_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": SHORT_PROMPT}],
            stream=False
        )
        return response
    
    response, duration = make_request()
    print(f"Latency for short prompt (non-streaming): {duration:.3f} seconds")
    print(f"Response structure: {response}")
    assert response is not None, "Response should not be None"
    assert "choices" in response and len(response.get("choices", [])) > 0, "Response should have at least one choice"

def test_chat_latency_long_prompt_non_streaming(api_key): # Use api_key fixture
    """Test latency for a long prompt without streaming."""
    # Instantiate a client with a longer timeout for this specific test
    custom_timeout_client = VeniceClient(api_key=api_key, timeout=180.0)

    @measure_time
    def make_request():
        try:
            response = custom_timeout_client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": LONG_PROMPT}],
                stream=False
            )
            return response
        except Exception as e:
            print(f"Timeout or error during API call for long prompt: {e}")
            print(f"Error details: {str(e)}")
            return (None, None) # Return tuple to match unpacking
        finally:
            custom_timeout_client.close()
    
    response, duration = make_request()
    print(f"Latency for long prompt (non-streaming): {duration:.3f} seconds")
    print(f"Response structure: {response}")
    assert response is not None, "Response should not be None"
    if response is not None: # Ensure response is not None for Pylance
        assert isinstance(response, dict) and "choices" in response and len(response.get("choices", [])) > 0, "Response should have at least one choice"

def test_chat_latency_short_prompt_streaming(venice_client):
    """Test latency for a short prompt with streaming, measuring time to first token and full response."""
    @measure_time
    def make_request():
        try:
            stream = venice_client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": SHORT_PROMPT}],
                stream=True
            )
            first_token_time = None
            full_response = []
            start_time = time.perf_counter()
            for chunk in stream:
                if first_token_time is None:
                    first_token_time = time.perf_counter() - start_time
                choices = chunk.get("choices", [])
                if choices and len(choices) > 0:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    full_response.append(content or "")
            return first_token_time, full_response
        except Exception as e:
            print(f"Error during API call for short prompt (streaming): {e}")
            return None, [] # Return None for time, empty list for response
    
    (first_token_time, full_response), total_duration = cast(tuple[Optional[float], list[str]], make_request()) # explicit cast for Pylance # type: ignore[reportGeneralTypeIssues]
    print(f"Latency to first token (short prompt, streaming): {first_token_time:.3f} seconds")
    print(f"Total latency for full response (short prompt, streaming): {total_duration:.3f} seconds")
    print(f"First chunk structure: {full_response[0] if full_response else 'No chunks received'}")
    assert first_token_time is not None, "First token time should be recorded"
    assert len(full_response) > 0, "Full response should not be empty"

# Asynchronous Tests
@pytest.mark.asyncio
async def test_chat_latency_short_prompt_non_streaming_async(async_venice_client):
    """Test latency for a short prompt without streaming using async client."""
    @measure_time_async
    async def make_request():
        response = await async_venice_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": SHORT_PROMPT}],
            stream=False
        )
        return response
    
    response, duration = await make_request()
    print(f"Latency for short prompt (non-streaming, async): {duration:.3f} seconds")
    assert response is not None, "Response should not be None"
    assert "choices" in response and len(response.get("choices", [])) > 0, "Response should have at least one choice"

@pytest.mark.asyncio
async def test_chat_latency_long_prompt_non_streaming_async(api_key): # Use api_key fixture
    """Test latency for a long prompt without streaming using async client."""
    # Instantiate an async client with a longer timeout for this specific test
    custom_timeout_async_client = AsyncVeniceClient(api_key=api_key, timeout=120.0)

    @measure_time_async
    async def make_request():
        try:
            response = await custom_timeout_async_client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": LONG_PROMPT}],
                stream=False
            )
            return response
        except Exception as e:
            print(f"Timeout or error during API call for long prompt (async): {e}")
            return (None, None) # Return tuple to match unpacking
        finally:
            await custom_timeout_async_client.close()
    
    response, duration = await make_request()
    print(f"Latency for long prompt (non-streaming, async): {duration:.3f} seconds")
    print(f"Response structure: {response}")
    assert response is not None, "Response should not be None"
    if response is not None: # Ensure response is not None for Pylance
        assert isinstance(response, dict) and "choices" in response and len(response.get("choices", [])) > 0, "Response should have at least one choice"

@pytest.mark.asyncio
async def test_chat_latency_short_prompt_streaming_async(async_venice_client):
    """Test latency for a short prompt with streaming using async client, measuring time to first token and full response."""
    @measure_time_async
    async def make_request():
        try:
            stream = await async_venice_client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": SHORT_PROMPT}],
                stream=True
            )
            first_token_time = None
            full_response = []
            start_time = time.perf_counter()
            async for chunk in stream:
                if first_token_time is None:
                    first_token_time = time.perf_counter() - start_time
                choices = chunk.get("choices", [])
                if choices and len(choices) > 0:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    full_response.append(content)
            return first_token_time, full_response
        except Exception as e:
            print(f"Error during API call for short prompt (streaming, async): {e}")
            return None, [] # Return None for time, empty list for response
    
    (first_token_time, full_response), total_duration = cast(tuple[Optional[float], list[str]], await make_request()) # explicit cast for Pylance # type: ignore[reportGeneralTypeIssues]
    print(f"Latency to first token (short prompt, streaming, async): {first_token_time:.3f} seconds")
    print(f"Total latency for full response (short prompt, streaming, async): {total_duration:.3f} seconds")
    assert first_token_time is not None, "First token time should be recorded"
    assert len(full_response) > 0, "Full response should not be empty"