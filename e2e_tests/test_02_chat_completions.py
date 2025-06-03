import pytest
import pytest_asyncio
from typing import List, Dict, Any
from venice_ai import VeniceClient, AsyncVeniceClient
from venice_ai.exceptions import VeniceError, InvalidRequestError
from venice_ai.types.chat import ChatCompletionChunk, MessageParam, Tool, FunctionDefinition

# Functional Tests for Chat Completions API

def test_create_completion_non_streaming_sync(venice_client: VeniceClient, default_chat_model_id: str):
    """Tests synchronous basic chat completion."""
    messages: List[MessageParam] = [{"role": "user", "content": "Hello, how are you?"}]
    response = venice_client.chat.completions.create(
        model=default_chat_model_id,
        messages=messages
    )
    assert isinstance(response, dict)
    assert "choices" in response
    assert isinstance(response["choices"], list)
    assert len(response["choices"]) > 0
    assert "message" in response["choices"][0]
    assert "content" in response["choices"][0]["message"]
    assert isinstance(response["choices"][0]["message"]["content"], str)

@pytest.mark.asyncio
async def test_create_completion_non_streaming_async(async_venice_client: AsyncVeniceClient, default_chat_model_id: str):
    """Tests asynchronous basic chat completion."""
    messages: List[MessageParam] = [{"role": "user", "content": "Hello, how are you?"}]
    response = await async_venice_client.chat.completions.create(
        model=default_chat_model_id,
        messages=messages
    )
    assert isinstance(response, dict)
    assert "choices" in response
    assert isinstance(response["choices"], list)
    assert len(response["choices"]) > 0
    assert "message" in response["choices"][0]
    assert "content" in response["choices"][0]["message"]
    assert isinstance(response["choices"][0]["message"]["content"], str)

def test_create_completion_streaming_sync(venice_client: VeniceClient, default_chat_model_id: str):
    """Tests synchronous streaming chat completion."""
    messages: List[MessageParam] = [{"role": "user", "content": "Tell me a short story."}]
    stream = venice_client.chat.completions.create(
        model=default_chat_model_id,
        messages=messages,
        stream=True
    )
    chunks = list(stream)
    
    # Check if chunks are returned
    assert len(chunks) > 0, "Streaming response returned no chunks"
    
    # Count content chunks (chunks with non-empty choices array and delta)
    content_chunks = 0
    for chunk in chunks:
        assert isinstance(chunk, dict)
        assert "choices" in chunk
        assert isinstance(chunk["choices"], list)
        
        # Skip assertion for empty choices arrays which may occur in final metadata chunks
        if "usage" in chunk or len(chunk["choices"]) == 0:
            continue
            
        content_chunks += 1
        assert "delta" in chunk["choices"][0]
        # Check for content or other delta fields
        assert ("content" in chunk["choices"][0]["delta"] or
                "tool_calls" in chunk["choices"][0]["delta"] or
                "role" in chunk["choices"][0]["delta"])
    
    # Ensure we received at least some content chunks
    assert content_chunks > 0, "No content chunks received in stream"

@pytest.mark.asyncio
async def test_create_completion_streaming_async(async_venice_client: AsyncVeniceClient, default_chat_model_id: str):
    """Tests asynchronous streaming chat completion."""
    messages: List[MessageParam] = [{"role": "user", "content": "Tell me a short story."}]
    stream = await async_venice_client.chat.completions.create(
        model=default_chat_model_id,
        messages=messages,
        stream=True
    )
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    # Check if chunks are returned
    assert len(chunks) > 0, "Streaming response returned no chunks"
    
    # Count content chunks (chunks with non-empty choices array and delta)
    content_chunks = 0
    for chunk in chunks:
        assert isinstance(chunk, dict)
        assert "choices" in chunk
        assert isinstance(chunk["choices"], list)
        
        # Skip assertion for empty choices arrays which may occur in final metadata chunks
        if "usage" in chunk or len(chunk["choices"]) == 0:
            continue
            
        content_chunks += 1
        assert "delta" in chunk["choices"][0]
        # Check for content or other delta fields
        assert ("content" in chunk["choices"][0]["delta"] or
                "tool_calls" in chunk["choices"][0]["delta"] or
                "role" in chunk["choices"][0]["delta"])
    
    # Ensure we received at least some content chunks
    assert content_chunks > 0, "No content chunks received in stream"


def test_create_completion_with_tool_calls_sync(venice_client: VeniceClient, default_chat_model_id: str):
    """Tests synchronous chat completion with tool call request."""
    # Define a simple mock tool
    tools: List[Tool] = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g., San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    messages: List[MessageParam] = [{"role": "user", "content": "What's the weather like in London?"}]

    # --- Debugging: List models and check capabilities ---
    print(f"--- Debugging: Listing models before sync tool call test ---")
    try:
        models_list = venice_client.models.list()
        print("Available Models (sync):")
        for model_info in models_list.get("data", []):
            model_id = model_info.get("id")
            capabilities = model_info.get("model_spec", {}).get("capabilities", {})
            supports_tool_calls = capabilities.get("supportsFunctionCalling", False)
            print(f"  Model: {model_id}, Supports Tool Calls: {supports_tool_calls}")
            if model_id == default_chat_model_id:
                 print(f"  {default_chat_model_id} supports tool calls: {supports_tool_calls}")
                 print(f"  {default_chat_model_id} capabilities: {capabilities}") # Added logging for full capabilities
    except Exception as e:
        print(f"Error listing models (sync): {e}")
    print(f"--- End Debugging ---")

    response = venice_client.chat.completions.create(
        model=default_chat_model_id, # Ensure this model supports tool calls
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    assert isinstance(response, dict)
    assert "choices" in response
    assert isinstance(response["choices"], list)
    assert len(response["choices"]) > 0
    message = response["choices"][0]["message"]
    assert "tool_calls" in message
    assert isinstance(message["tool_calls"], list)
    assert len(message["tool_calls"]) > 0
    tool_call = message["tool_calls"][0]
    assert "function" in tool_call
    assert "name" in tool_call["function"]
    assert "arguments" in tool_call["function"]
    assert tool_call["function"]["name"] == "get_current_weather"
    assert isinstance(tool_call["function"]["arguments"], str) # Arguments are typically a JSON string

@pytest.mark.asyncio
async def test_create_completion_with_tool_calls_async(async_venice_client: AsyncVeniceClient, default_chat_model_id: str):
    """Tests asynchronous chat completion with tool call request."""
    # Define a simple mock tool
    tools: List[Tool] = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g., San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    messages: List[MessageParam] = [{"role": "user", "content": "What's the weather like in Paris?"}]

    # --- Debugging: List models and check capabilities ---
    print(f"--- Debugging: Listing models before async tool call test ---")
    try:
        models_list = await async_venice_client.models.list()
        print("Available Models (async):")
        for model_info in models_list.get("data", []):
            model_id = model_info.get("id")
            capabilities = model_info.get("model_spec", {}).get("capabilities", {})
            supports_tool_calls = capabilities.get("supportsFunctionCalling", False)
            print(f"  Model: {model_id}, Supports Tool Calls: {supports_tool_calls}")
            if model_id == default_chat_model_id:
                 print(f"  {default_chat_model_id} supports tool calls: {supports_tool_calls}")
                 print(f"  {default_chat_model_id} capabilities: {capabilities}") # Added logging for full capabilities
    except Exception as e:
        print(f"Error listing models (async): {e}")
    print(f"--- End Debugging ---")

    response = await async_venice_client.chat.completions.create(
        model=default_chat_model_id, # Ensure this model supports tool calls
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    assert isinstance(response, dict)
    assert "choices" in response
    assert isinstance(response["choices"], list)
    assert len(response["choices"]) > 0
    message = response["choices"][0]["message"]
    assert "tool_calls" in message
    assert isinstance(message["tool_calls"], list)
    assert len(message["tool_calls"]) > 0
    tool_call = message["tool_calls"][0]
    assert "function" in tool_call
    assert "name" in tool_call["function"]
    assert "arguments" in tool_call["function"]
    assert tool_call["function"]["name"] == "get_current_weather"
    assert isinstance(tool_call["function"]["arguments"], str) # Arguments are typically a JSON string


def test_create_completion_with_various_params_sync(venice_client: VeniceClient, default_chat_model_id: str):
    """Tests synchronous chat completion with various parameters."""
    messages: List[MessageParam] = [{"role": "user", "content": "Write a short poem about nature."}]
    response = venice_client.chat.completions.create(
        model=default_chat_model_id,
        messages=messages,
        temperature=0.7,
        max_completion_tokens=50,
        top_p=0.9,
        # Add other relevant parameters as needed
    )
    assert isinstance(response, dict)
    assert "choices" in response
    assert len(response["choices"]) > 0
    assert "content" in response["choices"][0]["message"]
    # Add assertions to check if parameters influenced the response as expected (qualitative or quantitative checks)

@pytest.mark.asyncio
async def test_create_completion_with_various_params_async(async_venice_client: AsyncVeniceClient, default_chat_model_id: str):
    """Tests asynchronous chat completion with various parameters."""
    messages: List[MessageParam] = [{"role": "user", "content": "Write a short poem about the sea."}]
    response = await async_venice_client.chat.completions.create(
        model=default_chat_model_id,
        messages=messages,
        temperature=0.5,
        max_completion_tokens=75,
        top_p=0.8,
        # Add other relevant parameters as needed
    )
    assert isinstance(response, dict)
    assert "choices" in response
    assert len(response["choices"]) > 0
    assert "content" in response["choices"][0]["message"]
    # Add assertions to check if parameters influenced the response as expected


def test_create_completion_error_invalid_model_sync(venice_client: VeniceClient):
    """Tests synchronous chat completion with an invalid model."""
    messages: List[MessageParam] = [{"role": "user", "content": "This should fail."}]
    with pytest.raises(VeniceError) as excinfo: # Or a more specific API error like InvalidRequestError
        venice_client.chat.completions.create(
            model="non-existent-model-12345",
            messages=messages
        )
    # Assert on the error type and potentially the error message or status code
    assert isinstance(excinfo.value, VeniceError) # Check if it's a VeniceError or subclass
    # assert "Invalid model" in str(excinfo.value) # Example assertion on error message

@pytest.mark.asyncio
async def test_create_completion_error_invalid_model_async(async_venice_client: AsyncVeniceClient):
    """Tests asynchronous chat completion with an invalid model."""
    messages: List[MessageParam] = [{"role": "user", "content": "This should fail."}]
    with pytest.raises(VeniceError) as excinfo: # Or a more specific API error like InvalidRequestError
        await async_venice_client.chat.completions.create(
            model="another-invalid-model-67890",
            messages=messages
        )
    # Assert on the error type and potentially the error message or status code
    assert isinstance(excinfo.value, VeniceError) # Check if it's a VeniceError or subclass
    # assert "Invalid model" in str(excinfo.value) # Example assertion on error message