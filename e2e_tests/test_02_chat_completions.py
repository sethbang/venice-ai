import pytest
import pytest_asyncio
from typing import List, Dict, Any
from venice_ai import VeniceClient, AsyncVeniceClient
from venice_ai.exceptions import VeniceError, InvalidRequestError
from venice_ai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionChunkChoice, MessageParam, Tool, FunctionDefinition, ChatCompletionChoiceLogprobs, ChatCompletionTokenLogprob, ChatCompletionTopLogprob

# Functional Tests for Chat Completions API

def test_create_completion_non_streaming_sync(venice_client: VeniceClient, default_chat_model_id: str):
    """Tests synchronous basic chat completion."""
    messages: List[MessageParam] = [{"role": "user", "content": "Hello, how are you?"}]
    response = venice_client.chat.completions.create(
        model=default_chat_model_id,
        messages=messages
    )
    assert isinstance(response, ChatCompletion)
    assert isinstance(response.choices, list)
    assert len(response.choices) > 0
    assert response.choices[0].message is not None
    assert isinstance(response.choices[0].message.content, str)

@pytest.mark.asyncio
async def test_create_completion_non_streaming_async(async_venice_client: AsyncVeniceClient, default_chat_model_id: str):
    """Tests asynchronous basic chat completion."""
    messages: List[MessageParam] = [{"role": "user", "content": "Hello, how are you?"}]
    response = await async_venice_client.chat.completions.create(
        model=default_chat_model_id,
        messages=messages
    )
    assert isinstance(response, ChatCompletion)
    assert isinstance(response.choices, list)
    assert len(response.choices) > 0
    assert response.choices[0].message is not None
    assert isinstance(response.choices[0].message.content, str)

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
        assert isinstance(chunk, ChatCompletionChunk)
        assert isinstance(chunk.choices, list)
        
        # Skip assertion for empty choices arrays which may occur in final metadata chunks
        if hasattr(chunk, 'usage') and chunk.usage is not None or len(chunk.choices) == 0:
            continue
            
        content_chunks += 1
        assert chunk.choices[0].delta is not None
        # Check for content or other delta fields
        assert (chunk.choices[0].delta.content is not None or
                chunk.choices[0].delta.tool_calls is not None or
                chunk.choices[0].delta.role is not None)
    
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
        assert isinstance(chunk, ChatCompletionChunk)
        assert isinstance(chunk.choices, list)
        
        # Skip assertion for empty choices arrays which may occur in final metadata chunks
        if hasattr(chunk, 'usage') and chunk.usage is not None or len(chunk.choices) == 0:
            continue
            
        content_chunks += 1
        assert chunk.choices[0].delta is not None
        # Check for content or other delta fields
        assert (chunk.choices[0].delta.content is not None or
                chunk.choices[0].delta.tool_calls is not None or
                chunk.choices[0].delta.role is not None)
    
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

    assert isinstance(response, ChatCompletion)
    assert isinstance(response.choices, list)
    assert len(response.choices) > 0
    message = response.choices[0].message
    assert message.tool_calls is not None
    assert isinstance(message.tool_calls, list)
    assert len(message.tool_calls) > 0
    tool_call = message.tool_calls[0]
    assert tool_call.function is not None
    assert tool_call.function.name is not None
    assert tool_call.function.arguments is not None
    assert tool_call.function.name == "get_current_weather"
    assert isinstance(tool_call.function.arguments, str) # Arguments are typically a JSON string

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

    assert isinstance(response, ChatCompletion)
    assert isinstance(response.choices, list)
    assert len(response.choices) > 0
    message = response.choices[0].message
    assert message.tool_calls is not None
    assert isinstance(message.tool_calls, list)
    assert len(message.tool_calls) > 0
    tool_call = message.tool_calls[0]
    assert tool_call.function is not None
    assert tool_call.function.name is not None
    assert tool_call.function.arguments is not None
    assert tool_call.function.name == "get_current_weather"
    assert isinstance(tool_call.function.arguments, str) # Arguments are typically a JSON string


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
    assert isinstance(response, ChatCompletion)
    assert isinstance(response.choices, list)
    assert len(response.choices) > 0
    assert response.choices[0].message.content is not None
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
    assert isinstance(response, ChatCompletion)
    assert isinstance(response.choices, list)
    assert len(response.choices) > 0
    assert response.choices[0].message.content is not None
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


# E2E Tests for logprobs and top_logprobs parameters

def test_create_completion_with_logprobs_sync(venice_client: VeniceClient, default_chat_model_id: str):
    """Tests synchronous chat completion with logprobs and top_logprobs parameters."""
    
    # Test Case 1: logprobs=True, top_logprobs=2
    messages: List[MessageParam] = [{"role": "user", "content": "Explain logprobs."}]
    response = venice_client.chat.completions.create(
        model=default_chat_model_id,
        messages=messages,
        logprobs=True,
        top_logprobs=2
    )
    
    assert isinstance(response, ChatCompletion)
    assert response.choices is not None
    assert isinstance(response.choices, list)
    assert len(response.choices) > 0
    
    choice = response.choices[0]
    assert choice.logprobs is not None
    assert isinstance(choice.logprobs, ChatCompletionChoiceLogprobs)
    assert choice.logprobs.content is not None
    assert isinstance(choice.logprobs.content, list)
    assert len(choice.logprobs.content) > 0
    
    token_logprob = choice.logprobs.content[0]
    assert isinstance(token_logprob, ChatCompletionTokenLogprob)
    assert isinstance(token_logprob.token, str)
    assert isinstance(token_logprob.logprob, float)
    assert token_logprob.top_logprobs is not None
    assert isinstance(token_logprob.top_logprobs, list)
    assert len(token_logprob.top_logprobs) == 2
    
    top_lp = token_logprob.top_logprobs[0]
    assert isinstance(top_lp, ChatCompletionTopLogprob)
    assert isinstance(top_lp.token, str)
    assert isinstance(top_lp.logprob, float)
    assert top_lp.bytes is None or isinstance(top_lp.bytes, list)
    
    # Test Case 2: logprobs=True, top_logprobs=None
    messages2: List[MessageParam] = [{"role": "user", "content": "Explain logprobs again."}]
    response2 = venice_client.chat.completions.create(
        model=default_chat_model_id,
        messages=messages2,
        logprobs=True
    )
    
    assert isinstance(response2, ChatCompletion)
    assert response2.choices is not None
    assert len(response2.choices) > 0
    choice2 = response2.choices[0]
    assert choice2.logprobs is not None
    assert isinstance(choice2.logprobs, ChatCompletionChoiceLogprobs)
    assert choice2.logprobs.content is not None
    assert len(choice2.logprobs.content) > 0
    # top_logprobs should be None or empty list when not requested
    token_logprob2 = choice2.logprobs.content[0]
    assert token_logprob2.top_logprobs is None or token_logprob2.top_logprobs == []
    
    # Test Case 3: logprobs=False, top_logprobs=2 (should raise error or ignore top_logprobs)
    messages3: List[MessageParam] = [{"role": "user", "content": "Test invalid logprobs."}]
    try:
        response3 = venice_client.chat.completions.create(
            model=default_chat_model_id,
            messages=messages3,
            logprobs=False,
            top_logprobs=2
        )
        # If no error, logprobs should be None
        choice3 = response3.choices[0]
        assert choice3.logprobs is None
    except (InvalidRequestError, VeniceError):
        # Expected behavior if API rejects invalid combination
        pass
    
    # Test Case 4: logprobs=False, top_logprobs=None
    messages4: List[MessageParam] = [{"role": "user", "content": "No logprobs please."}]
    response4 = venice_client.chat.completions.create(
        model=default_chat_model_id,
        messages=messages4,
        logprobs=False
    )
    
    assert isinstance(response4, ChatCompletion)
    assert len(response4.choices) > 0
    choice4 = response4.choices[0]
    assert choice4.logprobs is None


@pytest.mark.asyncio
async def test_create_completion_with_logprobs_async(async_venice_client: AsyncVeniceClient, default_chat_model_id: str):
    """Tests asynchronous chat completion with logprobs and top_logprobs parameters."""
    
    # Test Case 1: logprobs=True, top_logprobs=2
    messages: List[MessageParam] = [{"role": "user", "content": "Explain logprobs."}]
    response = await async_venice_client.chat.completions.create(
        model=default_chat_model_id,
        messages=messages,
        logprobs=True,
        top_logprobs=2
    )
    
    assert isinstance(response, ChatCompletion)
    assert response.choices is not None
    assert isinstance(response.choices, list)
    assert len(response.choices) > 0
    
    choice = response.choices[0]
    assert choice.logprobs is not None
    assert isinstance(choice.logprobs, ChatCompletionChoiceLogprobs)
    assert choice.logprobs.content is not None
    assert isinstance(choice.logprobs.content, list)
    assert len(choice.logprobs.content) > 0
    
    token_logprob = choice.logprobs.content[0]
    assert isinstance(token_logprob, ChatCompletionTokenLogprob)
    assert isinstance(token_logprob.token, str)
    assert isinstance(token_logprob.logprob, float)
    assert token_logprob.top_logprobs is not None
    assert isinstance(token_logprob.top_logprobs, list)
    assert len(token_logprob.top_logprobs) == 2
    
    top_lp = token_logprob.top_logprobs[0]
    assert isinstance(top_lp, ChatCompletionTopLogprob)
    assert isinstance(top_lp.token, str)
    assert isinstance(top_lp.logprob, float)
    assert top_lp.bytes is None or isinstance(top_lp.bytes, list)
    
    # Test Case 2: logprobs=True, top_logprobs=None
    messages2: List[MessageParam] = [{"role": "user", "content": "Explain logprobs again."}]
    response2 = await async_venice_client.chat.completions.create(
        model=default_chat_model_id,
        messages=messages2,
        logprobs=True
    )
    
    assert isinstance(response2, ChatCompletion)
    assert response2.choices is not None
    assert len(response2.choices) > 0
    choice2 = response2.choices[0]
    assert choice2.logprobs is not None
    assert isinstance(choice2.logprobs, ChatCompletionChoiceLogprobs)
    assert choice2.logprobs.content is not None
    assert len(choice2.logprobs.content) > 0
    # top_logprobs should be None or empty list when not requested
    token_logprob2 = choice2.logprobs.content[0]
    assert token_logprob2.top_logprobs is None or token_logprob2.top_logprobs == []
    
    # Test Case 3: logprobs=False, top_logprobs=2 (should raise error or ignore top_logprobs)
    messages3: List[MessageParam] = [{"role": "user", "content": "Test invalid logprobs."}]
    try:
        response3 = await async_venice_client.chat.completions.create(
            model=default_chat_model_id,
            messages=messages3,
            logprobs=False,
            top_logprobs=2
        )
        # If no error, logprobs should be None
        choice3 = response3.choices[0]
        assert choice3.logprobs is None
    except (InvalidRequestError, VeniceError):
        # Expected behavior if API rejects invalid combination
        pass
    
    # Test Case 4: logprobs=False, top_logprobs=None
    messages4: List[MessageParam] = [{"role": "user", "content": "No logprobs please."}]
    response4 = await async_venice_client.chat.completions.create(
        model=default_chat_model_id,
        messages=messages4,
        logprobs=False
    )
    
    assert isinstance(response4, ChatCompletion)
    assert len(response4.choices) > 0
    choice4 = response4.choices[0]
    assert choice4.logprobs is None


def test_create_completion_streaming_with_logprobs_sync(venice_client: VeniceClient, default_chat_model_id: str):
    """Tests synchronous streaming chat completion with logprobs."""
    
    messages: List[MessageParam] = [{"role": "user", "content": "Stream logprobs."}]
    stream = venice_client.chat.completions.create(
        model=default_chat_model_id,
        messages=messages,
        stream=True,
        logprobs=True,
        top_logprobs=1
    )
    
    chunks = list(stream)
    assert len(chunks) > 0
    
    logprobs_found_in_stream = False
    for chunk in chunks:
        assert isinstance(chunk, ChatCompletionChunk)
        assert isinstance(chunk.choices, list)
        
        if len(chunk.choices) > 0:
            choice_chunk = chunk.choices[0]
            assert isinstance(choice_chunk, ChatCompletionChunkChoice)
            
            if choice_chunk.logprobs is not None:
                logprobs_found_in_stream = True
                assert isinstance(choice_chunk.logprobs, ChatCompletionChoiceLogprobs)
                assert choice_chunk.logprobs.content is not None
                assert isinstance(choice_chunk.logprobs.content, list)
                
                if len(choice_chunk.logprobs.content) > 0:
                    token_logprob_chunk = choice_chunk.logprobs.content[0]
                    assert isinstance(token_logprob_chunk, ChatCompletionTokenLogprob)
                    assert isinstance(token_logprob_chunk.token, str)
                    assert isinstance(token_logprob_chunk.logprob, float)
                    
                    if token_logprob_chunk.top_logprobs is not None:
                        assert len(token_logprob_chunk.top_logprobs) == 1
                        top_lp = token_logprob_chunk.top_logprobs[0]
                        assert isinstance(top_lp, ChatCompletionTopLogprob)
                        assert isinstance(top_lp.token, str)
                        assert isinstance(top_lp.logprob, float)
    
    # Crucial assertion: logprobs should be found in the stream
    assert logprobs_found_in_stream is True


@pytest.mark.asyncio
async def test_create_completion_streaming_with_logprobs_async(async_venice_client: AsyncVeniceClient, default_chat_model_id: str):
    """Tests asynchronous streaming chat completion with logprobs."""
    
    messages: List[MessageParam] = [{"role": "user", "content": "Stream logprobs."}]
    stream = await async_venice_client.chat.completions.create(
        model=default_chat_model_id,
        messages=messages,
        stream=True,
        logprobs=True,
        top_logprobs=1
    )
    
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    
    assert len(chunks) > 0
    
    logprobs_found_in_stream = False
    for chunk in chunks:
        assert isinstance(chunk, ChatCompletionChunk)
        assert isinstance(chunk.choices, list)
        
        if len(chunk.choices) > 0:
            choice_chunk = chunk.choices[0]
            assert isinstance(choice_chunk, ChatCompletionChunkChoice)
            
            if choice_chunk.logprobs is not None:
                logprobs_found_in_stream = True
                assert isinstance(choice_chunk.logprobs, ChatCompletionChoiceLogprobs)
                assert choice_chunk.logprobs.content is not None
                assert isinstance(choice_chunk.logprobs.content, list)
                
                if len(choice_chunk.logprobs.content) > 0:
                    token_logprob_chunk = choice_chunk.logprobs.content[0]
                    assert isinstance(token_logprob_chunk, ChatCompletionTokenLogprob)
                    assert isinstance(token_logprob_chunk.token, str)
                    assert isinstance(token_logprob_chunk.logprob, float)
                    
                    if token_logprob_chunk.top_logprobs is not None:
                        assert len(token_logprob_chunk.top_logprobs) == 1
                        top_lp = token_logprob_chunk.top_logprobs[0]
                        assert isinstance(top_lp, ChatCompletionTopLogprob)
                        assert isinstance(top_lp.token, str)
                        assert isinstance(top_lp.logprob, float)
    
    # Crucial assertion: logprobs should be found in the stream
    assert logprobs_found_in_stream is True