Client Utilities
================

This document describes the utility functions available in the Venice AI client library for Python.

Overview
--------

The Venice AI client library includes a set of utility functions that simplify common operations when working with the Venice AI API. These utilities are available in the ``venice_ai.utils`` module and can be used alongside the main client classes.

.. automodule:: venice_ai.utils
   :members:

Usage Examples
--------------

Model Utilities
~~~~~~~~~~~~~~~

.. code-block:: python

    # Find and filter models
    from venice_ai import VeniceClient
    from venice_ai.utils import get_filtered_models, find_model_by_id, get_model_capabilities
    import asyncio

    async def example_model_utilities():
        client = VeniceClient(api_key="your-api-key")
        
        # Get all text models that support function calling
        text_models = await get_filtered_models(
            client,
            model_type="text",
            supports_capabilities=["supportsFunctionCalling"]
        )
        print(f"Found {len(text_models)} text models with function calling")
        
        # Find specific model
        model = await find_model_by_id(client, "llama-3.3-70b")
        if model:
            print(f"Found model: {model['id']}, Type: {model.get('type')}")
            
            # Check model capabilities
            capabilities = await get_model_capabilities(client, model['id'])
            if capabilities:
                print(f"Model capabilities: {capabilities}")

    # Run the async example
    asyncio.run(example_model_utilities())

Chat Message Utilities
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from venice_ai.utils import validate_chat_messages, format_tool_response

    # Validate message structure
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in New York?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\":\"New York\"}"
                }
            }
        ]}
    ]
    
    # Validate messages
    result = validate_chat_messages(messages)
    if not result["errors"]:
        print("Messages are valid")
    else:
        print("Validation errors:", result["errors"])
    
    # Format a tool response
    weather_data = {"temperature": 22, "condition": "sunny"}
    tool_message = format_tool_response("call_abc123", weather_data)
    messages.append(tool_message)
    
    # Check that new message sequence is valid
    result = validate_chat_messages(messages)
    print("Updated messages valid:", not result["errors"])

Token Counting
~~~~~~~~~~~~~~

.. code-block:: python

    from venice_ai.utils import estimate_token_count

    # Count tokens in various text chunks
    text = "This is a sample text to count tokens in."
    token_count = estimate_token_count(text)
    print(f"Estimated token count: {token_count}")

Installation Requirements
-------------------------

Some utilities have optional dependencies:

- For ``estimate_token_count`` to use the accurate tokenization method: ``pip install tiktoken``

Best Practices
--------------

- **Error Handling**: Always check for errors when using validation functions.
- **Capabilities Checking**: Use the model utility functions to ensure you're working with models that support your required features.
- **Token Management**: Use ``estimate_token_count`` to proactively manage token usage within limits.
- **Async Consistency**: Most model utility functions are async and should be awaited properly.