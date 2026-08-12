#!/usr/bin/env python3
"""
Venice AI SDK - Pydantic Model Best Practices
==============================================

This file is THE definitive reference for proper Pydantic model usage in the Venice AI SDK.

PURPOSE
-------
Comprehensive guide to demonstrate the CORRECT patterns for working with all Venice AI SDK types.


HOW TO USE THIS FILE
--------------------
This file serves dual purposes:

1. **Executable Examples**: Run with `poetry run python examples/best_practices/pydantic_models.py`
   (requires VENICE_API_KEY environment variable) to see working examples.

2. **Reference Guide**: Read the code to learn proper patterns. Each section shows:
   - ✅ CORRECT patterns with detailed explanations
   - ❌ ANTI-PATTERNS (commented out) showing what NOT to do
   - 🔧 Tool definitions and usage patterns
   - 📊 Response handling and data extraction

KEY SECTIONS OVERVIEW
---------------------
1. Message Models - Proper construction of UserMessage, SystemMessage, AssistantMessage
2. Response Models - Safe access to ChatCompletion response fields
3. Tool Calling - CRITICAL: The correct way to define and use tools
4. Streaming - Type-safe async iteration over ChatCompletionChunk objects
5. Request Configuration - Using StreamOptions, VeniceParameters, JSONSchemaFormat
6. Type Safety - Full type annotations with TYPE_CHECKING imports
7. Common Pitfalls - Quick reference guide to all anti-patterns

IMPORTANT NOTES
---------------
- ALL request/response objects are Pydantic models - treat them as such!
- NEVER use dict-style access like obj['field'] - always use obj.field
- ALWAYS check for None before accessing optional fields
- USE proper type hints for better IDE support and type checking
- READ the inline comments - they explain WHY things work this way

"""

import asyncio
import json
import sys
from typing import TYPE_CHECKING

from pydantic import ValidationError

from venice_ai import VeniceClient
from venice_ai.types.api.requests import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from venice_ai.types.api.requests.common import (
    ImageContent,
    ImageUrl,
    JSONSchemaFormat,
    StreamOptions,
    TextContent,
    Tool,
    ToolFunction,
    VeniceParameters,
)

# Type-only imports to avoid circular dependencies
if TYPE_CHECKING:
    pass


# =============================================================================
# SECTION 1: MESSAGE MODELS
# =============================================================================
# Demonstrates proper construction and usage of message objects.
# Messages are the foundation of all chat interactions.
# =============================================================================


async def example_message_models_correct():
    """
    ✅ CORRECT: Demonstrates proper message model construction.

    All messages in the Venice AI SDK are Pydantic models, not dicts!
    This provides type safety, validation, and clear structure.
    """
    print("\n" + "=" * 70)
    print("Section 1: Message Models - CORRECT Patterns")
    print("=" * 70)

    # ✅ CORRECT: UserMessage with simple text content
    # This is the most common message type for user input
    user_msg = UserMessage(
        content="What is the capital of France?"  # Simple string content
    )
    print("\n✅ UserMessage created:")
    print(f"   role: {user_msg.role}")
    print(f"   content: {user_msg.content}")

    # ✅ CORRECT: SystemMessage for setting context
    # System messages guide the model's behavior and personality
    system_msg = SystemMessage(
        content="You are a helpful geography tutor. Provide concise, accurate answers.",
    )
    print("\n✅ SystemMessage created:")
    print(f"   role: {system_msg.role}")
    print(f"   content: {system_msg.content}")

    # ✅ CORRECT: AssistantMessage for model responses
    # Used when building conversation history or few-shot examples
    assistant_msg = AssistantMessage(
        content="The capital of France is Paris.",
    )
    print("\n✅ AssistantMessage created:")
    print(f"   role: {assistant_msg.role}")
    print(f"   content: {assistant_msg.content}")

    # ✅ CORRECT: UserMessage with multimodal content (text + image)
    # This demonstrates the more advanced content field structure
    multimodal_msg = UserMessage(
        content=[
            TextContent(type="text", text="What do you see in this image?"),
            ImageContent(
                type="image_url",
                image_url=ImageUrl(
                    url="https://example.com/image.jpg"
                    # Note: ImageUrl only has 'url' parameter
                ),
            ),
        ]
    )
    print("\n✅ Multimodal UserMessage created:")
    print(f"   role: {multimodal_msg.role}")
    print(f"   content type: {type(multimodal_msg.content)}")
    print(f"   content items: {len(multimodal_msg.content)} (text + image)")

    # ✅ CORRECT: Accessing message properties using Pydantic model attributes
    # This is type-safe and validated at runtime
    print("\n✅ Accessing message properties (Pydantic style):")
    print(f"   user_msg.role = '{user_msg.role}'")  # Attribute access
    print(f"   user_msg.content = '{user_msg.content}'")  # Attribute access

    # ✅ CORRECT: Building a conversation history
    # Messages are combined in a list for multi-turn conversations
    conversation = [system_msg, user_msg, assistant_msg]
    print("\n✅ Conversation history created:")
    print(f"   Total messages: {len(conversation)}")
    for i, msg in enumerate(conversation):
        print(f"   Message {i + 1}: {msg.role}")


def example_message_models_anti_patterns():
    """
    ❌ ANTI-PATTERNS: What NOT to do with message models.

    These patterns are WRONG and will cause runtime errors or bypass type safety.
    They are shown here for educational purposes only - DO NOT USE THEM!
    """
    print("\n" + "=" * 70)
    print("Section 1: Message Models - ANTI-PATTERNS (What NOT to Do)")
    print("=" * 70)

    print("\n❌ WRONG: Using plain dicts instead of Pydantic models")
    print("   # DON'T DO THIS:")
    print("   # user_msg = {'role': 'user', 'content': 'Hello'}")
    print("   # This bypasses all validation and type checking!")

    print("\n❌ WRONG: Dict-style access on Pydantic models")
    print("   # DON'T DO THIS:")
    print("   # content = user_msg['content']  # ❌ Wrong!")
    print("   # INSTEAD USE:")
    print("   # content = user_msg.content  # ✅ Correct!")

    print("\n❌ WRONG: Not checking content type before accessing")
    print("   # DON'T DO THIS:")
    print("   # text = user_msg.content  # Might be a list!")
    print("   # INSTEAD USE:")
    print("   # if isinstance(user_msg.content, str):")
    print("   #     text = user_msg.content")
    print("   # elif isinstance(user_msg.content, list):")
    print("   #     # Handle multimodal content")

    print("\n❌ WRONG: Mutating message fields directly")
    print("   # DON'T DO THIS:")
    print("   # user_msg.role = 'assistant'  # Wrong role!")
    print("   # INSTEAD: Create a new message object")

    print("\n💡 Key Takeaway: Always use Pydantic models, never plain dicts!")


def example_validation_rejections_live():
    """
    🛑 LIVE DEMO: Actually trigger Pydantic validation errors.

    The anti-pattern functions above are commented strings. Here we run the
    bad code in a try/except so you can see the rejected input and the error
    message Pydantic emits.
    """
    print("\n" + "=" * 70)
    print("Section 1b: Validation Rejections (LIVE)")
    print("=" * 70)

    # 1. Wrong role on UserMessage — Pydantic enforces role="user".
    bad_role_input = {"role": "assistant", "content": "Hello"}
    print(f"\n🛑 Attempting: UserMessage(**{bad_role_input})")
    try:
        UserMessage(**bad_role_input)  # type: ignore[arg-type]
        print("   ⚠️  Expected ValidationError but construction succeeded")
    except ValidationError as e:
        print(f"   ✅ Rejected: {e.errors()[0]['msg']}")
        print(f"      loc={e.errors()[0]['loc']}, input={e.errors()[0]['input']}")

    # 2. Missing required field — `name` on ToolFunction.
    print('\n🛑 Attempting: ToolFunction(description="missing name")')
    try:
        ToolFunction(description="missing name")  # type: ignore[call-arg]
        print("   ⚠️  Expected ValidationError but construction succeeded")
    except ValidationError as e:
        print(f"   ✅ Rejected: {e.errors()[0]['msg']}")
        print(f"      loc={e.errors()[0]['loc']}")

    # 3. Wrong type — `arguments` on a tool call must be a JSON string,
    #    `parameters` on ToolFunction must be a dict.
    bad_params = "this should be a dict, not a string"
    print(f'\n🛑 Attempting: ToolFunction(name="x", parameters={bad_params!r})')
    try:
        ToolFunction(name="x", parameters=bad_params)  # type: ignore[arg-type]
        print("   ⚠️  Expected ValidationError but construction succeeded")
    except ValidationError as e:
        print(f"   ✅ Rejected: {e.errors()[0]['msg']}")
        print(f"      loc={e.errors()[0]['loc']}")

    # 4. Out-of-range — StreamOptions.include_usage must be a bool.
    print('\n🛑 Attempting: StreamOptions(include_usage="not a bool")')
    try:
        StreamOptions(include_usage="not a bool")  # type: ignore[arg-type]
        print("   ⚠️  Expected ValidationError but construction succeeded")
    except ValidationError as e:
        print(f"   ✅ Rejected: {e.errors()[0]['msg']}")
        print(f"      loc={e.errors()[0]['loc']}")

    print("\n💡 Pydantic catches these at construction time, before any API call.")


# =============================================================================
# SECTION 2: RESPONSE MODELS
# =============================================================================
# Demonstrates safe access to ChatCompletion response objects.
# Responses contain choices, messages, usage stats, and more.
# =============================================================================


async def example_response_access_correct() -> bool:
    """
    ✅ CORRECT: Safe access to ChatCompletion response fields.

    Response objects are complex Pydantic models with nested structures.
    Always check for None and use proper attribute access.

    Returns True on success, False if the live request failed.
    """
    print("\n" + "=" * 70)
    print("Section 2: Response Access - CORRECT Patterns")
    print("=" * 70)

    async with VeniceClient() as client:
        try:
            # Make a simple request
            model = await client.models.resolve_chat()
            response = await client.chat.completions.create(
                model=model,
                messages=[UserMessage(content="Say 'Hello!' in exactly one word.")],
                max_completion_tokens=10,
            )

            # ✅ CORRECT: Safe access to response.choices
            # Always check if choices exist before accessing
            if response.choices:
                print("\n✅ Accessing response.choices safely:")
                print(f"   Number of choices: {len(response.choices)}")

                # ✅ CORRECT: Access first choice using attribute access
                first_choice = response.choices[0]
                print(f"   First choice index: {first_choice.index}")
                print(f"   Finish reason: {first_choice.finish_reason}")

                # ✅ CORRECT: Access message from choice
                message = first_choice.message
                print("\n✅ Accessing message:")
                print(f"   Role: {message.role}")

                # ✅ CORRECT: Safe access to optional content field
                # Content might be None if there were tool calls instead
                if message.content:
                    print(f"   Content: {message.content}")
                else:
                    print("   Content: (None - likely had tool calls)")

                # ✅ CORRECT: Check for tool_calls before accessing
                if message.tool_calls:
                    print(f"   Tool calls: {len(message.tool_calls)}")
                else:
                    print("   Tool calls: None")

            # ✅ CORRECT: Safe access to usage statistics
            # Usage is always present but good to be defensive
            if response.usage:
                usage = response.usage
                print("\n✅ Accessing usage statistics:")
                print(f"   Prompt tokens: {usage.prompt_tokens}")
                print(f"   Completion tokens: {usage.completion_tokens}")
                print(f"   Total tokens: {usage.total_tokens}")

            # ✅ CORRECT: Access response ID
            print("\n✅ Response metadata:")
            print(f"   ID: {response.id}")
            print(f"   Model: {response.model}")
            print(f"   Created: {response.created}")

            return True

        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False


def example_response_access_anti_patterns():
    """
    ❌ ANTI-PATTERNS: Unsafe response access patterns.

    These patterns can cause runtime errors or produce incorrect results.
    """
    print("\n" + "=" * 70)
    print("Section 2: Response Access - ANTI-PATTERNS (What NOT to Do)")
    print("=" * 70)

    print("\n❌ WRONG: Accessing without checking for None")
    print("   # DON'T DO THIS:")
    print("   # content = response.text")
    print("   # This will crash if content is None!")
    print("   # INSTEAD USE:")
    print("   # if response.choices and response.text:")
    print("   #     content = response.text")

    print("\n❌ WRONG: Dict-style access on response")
    print("   # DON'T DO THIS:")
    print("   # content = response['choices'][0]['message']['content']")
    print("   # INSTEAD USE:")
    print("   # content = response.text")

    print("\n❌ WRONG: Assuming tool_calls always exists")
    print("   # DON'T DO THIS:")
    print("   # for tool_call in message.tool_calls:")
    print("   #     # This crashes if tool_calls is None!")
    print("   # INSTEAD USE:")
    print("   # if message.tool_calls:")
    print("   #     for tool_call in message.tool_calls:")

    print("\n❌ WRONG: Not checking if choices array is empty")
    print("   # DON'T DO THIS:")
    print("   # message = response.choices[0].message  # IndexError if empty!")
    print("   # INSTEAD USE:")
    print("   # if response.choices:")
    print("   #     message = response.choices[0].message")

    print("\n💡 Key Takeaway: Always check for None and empty arrays!")


# =============================================================================
# SECTION 3: TOOL CALLING - CRITICAL SECTION
# =============================================================================
# This section demonstrates the CORRECT way to work with tools.
# =============================================================================


async def example_tool_definition_correct():
    """
    ✅ CORRECT: Define tools using Tool and ToolFunction Pydantic models.

    Tools must be defined as proper Pydantic models, not plain dicts.
    This ensures validation and type safety.
    """
    print("\n" + "=" * 70)
    print("Section 3: Tool Definition - CORRECT Patterns")
    print("=" * 70)

    # ✅ CORRECT: Tool definition using Pydantic models
    # This is the ONLY correct way to define tools
    weather_tool = Tool(
        type="function",  # Must be "function" for function calling
        function=ToolFunction(
            name="get_weather",  # Clear, descriptive function name
            description="Get current weather for a location",  # Helps model decide when to use it
            parameters={
                # JSON Schema for function parameters
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g. 'San Francisco, CA'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],  # Only location is required
            },
            strict=False,  # Set to True for strict schema validation
        ),
        id=None,  # Optional: tool ID
    )

    print("\n✅ Tool defined using Pydantic models:")
    print(f"   Type: {weather_tool.type}")
    assert weather_tool.function is not None
    print(f"   Function name: {weather_tool.function.name}")
    print(f"   Function description: {weather_tool.function.description}")
    print(f"   Strict validation: {weather_tool.function.strict}")

    # ✅ CORRECT: Accessing tool properties using Pydantic attributes
    func = weather_tool.function
    print("\n✅ Accessing tool properties (Pydantic style):")
    print(f"   func.name = '{func.name}'")  # Attribute access
    print(f"   func.description = '{func.description}'")  # Attribute access

    # ✅ CORRECT: Multiple tools in a list
    tools = [
        weather_tool,
        Tool(
            type="function",
            function=ToolFunction(
                name="search_web",
                description="Search the web for current information",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}},
                    "required": ["query"],
                },
                strict=False,
            ),
            id=None,  # Optional: tool ID
        ),
    ]

    print("\n✅ Multiple tools defined:")
    for i, tool in enumerate(tools):
        assert tool.function is not None
        print(f"   Tool {i + 1}: {tool.function.name}")


async def example_tool_calling_correct() -> bool:
    """
    🔧 CRITICAL: This demonstrates the CORRECT way to access tool calls.

    This is the correct, type-safe pattern for accessing tool calls.
    ALWAYS use Pydantic model property access, NEVER dict-style access.

    Returns True on success, False if the live request failed.
    """
    print("\n" + "=" * 70)
    print("Section 3: Tool Calling - CORRECT Patterns ⚡ CRITICAL")
    print("=" * 70)

    async with VeniceClient() as client:
        try:
            # Define tool
            calc_tool = Tool(
                type="function",
                function=ToolFunction(
                    name="calculate",
                    description="Perform basic math calculation",
                    parameters={
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Math expression to evaluate",
                            }
                        },
                        "required": ["expression"],
                    },
                    strict=False,
                ),
                id=None,  # Optional: tool ID
            )

            # Make request with tool
            model = await client.models.resolve_chat(require_function_calling=True)
            response = await client.chat.completions.create(
                model=model,
                messages=[UserMessage(content="What is 42 * 137?")],
                tools=[calc_tool],
                tool_choice="auto",
                max_completion_tokens=200,
            )

            # ✅ CORRECT: Check for tool calls and access using Pydantic properties
            if response.choices and response.choices[0].message.tool_calls:
                tool_calls = response.choices[0].message.tool_calls

                print(f"\n🔧 Tool calls received: {len(tool_calls)}")

                for i, tool_call in enumerate(tool_calls):
                    print(f"\n✅ CORRECT: Tool Call {i + 1} Access Pattern:")

                    # 🔧 CRITICAL: This is the CORRECT way!
                    # Use Pydantic model property access, NOT dict access
                    func_name = tool_call.function.name  # ✅ Pydantic property
                    func_args_str = tool_call.function.arguments  # ✅ Pydantic property
                    call_id = tool_call.id  # ✅ Pydantic property

                    print(f"   Function name: {func_name}")
                    print(f"   Call ID: {call_id}")
                    print(f"   Arguments (JSON string): {func_args_str}")

                    # ✅ CORRECT: Parse arguments from JSON string
                    try:
                        func_args = json.loads(func_args_str)
                        print(f"   Parsed arguments: {func_args}")
                    except json.JSONDecodeError as e:
                        print(f"   ⚠️  Failed to parse arguments: {e}")

                print("\n🎯 Pattern Summary:")
                print("   ✅ tool_call.function.name  (Pydantic property access)")
                print("   ✅ tool_call.function.arguments  (Pydantic property access)")
                print("   ✅ tool_call.id  (Pydantic property access)")
            else:
                print("\n⚠️  No tool calls in response")

            return True

        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False


def example_tool_calling_anti_patterns():
    """
    ❌ CRITICAL ANTI-PATTERNS: Tool-call access mistakes that don't work with Pydantic models.

    This section shows the WRONG patterns to avoid when accessing tool calls.
    NEVER use these patterns - they don't work with Pydantic models!
    """
    print("\n" + "=" * 70)
    print("Section 3: Tool Calling - ANTI-PATTERNS ⚠️  CRITICAL")
    print("=" * 70)

    print("\n❌ WRONG: Dict-style access on tool calls")
    print("   This is a common mistake when accessing tool calls on the response.")
    print("\n   # ❌ WRONG: Dict-style access (DOESN'T WORK!)")
    print("   # for tool_call in message.tool_calls:")
    print("   #     func_name = tool_call['function']['name']  # ❌ ERROR!")
    print("   #     func_args = tool_call['function']['arguments']  # ❌ ERROR!")
    print("   #     call_id = tool_call['id']  # ❌ ERROR!")
    print("\n   # ✅ CORRECT: Pydantic property access")
    print("   # for tool_call in message.tool_calls:")
    print("   #     func_name = tool_call.function.name  # ✅ Works!")
    print("   #     func_args = tool_call.function.arguments  # ✅ Works!")
    print("   #     call_id = tool_call.id  # ✅ Works!")

    print("\n❌ WRONG: Defining tools as plain dicts")
    print("   # DON'T DO THIS:")
    print("   # tool = {")
    print("   #     'type': 'function',")
    print("   #     'function': {")
    print("   #         'name': 'my_func',")
    print("   #         'description': '...',")
    print("   #         'parameters': {...}")
    print("   #     }")
    print("   # }")
    print("   # INSTEAD USE Tool() and ToolFunction() Pydantic models!")

    print("\n❌ WRONG: Not parsing arguments JSON string")
    print("   # DON'T DO THIS:")
    print("   # args = tool_call.function.arguments  # This is a JSON string!")
    print("   # value = args['param']  # ❌ Can't index a string!")
    print("   # INSTEAD USE:")
    print("   # args = json.loads(tool_call.function.arguments)")
    print("   # value = args['param']  # ✅ Now it's a dict!")

    print("\n❌ WRONG: Not checking if tool_calls exists")
    print("   # DON'T DO THIS:")
    print("   # for tool_call in message.tool_calls:  # Might be None!")
    print("   # INSTEAD USE:")
    print("   # if message.tool_calls:")
    print("   #     for tool_call in message.tool_calls:")

    print("\n💡 Critical Takeaway: tool_call.function.name, NOT tool_call['function']['name']!")


# =============================================================================
# SECTION 4: STREAMING
# =============================================================================
# Demonstrates type-safe streaming with ChatCompletionChunk objects.
# Streaming requires careful None checking and delta handling.
# =============================================================================


async def example_streaming_correct() -> bool:
    """
    ✅ CORRECT: Type-safe async streaming with ChatCompletionChunk.

    Streaming returns chunks with delta updates, not complete messages.
    Always check for None and accumulate content properly.

    Returns True on success, False if the live request failed.
    """
    print("\n" + "=" * 70)
    print("Section 4: Streaming - CORRECT Patterns")
    print("=" * 70)

    async with VeniceClient() as client:
        try:
            print("\n🌊 Starting streaming request...")

            # ✅ CORRECT: Create streaming request
            model = await client.models.resolve_chat()
            stream = await client.chat.completions.create(
                model=model,
                messages=[UserMessage(content="Count from 1 to 5, one number per line.")],
                stream=True,  # Enable streaming
                max_completion_tokens=50,
            )

            print("\n✅ Streaming response (Pydantic chunks):")
            print("   ", end="")

            accumulated_content = ""
            finish_reason = None

            # ✅ CORRECT: Type-annotated async iteration
            # Each chunk is a ChatCompletionChunk Pydantic model
            async for chunk in stream:
                # ✅ CORRECT: Check if choices exist
                if chunk.choices:
                    choice = chunk.choices[0]

                    # ✅ CORRECT: Access delta from choice
                    delta = choice.delta

                    # ✅ CORRECT: Check if content exists before accessing
                    # Content might be None for chunks without new text
                    if delta.content:
                        print(delta.content, end="", flush=True)
                        accumulated_content += delta.content

                    # ✅ CORRECT: Check finish_reason
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

            print()  # New line after streaming

            print("\n✅ Streaming complete:")
            print(f"   Finish reason: {finish_reason}")
            print(f"   Total content length: {len(accumulated_content)} chars")

            return True

        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False


def example_streaming_anti_patterns():
    """
    ❌ ANTI-PATTERNS: Unsafe streaming patterns.

    Streaming requires even more careful None checking than regular responses.
    """
    print("\n" + "=" * 70)
    print("Section 4: Streaming - ANTI-PATTERNS (What NOT to Do)")
    print("=" * 70)

    print("\n❌ WRONG: Not checking if delta.content is None")
    print("   # DON'T DO THIS:")
    print("   # async for chunk in stream:")
    print("   #     content = chunk.choices[0].delta.content")
    print("   #     print(content)  # Might print None!")
    print("   # INSTEAD USE:")
    print("   # async for chunk in stream:")
    print("   #     if chunk.choices and chunk.choices[0].delta.content:")
    print("   #         print(chunk.choices[0].delta.content)")

    print("\n❌ WRONG: Dict-style access on chunks")
    print("   # DON'T DO THIS:")
    print("   # async for chunk in stream:")
    print("   #     content = chunk['choices'][0]['delta']['content']")
    print("   # INSTEAD USE:")
    print("   # async for chunk in stream:")
    print("   #     content = chunk.choices[0].delta.content")

    print("\n❌ WRONG: Assuming choices always exists")
    print("   # DON'T DO THIS:")
    print("   # delta = chunk.choices[0].delta  # IndexError if empty!")
    print("   # INSTEAD USE:")
    print("   # if chunk.choices:")
    print("   #     delta = chunk.choices[0].delta")

    print("\n❌ WRONG: Not handling finish_reason")
    print("   # DON'T DO THIS:")
    print("   # Just accumulate content without checking if done")
    print("   # INSTEAD USE:")
    print("   # if choice.finish_reason:")
    print("   #     # Handle end of stream")

    print("\n💡 Key Takeaway: Streaming requires extra None checking!")


# =============================================================================
# SECTION 5: REQUEST CONFIGURATION
# =============================================================================
# Demonstrates advanced request configuration using Pydantic models.
# StreamOptions, VeniceParameters, and JSONSchemaFormat examples.
# =============================================================================


async def example_request_config_correct():
    """
    ✅ CORRECT: Advanced request configuration with Pydantic models.

    Venice AI supports various advanced features through configuration objects.
    All of these must be Pydantic models, not dicts.
    """
    print("\n" + "=" * 70)
    print("Section 5: Request Configuration - CORRECT Patterns")
    print("=" * 70)

    # ✅ CORRECT: StreamOptions for streaming with usage tracking
    stream_opts = StreamOptions(
        include_usage=True  # Include token usage in final chunk
    )
    print("\n✅ StreamOptions created:")
    print(f"   include_usage: {stream_opts.include_usage}")

    # ✅ CORRECT: VeniceParameters for Venice-specific features
    venice_params = VeniceParameters(
        enable_web_search="on",  # Enable web search ("auto", "off", or "on")
        include_venice_system_prompt=False,  # Use custom system prompt
        character_slug=None,  # Optional: character slug
        strip_thinking_response=False,  # Don't strip thinking blocks
        disable_thinking=False,  # Don't disable thinking
        enable_web_citations=False,  # Don't enable citations
        include_search_results_in_stream=False,  # Don't include in stream
        return_search_results_as_documents=None,  # Optional: return as documents
    )
    print("\n✅ VeniceParameters created:")
    print(f"   enable_web_search: {venice_params.enable_web_search}")
    print(f"   include_venice_system_prompt: {venice_params.include_venice_system_prompt}")

    # ✅ CORRECT: JSONSchemaFormat for structured outputs
    json_schema = JSONSchemaFormat(
        type="json_schema",
        json_schema={
            "name": "person_info",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "email": {"type": "string"},
                },
                "required": ["name", "age"],
                "additionalProperties": False,
            },
        },
    )
    print("\n✅ JSONSchemaFormat created:")
    print(f"   type: {json_schema.type}")
    print(f"   schema name: {json_schema.json_schema['name']}")

    print("\n✅ Example request with all configurations:")
    print("   model = await client.models.resolve_chat()")
    print("   await client.chat.completions.create(")
    print("       model=model,")
    print("       messages=[...],")
    print("       stream=True,")
    print("       stream_options=stream_opts,  # Pydantic model")
    print("       venice_parameters=venice_params,  # Pydantic model")
    print("       response_format=json_schema  # Pydantic model")
    print("   )")

    print("\n💡 All configuration objects are Pydantic models!")


# =============================================================================
# SECTION 6: TYPE SAFETY
# =============================================================================
# Demonstrates comprehensive type annotations and TYPE_CHECKING usage.
# Shows how to write fully typed Venice AI SDK code.
# =============================================================================


async def example_type_hints_correct():
    """
    ✅ CORRECT: Full type annotations for Venice AI SDK code.

    Use TYPE_CHECKING imports and proper type hints for maximum type safety.
    """
    print("\n" + "=" * 70)
    print("Section 6: Type Safety - CORRECT Patterns")
    print("=" * 70)

    print("\n✅ Import pattern with TYPE_CHECKING:")
    print("   from typing import TYPE_CHECKING")
    print("   if TYPE_CHECKING:")
    print("       from venice_ai.types.chat import ChatCompletionResponse, ChatCompletionChunk")
    print("\n   This avoids circular imports while providing type hints!")

    print("\n✅ Function with full type annotations:")
    print("   async def process_response(")
    print("       response: 'ChatCompletion'  # Type hint using string")
    print("   ) -> Optional[str]:")
    print("       if response.choices:")
    print("           return response.text")
    print("       return None")

    print("\n✅ Handling Union types in messages:")
    print("   from typing import Union, List")
    print("   messages: List[Union[UserMessage, SystemMessage, AssistantMessage]] = [")
    print("       SystemMessage(content='...'),")
    print("       UserMessage(content='...')")
    print("   ]")

    print("\n✅ Type-safe async iteration:")
    print("   from typing import AsyncIterable")
    print("   stream: AsyncIterable['ChatCompletionChunk'] = await client.chat.completions.create(")
    print("       ..., stream=True")
    print("   )")
    print("   async for chunk in stream:")
    print("       # chunk is properly typed as ChatCompletionChunk")

    print("\n💡 Type hints enable IDE autocomplete and catch errors early!")


# =============================================================================
# SECTION 7: COMMON PITFALLS REFERENCE
# =============================================================================
# Quick reference guide to all common pitfalls and their solutions.
# Scannable format for quick lookups.
# =============================================================================


def common_pitfalls_reference():
    """
    📚 QUICK REFERENCE: Common Pitfalls and Solutions

    A comprehensive, scannable list of all common mistakes and their fixes.
    Use this as a quick lookup when writing Venice AI SDK code.
    """
    print("\n" + "=" * 70)
    print("Section 7: Common Pitfalls Reference")
    print("=" * 70)

    pitfalls = """
1. MESSAGE CONSTRUCTION
   ❌ msg = {'role': 'user', 'content': 'Hello'}
   ✅ msg = UserMessage(content='Hello')

2. ACCESSING MESSAGE CONTENT
   ❌ content = msg['content']
   ✅ content = msg.content

3. RESPONSE ACCESS WITHOUT CHECKING
   ❌ content = response.text
   ✅ if response.choices and response.text:
       content = response.text

4. TOOL CALL ACCESS
   ❌ name = tool_call['function']['name']
   ✅ name = tool_call.function.name

5. TOOL DEFINITION
   ❌ tool = {'type': 'function', 'function': {...}}
   ✅ tool = Tool(type='function', function=ToolFunction(...))

6. TOOL ARGUMENTS PARSING
   ❌ args = tool_call.function.arguments['param']
   ✅ args = json.loads(tool_call.function.arguments)
       value = args['param']

7. STREAMING CONTENT ACCESS
   ❌ print(chunk.choices[0].delta.content)
   ✅ if chunk.choices and chunk.choices[0].delta.content:
       print(chunk.choices[0].delta.content)

8. CHECKING FOR TOOL CALLS
   ❌ for tc in message.tool_calls:
   ✅ if message.tool_calls:
       for tc in message.tool_calls:

9. MULTIMODAL CONTENT
   ❌ text = message.content  # Might be a list!
   ✅ if isinstance(message.content, str):
       text = message.content
       elif isinstance(message.content, list):
           # Handle multimodal

10. CONFIGURATION OBJECTS
    ❌ stream_options = {'include_usage': True}
    ✅ stream_options = StreamOptions(include_usage=True)

11. TYPE HINTS
    ❌ def process(response):
    ✅ def process(response: 'ChatCompletion') -> Optional[str]:

12. FINISH REASON IN STREAMING
    ❌ # Just accumulate without checking done
    ✅ if choice.finish_reason:
        # Handle end of stream

13. EMPTY CHOICES ARRAY
    ❌ message = response.choices[0].message
    ✅ if response.choices:
        message = response.choices[0].message

14. OPTIONAL FIELDS
    ❌ usage = response.usage.total_tokens
    ✅ if response.usage:
        usage = response.usage.total_tokens

15. MESSAGE HISTORY
    ❌ messages = [{'role': 'user', 'content': '...'}]
    ✅ messages = [UserMessage(content='...')]
"""

    print(pitfalls)
    print("\n" + "=" * 70)
    print("💡 Remember: Pydantic models, not dicts!")
    print("💡 Remember: Attribute access, not dict-style!")
    print("💡 Remember: Always check for None!")
    print("=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================


async def main() -> int:
    """
    Run all best practices examples.

    This demonstrates all the patterns in sequence, showing both correct
    approaches and anti-patterns (the anti-patterns are not executed, just shown).

    Returns 0 if every live section succeeded, 1 if any live section failed.
    """
    print("=" * 70)
    print("Venice AI SDK - Pydantic Model Best Practices")
    print("=" * 70)

    live_results: list[tuple[str, bool]] = []

    # Section 1: Message Models
    print("\n" + "=" * 70)
    print("Section 1: Message Models")
    print("=" * 70)
    await example_message_models_correct()
    example_message_models_anti_patterns()
    example_validation_rejections_live()

    # Section 2: Response Models
    print("\n" + "=" * 70)
    print("Section 2: Response Access")
    print("=" * 70)
    live_results.append(("Response Access", await example_response_access_correct()))
    example_response_access_anti_patterns()

    # Section 3: Tool Calling (CRITICAL)
    print("\n" + "=" * 70)
    print("Section 3: Tool Calling ⚡ CRITICAL")
    print("=" * 70)
    await example_tool_definition_correct()
    live_results.append(("Tool Calling", await example_tool_calling_correct()))
    example_tool_calling_anti_patterns()

    # Section 4: Streaming
    print("\n" + "=" * 70)
    print("Section 4: Streaming")
    print("=" * 70)
    live_results.append(("Streaming", await example_streaming_correct()))
    example_streaming_anti_patterns()

    # Section 5: Request Configuration
    print("\n" + "=" * 70)
    print("Section 5: Request Configuration")
    print("=" * 70)
    await example_request_config_correct()

    # Section 6: Type Safety
    print("\n" + "=" * 70)
    print("Section 6: Type Safety")
    print("=" * 70)
    await example_type_hints_correct()

    # Section 7: Common Pitfalls Reference
    common_pitfalls_reference()

    # Final summary
    print("\n" + "=" * 70)
    print("✅ Best Practices Examples Complete!")
    print("=" * 70)

    print("\n📚 What You Learned:")
    print("   ✅ All Venice AI types are Pydantic models, not dicts")
    print("   ✅ Always use attribute access (obj.field), never dict-style (obj['field'])")
    print("   ✅ Check for None before accessing optional fields")
    print("   ✅ Tool-call access: use tool_call.function.name, NOT tool_call['function']['name']")
    print("   ✅ Proper streaming with None checking")
    print("   ✅ Type hints for better IDE support")

    print("\n🎯 Next Steps:")
    print("   1. Review the code comments for detailed explanations")
    print("   2. Use this file as a reference when writing Venice AI code")
    print("   3. Copy the ✅ CORRECT patterns into your own code")
    print("   4. Avoid the ❌ ANTI-PATTERNS shown here")

    print("\n🔗 Related Examples:")
    print("   - examples/chat/tool_calling.py - More tool calling examples")
    print("   - examples/chat/streaming_chat.py - Advanced streaming patterns")
    print("   - examples/basic/quick_start.py - Getting started guide")

    print("\n" + "=" * 70)

    passed = sum(1 for _, ok in live_results if ok)
    failed = len(live_results) - passed
    if failed:
        print(f"\n⚠️ {passed}/{len(live_results)} live sections succeeded; {failed} failed")
        for name, ok in live_results:
            status = "✓" if ok else "✗"
            print(f"   {status} {name}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
    sys.exit(exit_code)
