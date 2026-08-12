#!/usr/bin/env python3
"""
Venice AI SDK - Function Calling Examples
=========================================

This example demonstrates function calling capabilities with Venice AI models.
It showcases:

- Basic function/tool definition and calling
- Multiple tools with different schemas
- Tool choice control (auto, none, specific function)
- Parallel function calling
- Error handling for function calls
- Processing function call results
"""

import asyncio
import sys
from typing import Literal

from venice_ai import UserMessage, VeniceClient, tool_from_function
from venice_ai.types.api.requests.common import Tool, ToolFunction


async def basic_function_calling() -> bool:
    """Demonstrate basic function calling with a simple weather tool.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print("🔧 Basic Function Calling")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            # Get a model that supports function calling
            model_id = await client.models.resolve_chat(require_function_calling=True)

            print(f"🤖 Using model: {model_id}")

            # ``tool_from_function`` introspects type hints + docstring to build
            # the JSON schema. The function body isn't called by the SDK —
            # only the signature matters.
            def get_weather(
                location: str,
                unit: Literal["celsius", "fahrenheit"] = "fahrenheit",
            ) -> str:
                """Get the current weather for a specific location."""
                raise NotImplementedError  # would call real weather API in production

            weather_tool = tool_from_function(get_weather)
            assert weather_tool.function is not None  # tool_from_function always sets .function

            print("\n🛠️ Tool Definition (built by tool_from_function):")
            print(f"   Function: {weather_tool.function.name}")
            print(f"   Description: {weather_tool.function.description}")

            # Make a request that should trigger function calling
            response = await client.chat.completions.create(
                model=model_id,
                messages=[
                    UserMessage(
                        content="What's the weather like in New York City? I prefer Celsius."
                    )
                ],
                tools=[weather_tool],
                tool_choice="auto",
                max_completion_tokens=200,
            )

            # Check if the model called the function
            if response.choices and response.choices[0].message.tool_calls:
                tool_calls = response.choices[0].message.tool_calls
                print("\n✅ Function called successfully!")
                print(f"   Number of tool calls: {len(tool_calls)}")

                for i, tool_call in enumerate(tool_calls):
                    print(f"\n   Tool Call {i + 1}:")
                    print(f"   🔧 Function: {tool_call.function.name}")
                    print(f"   🆔 Call ID: {tool_call.id}")

                    # arguments_dict parses the JSON `arguments` string for us.
                    args = tool_call.function.arguments_dict
                    print("   📊 Arguments:")
                    for key, value in args.items():
                        print(f"      {key}: {value}")

                print("\n💡 Tool call successful! In a real application, you would:")
                print("   1. Execute the actual weather API call")
                print("   2. Format the response")
                print("   3. Continue the conversation with the tool result")

            else:
                print("\n❌ No function calls made")
                print(f"   Response: {response.text}")

        except Exception as e:
            print(f"❌ Error in basic function calling: {e}")
            ok = False

    return ok


async def multiple_tools_example() -> bool:
    """Demonstrate using multiple tools with different complexity.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print("\n🛠️ Multiple Tools Example")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            model_id = await client.models.resolve_chat(require_function_calling=True)

            print(f"🤖 Using model: {model_id}")

            # Define multiple tools
            tools = [
                Tool(
                    type="function",
                    function=ToolFunction(
                        name="calculate_area",
                        description="Calculate the area of a rectangle",
                        parameters={
                            "type": "object",
                            "properties": {
                                "width": {
                                    "type": "number",
                                    "description": "Width in meters",
                                    "minimum": 0,
                                },
                                "height": {
                                    "type": "number",
                                    "description": "Height in meters",
                                    "minimum": 0,
                                },
                            },
                            "required": ["width", "height"],
                        },
                        strict=True,
                    ),
                ),
                Tool(
                    type="function",
                    function=ToolFunction(
                        name="get_random_fact",
                        description="Get a random interesting fact about a topic",
                        parameters={
                            "type": "object",
                            "properties": {
                                "topic": {
                                    "type": "string",
                                    "description": "The topic to get a fact about",
                                },
                                "category": {
                                    "type": "string",
                                    "enum": ["science", "history", "nature", "technology"],
                                    "description": "Category of fact",
                                },
                            },
                            "required": ["topic"],
                        },
                        strict=False,
                    ),
                ),
            ]

            print("\n📋 Available Tools:")
            for tool in tools:
                func = tool.function
                assert func is not None
                strict_status = "Strict" if func.strict else "Relaxed"
                print(f"   • {func.name} ({strict_status} validation)")

            # Test with a request that could use multiple tools
            response = await client.chat.completions.create(
                model=model_id,
                messages=[
                    UserMessage(
                        content="I need to calculate the area of a 5x3 meter rectangle, and also tell me an interesting science fact about mathematics."
                    )
                ],
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=True,
                max_completion_tokens=300,
            )

            if response.choices and response.choices[0].message.tool_calls:
                tool_calls = response.choices[0].message.tool_calls
                print(f"\n✅ {len(tool_calls)} tool call(s) made:")

                for i, tool_call in enumerate(tool_calls):
                    print(f"\n   Tool Call {i + 1}:")
                    print(f"   🔧 Function: {tool_call.function.name}")
                    print(f"   🆔 Call ID: {tool_call.id}")

                    args = tool_call.function.arguments_dict
                    print(f"   📊 Arguments: {args}")

            else:
                print("\n❌ No tool calls made")
                print(f"   Response: {response.text}")

        except Exception as e:
            print(f"❌ Error in multiple tools example: {e}")
            ok = False

    return ok


async def tool_choice_control() -> bool:
    """Demonstrate different tool choice strategies.

    Returns ``True`` on success, ``False`` if any request failed.
    """
    print("\n🎯 Tool Choice Control")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            model_id = await client.models.resolve_chat(require_function_calling=True)

            print(f"🤖 Using model: {model_id}")

            # Define a simple calculation tool
            calc_tool = Tool(
                type="function",
                function=ToolFunction(
                    name="calculate",
                    description="Perform basic mathematical calculations",
                    parameters={
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["add", "subtract", "multiply", "divide"],
                                "description": "Mathematical operation",
                            },
                            "a": {"type": "number", "description": "First number"},
                            "b": {"type": "number", "description": "Second number"},
                        },
                        "required": ["operation", "a", "b"],
                    },
                    strict=False,
                ),
            )

            test_cases = [
                {
                    "name": "Auto Choice",
                    "tool_choice": "auto",
                    "description": "Let model decide whether to use tools",
                },
                {
                    "name": "No Tools",
                    "tool_choice": "none",
                    "description": "Force model to not use tools",
                },
                {
                    "name": "Forced Tool",
                    "tool_choice": {"type": "function", "function": {"name": "calculate"}},
                    "description": "Force model to use specific tool",
                },
            ]

            user_message = "What is 15 multiplied by 8?"

            for test_case in test_cases:
                print(f"\n🧪 Testing: {test_case['name']}")
                print(f"   Strategy: {test_case['description']}")

                try:
                    response = await client.chat.completions.create(
                        model=model_id,
                        messages=[UserMessage(content=user_message)],
                        tools=[calc_tool],
                        tool_choice=test_case["tool_choice"],
                        max_completion_tokens=150,
                    )

                    if response.choices and response.choices[0].message.tool_calls:
                        tool_calls = response.choices[0].message.tool_calls
                        print(f"   ✅ Tool called: {tool_calls[0].function.name}")

                        args = tool_calls[0].function.arguments_dict
                        print(f"   📊 Arguments: {args}")
                    else:
                        content = response.text or ""
                        print(f"   💬 Direct response: {content[:100]}...")

                except Exception as e:
                    print(f"   ❌ Error: {str(e)[:100]}...")
                    ok = False

        except Exception as e:
            print(f"❌ Error in tool choice control: {e}")
            ok = False

    return ok


async def function_calling_best_practices() -> bool:
    """Show best practices for function calling.

    Pure informational output — always returns ``True``.
    """
    print("\n📚 Function Calling Best Practices")
    print("-" * 40)

    print("💡 Design Guidelines:")
    print("   ✅ Use descriptive function names and descriptions")
    print("   ✅ Define clear parameter schemas with types and constraints")
    print("   ✅ Include helpful descriptions for all parameters")
    print("   ✅ Mark required parameters appropriately")
    print("   ✅ Use enums for limited value sets")
    print("   ✅ Set appropriate validation (strict vs relaxed)")

    print("\n⚡ Performance Tips:")
    print("   ✅ Use parallel_tool_calls=True for independent operations")
    print("   ✅ Keep function descriptions concise but informative")
    print("   ✅ Limit the number of tools to avoid confusion")
    print("   ✅ Use tool_choice strategically to guide model behavior")

    print("\n🔧 Error Handling:")
    print("   ✅ Always validate function call arguments")
    print("   ✅ Handle JSON parsing errors gracefully")
    print("   ✅ Provide meaningful error messages in tool responses")
    print("   ✅ Implement fallback strategies for tool failures")

    return True


async def main() -> int:
    """Run all function calling examples.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    print("🚀 Venice AI Function Calling Examples")
    print("=" * 50)

    results: list[tuple[str, bool]] = [
        ("basic_function_calling", await basic_function_calling()),
        ("multiple_tools_example", await multiple_tools_example()),
        ("tool_choice_control", await tool_choice_control()),
        ("function_calling_best_practices", await function_calling_best_practices()),
    ]

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("\n✨ Function calling examples completed!")

    print("\n💡 Key concepts demonstrated:")
    print("   - Basic function/tool definition and calling")
    print("   - Multiple tools with different validation levels")
    print("   - Tool choice control strategies")
    print("   - Parallel function calling")
    print("   - Best practices for production use")

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print(
            "Check that your API key is valid and you have a model that supports function calling.",
            file=sys.stderr,
        )
        sys.exit(1)
