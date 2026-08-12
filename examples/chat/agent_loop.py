#!/usr/bin/env python3
"""
Venice AI SDK - Agent Loop with run_with_tools
==============================================

This example demonstrates the agent-loop convenience method
``client.chat.completions.run_with_tools(...)``, which drives the canonical
"call model -> execute requested tools -> feed results back -> repeat until a
final answer" loop for you. Compared with ``tool_calling.py`` (which builds
tool definitions and dispatches calls by hand), this is the batteries-included
path: hand it plain Python functions and it does the orchestration.

It showcases:

- Defining simple Python tools (a calculator and a mock weather lookup) and
  letting the SDK introspect them via ``tool_from_function``
- Running a multi-step agent loop where the model actually calls the tools and
  the SDK executes them, feeding results back automatically
- Observing each tool call as it happens with the ``on_tool_call`` hook
- Inspecting the ``ToolLoopResult`` (final text, round-trip count, token usage)

Prerequisites:
- Install: pip install venice-ai
- Set API key: export VENICE_API_KEY="your-api-key"
"""

import asyncio
import sys
from typing import Any, Literal

from venice_ai import (
    MaxIterationsExceededError,
    UserMessage,
    VeniceClient,
    tool_from_function,
)
from venice_ai.types.api.chat import ToolCall

# ---------------------------------------------------------------------------
# Tool implementations
#
# These are plain Python functions. ``run_with_tools`` introspects each one
# with ``tool_from_function`` (type hints + docstring -> JSON schema) and also
# registers the function body as the dispatch handler, so the SDK both
# advertises the tool to the model AND runs it when the model calls it. The
# return value is stringified and sent back to the model as a tool result.
# ---------------------------------------------------------------------------


def calculator(
    operation: Literal["add", "subtract", "multiply", "divide"],
    a: float,
    b: float,
) -> float:
    """Perform a basic arithmetic operation on two numbers."""
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b == 0:
            raise ValueError("division by zero")
        return a / b
    raise ValueError(f"unknown operation: {operation!r}")


def get_weather(
    city: str,
    unit: Literal["celsius", "fahrenheit"] = "celsius",
) -> str:
    """Look up the current weather for a city.

    This is a mock implementation that returns canned data; a real tool would
    call a weather API here. The model never sees this body — only the schema
    derived from the signature and docstring.
    """
    # Canned "forecast" so the example runs offline and deterministically.
    table = {
        "san francisco": (18, "foggy"),
        "tokyo": (24, "clear"),
        "london": (12, "rainy"),
    }
    temp_c, sky = table.get(city.lower(), (21, "partly cloudy"))
    temp = temp_c if unit == "celsius" else round(temp_c * 9 / 5 + 32)
    symbol = "°C" if unit == "celsius" else "°F"
    return f"{city.title()}: {temp}{symbol}, {sky}"


async def agent_loop_demo() -> bool:
    """Run a multi-step agent loop that exercises both tools.

    Returns ``True`` on success, ``False`` if the API call failed. If the loop
    runs but the account/model simply chooses not to call any tool, that is
    still a successful exercise of ``run_with_tools`` (it returned a final
    answer), so we don't fail the demo on that alone.
    """
    print("🤖 Agent Loop (run_with_tools)")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            # Resolver-based selection — NEVER hardcode a model id. We need a
            # model that supports function calling for the loop to do anything.
            model_id = await client.models.resolve_chat(require_function_calling=True)
            print(f"🤖 Using model: {model_id}")

            # Observation hook: called after each tool runs successfully with the
            # ToolCall the model emitted and the handler's return value. It is
            # read-only and does not change what the model sees.
            tool_calls_seen: list[tuple[str, dict[str, Any], Any]] = []

            def on_tool_call(call: ToolCall, result: Any) -> None:
                args = call.function.arguments_dict
                tool_calls_seen.append((call.function.name, args, result))
                print(f"   🔧 {call.function.name}({args}) -> {result!r}")

            print("\n💬 Question: a multi-step task that needs both tools.")
            print("   (compute 137 * 4, then get the weather in Tokyo)\n")

            # The SDK runs the full loop: it calls the model, executes any tools
            # the model requests, feeds the results back, and repeats until the
            # model produces a final (non-tool-call) answer.
            result = await client.chat.completions.run_with_tools(
                model=model_id,
                messages=[
                    UserMessage(
                        content=(
                            "First, what is 137 multiplied by 4? "
                            "Then tell me the current weather in Tokyo in celsius. "
                            "Use the tools provided, then summarize both answers."
                        )
                    )
                ],
                # Bare callables — auto-converted with tool_from_function and
                # registered as their own dispatch handlers.
                tools=[calculator, get_weather],
                on_tool_call=on_tool_call,
                max_iterations=6,
                max_completion_tokens=400,
            )

            print(f"\n✅ Loop converged in {result.iterations} round-trip(s).")
            print(f"   Tool calls executed: {len(tool_calls_seen)}")
            print(f"\n📝 Final answer:\n   {result.text}")

            if result.usage:
                print(
                    f"\n📊 Token usage: prompt={result.usage.prompt_tokens}, "
                    f"completion={result.usage.completion_tokens}, "
                    f"total={result.usage.total_tokens}"
                )

            if not tool_calls_seen:
                print(
                    "\nℹ️  The model answered without calling any tool this run. "
                    "run_with_tools still drove the loop to a final answer."
                )

        except MaxIterationsExceededError as e:
            # The loop ran but never converged — surface it honestly.
            print(f"❌ Tool loop did not converge: {e}")
            ok = False
        except Exception as e:
            print(f"❌ Error in agent loop: {e}")
            ok = False

    return ok


async def inspect_tool_schema_demo() -> bool:
    """Show what ``tool_from_function`` builds from a plain function.

    Pure introspection — no network call — so this always returns ``True``.
    It makes the magic behind ``run_with_tools`` visible: the same conversion
    happens automatically when you pass bare callables to the loop.
    """
    print("\n🔍 Inspecting tool_from_function output")
    print("-" * 40)

    weather_tool = tool_from_function(get_weather)
    assert weather_tool.function is not None  # tool_from_function always sets .function

    print(f"   Function name: {weather_tool.function.name}")
    print(f"   Description:   {weather_tool.function.description}")
    print(f"   Parameters:    {weather_tool.function.parameters}")

    return True


async def main() -> int:
    """Run all agent-loop demos.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    print("🚀 Venice AI Agent Loop Examples")
    print("=" * 50)

    results: list[tuple[str, bool]] = [
        ("inspect_tool_schema_demo", await inspect_tool_schema_demo()),
        ("agent_loop_demo", await agent_loop_demo()),
    ]

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("\n✨ Agent loop examples completed!")

    print("\n💡 Key concepts demonstrated:")
    print("   - run_with_tools drives the full tool-call loop for you")
    print("   - Plain Python functions become tools via tool_from_function")
    print("   - on_tool_call observes each tool the model invokes")
    print("   - ToolLoopResult carries the final text, iterations, and usage")

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
