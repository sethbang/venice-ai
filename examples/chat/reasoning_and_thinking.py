#!/usr/bin/env python3
"""
Venice AI SDK - Reasoning and Thinking Examples
===============================================

This example demonstrates how to use models with reasoning capabilities that
support <thinking> blocks for transparent thought processes using VeniceParameters.

Learn how to:
- Select models that support reasoning
- Use VeniceParameters to control thinking behavior
- Use strip_thinking_response and disable_thinking properly
- Parse and display thinking blocks separately from responses
"""

import asyncio
import sys

from venice_ai import VeniceClient, extract_thinking_blocks
from venice_ai.types.api import SystemMessage, UserMessage
from venice_ai.types.api.requests import VeniceParameters


def format_thinking_output(thinking_blocks: list[str], message: str, title: str = "Response"):
    """Format and display thinking blocks and message in a clear way."""
    print("\n" + "=" * 70)

    if thinking_blocks:
        print("💭 THINKING PROCESS")
        print("-" * 70)
        for i, block in enumerate(thinking_blocks, 1):
            block_lines = block.strip().split("\n")
            # Show first few lines of thinking
            print(f"  Block {i}:")
            for _j, line in enumerate(block_lines[:5]):
                print(f"    {line}")
            if len(block_lines) > 5:
                print(f"    ... ({len(block_lines) - 5} more lines)")
            if i < len(thinking_blocks):
                print()  # Space between blocks
        print("-" * 70)
    else:
        print("ℹ️ No thinking blocks detected in response")
        print("-" * 70)

    print(f"📝 {title.upper()}")
    print("-" * 70)
    # Ensure message is not too long for display
    if len(message) > 1000:
        print(message[:1000])
        print(f"... (truncated {len(message) - 1000} characters)")
    else:
        print(message if message else "(No message content)")
    print("-" * 70)
    print("=" * 70)


async def find_reasoning_model(client: VeniceClient) -> str | None:
    """Find a model that supports reasoning using only dynamic detection."""
    from venice_ai.types.api import TextModelSpec

    try:
        # ``capabilities`` only lives on text models; filter the call.
        all_models = await client.models.list(type="text")

        # Look for models with reasoning support - ONLY dynamic detection
        for model in all_models.data:
            spec = model.model_spec
            if not isinstance(spec, TextModelSpec):
                continue
            caps = spec.capabilities
            if caps is not None and caps.supportsReasoning:
                return model.id

    except Exception as e:
        print(f"⚠️ Error finding reasoning model: {e}")

    return None


async def basic_reasoning_with_venice_parameters() -> bool:
    """Demonstrate reasoning using VeniceParameters.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print("\n🧠 Basic Reasoning with VeniceParameters")
    print("=" * 70)

    async with VeniceClient() as client:
        try:
            # Find a model that supports reasoning
            reasoning_model = await find_reasoning_model(client)

            if not reasoning_model:
                print("⚠️ No models with reasoning support found, using standard model")
                model = await client.models.resolve_chat(require_reasoning=True)
            else:
                model = reasoning_model
                print(f"📍 Using reasoning model: {model}")

            # Complex problem that benefits from step-by-step thinking
            problem = """
            I have a sequence: 2, 6, 12, 20, 30, ...
            What are the next three numbers in this sequence?
            Please think through the pattern step by step.
            """

            # Create VeniceParameters to keep thinking blocks
            venice_params = VeniceParameters(
                character_slug=None,
                strip_thinking_response=False,  # Keep thinking blocks visible
                disable_thinking=False,  # Enable thinking (default)
                enable_web_search="off",
                enable_web_citations=False,
                include_search_results_in_stream=False,
                return_search_results_as_documents=None,
                include_venice_system_prompt=True,
            )

            print("\n🔬 Testing with VeniceParameters(strip_thinking_response=False):")
            # Print only relevant parameters
            params_dict = venice_params.model_dump()
            important_params = {
                k: v
                for k, v in params_dict.items()
                if k in ["strip_thinking_response", "disable_thinking"]
                or v not in [None, False, "off"]
            }
            print(f"   Key parameters: {' '.join(f'{k}={v}' for k, v in important_params.items())}")

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    SystemMessage(
                        content="You are a mathematical problem solver. Show your reasoning process step by step."
                    ),
                    UserMessage(content=problem),
                ],
                venice_parameters=venice_params,  # Pass VeniceParameters object
                # A modest cap keeps reasoning + visible output bounded so the
                # demo finishes quickly. Reasoning models can otherwise burn a
                # large budget thinking before emitting the final answer.
                max_completion_tokens=1024,
                temperature=0.2,
            )

            msg = response.choices[0].message
            reasoning = getattr(msg, "reasoning_content", None)
            content = str(msg.content or "")

            if reasoning:
                thinking_blocks = [str(reasoning)]
                clean_response = content
            else:
                thinking_blocks, clean_response = extract_thinking_blocks(content)

            # Check if response metadata includes venice_parameters
            if hasattr(response, "venice_parameters") and response.venice_parameters:
                print("\n✅ VeniceParameters in response:")
                print(
                    f"   strip_thinking_response: {response.venice_parameters.strip_thinking_response}"
                )
                print(f"   disable_thinking: {response.venice_parameters.disable_thinking}")

            # Display results
            format_thinking_output(thinking_blocks, clean_response, "FINAL ANSWER")

            # Token usage
            if response.usage:
                usage = response.usage
                print(
                    f"📊 Token Usage: Input={usage.prompt_tokens}, Output={usage.completion_tokens}, Total={usage.total_tokens}"
                )
        except Exception as e:
            print(f"❌ Error in basic reasoning: {e}")
            return False

    return True


async def test_strip_thinking_parameter() -> bool:
    """Test strip_thinking_response parameter using VeniceParameters.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print("\n🎭 Testing strip_thinking_response with VeniceParameters")
    print("=" * 70)

    async with VeniceClient() as client:
        try:
            # Find a model that supports reasoning
            reasoning_model = await find_reasoning_model(client)

            if not reasoning_model:
                print("⚠️ No models with reasoning support found, using standard model")
                model = await client.models.resolve_chat(require_reasoning=True)
            else:
                model = reasoning_model
                print(f"📍 Using reasoning model: {model}")

            # Logic puzzle requiring careful reasoning
            puzzle = """
            Three friends - Alice, Bob, and Charlie - each have a different favorite color (red, blue, green).
            - Alice doesn't like red
            - The person who likes blue is not Charlie
            - Bob's favorite color comes before Charlie's alphabetically

            What is each person's favorite color?
            """

            # Create VeniceParameters to strip thinking blocks
            venice_params = VeniceParameters(
                character_slug=None,
                strip_thinking_response=True,  # Strip thinking blocks
                disable_thinking=False,  # Still allow thinking, just strip it
                enable_web_search="off",
                enable_web_citations=False,
                include_search_results_in_stream=False,
                return_search_results_as_documents=None,
                include_venice_system_prompt=True,
            )

            print("\n🔬 Testing with VeniceParameters(strip_thinking_response=True):")
            # Print parameters in a cleaner format
            params_dict = venice_params.model_dump()
            important_params = {
                k: v
                for k, v in params_dict.items()
                if k in ["strip_thinking_response", "disable_thinking"]
                or v not in [None, False, "off"]
            }
            print(f"   Key parameters: {' '.join(f'{k}={v}' for k, v in important_params.items())}")

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    SystemMessage(
                        content="You are a logic puzzle solver. Think through the problem systematically, then provide a clear answer."
                    ),
                    UserMessage(content=puzzle),
                ],
                venice_parameters=venice_params,  # Pass VeniceParameters object
                max_completion_tokens=1024,
                temperature=0.1,
            )

            msg = response.choices[0].message
            reasoning = getattr(msg, "reasoning_content", None)
            content = str(msg.content or "")

            if reasoning:
                thinking_blocks = [str(reasoning)]
                clean_response = content
            else:
                thinking_blocks, clean_response = extract_thinking_blocks(content)

            # Check response metadata
            if hasattr(response, "venice_parameters") and response.venice_parameters:
                if response.venice_parameters.strip_thinking_response:
                    print("\n✅ Server confirmed: strip_thinking_response=True")
                    if thinking_blocks:
                        print("⚠️ But thinking blocks still found in response!")
                        print(
                            "   This may indicate the model doesn't fully support this parameter."
                        )
                else:
                    print("\n⚠️ Server returned strip_thinking_response=False")

            # Display results
            print("=" * 70)

            # Check if response actually contains thinking tags
            content_str = str(content) if not isinstance(content, str) else content
            contains_thinking_tags = (
                "<think" in content_str.lower() or "<thinking" in content_str.lower()
            )

            if contains_thinking_tags and venice_params.strip_thinking_response:
                print("❌ ERROR: strip_thinking_response=True but thinking blocks still present!")
                print("   The server did NOT strip thinking blocks despite the parameter.")
                print("   This indicates the parameter is not working as expected.\n")
                # Show the raw content to demonstrate the issue
                print("📝 RAW RESPONSE (first 500 chars showing <think> tags):")
                print("-" * 70)
                if len(content_str) > 500:
                    print(content_str[:500] + "...")
                else:
                    print(content_str)
                print("-" * 70)
            elif thinking_blocks:
                # Thinking blocks were found and extracted
                if venice_params.strip_thinking_response:
                    print("⚠️ WARNING: strip_thinking_response=True but thinking blocks detected!")
                    print("   The API may not fully support this parameter.")
                format_thinking_output(thinking_blocks, clean_response, "SOLUTION")
            else:
                # No thinking blocks found
                if venice_params.strip_thinking_response and not contains_thinking_tags:
                    print("✅ Thinking blocks successfully stripped from response")
                elif not venice_params.strip_thinking_response and not contains_thinking_tags:
                    print("ℹ️ No thinking blocks in response (model may not have used reasoning)")
                print("-" * 70)
                print("📝 SOLUTION")
                print("-" * 70)
                print(clean_response)

            # Token usage
            if response.usage:
                usage = response.usage
                print(
                    f"📊 Token Usage: Input={usage.prompt_tokens}, Output={usage.completion_tokens}, Total={usage.total_tokens}"
                )
        except Exception as e:
            print(f"❌ Error in strip_thinking test: {e}")
            return False

    return True


async def test_disable_thinking_parameter() -> bool:
    """Test disable_thinking parameter using VeniceParameters.

    Returns ``True`` on success. A timeout on the ``disable_thinking=True`` call
    is the *expected, documented* outcome for some reasoning models (they need
    thinking to respond), so it is reported and still counts as success.
    Returns ``False`` only on a genuine API error.
    """
    print("\n🔄 Testing disable_thinking with VeniceParameters")
    print("=" * 70)

    async with VeniceClient() as client:
        try:
            # Find a model that supports reasoning
            reasoning_model = await find_reasoning_model(client)

            if not reasoning_model:
                print("⚠️ No models with reasoning support found, using standard model")
                model = await client.models.resolve_chat(require_reasoning=True)
            else:
                model = reasoning_model
                print(f"📍 Using reasoning model: {model}")

            # Question that benefits from reasoning
            question = """
            If it takes 5 machines 5 minutes to make 5 widgets,
            how long would it take 100 machines to make 100 widgets?
            """

            # Test 1: With reasoning enabled
            print("\n📍 Test 1: With thinking enabled (disable_thinking=False)")
            venice_params_enabled = VeniceParameters(
                character_slug=None,
                disable_thinking=False,
                strip_thinking_response=False,  # Keep thinking visible
                enable_web_search="off",
                enable_web_citations=False,
                include_search_results_in_stream=False,
                return_search_results_as_documents=None,
                include_venice_system_prompt=True,
            )

            response_with = await client.chat.completions.create(
                model=model,
                messages=[UserMessage(content=question)],
                venice_parameters=venice_params_enabled,
                max_completion_tokens=400,
                temperature=0.1,
            )

            msg_with = response_with.choices[0].message
            reasoning_with = getattr(msg_with, "reasoning_content", None)
            content_with = msg_with.content or ""

            if reasoning_with:
                thinking_blocks_with = [str(reasoning_with)]
                clean_with = content_with
            else:
                thinking_blocks_with, clean_with = extract_thinking_blocks(content_with)

            print(f"   Thinking blocks found: {'Yes' if thinking_blocks_with else 'No'}")
            print(f"   Response length: {len(content_with)} characters")
            if response_with.usage:
                print(f"   Tokens used: {response_with.usage.total_tokens}")

            # Test 2: With reasoning disabled
            print("\n📍 Test 2: With thinking disabled (disable_thinking=True)")
            print("   ⚠️ Note: Some models may timeout with disable_thinking=True")
            venice_params_disabled = VeniceParameters(
                character_slug=None,
                disable_thinking=True,  # Disable thinking
                strip_thinking_response=False,  # Shouldn't matter if thinking is disabled
                enable_web_search="off",
                enable_web_citations=False,
                include_search_results_in_stream=False,
                return_search_results_as_documents=None,
                include_venice_system_prompt=True,
            )

            try:
                # Add timeout to prevent hanging when disable_thinking=True
                response_without = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[UserMessage(content=question)],
                        venice_parameters=venice_params_disabled,
                        max_completion_tokens=400,
                        temperature=0.1,
                    ),
                    timeout=15.0,  # 15 second timeout
                )
            except TimeoutError:
                # This is an EXPECTED outcome for models that require thinking —
                # it's exactly what this demo is documenting, not a failure.
                print("\n⚠️ Request timed out with disable_thinking=True")
                print("   This suggests the model may not support disabling thinking.")
                print("   Some models require thinking to function properly.")
                return True

            msg_without = response_without.choices[0].message
            reasoning_without = getattr(msg_without, "reasoning_content", None)
            content_without = msg_without.content or ""

            if reasoning_without:
                thinking_blocks_without = [str(reasoning_without)]
                clean_without = content_without
            else:
                thinking_blocks_without, clean_without = extract_thinking_blocks(content_without)

            print(f"   Thinking blocks found: {'Yes' if thinking_blocks_without else 'No'}")
            print(f"   Response length: {len(content_without)} characters")
            if response_without.usage:
                print(f"   Tokens used: {response_without.usage.total_tokens}")

            # Compare results
            print("\n📊 Comparison Results:")
            if response_with.usage and response_without.usage:
                print(
                    f"   Token difference: {response_with.usage.total_tokens - response_without.usage.total_tokens}"
                )
            print(f"   Character difference: {len(content_with) - len(content_without)}")

            if thinking_blocks_with and thinking_blocks_without:
                print("\n⚠️ Both responses contain thinking blocks.")
                print("   The disable_thinking parameter may not be fully supported by this model.")
            elif thinking_blocks_with and not thinking_blocks_without:
                print("\n✅ disable_thinking parameter working correctly!")
                print("   Reasoning was successfully disabled in the second response.")
            elif not thinking_blocks_with and not thinking_blocks_without:
                print("\n⚠️ No thinking blocks in either response.")
                print("   The model might not be generating visible thinking blocks.")

            # Show sample of each response
            print("\n📝 Response Comparison:")
            print("   With thinking enabled (first 200 chars):")
            print(f"   {clean_with[:200]}...")
            print("\n   With thinking disabled (first 200 chars):")
            print(f"   {clean_without[:200]}...")
        except Exception as e:
            print(f"❌ Error in disable_thinking test: {e}")
            return False

    return True


async def combined_venice_parameters_example() -> bool:
    """Demonstrate using multiple VeniceParameters together and show proper usage.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print("\n🔧 Combined VeniceParameters Example")
    print("=" * 70)

    async with VeniceClient() as client:
        try:
            # Find a model that supports reasoning
            reasoning_model = await find_reasoning_model(client)

            if not reasoning_model:
                print("⚠️ No models with reasoning support found, using standard model")
                model = await client.models.resolve_chat(require_reasoning=True)
            else:
                model = reasoning_model
                print(f"📍 Using model: {model}")

            # Multi-step problem
            problem = """
            A store sells apples for $2 each and oranges for $3 each.
            Sarah buys some apples and oranges for a total of $23.
            She buys 9 fruits in total.

            How many apples and oranges did she buy?
            Show your work step by step.
            """

            # Create comprehensive VeniceParameters
            venice_params = VeniceParameters(
                character_slug=None,
                strip_thinking_response=True,  # Strip thinking for clean output
                disable_thinking=False,  # Still allow model to think
                enable_web_search="off",
                enable_web_citations=False,
                include_search_results_in_stream=False,
                return_search_results_as_documents=None,
                include_venice_system_prompt=True,
            )

            print("\n🔬 Using comprehensive VeniceParameters:")
            print(f"   strip_thinking_response: {venice_params.strip_thinking_response}")
            print(f"   disable_thinking: {venice_params.disable_thinking}")

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    SystemMessage(
                        content="You are a math tutor. Show your work step by step clearly."
                    ),
                    UserMessage(content=problem),
                ],
                venice_parameters=venice_params,
                max_completion_tokens=1024,
                temperature=0.1,
            )

            msg = response.choices[0].message
            reasoning = getattr(msg, "reasoning_content", None)
            content = str(msg.content or "")

            if reasoning:
                thinking_blocks = [str(reasoning)]
                solution = str(content)
            else:
                thinking_blocks, solution = extract_thinking_blocks(str(content))

            # Check server response
            if hasattr(response, "venice_parameters") and response.venice_parameters:
                print("\n📡 Server Response Metadata:")
                print(
                    f"   strip_thinking_response: {response.venice_parameters.strip_thinking_response}"
                )
                print(f"   disable_thinking: {response.venice_parameters.disable_thinking}")

            # Display results
            format_thinking_output(thinking_blocks, solution, "SOLUTION")

            # Token usage
            if response.usage:
                usage = response.usage
                print(
                    f"📊 Token Usage: Input={usage.prompt_tokens}, Output={usage.completion_tokens}, Total={usage.total_tokens}"
                )
        except Exception as e:
            print(f"❌ Error in combined example: {e}")
            return False

    return True


async def usage_breakdown_with_typed_access() -> bool:
    """Read the typed ``completion_tokens_details`` + ``cache_read_input_tokens``.

    Reasoning models populate the ``completion_tokens_details`` object so callers
    can separate visible output from internal reasoning tokens. Venice also
    surfaces ``cache_read_input_tokens`` at the top level of ``usage`` to mirror
    ``prompt_tokens_details.cached_tokens``.

    Both fields are modelled on :class:`venice_ai.ChatUsage`, so there's no
    need to drop down to raw dicts anymore.

    Returns ``True`` on success (including a graceful skip when no reasoning
    model is available), ``False`` if the API call failed.
    """
    print("\n🧮 Typed Usage Breakdown (reasoning + cache)")
    print("=" * 70)

    async with VeniceClient() as client:
        try:
            reasoning_model = await find_reasoning_model(client)
            if not reasoning_model:
                print("⚠️ No reasoning-capable model available; skipping.")
                return True

            print(f"📍 Using reasoning model: {reasoning_model}")

            response = await client.chat.completions.create(
                model=reasoning_model,
                messages=[UserMessage(content="What is 47 * 83? Think step by step.")],
                reasoning_effort="medium",
                max_completion_tokens=1024,
            )

            usage = response.usage
            if not usage:
                print("⚠️ No usage info returned.")
                return True

            print(
                "📊 Token Usage: "
                f"Input={usage.prompt_tokens}, Output={usage.completion_tokens}, "
                f"Total={usage.total_tokens}"
            )

            if usage.completion_tokens_details is not None:
                details = usage.completion_tokens_details
                print(
                    "🧠 Completion details: "
                    f"reasoning_tokens={details.reasoning_tokens}, "
                    f"audio_tokens={details.audio_tokens}, "
                    f"image_tokens={details.image_tokens}"
                )
            else:
                print("🧠 Completion details: none reported for this model.")

            if usage.prompt_tokens_details is not None:
                prompt_details = usage.prompt_tokens_details
                print(
                    "📥 Prompt details: "
                    f"cached_tokens={prompt_details.cached_tokens}, "
                    f"audio_tokens={prompt_details.audio_tokens}"
                )

            if usage.cache_read_input_tokens is not None:
                print(f"💾 cache_read_input_tokens={usage.cache_read_input_tokens}")
        except Exception as e:
            print(f"❌ Error in usage breakdown: {e}")
            return False

    return True


async def model_capability_exploration() -> bool:
    """Explore and display reasoning capabilities of available models.

    Returns ``True`` on success, ``False`` if the model listing failed.
    """
    print("\n🔍 Model Reasoning Capabilities")
    print("=" * 70)

    from venice_ai.types.api import TextModelSpec

    async with VeniceClient() as client:
        try:
            # ``capabilities`` only exists on text models; restrict the listing.
            all_models = await client.models.list(type="text")

            # Find models with reasoning support
            reasoning_models = []
            for model in all_models.data:
                spec = model.model_spec
                if not isinstance(spec, TextModelSpec):
                    continue
                caps = spec.capabilities
                if caps is not None and caps.supportsReasoning:
                    reasoning_models.append(model)

            if reasoning_models:
                print(f"📊 Found {len(reasoning_models)} models with reasoning support:\n")

                for model in reasoning_models[:5]:  # Show first 5
                    print(f"🤖 Model: {model.id}")

                    # Display model capabilities — known TextModelSpec by filter above.
                    spec = model.model_spec
                    assert isinstance(spec, TextModelSpec)
                    caps = spec.capabilities
                    if caps is not None:
                        print(f"   ✓ Reasoning: {caps.supportsReasoning}")
                        print(f"   ✓ Vision: {caps.supportsVision}")
                        print(f"   ✓ Functions: {caps.supportsFunctionCalling}")
                        print(f"   ✓ Web Search: {caps.supportsWebSearch}")
                        print(f"   ✓ Code Optimized: {caps.optimizedForCode}")
                    print()
            else:
                print("⚠️ No models with explicit reasoning support found")
                print("\n📝 Available text models (can still use prompting techniques):")
                text_count = 0
                for model in all_models.data:
                    if hasattr(model, "type") and model.type == "text":
                        print(f"   - {model.id}")
                        text_count += 1
                        if text_count >= 5:
                            break

        except Exception as e:
            print(f"❌ Error exploring models: {e}")
            return False

    return True


async def main() -> int:
    """Run all reasoning and thinking examples.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    # Line-buffer stdout so the header printed *before* each (slow) completion
    # survives even if the process is killed mid-request — making it obvious
    # which call was in flight.
    sys.stdout.reconfigure(line_buffering=True)

    print("🚀 Venice AI Reasoning & Thinking Examples with VeniceParameters")
    print("=" * 70)
    print("This example demonstrates the PROPER way to use reasoning control")
    print("parameters through the VeniceParameters model.\n")

    # Run examples, tracking an honest pass/fail tally.
    results: list[tuple[str, bool]] = [
        ("model_capability_exploration", await model_capability_exploration()),
        ("basic_reasoning_with_venice_parameters", await basic_reasoning_with_venice_parameters()),
        ("test_strip_thinking_parameter", await test_strip_thinking_parameter()),
        ("test_disable_thinking_parameter", await test_disable_thinking_parameter()),
        ("combined_venice_parameters_example", await combined_venice_parameters_example()),
        ("usage_breakdown_with_typed_access", await usage_breakdown_with_typed_access()),
    ]

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("\n✨ Examples completed!")

    print("\n💡 Key Insights:")
    print("   - Use VeniceParameters for controlling reasoning behavior")
    print("   - Pass VeniceParameters object directly to chat.completions.create()")
    print("   - strip_thinking_response controls visibility of thinking blocks")
    print("   - disable_thinking controls whether model uses reasoning at all")
    print("   - Check response.venice_parameters for server confirmation")
    print("\n📚 Correct Usage:")
    print("   ```python")
    print("   venice_params = VeniceParameters(")
    print("       strip_thinking_response=True,  # Hide thinking blocks")
    print("       disable_thinking=False,        # Allow thinking")
    print("       include_venice_system_prompt=True")
    print("   )")
    print("   response = await client.chat.completions.create(")
    print("       model=model,")
    print("       messages=messages,")
    print("       venice_parameters=venice_params  # Pass object directly")
    print("   )")
    print("   ```")
    print("\n⚠️ Note: The effectiveness of these parameters depends on")
    print("   model support. Always check the response metadata to confirm")
    print("   which parameters were actually applied by the server.")

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
            "Check that your API key is valid and you have a model that supports reasoning.",
            file=sys.stderr,
        )
        sys.exit(1)
