#!/usr/bin/env python3
"""
Venice AI SDK - Streaming Chat Completions
==========================================

This example demonstrates streaming chat completions using the SDK's high-level
streaming API. Learn how to handle real-time token streaming, track usage, and
implement various streaming patterns.

Features Demonstrated:
    - High-level stream() API with text_deltas() and collect()
    - Manual chunk iteration for advanced use cases
    - Stream options for usage tracking
    - Concurrent streaming
    - Multi-turn streaming conversations
    - Error handling during streams
"""

import asyncio
import sys
import time

from venice_ai import VeniceClient
from venice_ai.types.api import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from venice_ai.types.api.requests import StreamOptions

# =============================================================================
# Streaming Examples
# =============================================================================


async def basic_streaming():
    """Demonstrate basic streaming with the high-level stream() API."""
    print("🌊 Basic Streaming Example")
    print("-" * 40)

    async with VeniceClient() as client:
        chat_model = await client.models.resolve_chat()
        print(f"📍 Using model: {chat_model}")

        print("\n🤖 Assistant (streaming): ", end="", flush=True)

        # Use the high-level stream() API with text_deltas()
        stream = await client.chat.completions.stream(
            model=chat_model,
            messages=[
                UserMessage(
                    content="Write a short story about a robot learning to paint. Make it exactly 3 sentences."
                )
            ],
            max_completion_tokens=200,
            temperature=0.8,
        )
        async with stream:
            async for text in stream.text_deltas():
                print(text, end="", flush=True)

        print()


async def streaming_with_collect():
    """Demonstrate ``collect_with_deltas()`` — live deltas and the final response.

    ``ChatStream.collect_with_deltas()`` yields each text delta as it arrives
    AND populates ``stream.final_response`` once iteration completes — one
    consumption gets you both signals, no duplicate request needed.
    (For display-only streaming use ``text_deltas()``; for "give me only the
    final response, don't render deltas" use ``collect()``.)
    """
    print("\n📦 Streaming with collect_with_deltas()")
    print("-" * 40)

    async with VeniceClient() as client:
        chat_model = await client.models.resolve_chat()
        print(f"📍 Using model: {chat_model}")

        messages = [
            SystemMessage(content="You are a helpful assistant that provides concise answers."),
            UserMessage(content="What are the three primary colors?"),
        ]

        print("\n🤖 Assistant: ", end="", flush=True)
        stream = await client.chat.completions.stream(
            model=chat_model,
            messages=messages,
            stream_options=StreamOptions(include_usage=True),
            max_completion_tokens=100,
            temperature=0.3,
        )
        async with stream:
            async for text in stream.collect_with_deltas():
                print(text, end="", flush=True)
        print()

        # final_response is populated once iteration completes.
        response = stream.final_response
        if response and response.usage:
            print("\n📊 Token Usage:")
            print(f"   Prompt tokens: {response.usage.prompt_tokens}")
            print(f"   Completion tokens: {response.usage.completion_tokens}")
            print(f"   Total tokens: {response.usage.total_tokens}")


async def animated_streaming():
    """Demonstrate streaming with animated display patterns."""
    print("\n🎨 Animated Streaming Display")
    print("-" * 40)

    async with VeniceClient() as client:
        chat_model = await client.models.resolve_chat()
        print(f"📍 Using model: {chat_model}")

        messages = [
            UserMessage(
                content="Give me 3 short facts about the ocean. Keep each fact to one sentence."
            )
        ]

        # Word-by-word streaming with animation
        print("\n🎭 Word-by-word streaming:")
        print("   ", end="", flush=True)

        stream = await client.chat.completions.stream(
            model=chat_model, messages=messages, max_completion_tokens=200, temperature=0.5
        )
        async with stream:
            buffer = ""
            word_count = 0
            async for text in stream.text_deltas():
                buffer += text

                # Process complete words
                while " " in buffer or "\n" in buffer:
                    space_idx = buffer.find(" ")
                    newline_idx = buffer.find("\n")

                    if space_idx == -1:
                        delimiter_idx = newline_idx
                    elif newline_idx == -1:
                        delimiter_idx = space_idx
                    else:
                        delimiter_idx = min(space_idx, newline_idx)

                    word = buffer[: delimiter_idx + 1]
                    print(word, end="", flush=True)
                    buffer = buffer[delimiter_idx + 1 :]
                    word_count += 1

                    # Only animate first 50 words
                    if word_count < 50:
                        await asyncio.sleep(0.05)

            # Print remaining buffer
            if buffer:
                print(buffer, end="", flush=True)
        print()


async def concurrent_streams():
    """Demonstrate handling multiple concurrent streams."""
    print("\n🔀 Concurrent Streaming")
    print("-" * 40)

    async with VeniceClient() as client:
        chat_model = await client.models.resolve_chat()
        print(f"📍 Using model: {chat_model}")

        prompts = [
            "Tell me a joke about programming.",
            "Give me a fun fact about space.",
            "Share a cooking tip.",
        ]

        async def process_stream(prompt: str, index: int) -> tuple[int, str, float]:
            """Process a single stream and return results."""
            wall_start = time.perf_counter()

            stream = await client.chat.completions.stream(
                model=chat_model,
                messages=[UserMessage(content=prompt)],
                max_completion_tokens=500,
                temperature=0.8,
            )
            async with stream:
                response = await stream.collect()

            content = str(response.text or "")
            wall_time = time.perf_counter() - wall_start
            return index, content, wall_time

        print("\n🚀 Starting concurrent streams...")

        start_time = time.time()
        tasks = [process_stream(prompt, i) for i, prompt in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        for index, content, wall_time in results:
            print(f"\n📝 Stream {index + 1} - {prompts[index][:30]}...")
            clean_content = content.replace("\n", " ").strip()
            print(f"   Response: {clean_content}")
            print(f"   Duration: {wall_time:.3f}s")

        print(f"\n⏱️ Total wall time for {len(prompts)} concurrent streams: {total_time:.3f}s")


async def error_handling_in_streams():
    """Demonstrate proper error handling during streaming."""
    print("\n⚠️ Error Handling in Streams")
    print("-" * 40)

    async with VeniceClient() as client:
        chat_model = await client.models.resolve_chat()
        print(f"📍 Using model: {chat_model}")

        test_cases = [
            {
                "name": "Normal streaming",
                "messages": [UserMessage(content="Say hello")],
                "max_completion_tokens": 10,
            },
            {
                "name": "Very long response (might hit limit)",
                "messages": [UserMessage(content="Count from 1 to 1000")],
                "max_completion_tokens": 50,
            },
            {
                "name": "Empty message handling",
                "messages": [UserMessage(content="")],
                "max_completion_tokens": 10,
            },
        ]

        for test_case in test_cases:
            print(f"\n🧪 Test: {test_case['name']}")
            try:
                stream = await client.chat.completions.stream(
                    model=chat_model,
                    messages=test_case["messages"],
                    max_completion_tokens=test_case["max_completion_tokens"],
                    temperature=0.5,
                )
                async with stream:
                    response = await stream.collect()

                content = response.text or ""
                finish_reason = response.choices[0].finish_reason if response.choices else None
                print("   ✅ Success")
                print(f"   Content: {content[:100]}{'...' if len(content) > 100 else ''}")
                print(f"   Finish reason: {finish_reason}")

            except Exception as e:
                error_msg = str(e)[:200]
                print(f"   ❌ Error occurred: {type(e).__name__}: {error_msg}")


async def multi_turn_streaming():
    """Demonstrate multi-turn conversation with streaming."""
    print("\n💬 Multi-turn Streaming Conversation")
    print("-" * 40)

    async with VeniceClient() as client:
        chat_model = await client.models.resolve_chat()
        print(f"📍 Using model: {chat_model}")

        # Initialize conversation
        messages: list[SystemMessage | UserMessage | AssistantMessage] = [
            SystemMessage(
                content="You are a helpful tutor teaching about planets. Keep responses brief."
            )
        ]

        # Conversation turns
        user_inputs = [
            "What is the largest planet?",
            "How many moons does it have?",
            "What about the second largest?",
        ]

        for turn, user_input in enumerate(user_inputs, 1):
            print(f"\n🔄 Turn {turn}")
            print(f"👤 User: {user_input}")

            messages.append(UserMessage(content=user_input))

            print("🤖 Assistant: ", end="", flush=True)

            stream = await client.chat.completions.stream(
                model=chat_model, messages=messages, max_completion_tokens=100, temperature=0.5
            )
            async with stream:
                async for text in stream.collect_with_deltas():
                    print(text, end="", flush=True)

            print()

            # Add the streamed content to conversation history — collect_with_deltas
            # populated stream.final_response, so we can use AssistantMessage.from_response
            # instead of a manually accumulated string.
            if stream.final_response is not None:
                messages.append(AssistantMessage.from_response(stream.final_response))

        print(f"\n📝 Final conversation length: {len(messages)} messages")


async def main():
    """Run all streaming examples."""
    print("🚀 Venice AI Streaming Chat Examples")
    print("=" * 50)

    await basic_streaming()
    await streaming_with_collect()
    await animated_streaming()
    await concurrent_streams()
    await error_handling_in_streams()
    await multi_turn_streaming()

    print("\n✨ Streaming examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - High-level stream() API")
    print("   - text_deltas() for real-time display")
    print("   - collect() for assembled responses")
    print("   - Concurrent stream processing")
    print("   - Error handling in streams")
    print("   - Multi-turn conversations with streaming")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Stream interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
