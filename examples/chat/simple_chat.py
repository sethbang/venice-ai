#!/usr/bin/env python3
"""
Venice AI SDK - Simple Chat Completions
=======================================

This example demonstrates basic chat completion functionality with the Venice AI SDK.
Learn how to create simple conversational AI interactions.
"""

import asyncio
import sys

from venice_ai import VeniceClient
from venice_ai.types.api import SystemMessage, UserMessage


async def basic_chat_completion():
    """Create a simple chat completion."""
    print("💬 Basic Chat Completion")
    print("-" * 30)

    async with VeniceClient() as client:
        # Get available chat model dynamically
        chat_model = await client.models.resolve_chat()
        print(f"📍 Using model: {chat_model}")

        # Simple user message
        response = await client.chat.completions.create(
            model=chat_model,
            messages=[UserMessage(content="Explain quantum computing in simple terms.")],
            max_completion_tokens=200,
            temperature=0.7,
        )

        # response.text handles the str | list[Content] | None union for you,
        # collapsing multimodal lists to their text parts and returning None
        # if the response has no choices/content.
        print(f"🤖 Assistant: {response.text or ''}")

        # Show usage information
        if response.usage:
            usage = response.usage
            print("\n📊 Token Usage:")
            print(f"   Input tokens: {usage.prompt_tokens}")
            print(f"   Output tokens: {usage.completion_tokens}")
            print(f"   Total tokens: {usage.total_tokens}")


async def chat_with_system_message():
    """Create a chat completion with a system message."""
    print("\n🎭 Chat with System Message")
    print("-" * 30)

    async with VeniceClient() as client:
        # Get available chat model dynamically
        chat_model = await client.models.resolve_chat()
        print(f"📍 Using model: {chat_model}")

        # Chat with system message to set personality
        response = await client.chat.completions.create(
            model=chat_model,
            messages=[
                SystemMessage(
                    content="You are a helpful assistant that explains things like you're talking to a 10-year-old. Use simple words and fun analogies."
                ),
                UserMessage(content="What is gravity?"),
            ],
            max_completion_tokens=150,
            temperature=0.8,
        )

        print(f"🤖 Kid-friendly Assistant: {response.text or ''}")


async def different_models_comparison():
    """Compare responses from different models."""
    print("\n🔬 Different Models Comparison")
    print("-" * 30)

    question = "What's the meaning of life?"

    async with VeniceClient() as client:
        # Get multiple models for comparison
        try:
            models_response = await client.models.list(type="chat")
            models = [m.id for m in models_response.data[:2]]
            print(f"📍 Comparing models: {models}")
        except Exception as e:
            print(f"⚠️ Could not get multiple models: {e}")
            # Fallback to single model
            models = [await client.models.resolve_chat()]
            print(f"📍 Using single model: {models}")

        for model in models:
            try:
                print(f"\n🤖 {model} says:")
                response = await client.chat.completions.create(
                    model=model,
                    messages=[UserMessage(content=question)],
                    max_completion_tokens=100,
                    temperature=0.7,
                )

                content = response.text or ""
                if not content.strip():
                    print("   (model returned empty response)")
                else:
                    print(f"   {content}")

            except Exception as e:
                print(f"   ❌ Error with {model}: {e}")


async def parameter_variations():
    """Demonstrate different parameter settings."""
    print("\n⚙️ Parameter Variations")
    print("-" * 30)

    prompt = "Write a haiku about programming."

    # Different temperature settings
    temperatures = [0.1, 0.7, 1.5]

    async with VeniceClient() as client:
        # Get available chat model dynamically
        chat_model = await client.models.resolve_chat()
        print(f"📍 Using model: {chat_model}")

        for temp in temperatures:
            print(f"\n🌡️ Temperature {temp} (creativity level):")

            response = await client.chat.completions.create(
                model=chat_model,
                messages=[UserMessage(content=prompt)],
                max_completion_tokens=100,
                temperature=temp,
            )

            print(f"   {response.text or ''}")


async def main():
    """Run all chat completion examples."""
    print("🚀 Venice AI Simple Chat Examples")
    print("=" * 50)

    await basic_chat_completion()
    await chat_with_system_message()
    await different_models_comparison()
    await parameter_variations()

    print("\n✨ Simple chat examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Basic chat completions")
    print("   - System messages for personality")
    print("   - Model selection and comparison")
    print("   - Parameter tuning (temperature)")
    print("   - Token usage monitoring")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
