#!/usr/bin/env python3
"""
Venice AI SDK - Quick Start Example
==================================

This example demonstrates the most basic usage of the Venice AI SDK.
Get up and running with just a few lines of code!

Prerequisites:
- Install: pip install venice-py
- Set API key: export VENICE_API_KEY="your-api-key"
"""

import asyncio
import sys

from venice_ai import VeniceClient
from venice_ai.types.api import UserMessage


async def main():
    """Quick start example showing basic Venice AI usage."""

    print("🚀 Venice AI Quick Start Example")
    print("=" * 40)

    # Create a Venice AI client (reads VENICE_API_KEY from environment)
    async with VeniceClient() as client:
        print("✅ Client created successfully")

        # Get available chat model dynamically
        chat_model = await client.models.resolve_chat()
        print(f"📍 Using model: {chat_model}")

        # Simple chat completion
        print("\n💬 Creating a simple chat completion...")

        response = await client.chat.completions.create(
            model=chat_model,
            messages=[UserMessage(content="Say hello and introduce yourself briefly.")],
            max_completion_tokens=100,
        )

        # Extract and display the response
        message = response.text
        print(f"🤖 Assistant: {message}")

        # Show token usage
        if response.usage:
            usage = response.usage
            print("\n📊 Token Usage:")
            print(f"   Prompt tokens: {usage.prompt_tokens}")
            print(f"   Completion tokens: {usage.completion_tokens}")
            print(f"   Total tokens: {usage.total_tokens}")

    print("\n✨ Quick start completed successfully!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("Check that your API key is valid and you have internet connection.", file=sys.stderr)
        sys.exit(1)
