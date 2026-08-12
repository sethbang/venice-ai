#!/usr/bin/env python3
"""
Venice AI SDK - Multi-turn Conversation
========================================

This example demonstrates how to maintain context across multiple conversation turns.
Learn how to build conversational AI that remembers previous interactions.
"""

import asyncio
import sys

from venice_ai import Conversation, VeniceClient
from venice_ai.types.api import AssistantMessage, SystemMessage, UserMessage


async def simple_conversation():
    """Demonstrate a simple multi-turn conversation with context preservation.

    Uses :class:`venice_ai.Conversation` — a thin wrapper that manages the
    message list and exposes ``add_user`` / ``add_response`` for the
    canonical loop. For variations (sliding-window history, mid-conversation
    context reset, etc.) see the other functions in this file.
    """
    print("💬 Simple Multi-turn Conversation")
    print("-" * 40)

    async with VeniceClient() as client:
        # Get available chat model dynamically
        chat_model = await client.models.resolve_chat()
        print(f"📍 Using model: {chat_model}")

        # System message sets the context
        conv = Conversation(
            system="You are a helpful assistant. Keep your responses brief and friendly."
        )

        # Conversation turns
        conversation_turns = [
            "Hi! What's your favorite programming language?",
            "Why do you like it?",
            "Can you give me a simple code example?",
        ]

        for turn_num, user_input in enumerate(conversation_turns, 1):
            print(f"\n🔄 Turn {turn_num}")
            print(f"👤 User: {user_input}")

            conv.add_user(user_input)

            response = await client.chat.completions.create(
                model=chat_model, messages=conv.messages, max_completion_tokens=200, temperature=0.7
            )

            assistant_content = response.text or ""
            print(f"🤖 Assistant: {assistant_content}")

            # add_response unwraps the choice and appends an AssistantMessage.
            conv.add_response(response)

        msgs = conv.messages
        print(f"\n📝 Conversation history: {len(msgs)} messages")
        print("   System messages: 1")
        print(f"   User messages: {len(conversation_turns)}")
        print(f"   Assistant messages: {len(conversation_turns)}")


async def conversation_with_personality():
    """Demonstrate conversation with different assistant personalities."""
    print("\n🎭 Conversation with Personality")
    print("-" * 40)

    async with VeniceClient() as client:
        # Get available chat model dynamically
        chat_model = await client.models.resolve_chat()
        print(f"📍 Using model: {chat_model}")

        # Initialize with personality-defining system message
        messages: list[SystemMessage | UserMessage | AssistantMessage] = [
            SystemMessage(
                content="You are a pirate captain assistant. Speak like a pirate and make nautical references. Keep responses brief.",
                name="Captain",
            )
        ]

        # Ask questions about different topics
        questions = [
            "How do I learn programming?",
            "What's the weather like today?",
            "Tell me about databases.",
        ]

        for question in questions:
            print(f"\n👤 User: {question}")

            messages.append(UserMessage(content=question))

            response = await client.chat.completions.create(
                model=chat_model,
                messages=messages,
                max_completion_tokens=150,
                temperature=0.9,  # Higher temperature for more creative responses
            )

            assistant_content = response.text or ""
            print(f"🏴‍☠️ Pirate Assistant: {assistant_content}")

            messages.append(AssistantMessage.from_response(response))


async def context_window_management():
    """Demonstrate managing context window by limiting conversation history."""
    print("\n🪟 Context Window Management")
    print("-" * 40)

    async with VeniceClient() as client:
        # Get available chat model dynamically
        chat_model = await client.models.resolve_chat()
        print(f"📍 Using model: {chat_model}")

        # System message
        system_message = SystemMessage(
            content="You are a helpful math tutor. Provide clear, concise explanations."
        )

        # Keep only last N conversation turns (sliding window)
        MAX_HISTORY_PAIRS = 2  # Keep last 2 user-assistant pairs
        conversation_history: list[UserMessage | AssistantMessage] = []

        # Series of math questions
        questions = [
            "What is 15 + 27?",
            "What's the answer multiplied by 2?",  # Refers to previous
            "Now divide that by 3.",  # Refers to previous
            "What was my first question?",  # Should not remember if outside window
        ]

        for turn_num, question in enumerate(questions, 1):
            print(f"\n🔄 Turn {turn_num}")
            print(f"👤 User: {question}")

            # Build messages with system + limited history
            messages: list[SystemMessage | UserMessage | AssistantMessage] = [system_message]
            messages.extend(conversation_history)
            messages.append(UserMessage(content=question))

            response = await client.chat.completions.create(
                model=chat_model, messages=messages, max_completion_tokens=100, temperature=0.3
            )

            assistant_content = response.text or ""
            print(f"🤖 Assistant: {assistant_content}")

            # Add to conversation history
            conversation_history.append(UserMessage(content=question))
            conversation_history.append(AssistantMessage.from_response(response))

            # Trim history to maintain window size (keep last N pairs)
            # Each pair = 2 messages (user + assistant)
            max_messages = MAX_HISTORY_PAIRS * 2
            if len(conversation_history) > max_messages:
                conversation_history = conversation_history[-max_messages:]

            print(
                f"📊 History size: {len(conversation_history)} messages ({len(conversation_history) // 2} pairs)"
            )


async def conversation_with_context_reset():
    """Demonstrate resetting conversation context."""
    print("\n🔄 Conversation with Context Reset")
    print("-" * 40)

    async with VeniceClient() as client:
        # Get available chat model dynamically
        chat_model = await client.models.resolve_chat()
        print(f"📍 Using model: {chat_model}")

        # First conversation topic
        print("\n📚 Topic 1: Science")
        messages: list[SystemMessage | UserMessage | AssistantMessage] = [
            SystemMessage(content="You are a science teacher. Keep responses brief.")
        ]

        # Talk about science
        science_questions = [
            "What is photosynthesis?",
            "Why is it important?",
        ]

        for question in science_questions:
            print(f"👤 User: {question}")
            messages.append(UserMessage(content=question))

            response = await client.chat.completions.create(
                model=chat_model, messages=messages, max_completion_tokens=100, temperature=0.5
            )

            content = response.text or ""
            print(f"🤖 Science Teacher: {content}\n")

            messages.append(AssistantMessage.from_response(response))

        # Reset context and switch topics
        print("🔄 Resetting context and switching topics...\n")
        print("🎨 Topic 2: Art")

        messages: list[SystemMessage | UserMessage | AssistantMessage] = [
            SystemMessage(content="You are an art historian. Keep responses brief.")
        ]

        # Talk about art
        art_questions = [
            "Who painted the Mona Lisa?",
            "What was special about their technique?",
        ]

        for question in art_questions:
            print(f"👤 User: {question}")
            messages.append(UserMessage(content=question))

            response = await client.chat.completions.create(
                model=chat_model, messages=messages, max_completion_tokens=100, temperature=0.5
            )

            content = response.text or ""
            print(f"🤖 Art Historian: {content}\n")

            messages.append(AssistantMessage.from_response(response))


async def interactive_conversation_example():
    """Demonstrate pattern for interactive user input (simulated)."""
    print("\n💻 Interactive Conversation Pattern")
    print("-" * 40)

    async with VeniceClient() as client:
        # Get available chat model dynamically
        chat_model = await client.models.resolve_chat()
        print(f"📍 Using model: {chat_model}")

        # Initialize conversation
        messages: list[SystemMessage | UserMessage | AssistantMessage] = [
            SystemMessage(
                content="You are a helpful coding assistant. Provide brief, practical advice."
            )
        ]

        # Simulated user inputs (in real app, these would come from input())
        simulated_inputs = [
            "How do I read a file in Python?",
            "What about error handling?",
            "exit",  # Exit command
        ]

        print("💡 Type 'exit' to end the conversation (simulated)\n")

        for user_input in simulated_inputs:
            # Simulate user input
            print(f"👤 User: {user_input}")

            # Check for exit command
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("👋 Goodbye!")
                break

            # Add user message
            messages.append(UserMessage(content=user_input))

            # Get response
            response = await client.chat.completions.create(
                model=chat_model, messages=messages, max_completion_tokens=200, temperature=0.7
            )

            # Display response
            content = response.text or ""
            print(f"🤖 Assistant: {content}\n")

            # Add to history
            messages.append(AssistantMessage.from_response(response))


async def main():
    """Run all multi-turn conversation examples."""
    print("🚀 Venice AI Multi-turn Conversation Examples")
    print("=" * 50)

    await simple_conversation()
    await conversation_with_personality()
    await context_window_management()
    await conversation_with_context_reset()
    await interactive_conversation_example()

    print("\n✨ Multi-turn conversation examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Maintaining conversation history")
    print("   - Context preservation across turns")
    print("   - System messages for personality")
    print("   - Managing context window size")
    print("   - Resetting conversation context")
    print("   - Interactive conversation patterns")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
