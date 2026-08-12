#!/usr/bin/env python3
"""
Venice AI SDK - Venice Parameters Showcase
==========================================

This example demonstrates the Venice-specific parameters available through the VeniceParameters class.
These parameters provide fine-grained control over Venice AI's unique features including character personalities,
reasoning capabilities, web search integration, and system prompt behavior.

Venice Parameters Covered:
    • character_slug: Use public Venice character personalities
    • strip_thinking_response: Control display of reasoning <think> blocks
    • disable_thinking: Disable reasoning on reasoning-capable models
    • enable_web_search: Enable web search for current knowledge
    • enable_web_citations: Request citation formatting in responses
    • include_search_results_in_stream: Stream search results as first chunk
    • return_search_results_as_documents: Surface search as tool calls
    • include_venice_system_prompt: Control Venice system prompt inclusion

Requirements:
    - Venice AI API key (set as VENICE_API_KEY environment variable)
    - Python 3.13+
    - venice-ai SDK

Features Demonstrated:
    - Dynamic character discovery and usage
    - Reasoning model detection and thinking control
    - Web search integration with citations
    - Search result streaming and document formatting
    - System prompt behavior control
    - Real-world parameter combinations

Performance note:
    To keep this showcase fast and within typical rate limits, completions are
    capped at modest ``max_completion_tokens`` and the independent calls inside a
    demo are dispatched concurrently via ``client.gather(max_concurrency=...)``.
"""

import asyncio
import re
import sys
from typing import Any

from venice_ai import VeniceClient, extract_thinking_blocks
from venice_ai.types.api import SystemMessage, UserMessage
from venice_ai.types.api.requests import VeniceParameters

# =============================================================================
# Helper Functions
# =============================================================================


def _print_usage(response: Any) -> None:
    """Print token usage from a response, or note when usage is unavailable."""
    usage = getattr(response, "usage", None)
    if usage is None:
        print("\n📊 Token Usage: not provided by the API", flush=True)
        return
    print(
        f"\n📊 Token Usage: Input={usage.prompt_tokens}, "
        f"Output={usage.completion_tokens}, Total={usage.total_tokens}",
        flush=True,
    )


def display_body(response: Any, visible: str) -> None:
    """Print a response body, falling back to ``reasoning_content`` when empty.

    Reasoning/web-search models on a tight ``max_completion_tokens`` budget often
    spend it entirely on the dedicated ``reasoning_content`` field and emit empty
    user-visible ``content``. To keep the demo's output substantive we surface the
    reasoning trace (truncated) rather than printing a blank section.
    """
    if visible.strip():
        print(visible, flush=True)
        return

    message = None
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
    reasoning = (getattr(message, "reasoning_content", None) or "").strip()
    if reasoning:
        print("ℹ️ (visible content empty — showing reasoning_content trace)", flush=True)
        print(reasoning[:1200] + ("…" if len(reasoning) > 1200 else ""), flush=True)
    else:
        print("[no visible content — token budget consumed by reasoning]", flush=True)


def display_parameters(params: VeniceParameters, title: str = "VeniceParameters") -> None:
    """Display VeniceParameters in a formatted way."""
    print(f"\n🔧 {title}:", flush=True)
    params_dict = params.model_dump()

    # Show all non-default values
    for key, value in params_dict.items():
        if value is not None and value is not False and value != "off":
            print(f"   {key}: {value}", flush=True)


def format_web_citations(response) -> list[str]:
    """Extract and format web citations from response."""
    citations = []

    vp = getattr(response, "venice_parameters", None)
    web_citations = getattr(vp, "web_search_citations", None) if vp else None
    if web_citations:
        for i, citation in enumerate(web_citations, 1):
            # Handle both object attributes and dict keys
            if isinstance(citation, dict):
                title = citation.get("title", "Unknown")
                url = citation.get("url", "")
            else:
                title = getattr(citation, "title", "Unknown")
                url = getattr(citation, "url", "")
            citations.append(f"[{i}] {title} - {url}")

    return citations


def print_section_header(title: str, emoji: str = "📋") -> None:
    """Print a formatted section header."""
    print(f"\n{emoji} {title}", flush=True)
    print("=" * 70, flush=True)


def print_subsection(title: str, emoji: str = "📍") -> None:
    """Print a formatted subsection header."""
    print(f"\n{emoji} {title}", flush=True)
    print("-" * 50, flush=True)


# =============================================================================
# Model and Resource Discovery Functions
# =============================================================================


async def find_web_search_model(client: VeniceClient) -> str | None:
    """
    Find a model that supports web search capabilities.

    The resolver has no ``require_web_search`` flag, so we inspect the text
    model specs directly.

    Args:
        client: Venice AI client instance

    Returns:
        Model ID that supports web search, or None if not found
    """
    from venice_ai.types.api import TextModelSpec

    try:
        models_response = await client.models.list(type="text")

        for model in models_response.data:
            spec = model.model_spec
            if (
                isinstance(spec, TextModelSpec)
                and spec.capabilities
                and spec.capabilities.supportsWebSearch
            ):
                return model.id

    except Exception as e:
        print(f"⚠️ Error finding web search model: {e}", flush=True)

    return None


async def find_available_characters(client: VeniceClient) -> list[Any]:
    """
    Fetch available characters from the Venice API.

    Args:
        client: Venice AI client instance

    Returns:
        List of character objects, or empty list if unavailable
    """
    try:
        characters_response = await client.characters.list()
        return characters_response.data if characters_response.data else []
    except Exception as e:
        print(f"⚠️ Error fetching characters: {e}", flush=True)
        return []


# =============================================================================
# Example Functions
# =============================================================================


async def character_example(client: VeniceClient) -> bool:
    """Demonstrate character_slug parameter with real Venice characters.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print_section_header("Character-Based Chat Example", "🎭")

    # Fetch available characters
    characters = await find_available_characters(client)

    if not characters:
        print("⚠️ No characters available, skipping character example", flush=True)
        return True  # Not a failure: account simply has no characters.

    # Use the first available character
    character = characters[0]
    print(f"📍 Using character: {character.slug} ({character.name})", flush=True)
    if hasattr(character, "description") and character.description:
        print(f"   Description: {character.description}", flush=True)
    if hasattr(character, "tags") and character.tags:
        print(f"   Tags: {character.tags}", flush=True)

    # Get a suitable model
    model = await client.models.resolve_chat()
    print(f"📍 Using model: {model}", flush=True)

    # Create VeniceParameters with character
    venice_params = VeniceParameters(
        character_slug=character.slug,
        enable_web_search="off",
        include_venice_system_prompt=True,
    )

    display_parameters(venice_params)

    # Ask a question that showcases the character's personality
    question = "In one short paragraph, what's your perspective on the nature of reality?"
    print(f"\n💬 Question: {question}", flush=True)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[UserMessage(content=question)],
            venice_parameters=venice_params,
            max_completion_tokens=256,
            temperature=0.7,
        )

        content = response.text or ""
        thinking_blocks, clean_response = extract_thinking_blocks(content)

        if thinking_blocks:
            print("\n💭 Thinking Process:", flush=True)
            for i, block in enumerate(thinking_blocks, 1):
                print(f"   Block {i}: {block[:100]}...", flush=True)

        print_subsection("Character Response", "📝")
        print(clean_response, flush=True)

        _print_usage(response)
        return True

    except Exception as e:
        print(f"❌ Error in character example: {e}", flush=True)
        return False


async def thinking_control_example(client: VeniceClient) -> bool:
    """Demonstrate thinking control parameters with a reasoning-capable model.

    Runs the two test completions concurrently via ``client.gather``.

    Returns ``True`` on success, ``False`` if a request failed.
    """
    print_section_header("Thinking Control Example", "🧠")

    # Pick a reasoning model. Venice reasoning models surface chain-of-thought
    # via the dedicated `reasoning_content` field on the assistant message,
    # alongside the user-visible `content`. We compare the two test runs by
    # length of those fields plus a check for `<think>` tags inside content.
    try:
        reasoning_model = await client.models.resolve_chat(require_reasoning=True)
    except Exception as e:
        print(f"⚠️ Could not resolve a reasoning model: {e}", flush=True)
        reasoning_model = await client.models.resolve_chat()

    print(f"📍 Using model: {reasoning_model}", flush=True)

    # A logic puzzle pushes the model into chain-of-thought reasoning.
    puzzle = (
        "Three friends — Alice, Bob, and Carol — each picked one of red, "
        "green, or blue. Alice did not pick red. Bob's color comes "
        "alphabetically before Carol's. What did each person pick? "
        "Answer in one or two sentences."
    )

    print(f"\n💭 Question: {puzzle}", flush=True)

    # Test 1 config: Show thinking blocks (disable_thinking=False)
    venice_params_show = VeniceParameters(
        strip_thinking_response=False,  # Keep thinking blocks visible
        disable_thinking=False,  # Enable thinking
        enable_web_search="off",
        include_venice_system_prompt=True,
    )

    # Test 2 config: Disable thinking entirely (disable_thinking=True)
    venice_params_strip = VeniceParameters(
        strip_thinking_response=True,  # Strip <think> tags if any leak into content
        disable_thinking=True,  # Tell server to skip the reasoning step
        enable_web_search="off",
        include_venice_system_prompt=True,
    )

    print_subsection("Test 1: Show Thinking Process", "🔍")
    display_parameters(venice_params_show)
    print_subsection("Test 2: Disable Thinking", "✂️")
    display_parameters(venice_params_strip)
    print("\n⏳ Dispatching both completions concurrently...", flush=True)

    try:
        response_show, response_strip = await client.gather(
            [
                client.chat.completions.create(
                    model=reasoning_model,
                    messages=[UserMessage(content=puzzle)],
                    venice_parameters=venice_params_show,
                    max_completion_tokens=512,
                    temperature=0.1,
                ),
                client.chat.completions.create(
                    model=reasoning_model,
                    messages=[UserMessage(content=puzzle)],
                    venice_parameters=venice_params_strip,
                    max_completion_tokens=512,
                    temperature=0.1,
                ),
            ],
            max_concurrency=2,
            return_exceptions=False,
        )
    except Exception as e:
        print(f"❌ Error in thinking control example: {e}", flush=True)
        return False

    # --- Test 1 output ---
    content_show = response_show.text or ""
    thinking_blocks_show, clean_show = extract_thinking_blocks(content_show)
    print_subsection("Test 1: Model Response (thinking on)", "📝")
    if thinking_blocks_show:
        print("💭 Thinking Process (visible):", flush=True)
        for block in thinking_blocks_show:
            lines = block.strip().split("\n")
            for line in lines[:2]:
                print(f"   {line}", flush=True)
            if len(lines) > 2:
                print(f"   ... ({len(lines) - 2} more lines of reasoning)", flush=True)
    display_body(response_show, clean_show if clean_show else content_show)
    _print_usage(response_show)

    # --- Test 2 output ---
    content_strip = response_strip.text or ""
    thinking_blocks_strip, clean_strip = extract_thinking_blocks(content_strip)
    print_subsection("Test 2: Model Response (thinking off)", "📝")
    if thinking_blocks_strip:
        print(
            f"ℹ️ Note: Found {len(thinking_blocks_strip)} thinking block(s) in response", flush=True
        )
        print("   (strip_thinking_response may not be supported by this model)", flush=True)
    else:
        print("✅ Response contains no thinking blocks", flush=True)
    print(content_strip if not thinking_blocks_strip else clean_strip, flush=True)
    _print_usage(response_strip)

    # --- Verify the two outputs actually differ ---
    print_subsection("Reasoning On vs Off Comparison", "🔬")
    with_msg = response_show.choices[0].message
    without_msg = response_strip.choices[0].message
    with_reason = getattr(with_msg, "reasoning_content", None) or ""
    without_reason = getattr(without_msg, "reasoning_content", None) or ""
    with_content = with_msg.content or ""
    without_content = without_msg.content or ""
    with_tags = "<think" in str(with_content).lower()
    without_tags = "<think" in str(without_content).lower()

    print(
        f"   Test 1 (disable_thinking=False): "
        f"reasoning_content={len(with_reason)} chars, "
        f"content={len(with_content)} chars, "
        f"<think> tags in content: {with_tags}",
        flush=True,
    )
    print(
        f"   Test 2 (disable_thinking=True):  "
        f"reasoning_content={len(without_reason)} chars, "
        f"content={len(without_content)} chars, "
        f"<think> tags in content: {without_tags}",
        flush=True,
    )

    if len(with_reason) > 0 and len(without_reason) == 0:
        print("   ✅ disable_thinking eliminated the reasoning_content field", flush=True)
    elif len(with_reason) > len(without_reason) * 2:
        print("   ✅ disable_thinking substantially reduced the reasoning trace", flush=True)
    elif with_tags and not without_tags:
        print("   ✅ strip_thinking_response removed <think> tags", flush=True)
    else:
        print(
            "   ⚠️ The two outputs are similar — flags had little visible effect on this model",
            flush=True,
        )

    return True


async def web_search_example(client: VeniceClient) -> bool:
    """Demonstrate web search parameters for current information.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print_section_header("Web Search Example", "🌐")

    # Find a web search capable model
    web_model = await find_web_search_model(client)

    if not web_model:
        print("⚠️ No web search models found, using standard model", flush=True)
        web_model = await client.models.resolve_chat()

    print(f"📍 Using model: {web_model}", flush=True)

    # Question requiring current information
    question = "Briefly, what are notable recent developments in artificial intelligence?"
    print(f"\n🔍 Question requiring current info: {question}", flush=True)

    # Create VeniceParameters with web search enabled
    venice_params = VeniceParameters(
        enable_web_search="on",  # Enable web search
        enable_web_citations=True,  # Request citations
        include_venice_system_prompt=True,
    )

    display_parameters(venice_params)

    try:
        response = await client.chat.completions.create(
            model=web_model,
            messages=[
                SystemMessage(
                    content="Provide current, accurate information and cite your sources."
                ),
                UserMessage(content=question),
            ],
            venice_parameters=venice_params,
            max_completion_tokens=384,
            temperature=0.3,
        )

        content = response.text or ""
        thinking_blocks, clean_response = extract_thinking_blocks(content)

        if thinking_blocks:
            print_subsection("Search Reasoning", "💭")
            for block in thinking_blocks:
                print(f"   {block[:200]}...", flush=True)

        print_subsection("Response with Web Search", "📝")
        display_body(response, clean_response)

        # Show citations if available
        citations = format_web_citations(response)
        if citations:
            print_subsection("Web Citations", "📚")
            for citation in citations:
                print(f"   {citation}", flush=True)

        # Check for REF citations in content
        content_str = str(content) if not isinstance(content, str) else content
        ref_citations = re.findall(r"\[REF\](\d+)\[/REF\]", content_str)
        if ref_citations:
            print_subsection("Inline Citations Found", "🔗")
            print(f"   Found {len(ref_citations)} inline citations: {ref_citations}", flush=True)

        _print_usage(response)
        return True

    except Exception as e:
        print(f"❌ Error in web search example: {e}", flush=True)
        return False


async def search_results_streaming_example(client: VeniceClient) -> bool:
    """Demonstrate search results streaming and document formatting.

    Runs the two test completions concurrently via ``client.gather``.

    Returns ``True`` on success, ``False`` if a request failed.
    """
    print_section_header("Search Results Streaming Example", "📊")

    # Find a web search capable model
    web_model = await find_web_search_model(client)

    if not web_model:
        print("⚠️ No web search models found, using standard model", flush=True)
        web_model = await client.models.resolve_chat()

    print(f"📍 Using model: {web_model}", flush=True)

    question = "Briefly, what is the current status of renewable energy adoption globally?"
    print(f"\n🔍 Question: {question}", flush=True)

    # Test 1 config: Include search results in stream
    venice_params_stream = VeniceParameters(
        enable_web_search="on",
        enable_web_citations=True,
        include_search_results_in_stream=True,  # Include in stream
        include_venice_system_prompt=True,
    )

    # Test 2 config: Return search results as documents (OpenAI tool call format)
    venice_params_docs = VeniceParameters(
        enable_web_search="on",
        enable_web_citations=True,
        return_search_results_as_documents=True,  # Return as tool calls
        include_venice_system_prompt=True,
    )

    print_subsection("Test 1: Search Results in Stream", "📡")
    display_parameters(venice_params_stream)
    print_subsection("Test 2: Search Results as Documents", "📄")
    display_parameters(venice_params_docs)
    print("\n⏳ Dispatching both completions concurrently...", flush=True)

    try:
        response_stream, response_docs = await client.gather(
            [
                client.chat.completions.create(
                    model=web_model,
                    messages=[UserMessage(content=question)],
                    venice_parameters=venice_params_stream,
                    max_completion_tokens=384,
                    temperature=0.3,
                ),
                client.chat.completions.create(
                    model=web_model,
                    messages=[UserMessage(content=question)],
                    venice_parameters=venice_params_docs,
                    max_completion_tokens=384,
                    temperature=0.3,
                ),
            ],
            max_concurrency=2,
            return_exceptions=False,
        )
    except Exception as e:
        print(f"❌ Error in search results streaming example: {e}", flush=True)
        return False

    # --- Test 1 output ---
    content_stream = response_stream.text or ""
    print_subsection("Test 1: Response with Streamed Search Results", "📝")
    display_body(response_stream, content_stream)
    _print_usage(response_stream)

    # --- Test 2 output ---
    content_docs = response_docs.text or ""
    print_subsection("Test 2: Response with Document Results", "📝")
    display_body(response_docs, content_docs)

    # Check for tool calls in response
    message = response_docs.choices[0].message
    if hasattr(message, "tool_calls") and message.tool_calls:
        print_subsection("Search Results as Tool Calls", "🔧")
        for i, tool_call in enumerate(message.tool_calls, 1):
            print(f"   Tool Call {i}: {tool_call}", flush=True)
    _print_usage(response_docs)

    return True


async def system_prompt_control_example(client: VeniceClient) -> bool:
    """Demonstrate include_venice_system_prompt parameter.

    Runs the with/without completions concurrently via ``client.gather``.

    Returns ``True`` on success, ``False`` if a request failed.
    """
    print_section_header("System Prompt Control Example", "🔧")

    # Get a suitable model
    model = await client.models.resolve_chat()
    print(f"📍 Using model: {model}", flush=True)

    question = "Explain quantum computing in two or three simple sentences."
    print(f"\n❓ Question: {question}", flush=True)

    # Test 1 config: With Venice system prompts
    venice_params_with = VeniceParameters(
        enable_web_search="off",
        include_venice_system_prompt=True,  # Include Venice prompts
    )

    # Test 2 config: Without Venice system prompts
    venice_params_without = VeniceParameters(
        enable_web_search="off",
        include_venice_system_prompt=False,  # Exclude Venice prompts
    )

    print_subsection("Test 1: With Venice System Prompts", "✅")
    display_parameters(venice_params_with)
    print_subsection("Test 2: Without Venice System Prompts", "❌")
    display_parameters(venice_params_without)
    print("\n⏳ Dispatching both completions concurrently...", flush=True)

    system_msg = SystemMessage(content="You are a helpful technical assistant.")

    try:
        response_with, response_without = await client.gather(
            [
                client.chat.completions.create(
                    model=model,
                    messages=[system_msg, UserMessage(content=question)],
                    venice_parameters=venice_params_with,
                    max_completion_tokens=256,
                    temperature=0.7,
                ),
                client.chat.completions.create(
                    model=model,
                    messages=[system_msg, UserMessage(content=question)],
                    venice_parameters=venice_params_without,
                    max_completion_tokens=256,
                    temperature=0.7,
                ),
            ],
            max_concurrency=2,
            return_exceptions=False,
        )
    except Exception as e:
        print(f"❌ Error in system prompt control example: {e}", flush=True)
        return False

    content_with = response_with.text or ""
    print_subsection("Response With Venice Prompts", "📝")
    print(content_with, flush=True)
    _print_usage(response_with)

    content_without = response_without.text or ""
    print_subsection("Response Without Venice Prompts", "📝")
    print(content_without, flush=True)
    _print_usage(response_without)

    # Compare response lengths
    print_subsection("Comparison", "📏")
    print(f"   With Venice prompts: {len(content_with)} characters", flush=True)
    print(f"   Without Venice prompts: {len(content_without)} characters", flush=True)
    print(f"   Difference: {abs(len(content_with) - len(content_without))} characters", flush=True)

    return True


async def comprehensive_example(client: VeniceClient) -> bool:
    """Demonstrate multiple VeniceParameters working together in a single call.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print_section_header("Comprehensive Example - All Parameters", "🎯")

    # Get available resources (prefer a web-search model for this example).
    characters = await find_available_characters(client)
    web_model = await find_web_search_model(client)
    model = web_model if web_model else await client.models.resolve_chat()

    print(f"📍 Using model: {model}", flush=True)

    # Choose character if available
    character_slug = characters[0].slug if characters else None
    if character_slug:
        print(f"📍 Using character: {character_slug} ({characters[0].name})", flush=True)

    # Complex question that benefits from multiple features
    question = (
        "I'm planning to start a renewable energy company. In a few sentences, "
        "what are the key technologies to focus on and what makes a successful "
        "clean energy startup?"
    )

    print(f"\n💼 Business Question:\n{question}", flush=True)

    # Comprehensive VeniceParameters configuration
    venice_params = VeniceParameters(
        character_slug=character_slug,  # Use character if available
        strip_thinking_response=False,  # Show reasoning process
        disable_thinking=False,  # Enable thinking for complex analysis
        enable_web_search="on",  # Get current market data
        enable_web_citations=True,  # Cite sources
        include_venice_system_prompt=True,  # Full capabilities
    )

    display_parameters(venice_params, "Comprehensive Configuration")

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                SystemMessage(
                    content=(
                        "You are an expert business consultant specializing in clean "
                        "energy and startups. Be concise."
                    )
                ),
                UserMessage(content=question),
            ],
            venice_parameters=venice_params,
            max_completion_tokens=512,
            temperature=0.4,
        )

        content = response.text or ""
        thinking_blocks, clean_response = extract_thinking_blocks(content)

        # Show thinking process if present
        if thinking_blocks:
            print_subsection("Strategic Analysis Process", "🧠")
            for i, block in enumerate(thinking_blocks, 1):
                lines = block.strip().split("\n")
                print(f"   Analysis Step {i}:", flush=True)
                for line in lines[:4]:  # Show first 4 lines
                    print(f"     {line}", flush=True)
                if len(lines) > 4:
                    print(f"     ... ({len(lines) - 4} more lines)", flush=True)
                print(flush=True)

        print_subsection("Comprehensive Business Analysis", "📊")
        print(clean_response, flush=True)

        # Show citations if present
        citations = format_web_citations(response)
        if citations:
            print_subsection("Market Research Sources", "📚")
            for citation in citations:
                print(f"   {citation}", flush=True)

        # Show response metadata
        if hasattr(response, "venice_parameters") and response.venice_parameters:
            params_response = response.venice_parameters
            print_subsection("Response Metadata", "📋")
            print(f"   Web search used: {params_response.enable_web_search}", flush=True)
            print(f"   Citations enabled: {params_response.enable_web_citations}", flush=True)
            if character_slug:
                print(f"   Character used: {params_response.character_slug}", flush=True)
            print(f"   Venice prompts: {params_response.include_venice_system_prompt}", flush=True)

        _print_usage(response)
        return True

    except Exception as e:
        print(f"❌ Error in comprehensive example: {e}", flush=True)
        return False


# =============================================================================
# Main Function
# =============================================================================


async def main() -> int:
    """Run all Venice Parameters examples.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    print("🚀 Venice AI Parameters Showcase", flush=True)
    print("=" * 70, flush=True)
    print("Demonstrating Venice-specific parameters with real API data\n", flush=True)

    async with VeniceClient() as client:
        results: list[tuple[str, bool]] = [
            ("character_example", await character_example(client)),
            ("thinking_control_example", await thinking_control_example(client)),
            ("web_search_example", await web_search_example(client)),
            ("search_results_streaming_example", await search_results_streaming_example(client)),
            ("system_prompt_control_example", await system_prompt_control_example(client)),
            ("comprehensive_example", await comprehensive_example(client)),
        ]

    failed = [name for name, ok in results if not ok]

    if failed:
        print_section_header(f"{len(failed)} of {len(results)} demos failed", "⚠️")
        print(f"   Failed: {', '.join(failed)}", flush=True)
    else:
        print_section_header("Examples Completed Successfully! ✨", "🎉")

    print("\n💡 Key Venice Parameters demonstrated:", flush=True)
    print("   • character_slug: Leverage pre-built AI personalities", flush=True)
    print("   • strip_thinking_response: Control reasoning visibility", flush=True)
    print("   • disable_thinking: Toggle reasoning capabilities", flush=True)
    print("   • enable_web_search: Access current information", flush=True)
    print("   • enable_web_citations: Get source attribution", flush=True)
    print("   • include_search_results_in_stream: Control search display", flush=True)
    print("   • return_search_results_as_documents: OpenAI-compatible tools", flush=True)
    print("   • include_venice_system_prompt: Fine-tune system behavior", flush=True)
    print("\n🔗 Combine these parameters for powerful, customized AI interactions!", flush=True)

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
