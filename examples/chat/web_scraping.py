#!/usr/bin/env python3
"""
Venice AI SDK - Web Scraping with Chat Completions
===================================================

This example demonstrates the ``enable_web_scraping`` Venice parameter, which
automatically detects URLs in user messages, fetches their content, and augments
the model's context so it can answer questions about those pages.

**Web Scraping vs Web Search**

+-----------------------+--------------------------------------------------+
| Feature               | Behaviour                                        |
+=======================+==================================================+
| ``enable_web_scraping`` | Scrapes the **specific URLs** you include in     |
|                       | your message. The model sees the page content.   |
+-----------------------+--------------------------------------------------+
| ``enable_web_search`` | Searches the **open web** for relevant results   |
|                       | related to your prompt (no URL required).        |
+-----------------------+--------------------------------------------------+

Sections:
    1. Basic URL scraping
    2. Multiple URLs (compare / contrast)
    3. Extracting specific information from a page
    4. Web scraping vs web search side-by-side
    5. Streaming with web scraping
"""

import asyncio
import sys
from collections.abc import AsyncIterable
from typing import Any

from venice_ai import VeniceClient
from venice_ai.types.api import UserMessage
from venice_ai.types.api.requests import VeniceParameters
from venice_ai.types.chat import ChatCompletionChunk


def _print_usage(response: Any, *, indent: str = "") -> None:
    """Print token usage from a response, or note when unavailable."""
    usage = getattr(response, "usage", None)
    if usage is None:
        print(f"{indent}📊 Token usage: not provided by the API")
        return
    print(
        f"{indent}📊 Token Usage: Input={usage.prompt_tokens}, "
        f"Output={usage.completion_tokens}, Total={usage.total_tokens}"
    )


# =============================================================================
# 1. Basic URL Scraping
# =============================================================================


async def basic_url_scraping():
    """Send a message containing a URL and ask the model to summarise it."""
    print("🌐 Basic URL Scraping")
    print("-" * 40)

    async with VeniceClient() as client:
        chat_model = await client.models.resolve_chat()
        print(f"📍 Using model: {chat_model}")

        try:
            response = await client.chat.completions.create(
                model=chat_model,
                messages=[
                    UserMessage(
                        content=(
                            "Summarize the content of this page: "
                            "https://docs.venice.ai/api-reference/endpoint/chat/completions"
                        ),
                    )
                ],
                # Only enable_web_scraping differs from VeniceParameters defaults
                venice_parameters=VeniceParameters(
                    enable_web_scraping=True,
                ),
                max_completion_tokens=400,
                temperature=0.3,
            )

            content = response.text or ""
            print(f"\n📄 Summary:\n{content}")

            print()
            _print_usage(response)
            return True

        except Exception as e:
            print(f"❌ Error in basic scraping example: {e}")
            return False


# =============================================================================
# 2. Multiple URLs — Compare / Contrast
# =============================================================================


async def multiple_urls():
    """Scrape multiple URLs in a single message and compare them."""
    print("\n🔗 Multiple URL Scraping")
    print("-" * 40)

    async with VeniceClient() as client:
        chat_model = await client.models.resolve_chat()
        print(f"📍 Using model: {chat_model}")

        try:
            response = await client.chat.completions.create(
                model=chat_model,
                messages=[
                    UserMessage(
                        content=(
                            "Compare the topics covered by these two pages and list "
                            "the key differences:\n"
                            "1. https://en.wikipedia.org/wiki/Python_%28programming_language%29\n"
                            "2. https://en.wikipedia.org/wiki/Rust_%28programming_language%29"
                        ),
                    )
                ],
                venice_parameters=VeniceParameters(
                    enable_web_scraping=True,
                ),
                max_completion_tokens=600,
                temperature=0.4,
            )

            content = response.text or ""
            print(f"\n🔍 Comparison:\n{content}")

            print()
            _print_usage(response)
            return True

        except Exception as e:
            print(f"❌ Error in multiple URLs example: {e}")
            return False


# =============================================================================
# 3. Extracting Specific Information
# =============================================================================


async def extract_specific_info():
    """Scrape a URL and ask for specific data extraction."""
    print("\n🎯 Extracting Specific Information")
    print("-" * 40)

    async with VeniceClient() as client:
        chat_model = await client.models.resolve_chat()
        print(f"📍 Using model: {chat_model}")

        try:
            response = await client.chat.completions.create(
                model=chat_model,
                messages=[
                    UserMessage(
                        content=(
                            "From this page, extract and list: "
                            "(a) all available API endpoints, "
                            "(b) required authentication method, and "
                            "(c) any rate limit information mentioned.\n\n"
                            "https://docs.venice.ai/api-reference/endpoint/chat/completions"
                        ),
                    )
                ],
                venice_parameters=VeniceParameters(
                    enable_web_scraping=True,
                ),
                max_completion_tokens=500,
                temperature=0.2,
            )

            content = response.text or ""
            print(f"\n📋 Extracted Data:\n{content}")

            print()
            _print_usage(response)
            return True

        except Exception as e:
            print(f"❌ Error in extraction example: {e}")
            return False


# =============================================================================
# 4. Web Scraping vs Web Search — Side-by-Side
# =============================================================================


async def scraping_vs_search():
    """Show the difference between enable_web_scraping and enable_web_search."""
    print("\n⚖️  Web Scraping vs Web Search")
    print("-" * 40)
    print("   🌐 enable_web_scraping=True  → fetches the specific URL you provide")
    print("   🔍 enable_web_search='on'    → searches the web for relevant info")

    async with VeniceClient() as client:
        chat_model = await client.models.resolve_chat()
        print(f"📍 Using model: {chat_model}")

        question = (
            "What models does Venice AI support? https://docs.venice.ai/api-reference/api-spec"
        )

        # --- Web Scraping (fetches the URL directly) ---
        try:
            print("\n🌐 With enable_web_scraping=True (scrapes the URL):")
            response_scrape = await client.chat.completions.create(
                model=chat_model,
                messages=[UserMessage(content=question)],
                venice_parameters=VeniceParameters(
                    enable_web_scraping=True,
                ),
                max_completion_tokens=300,
                temperature=0.3,
            )

            content_scrape = response_scrape.text or ""
            print(f"   📄 {content_scrape[:300]}{'...' if len(content_scrape) > 300 else ''}")

            _print_usage(response_scrape, indent="   ")
            ok_scrape = True

        except Exception as e:
            print(f"   ❌ Error with web scraping: {e}")
            ok_scrape = False

        # --- Web Search (searches the internet) ---
        try:
            print("\n🔍 With enable_web_search='on' (searches the web):")
            response_search = await client.chat.completions.create(
                model=chat_model,
                messages=[
                    UserMessage(
                        content="What models does Venice AI support?",
                    )
                ],
                venice_parameters=VeniceParameters(
                    enable_web_search="on",
                ),
                max_completion_tokens=300,
                temperature=0.3,
            )

            content_search = response_search.text or ""
            print(f"   📄 {content_search[:300]}{'...' if len(content_search) > 300 else ''}")

            _print_usage(response_search, indent="   ")
            ok_search = True

        except Exception as e:
            print(f"   ❌ Error with web search: {e}")
            ok_search = False

        print("\n💡 Key takeaway:")
        print("   • Web scraping reads the exact page you link to")
        print("   • Web search finds relevant pages on its own")

        return ok_scrape and ok_search


# =============================================================================
# 5. Streaming with Web Scraping
# =============================================================================


async def streaming_with_scraping():
    """Demonstrate that web scraping works with streaming responses."""
    print("\n🌊 Streaming with Web Scraping")
    print("-" * 40)

    async with VeniceClient() as client:
        chat_model = await client.models.resolve_chat()
        print(f"📍 Using model: {chat_model}")

        try:
            print("\n🤖 Assistant (streaming): ", end="", flush=True)

            stream: AsyncIterable[ChatCompletionChunk] = await client.chat.completions.create(
                model=chat_model,
                messages=[
                    UserMessage(
                        content=(
                            "Give me a brief overview of this page in 3 bullet points: "
                            "https://en.wikipedia.org/wiki/Large_language_model"
                        ),
                    )
                ],
                venice_parameters=VeniceParameters(
                    enable_web_scraping=True,
                ),
                stream=True,
                max_completion_tokens=300,
                temperature=0.3,
            )

            content = ""
            chunk_count = 0
            async for chunk in stream:
                chunk_count += 1
                token = chunk.text
                if token:
                    content += token
                    print(token, end="", flush=True)

            print()  # newline after streaming
            print(f"\n📊 Streaming Stats: {chunk_count} chunks, {len(content)} chars of content")
            return True

        except Exception as e:
            print(f"\n❌ Error in streaming scraping example: {e}")
            return False


# =============================================================================
# Main
# =============================================================================


async def main() -> int:
    """Run all web scraping examples, returning a nonzero exit code on any failure."""
    print("🚀 Venice AI Web Scraping Examples")
    print("=" * 50)

    # Each section catches its own API errors and reports success/failure, so the
    # whole suite keeps running. The aggregated tally below drives the exit code.
    results = [
        await basic_url_scraping(),
        await multiple_urls(),
        await extract_specific_info(),
        await scraping_vs_search(),
        await streaming_with_scraping(),
    ]

    completed = sum(results)
    total = len(results)
    print(f"\n✨ Web scraping examples: {completed}/{total} completed")

    if all(results):
        print("\n💡 Key concepts demonstrated:")
        print("   - Basic URL scraping with enable_web_scraping=True")
        print("   - Multi-URL scraping for comparison tasks")
        print("   - Targeted data extraction from scraped pages")
        print("   - Difference between web scraping and web search")
        print("   - Streaming responses with web scraping enabled")
    else:
        print("\n⚠️  Some sections failed — see the ❌ messages above.", file=sys.stderr)

    return 0 if all(results) else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
