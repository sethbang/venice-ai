#!/usr/bin/env python3
"""
Venice AI SDK - Augment: Web Search
===================================

Demonstrates ``client.augment.search(...)`` — Venice's structured web-search
endpoint. Returns a list of results (title, URL, content snippet, date)
from either Brave (default, Zero Data Retention) or Google (anonymised
proxy).

**Pricing:** $0.01 per request.

**Note:** The Augment API is marked experimental in the Venice docs; request
and response shapes may change without notice.
"""

import asyncio
import sys

from venice_ai import VeniceClient
from venice_ai.exceptions import APIError, VeniceError
from venice_ai.types.api import SystemMessage, UserMessage

# ---------------------------------------------------------------------------
# 1. Basic search (Brave default)
# ---------------------------------------------------------------------------


async def basic_search() -> bool:
    """Run a simple search and print the first few results.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print("🔎 Basic Search (Brave, default)")
    print("-" * 30)

    async with VeniceClient() as client:
        try:
            response = await client.augment.search(query="venice.ai API features")
        except (VeniceError, APIError) as e:
            print(f"❌ Search failed: {e}")
            return False

        print(f"📍 Query: {response.query}")
        print(f"🔢 Results: {len(response.results)}")

        for idx, result in enumerate(response.results[:5], start=1):
            print(f"\n{idx}. {result.title}")
            print(f"   🔗 {result.url}")
            if result.date:
                print(f"   📅 {result.date}")
            print(f"   📝 {result.content[:150].strip()}...")

    return True


# ---------------------------------------------------------------------------
# 2. Compare Brave vs Google providers
# ---------------------------------------------------------------------------


async def compare_providers() -> bool:
    """Run the same query through both providers and compare.

    Returns ``True`` only if both providers succeeded, ``False`` otherwise.
    """
    print("\n🆚 Compare Brave vs Google")
    print("-" * 30)

    query = "EIP-4361 Sign-In-With-Ethereum"

    ok = True
    async with VeniceClient() as client:
        for provider in ("brave", "google"):
            try:
                response = await client.augment.search(
                    query=query,
                    limit=3,
                    search_provider=provider,  # type: ignore[arg-type]
                )
                print(f"\n🔎 {provider.title()} ({len(response.results)} results)")
                for result in response.results:
                    print(f"   • {result.title[:80]}")
                    print(f"     {result.url}")
            except (VeniceError, APIError) as e:
                print(f"❌ {provider} failed: {e}")
                ok = False

    return ok


# ---------------------------------------------------------------------------
# 3. Use search results to ground a chat completion (RAG-lite)
# ---------------------------------------------------------------------------


async def search_grounded_chat() -> bool:
    """Combine augment.search() with chat.completions.create() for a
    lightweight retrieval-augmented-generation flow.

    Returns ``True`` on success, ``False`` if any API call failed.
    """
    print("\n🧠 Search-Grounded Chat Completion")
    print("-" * 30)

    question = "What are the latest best practices for Python type hinting?"

    async with VeniceClient() as client:
        try:
            # Step 1 — search for recent context.
            search = await client.augment.search(query=question, limit=4)

            # Step 2 — build a grounding prompt with the snippets.
            snippets = "\n".join(f"- {r.title}: {r.content[:200].strip()}" for r in search.results)

            # Step 3 — ask a chat model to synthesise an answer.
            chat_model = await client.models.resolve_chat()
            print(f"📍 Chat model: {chat_model}")

            response = await client.chat.completions.create(
                model=chat_model,
                messages=[
                    SystemMessage(
                        content=(
                            "You are a concise assistant. Answer the user's "
                            "question using the web snippets below. Cite sources "
                            "inline as [1], [2], … matching the order you see them."
                        ),
                    ),
                    UserMessage(
                        content=f"Question: {question}\n\nSnippets:\n{snippets}",
                    ),
                ],
                temperature=0.2,
                max_completion_tokens=300,
            )

            print("\n💬 Grounded answer:")
            print(response.text)

        except (VeniceError, APIError) as e:
            print(f"❌ Grounded chat failed: {e}")
            return False

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    """Run all augment-search examples.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    print("🚀 Venice AI Augment — Web Search Examples")
    print("=" * 50)

    results: list[tuple[str, bool]] = [
        ("basic_search", await basic_search()),
        ("compare_providers", await compare_providers()),
        ("search_grounded_chat", await search_grounded_chat()),
    ]

    failed = [name for name, ok in results if not ok]

    print("\n" + "=" * 50)
    if failed:
        print(f"⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("✨ Search examples completed!")

    print("\n💡 Key concepts demonstrated:")
    print("   - Structured search results (title, url, content, date)")
    print("   - Provider selection (brave default, google proxied)")
    print("   - Pairing search with chat for grounded answers")

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
