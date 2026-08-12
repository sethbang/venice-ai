#!/usr/bin/env python3
"""
Venice AI SDK - Augment: Web Scrape
===================================

Demonstrates ``client.augment.scrape(url=...)`` — Venice's web-scraping
endpoint that returns page content as markdown. First tries Cloudflare's
native markdown extraction, then falls back to a headless browser. Some
sites that block automated access (X/Twitter, Reddit) are rejected
immediately with a ``400`` error.

**Pricing:** $0.01 per request.

**Note:** The Augment API is marked experimental in the Venice docs; request
and response shapes may change without notice.
"""

import asyncio
import sys

from venice_ai import VeniceClient
from venice_ai.exceptions import APIError, VeniceError

# ---------------------------------------------------------------------------
# 1. Basic scrape
# ---------------------------------------------------------------------------


async def basic_scrape() -> bool:
    """Scrape a public page and print the markdown output."""
    print("🌐 Basic Web Scrape")
    print("-" * 30)

    async with VeniceClient() as client:
        try:
            result = await client.augment.scrape(url="https://example.com")
        except (VeniceError, APIError) as e:
            print(f"❌ Scrape failed: {e}")
            return False

        print(f"📍 URL: {result.url}")
        print(f"📄 Format: {result.format}")
        print(f"📏 Content length: {len(result.content)} chars")
        print("\n📝 First 200 chars of content:")
        print("-" * 30)
        print(result.content[:200])
        return True


# ---------------------------------------------------------------------------
# 2. Scrape a docs page (longer, real content)
# ---------------------------------------------------------------------------


async def scrape_documentation_page() -> bool:
    """Scrape a docs-style page to see meaningful markdown."""
    print("\n📚 Scrape a Docs Page")
    print("-" * 30)

    docs_url = "https://www.python.org/"

    async with VeniceClient() as client:
        try:
            result = await client.augment.scrape(url=docs_url)
        except (VeniceError, APIError) as e:
            print(f"❌ Scrape failed: {e}")
            return False

        print(f"📍 Scraped: {result.url}")
        print(f"📏 Returned {len(result.content)} chars of markdown")

        # Count headings (markdown sections)
        headings = [line for line in result.content.splitlines() if line.lstrip().startswith("#")]
        print(f"📑 Headings detected: {len(headings)}")
        for heading in headings[:5]:
            print(f"   • {heading.strip()}")
        return True


# ---------------------------------------------------------------------------
# 3. Error handling for blocked sites
# ---------------------------------------------------------------------------


async def error_handling() -> bool:
    """Demonstrate how the scraper surfaces errors from blocked sites.

    This is an *intentional* error demonstration: blocked domains are
    expected to raise, and catching them here is the point. The function
    always returns ``True`` because "ran the demo" is success — a handled,
    expected error must not flip the overall run to a false failure.
    """
    print("\n⚠️  Error Handling")
    print("-" * 30)

    # Venice rejects some automated-access-blocked domains with 400. The
    # SDK surfaces those as APIError — catch and report.
    blocked_examples = [
        "https://x.com/",
        "https://reddit.com/",
    ]

    async with VeniceClient() as client:
        for url in blocked_examples:
            try:
                result = await client.augment.scrape(url=url)
                print(f"✅ {url} → {len(result.content)} chars")
            except (VeniceError, APIError) as e:
                print(f"🚫 {url} → {type(e).__name__}: {e}")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    """Run all augment-scrape examples and report an honest tally."""
    print("🚀 Venice AI Augment — Web Scrape Examples")
    print("=" * 50)

    results = {
        "Basic scrape": await basic_scrape(),
        "Docs page scrape": await scrape_documentation_page(),
        "Error handling demo": await error_handling(),
    }

    print("\n" + "=" * 50)
    if all(results.values()):
        print("✨ Scrape examples completed!")
        print("\n💡 Key concepts demonstrated:")
        print("   - Basic URL → markdown scraping")
        print("   - Real-world content (docs page with headings)")
        print("   - Graceful handling of blocked-domain errors")
        return 0

    failed = [name for name, ok in results.items() if not ok]
    print(f"❌ {len(failed)} of {len(results)} examples failed: {', '.join(failed)}")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
