#!/usr/bin/env python3
"""
Venice AI SDK - Augment: Text Parser
====================================

Demonstrates ``client.augment.parse_text(file=..., response_format=...)`` —
Venice's multipart document-parsing endpoint. Accepts PDF, DOCX, XLSX, and
plain text files up to 25 MB.

Two return shapes:

- ``response_format="json"`` (default) returns an
  :class:`AugmentTextParserResponse` with ``.text`` and ``.tokens``.
- ``response_format="text"`` returns a plain ``str`` of the extracted text.

**Privacy:** Parsing runs in-memory on Venice's infrastructure with zero
data retention — documents are processed then immediately discarded.

**Pricing:** $0.01 per request.

**Note:** The Augment API is marked experimental in the Venice docs; request
and response shapes may change without notice.
"""

import asyncio
import sys
from pathlib import Path

from venice_ai import VeniceClient
from venice_ai.exceptions import APIError, VeniceError

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Parse a plain text file (JSON response)
# ---------------------------------------------------------------------------


async def parse_plain_text() -> None:
    """Upload a small .txt file and get back text + token count."""
    print("📄 Parse Plain Text → JSON")
    print("-" * 30)

    # Create a small sample file
    sample = RESULTS_DIR / "augment_sample.txt"
    sample.write_text(
        "Venice AI is a privacy-first inference platform.\n"
        "It exposes a Python SDK at venice-py on PyPI.\n"
        "The Augment API lets you scrape, search, and parse documents.\n"
    )
    print(f"📎 Uploading: {sample}")

    async with VeniceClient() as client:
        result = await client.augment.parse_text(file=str(sample))

        print(f"✅ Tokens: {result.tokens}")
        print(f"📝 Extracted text ({len(result.text)} chars):")
        print("-" * 30)
        print(result.text)


# ---------------------------------------------------------------------------
# 2. Parse via raw bytes (no file path)
# ---------------------------------------------------------------------------


async def parse_raw_bytes() -> None:
    """Upload raw bytes directly — no temp file needed."""
    print("\n💾 Parse Raw Bytes")
    print("-" * 30)

    content = b"The quick brown fox jumps over the lazy dog.\n" * 10

    async with VeniceClient() as client:
        result = await client.augment.parse_text(
            file=content,
            content_type="text/plain",
            filename="fox.txt",
        )

        print(f"✅ Tokens: {result.tokens}")
        print(f"📝 Extracted text (first 80 chars): {result.text[:80]}")


# ---------------------------------------------------------------------------
# 3. Plain text response format
# ---------------------------------------------------------------------------


async def parse_with_text_response() -> None:
    """Use ``response_format='text'`` to get a plain ``str`` back.

    Unlike the default ``response_format='json'`` (which returns an
    ``AugmentTextParserResponse`` with ``.text``/``.tokens``), the ``"text"``
    format returns the extracted text directly as a ``str``.
    """
    print("\n🧾 Parse with response_format='text'")
    print("-" * 30)

    content = (
        b"Venice AI Augment text-parser demo.\n"
        b"This document is uploaded as raw bytes and parsed back as plain text.\n"
    )

    async with VeniceClient() as client:
        text = await client.augment.parse_text(
            file=content,
            response_format="text",
            content_type="text/plain",
            filename="demo.txt",
        )

        # response_format="text" returns a plain str (no .text / .tokens).
        print(f"✅ Got plain str ({len(text)} chars):")
        print("-" * 30)
        print(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    """Run all augment text-parser examples."""
    print("🚀 Venice AI Augment — Text Parser Examples")
    print("=" * 50)

    sub_examples = [
        ("parse_plain_text", parse_plain_text),
        ("parse_raw_bytes", parse_raw_bytes),
        ("parse_with_text_response", parse_with_text_response),
    ]

    results: list[tuple[str, bool]] = []
    for name, fn in sub_examples:
        try:
            await fn()
            results.append((name, True))
        except (VeniceError, APIError) as e:
            print(f"❌ {name} failed: {e}")
            results.append((name, False))

    print("\n" + "=" * 50)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"✨ {passed}/{total} text-parser sub-examples completed")
    for name, ok in results:
        status = "✅" if ok else "❌"
        print(f"   {status} {name}")
    print("\n💡 Key concepts demonstrated:")
    print("   - File-path upload with JSON response")
    print("   - Raw-bytes upload (no temp file)")
    print("   - response_format='text' for plain-string return")
    print("\n📁 Temp files in examples/results/:")
    print("   - augment_sample.txt (small plain-text sample)")

    return 0 if passed == total else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
