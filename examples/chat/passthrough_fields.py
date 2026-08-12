#!/usr/bin/env python3
"""
Venice AI SDK - Chat Completion Passthrough Fields
==================================================

Demonstrates the OpenAI-compatible passthrough fields plus Venice's
``prompt_cache_retention`` control on ``client.chat.completions.create()``:

- ``store`` — OpenAI-compat flag for server-side storage.
- ``text`` — OpenAI-compat text-config object (e.g. ``{"verbosity": "low"}``).
- ``include`` — OpenAI-compat include-specifier for response enrichment.
- ``metadata`` — free-form dict attached to the request (observability).
- ``prompt_cache_retention`` — ``"default"`` (standard TTL), ``"extended"``,
  or ``"24h"``. Higher tiers keep the prompt cached longer, improving hit
  rates for long-running agents at a small storage premium.

All five are forwarded verbatim; the server / upstream model interprets
them.
"""

import asyncio
import sys

from venice_ai import VeniceClient
from venice_ai.types.api import UserMessage


async def demo_passthrough_fields() -> bool:
    """Send a request with every passthrough field set.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print("🧾 Chat Completion — Passthrough Fields")
    print("-" * 30)

    ok = True
    async with VeniceClient() as client:
        try:
            model = await client.models.resolve_chat()
            print(f"📍 Using model: {model}")

            response = await client.chat.completions.create(
                model=model,
                messages=[UserMessage(content="Explain the solar system in 3 bullet points.")],
                max_completion_tokens=200,
                temperature=0.3,
                # --- passthrough fields ---
                store=False,
                text={"verbosity": "low"},
                include=[],
                metadata={
                    "workflow": "sdk_example",
                    "trace_id": "example-passthrough-001",
                },
                prompt_cache_retention="extended",
            )

            msg = response.choices[0].message
            print("\n💬 Response:")
            print(msg.content or msg.reasoning_content or "(no content)")

            print("\n📊 Passthrough fields forwarded:")
            print("   store                 = False")
            print("   text                  = {'verbosity': 'low'}")
            print("   include               = []")
            print("   metadata              = workflow/trace_id (for your own logs)")
            print("   prompt_cache_retention = 'extended' (longer cache TTL)")
        except Exception as e:
            print(f"❌ Error in passthrough fields demo: {e}")
            ok = False

    return ok


async def demo_cache_retention_tiers() -> bool:
    """Issue the same prompt under each retention tier to compare.

    Returns ``True`` only if every tier succeeded, ``False`` otherwise, so a
    real per-tier failure surfaces instead of being masked.
    """
    print("\n🧊 prompt_cache_retention Tiers")
    print("-" * 30)

    ok = True
    async with VeniceClient() as client:
        try:
            model = await client.models.resolve_chat()
        except Exception as e:
            print(f"❌ Error resolving model for retention tiers: {e}")
            return False

        for tier in ("default", "extended", "24h"):
            print(f"\n🕰️  Tier: {tier}")
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[UserMessage(content="Reply with the word OK.")],
                    max_completion_tokens=16,
                    temperature=0.0,
                    prompt_cache_retention=tier,  # type: ignore[arg-type]
                    prompt_cache_key=f"sdk-example-{tier}",
                )
                msg = response.choices[0].message
                text = msg.content if isinstance(msg.content, str) else None
                print(f"   {(text or msg.reasoning_content or '').strip()}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                ok = False

    return ok


async def main() -> int:
    """Run all passthrough-field demos.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    print("🚀 Venice AI Chat — Passthrough Fields Example")
    print("=" * 50)

    results: list[tuple[str, bool]] = [
        ("demo_passthrough_fields", await demo_passthrough_fields()),
        ("demo_cache_retention_tiers", await demo_cache_retention_tiers()),
    ]

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("\n✨ Done.")

    print("\n💡 Key concepts demonstrated:")
    print("   - store / text / include / metadata (OpenAI-compat passthroughs)")
    print("   - prompt_cache_retention tiers and prompt_cache_key pairing")

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
