#!/usr/bin/env python3
"""
Venice AI SDK - Responses API Example (Alpha)
=============================================

Venice exposes an OpenAI-compatible **Responses API** at ``POST /responses``,
wrapped by ``client.responses.create(...)``. Unlike ``/chat/completions``, the
endpoint is *stateless* (no conversation state is persisted between calls) and
returns a typed ``output`` array of blocks — ``reasoning``, ``message``,
``function_call``, ``web_search_call`` — rather than ``choices``.

This example demonstrates:

- A basic ``client.responses.create`` call with a plain-string ``input``
- Reading the typed ``output`` blocks and extracting the assistant text
- A structured (multi-message) ``input`` with ``max_output_tokens`` capping
- Inspecting the ``usage`` token block on the result

The endpoint is tagged **Alpha** by Venice; request/response shapes may change
without notice, and some accounts may not be entitled. If the account lacks
access, the demos degrade to a clear skip rather than a hard failure.

Prerequisites:
- Install: pip install venice-py
- Set API key: export VENICE_API_KEY="your-api-key"
"""

import asyncio
import sys

from venice_ai import VeniceClient
from venice_ai.exceptions import NotFoundError, PermissionDeniedError
from venice_ai.types.api.responses import (
    ResponsesMessageOutput,
    ResponsesReasoningOutput,
    ResponsesResponse,
)


def extract_text(response: ResponsesResponse) -> str:
    """Pull the assistant text out of a Responses ``output`` array.

    The Responses API returns a list of typed blocks instead of a single
    ``message``. Assistant text lives inside ``message`` blocks, whose
    ``content`` is a list of ``output_text`` items. We concatenate every
    ``output_text`` we find across all ``message`` blocks.
    """
    parts: list[str] = []
    for block in response.output:
        if isinstance(block, ResponsesMessageOutput):
            for item in block.content:
                # Each content item is an ``output_text`` block with a ``.text``.
                parts.append(item.text)
    return "\n".join(parts).strip()


def summarize_output_blocks(response: ResponsesResponse) -> None:
    """Print a short breakdown of the typed blocks in ``response.output``."""
    print(f"   📦 {len(response.output)} output block(s):")
    for i, block in enumerate(response.output):
        # ``.type`` is present on every block variant (including the
        # forward-compat fallback for unmodeled types).
        label = block.type
        if isinstance(block, ResponsesReasoningOutput):
            summary = block.summary or []
            label += f" (summary lines: {len(summary)})"
        elif isinstance(block, ResponsesMessageOutput):
            label += f" (role={block.role}, status={block.status})"
        print(f"      {i + 1}. {label}")


async def basic_responses_create() -> bool:
    """Demonstrate a basic ``client.responses.create`` call (string input).

    Returns ``True`` on success, ``False`` on a genuine failure, and skips
    (returning ``True``) if the account is not entitled to the Alpha endpoint.
    """
    print("💬 Basic Responses API Call")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            # Resolve a chat model dynamically — never hardcode a model id.
            # E2EE-capable models are not supported by /responses, so we keep
            # the default (require_private=False, exclude_beta=True).
            model = await client.models.resolve_chat()
            print(f"🤖 Using model: {model}")

            # The simplest form: ``input`` is a plain string prompt.
            response = await client.responses.create(
                model=model,
                input="Summarize the Treaty of Versailles in two sentences.",
                max_output_tokens=200,
            )

            print(f"\n🆔 Response id: {response.id}")
            print(f"   Object:  {response.object}")
            print(f"   Status:  {response.status}")
            print(f"   Model:   {response.model}")

            summarize_output_blocks(response)

            text = extract_text(response)
            if text:
                print(f"\n🗣️ Assistant:\n{text}")
            else:
                # A completed response should carry assistant text; treat an
                # empty extraction on a completed call as a soft warning.
                print("\n⚠️ No assistant text found in output blocks.")

            if response.usage:
                u = response.usage
                print("\n📊 Token Usage:")
                print(f"   Input tokens:  {u.input_tokens}")
                print(f"   Output tokens: {u.output_tokens}")
                print(f"   Total tokens:  {u.total_tokens}")

            # A completed status with text is the success signal we care about.
            return response.status == "completed" and bool(text)

        except (PermissionDeniedError, NotFoundError) as e:
            # The Responses API is Alpha; not every account / model is
            # entitled. Degrade to a clear skip instead of a hard failure.
            print(f"\n⏭️ Skipping — Responses API unavailable on this account: {e}")
            return True
        except Exception as e:
            print(f"\n❌ Error in basic responses call: {e}")
            return False


async def structured_input_responses() -> bool:
    """Demonstrate a structured (multi-message) ``input`` payload.

    ``input`` also accepts a list of structured input items (OpenAI Responses
    shape) instead of a plain string. Here we pass a system-style instruction
    followed by a user turn.

    Returns ``True`` on success/skip, ``False`` on a genuine failure.
    """
    print("\n🧱 Structured Input (message list)")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            model = await client.models.resolve_chat()
            print(f"🤖 Using model: {model}")

            # Each item mirrors the OpenAI Responses input-message shape.
            structured_input = [
                {
                    "role": "system",
                    "content": "You are a terse assistant. Answer in one short sentence.",
                },
                {
                    "role": "user",
                    "content": "What is the capital of Japan?",
                },
            ]

            response = await client.responses.create(
                model=model,
                input=structured_input,
                max_output_tokens=64,
                temperature=0.2,
            )

            print(f"\n🆔 Response id: {response.id}  (status={response.status})")
            summarize_output_blocks(response)

            text = extract_text(response)
            if text:
                print(f"\n🗣️ Assistant: {text}")

            return response.status == "completed" and bool(text)

        except (PermissionDeniedError, NotFoundError) as e:
            print(f"\n⏭️ Skipping — Responses API unavailable on this account: {e}")
            return True
        except Exception as e:
            print(f"\n❌ Error in structured input responses call: {e}")
            return False


async def main() -> int:
    """Run all Responses API demos.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    print("🚀 Venice AI Responses API Example (Alpha)")
    print("=" * 50)

    results: list[tuple[str, bool]] = [
        ("basic_responses_create", await basic_responses_create()),
        ("structured_input_responses", await structured_input_responses()),
    ]

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("\n✨ Responses API example completed!")

    print("\n💡 Key concepts demonstrated:")
    print("   - client.responses.create with a plain-string input")
    print("   - client.responses.create with a structured message list")
    print("   - Reading typed output blocks (reasoning / message / etc.)")
    print("   - Inspecting the usage token block")

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print(
            "Check that your API key is valid and your account has access to the "
            "Responses API (Alpha).",
            file=sys.stderr,
        )
        sys.exit(1)
