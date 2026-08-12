#!/usr/bin/env python3
"""
Venice AI SDK - Model Feature Suffixes
======================================

This example demonstrates how to use model feature suffixes — parameters appended
directly to model IDs using a colon separator. The Venice API parses these suffixes
and applies them as if they were set in the request body.

Suffix Format:
    model_id:param1=value1&param2=value2

This is useful for:
    • OpenAI-compatible clients that don't support extra request body parameters
    • Simple integrations where you can only configure the model name
    • Quick testing without modifying request structure

The SDK also provides a ``build_model_id()`` helper to construct suffixed model
strings programmatically.

Available Suffix Parameters:
    • enable_web_search       — "on", "off", or "auto"
    • strip_thinking_response — "true" or "false"
    • disable_thinking        — "true" or "false"
    • enable_web_scraping     — "true" or "false"
    • include_venice_system_prompt — "true" or "false"

Requirements:
    - Venice AI API key (set as VENICE_API_KEY environment variable)
    - Python 3.13+
    - venice-ai SDK
"""

import asyncio
import sys

from venice_ai import VeniceClient, build_model_id
from venice_ai.types.api import UserMessage
from venice_ai.types.api.requests import VeniceParameters

# =============================================================================
# Helper Functions
# =============================================================================


def print_section_header(title: str, emoji: str = "📋") -> None:
    """Print a formatted section header."""
    print(f"\n{emoji} {title}")
    print("=" * 70)


def print_subsection(title: str, emoji: str = "📍") -> None:
    """Print a formatted subsection header."""
    print(f"\n{emoji} {title}")
    print("-" * 50)


def print_response(response, label: str = "Response") -> None:
    """Print a chat completion response with usage stats."""
    content = response.text or ""
    print(f"\n📝 {label}:")
    print(content[:500])
    if len(content) > 500:
        print(f"   ... ({len(content) - 500} more characters)")

    usage = response.usage
    print(
        f"\n📊 Tokens: Input={usage.prompt_tokens}, "
        f"Output={usage.completion_tokens}, Total={usage.total_tokens}"
    )


# =============================================================================
# Example Functions
# =============================================================================


async def basic_suffix_example(client: VeniceClient, base_model: str) -> bool:
    """Demonstrate basic model feature suffix usage.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print_section_header("Basic Model Feature Suffix", "🏷️")

    # Append :enable_web_search=auto directly to the model string
    model_with_suffix = f"{base_model}:enable_web_search=auto"

    print(f"📍 Model string: {model_with_suffix}")
    print("   The API will parse the suffix and enable web search automatically.")

    try:
        response = await client.chat.completions.create(
            model=model_with_suffix,
            messages=[
                UserMessage(
                    content="What is the latest news about AI?",
                )
            ],
            max_completion_tokens=300,
            temperature=0.5,
        )

        print_response(response, "Web Search Response (via suffix)")
        print("\n✅ The API parsed enable_web_search=auto from the model string")

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True


async def multiple_params_example(client: VeniceClient, base_model: str) -> bool:
    """Demonstrate combining multiple parameters with & separator.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print_section_header("Multiple Parameters via Suffix", "🔧")

    # Combine multiple parameters with & separator
    model_with_params = f"{base_model}:enable_web_search=auto&strip_thinking_response=true"

    print(f"📍 Model string: {model_with_params}")
    print("   Parameters parsed from suffix:")
    print("     • enable_web_search = auto")
    print("     • strip_thinking_response = true")

    try:
        response = await client.chat.completions.create(
            model=model_with_params,
            messages=[
                UserMessage(
                    content="Briefly explain quantum computing.",
                )
            ],
            max_completion_tokens=300,
            temperature=0.5,
        )

        print_response(response, "Multi-param Suffix Response")
        print("\n✅ Both parameters applied via a single model string")

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True


async def build_model_id_example(client: VeniceClient, base_model: str) -> bool:
    """Demonstrate the SDK's build_model_id() helper.

    Returns ``True`` on success, ``False`` if the live API call failed. The
    double-suffix ``ValueError`` demo below is an intentional, expected error
    and does not affect the success result.
    """
    print_section_header("Using build_model_id() Helper", "⚡")

    # Build model ID programmatically
    model_id = build_model_id(
        base_model,
        enable_web_search="auto",
        strip_thinking_response="true",
    )

    print(f"📍 Built model ID: {model_id}")
    print("   build_model_id() constructs the suffix string for you.")

    ok = True
    try:
        response = await client.chat.completions.create(
            model=model_id,
            messages=[
                UserMessage(
                    content="What are the benefits of renewable energy?",
                )
            ],
            max_completion_tokens=300,
            temperature=0.5,
        )

        print_response(response, "build_model_id() Response")
        print("\n✅ build_model_id() produced a valid suffixed model string")

    except Exception as e:
        print(f"❌ Error: {e}")
        ok = False

    # Show that build_model_id rejects already-suffixed models.
    # This is an INTENTIONAL input-validation demo: catching the ValueError
    # here is the expected success path and must not flip ``ok``.
    print_subsection("Error Handling", "🛡️")
    try:
        build_model_id(f"{base_model}:enable_web_search=auto", disable_thinking="true")
    except ValueError as e:
        print(f"✅ Expected error for double suffix: {e}")

    return ok


async def comparison_example(client: VeniceClient, base_model: str) -> bool:
    """Side-by-side comparison: model suffix vs venice_parameters.

    Returns ``True`` only if both approach requests succeeded, ``False`` if
    either one failed.
    """
    print_section_header("Suffix vs venice_parameters Comparison", "⚖️")

    question = "What is the capital of France?"
    ok = True

    # --- Approach 1: Model suffix ---
    print_subsection("Approach 1: Model Feature Suffix", "🏷️")

    suffix_model = f"{base_model}:enable_web_search=off"
    print(f"📍 Model string: {suffix_model}")

    try:
        response_suffix = await client.chat.completions.create(
            model=suffix_model,
            messages=[UserMessage(content=question)],
            max_completion_tokens=200,
            temperature=0.3,
        )

        print_response(response_suffix, "Suffix Approach")

    except Exception as e:
        print(f"❌ Error: {e}")
        ok = False

    # --- Approach 2: venice_parameters ---
    print_subsection("Approach 2: venice_parameters Object", "🔧")

    print(f"📍 Model: {base_model}")
    print("📍 venice_parameters: VeniceParameters(enable_web_search='off')")

    try:
        response_params = await client.chat.completions.create(
            model=base_model,
            messages=[UserMessage(content=question)],
            # Only specifying enable_web_search to mirror the suffix above;
            # all other fields use their defaults.
            venice_parameters=VeniceParameters(
                enable_web_search="off",
            ),
            max_completion_tokens=200,
            temperature=0.3,
        )

        print_response(response_params, "venice_parameters Approach")

    except Exception as e:
        print(f"❌ Error: {e}")
        ok = False

    # Summary
    print_subsection("Comparison Summary", "📊")
    print("   Both approaches send the same parameters to the API.")
    print("   • Suffix approach:  Set params in the model string itself")
    print("   • venice_parameters: Set params in the request body object")
    print("   Choose based on your integration constraints (see Use Cases below).")

    return ok


async def use_cases_example() -> bool:
    """Document when to use suffixes vs venice_parameters (no API call needed).

    Pure informational output — always returns ``True``.
    """
    print_section_header("When to Use Model Feature Suffixes", "💡")

    print("""
   ┌─────────────────────────────────────────────────────────────────┐
   │  USE SUFFIXES WHEN:                                            │
   │                                                                │
   │  🏷️  Using an OpenAI-compatible client that only exposes the  │
   │      "model" field (e.g., Cursor, ChatGPT plugins, LiteLLM)   │
   │                                                                │
   │  ⚡  Quick testing via curl or simple scripts where you only   │
   │      want to change the model string                           │
   │                                                                │
   │  🔌  Integrations that proxy requests and can't modify the     │
   │      request body but can set the model name                   │
   │                                                                │
   ├─────────────────────────────────────────────────────────────────┤
   │  USE venice_parameters WHEN:                                   │
   │                                                                │
   │  🔧  You have full control over the request body               │
   │                                                                │
   │  📋  You want type-safe, IDE-autocompleted parameter names     │
   │                                                                │
   │  🧪  You prefer explicit, readable configuration               │
   │                                                                │
   │  📦  You need to combine with other VeniceParameters fields    │
   │      like character_slug or include_venice_system_prompt        │
   └─────────────────────────────────────────────────────────────────┘
""")

    print("🏷️  Available suffix parameters:")
    print("     enable_web_search          — 'on', 'off', or 'auto'")
    print("     strip_thinking_response    — 'true' or 'false'")
    print("     disable_thinking           — 'true' or 'false'")
    print("     enable_web_scraping        — 'true' or 'false'")
    print("     include_venice_system_prompt — 'true' or 'false'")

    return True


# =============================================================================
# Main Function
# =============================================================================


async def main() -> int:
    """Run all model feature suffix examples.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    print("🚀 Venice AI - Model Feature Suffixes")
    print("=" * 70)
    print("Append parameters directly to model IDs for lightweight configuration\n")

    async with VeniceClient() as client:
        # Dynamically select a chat model to use as the base for suffix demos.
        # A failure here means the run cannot proceed, so let it propagate to
        # the __main__ handler (loud, non-zero exit) rather than swallowing it.
        base_model = await client.models.resolve_chat()
        print(f"🤖 Selected base model: {base_model}\n")

        results: list[tuple[str, bool]] = [
            ("basic_suffix_example", await basic_suffix_example(client, base_model)),
            ("multiple_params_example", await multiple_params_example(client, base_model)),
            ("build_model_id_example", await build_model_id_example(client, base_model)),
            ("comparison_example", await comparison_example(client, base_model)),
        ]

    # This section doesn't need an API call.
    results.append(("use_cases_example", await use_cases_example()))

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print_section_header("Examples Completed! ✨", "🎉")

    print("\n💡 Key takeaways:")
    print("   • Append params to model IDs with : separator  (model:key=val)")
    print("   • Combine multiple params with & separator      (model:a=1&b=2)")
    print("   • Use build_model_id() for programmatic construction")
    print("   • Suffixes and venice_parameters produce equivalent results")
    print("   • Prefer suffixes for OpenAI-compat clients; venice_parameters otherwise")

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
