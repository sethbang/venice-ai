#!/usr/bin/env python3
"""
Venice AI SDK - Header Access Example
====================================

This example demonstrates how to access HTTP response headers and extract
useful information like rate limits, deprecation warnings, and account balance
from Venice AI API responses.

Features demonstrated:
- Direct header access
- Rate limit information
- Model deprecation warnings
- Account balance information
- API version tracking

Usage:
    python examples/headers/header_access_example.py
"""

import asyncio
import sys

from venice_ai import VeniceClient
from venice_ai.types.api.requests import UserMessage


async def discover_and_test_models(client: VeniceClient):
    """Discover available models and categorize by type."""
    print("📋 Discovering available models...")

    try:
        models_response = await client.models.list(type="all")
        models = models_response.data

        # Categorize models by type
        models_by_type = {}
        for model in models:
            model_type = model.type or "unknown"
            if model_type not in models_by_type:
                models_by_type[model_type] = []
            models_by_type[model_type].append(model.id)

        print(f"✅ Found {len(models)} models across {len(models_by_type)} types")
        for model_type, model_list in models_by_type.items():
            print(f"   • {model_type.upper()}: {len(model_list)} models")

        return models_by_type

    except Exception as e:
        print(f"❌ Error discovering models: {e}")
        return {}


async def demonstrate_header_access() -> bool:
    """Demonstrate accessing headers from API responses.

    Returns True if every *attempted* live call succeeded, False if any
    attempted call raised. Calls that are skipped because no model of that
    type is available do not count as failures.
    """
    print("🚀 Venice AI Header Access Example")
    print("=" * 50)

    async with VeniceClient() as client:
        # First discover available models
        models_by_type = await discover_and_test_models(client)

        if not models_by_type:
            print("❌ Could not discover models")
            return False

        ok = True

        # Test with a text model for chat completions
        if "text" in models_by_type and models_by_type["text"]:
            text_model = await client.models.resolve_chat()
            print(f"\n💬 Testing Chat Completion with {text_model}...")
            try:
                chat_response = await client.chat.completions.create(
                    model=text_model,
                    messages=[UserMessage(content="Hello!")],
                    max_completion_tokens=10,
                )

                content = chat_response.text
                content_str = str(content) if content else ""
                content_preview = content_str[:50] + "..." if len(content_str) > 50 else content_str
                print(f"✅ Response received: {content_preview}")

                demonstrate_response_headers(chat_response, "Chat Response")

            except Exception as e:
                print(f"❌ Chat completion error: {e}")
                ok = False
        else:
            print("\n💬 No text models available for chat testing")

        # Test with an image model
        if "image" in models_by_type and models_by_type["image"]:
            image_model = await client.models.resolve_image()
            print(f"\n🖼️ Testing Image Generation with {image_model}...")
            try:
                image_response = await client.image.create(
                    model=image_model,
                    prompt="A simple test image",
                    width=256,
                    height=256,
                    num_images=1,
                )

                if image_response.images:
                    print(f"✅ Image generated: {len(image_response.images)} image(s)")
                else:
                    print("✅ Image generated successfully")

                demonstrate_response_headers(image_response, "Image Response")

            except Exception as e:
                print(f"❌ Image generation error: {e}")
                ok = False
        else:
            print("\n🖼️ No image models available for testing")

        # Test with embeddings model
        if "embedding" in models_by_type and models_by_type["embedding"]:
            embed_model = await client.models.resolve_embedding()
            print(f"\n🔢 Testing Embeddings with {embed_model}...")
            try:
                embed_response = await client.embeddings.create(
                    model=embed_model, input="Hello world"
                )

                if hasattr(embed_response, "data") and embed_response.data:
                    print(
                        f"✅ Embedding created: {len(embed_response.data[0].embedding)} dimensions"
                    )
                else:
                    print("✅ Embedding created successfully")

                demonstrate_response_headers(embed_response, "Embeddings Response")

            except Exception as e:
                print(f"❌ Embeddings error: {e}")
                ok = False
        else:
            print("\n🔢 No embedding models available for testing")

        return ok


def demonstrate_response_headers(response, response_type: str):
    """Demonstrate header access for any response object."""
    print(f"\n📊 {response_type} Headers:")
    print("-" * 30)

    # 1. Raw headers access
    headers = response.headers
    if headers:
        print(f"📋 Raw Headers ({len(headers)} total):")
        venice_headers = {k: v for k, v in headers.items() if k.lower().startswith("x-venice")}
        rate_headers = {k: v for k, v in headers.items() if k.lower().startswith("x-ratelimit")}

        if venice_headers:
            print("   Venice Headers:")
            for key, value in venice_headers.items():
                print(f"     • {key}: {value}")

        if rate_headers:
            print("   Rate Limit Headers:")
            for key, value in rate_headers.items():
                print(f"     • {key}: {value}")
    else:
        print("❌ No headers available")
        return

    # 2. Rate limit information
    rate_limits = response.response_rate_limits
    if rate_limits:
        print("\n🚦 Rate Limits:")
        if rate_limits.limit_requests:
            print(f"   • Requests: {rate_limits.remaining_requests}/{rate_limits.limit_requests}")
            if rate_limits.reset_requests:
                print(f"     Reset: {rate_limits.reset_requests}")

        if rate_limits.limit_tokens:
            print(f"   • Tokens: {rate_limits.remaining_tokens}/{rate_limits.limit_tokens}")
            if rate_limits.reset_tokens:
                # NOTE: reset_tokens is now an ABSOLUTE Unix timestamp in
                # seconds (normalized from the ms-epoch x-ratelimit-reset-tokens
                # header) — NOT a duration/seconds-until-reset. Compare against
                # time.time() to get the wait, e.g. max(0, reset_tokens - time.time()).
                print(f"     Reset (Unix epoch seconds): {rate_limits.reset_tokens}")
    else:
        print("ℹ️ No rate limit information available")

    # 3. Deprecation warnings
    deprecation = response.deprecation_info
    if deprecation and deprecation.is_deprecated:
        print("\n⚠️ DEPRECATION WARNING:")
        if deprecation.warning:
            print(f"   Message: {deprecation.warning}")
        if deprecation.date:
            print(f"   Date: {deprecation.date}")
    else:
        print("✅ No deprecation warnings")

    # 4. Balance information
    balance = response.balance_info
    if balance:
        print("\n💰 Account Balance:")
        if balance.diem is not None:
            print(f"   • DIEM: {balance.diem:.4f}")
        if balance.usd is not None:
            print(f"   • USD: ${balance.usd:.4f}")
    else:
        print("ℹ️ No balance information available")

    # 5. API version
    version = response.venice_version
    if version:
        print(f"\n🏷️ API Version: {version}")
    else:
        print("ℹ️ No API version information")


def show_header_properties_usage():
    """Show example code for using header properties."""
    print("\n📖 Header Properties Usage Examples:")
    print("-" * 40)

    code_examples = [
        (
            "Check Rate Limits",
            """
# Check if you're approaching rate limits
response = await client.chat.completions.create(...)
if response.response_rate_limits:
    remaining = response.response_rate_limits.remaining_requests
    if remaining and remaining < 10:
        print(f"Warning: Only {remaining} requests remaining!")
        """,
        ),
        (
            "Handle Deprecation",
            """
# Check for model deprecation
image_model = await client.models.resolve_image()
response = await client.image.create(model=image_model, ...)
if response.deprecation_info and response.deprecation_info.is_deprecated:
    print(f"Model deprecated: {response.deprecation_info.warning}")
    # Switch to alternative model
        """,
        ),
        (
            "Monitor Balance",
            """
# Monitor account balance
response = await client.chat.completions.create(...)
if response.balance_info and response.balance_info.diem:
    if response.balance_info.diem < 1.0:
        print("Low balance warning!")
        """,
        ),
        (
            "Direct Header Access",
            """
# Access any header directly
response = await client.embeddings.create(...)
headers = response.headers
custom_header = headers.get('x-custom-header') if headers else None
        """,
        ),
    ]

    for title, code in code_examples:
        print(f"\n{title}:")
        print(code.strip())


async def main() -> int:
    """Main example function.

    Returns 0 if all attempted live calls succeeded, 1 otherwise.
    """
    ok = await demonstrate_header_access()
    show_header_properties_usage()

    print("\n✨ Header access examples completed!")
    print("\nKey Benefits:")
    print("• Easy access to rate limit information for throttling")
    print("• Automatic detection of deprecated models")
    print("• Real-time balance monitoring")
    print("• Access to all HTTP headers for debugging")

    if not ok:
        print("\n❌ One or more attempted API calls failed.", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Example cancelled!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("Check that your API key is valid and you have model access.", file=sys.stderr)
        sys.exit(1)
    sys.exit(exit_code)
