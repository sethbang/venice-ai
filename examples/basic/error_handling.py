#!/usr/bin/env python3
"""
Venice AI SDK - Error Handling Examples
=======================================

This example demonstrates comprehensive error handling patterns for the Venice AI SDK.
Learn how to gracefully handle different types of errors and implement retry logic.
"""

import asyncio
import sys

from venice_ai import VeniceClient
from venice_ai.exceptions import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    InvalidRequestError,
    ModelGoneError,
    NotFoundError,
    RateLimitError,
    VeniceError,
)
from venice_ai.types.api import UserMessage


async def basic_error_handling() -> bool:
    """Demonstrate basic error handling — happy path + a deliberate auth failure.

    Returns ``True`` if the happy-path call succeeded and both deliberate error
    branches surfaced an expected exception, ``False`` otherwise.
    """
    print("🛡️ Basic Error Handling")
    print("-" * 30)

    ok = True

    # --- Happy-path branch ---
    async with VeniceClient() as client:
        try:
            chat_model = await client.models.resolve_chat()
            response = await client.chat.completions.create(
                model=chat_model, messages=[UserMessage(content="Hello!")], max_completion_tokens=50
            )
            content = response.text or ""
            print(f"✅ Success: {content[:50]}...")

        except APIError as e:
            print(f"🚨 Unexpected API error: {e}")
            ok = False

    # --- AuthenticationError branch: deliberately use a bad key ---
    # Auth fails server-side before the model name is validated, so any
    # placeholder works — keeps the demo focused on the auth failure path
    # without coupling to a specific model ID.
    print("\n🔐 Triggering AuthenticationError (bad API key)...")
    async with VeniceClient(api_key="sk-invalid-key-for-demo") as bad_client:
        try:
            await bad_client.chat.completions.create(
                model="placeholder-auth-fails-first",
                messages=[UserMessage(content="Hello!")],
                max_completion_tokens=10,
            )
            print("⚠️ Expected AuthenticationError but request succeeded")
            ok = False

        except AuthenticationError as e:
            print(f"🔐 Authentication failed (as expected): {e}")
            print("   ✓ This is how AuthenticationError surfaces from the SDK")

        except InvalidRequestError as e:
            print(f"❌ Invalid request: {e}")

        except RateLimitError as e:
            print(f"⏱️ Rate limit exceeded: {e}")

        except APITimeoutError as e:
            print(f"⏰ Request timed out: {e}")

        except APIConnectionError as e:
            print(f"🌐 Connection error: {e}")

        except APIError as e:
            # Some servers return 400 instead of 401 for bad keys; still show the branch firing.
            print(f"🚨 API error (bad-key path): status={getattr(e, 'status_code', '?')}, {e}")

        except VeniceError as e:
            print(f"⚠️ Venice SDK error: {e}")

    # --- NotFoundError branch: ask for a model that doesn't exist ---
    print("\n❌ Triggering NotFoundError (nonexistent model)...")
    async with VeniceClient() as client:
        try:
            await client.chat.completions.create(
                model="this-model-does-not-exist-2026",
                messages=[UserMessage(content="Hi")],
                max_completion_tokens=10,
            )
            print("⚠️ Expected error but request succeeded")
            ok = False

        except NotFoundError as e:
            print(f"❌ Not found (as expected): {e}")
            print("   ✓ This is how NotFoundError surfaces from the SDK")

        except InvalidRequestError as e:
            # Some servers return 400 instead of 404 for unknown models.
            print(f"❌ Invalid request (as expected): {e}")
            print("   ✓ This is how InvalidRequestError surfaces from the SDK")

        except APIError as e:
            print(f"🚨 API error: status={getattr(e, 'status_code', '?')}, {e}")

    return ok


async def removed_model_lifecycle() -> bool:
    """Demonstrate the deprecation → removal lifecycle for a vanished model.

    A model's identifier moves through a lifecycle:

    1. **Active** — routable, returns ``200``.
    2. **Deprecated** — still routable; Venice auto-routes to a replacement and
       advertises the sunset via ``deprecation_info`` response headers.
    3. **Retired (410 Gone)** — no longer routable but still *recognised*; the
       SDK maps this to :class:`ModelGoneError`. This is the "migrate to a
       replacement" signal.
    4. **Removed (404 Not Found)** — dropped from the catalog entirely, now
       indistinguishable from a model that never existed; the SDK maps this to
       :class:`NotFoundError`.

    The exact status a given gone-model returns therefore *drifts over time*:
    it 410s right after retirement, then 404s once fully removed. Pinning a
    "currently-410" model ID would make this demo brittle — the moment that
    model is removed, the 410 branch goes dead. So we use an obviously-synthetic
    identifier and handle the whole tail of the lifecycle (410 **and** 404, plus
    a 400 fallback for servers that reject unknown models that way). Whichever
    fires, the demo passes and labels the lifecycle stage.

    Returns ``True`` if the gone model surfaced as any of
    ModelGone/NotFound/InvalidRequest (the expected lifecycle outcomes),
    ``False`` if the request unexpectedly succeeded.
    """
    print("\n🪦 Removed-model lifecycle (410 Gone → 404 Not Found)...")
    print("-" * 30)

    # Synthetic identifier: not a real model, so it can never be "resolved" and
    # its status can't drift on us. A genuinely retired/removed model would
    # behave the same way — that is exactly the point of this demo.
    gone_model = "retired-model-please-migrate-2099"
    async with VeniceClient() as client:
        try:
            await client.chat.completions.create(
                model=gone_model,
                messages=[UserMessage(content="Hi")],
                max_completion_tokens=10,
            )
            print(f"⚠️ Expected the request to fail for gone model '{gone_model}'")
            return False

        except ModelGoneError as e:
            # Lifecycle stage 3: retired but still recognised → 410.
            print(f"🪦 Model gone (410, as expected): status={getattr(e, 'status_code', '?')}, {e}")
            print("   ✓ A retired-but-recognised model maps to ModelGoneError (410)")
            print(
                "   💡 Migrate: pick a current model via client.models.resolve_*() / models.list()"
            )
            return True

        except NotFoundError as e:
            # Lifecycle stage 4: fully removed → now indistinguishable from a typo.
            print(f"❌ Not found (404, as expected): status={getattr(e, 'status_code', '?')}, {e}")
            print("   ✓ A fully-removed model maps to NotFoundError (404)")
            print("   ℹ️ Once a model leaves the catalog it 404s, not 410s — same fix: migrate.")
            return True

        except InvalidRequestError as e:
            # Some deployments reject an unknown/removed model with 400 instead.
            print(
                f"❌ Invalid request (400, as expected): status={getattr(e, 'status_code', '?')}, {e}"
            )
            print("   ✓ Some servers return 400 for an unknown/removed model — still 'migrate'.")
            return True


async def retry_with_backoff() -> bool:
    """Demonstrate retry logic with exponential backoff.

    Returns ``True`` if the (possibly retried) request ultimately succeeded,
    ``False`` if it exhausted retries or hit a non-retryable error.
    """
    print("\n🔄 Retry with Exponential Backoff")
    print("-" * 30)

    async def make_request_with_retry(client: VeniceClient, max_retries: int = 3):
        """Make a request with retry logic."""

        # Get available chat model dynamically
        chat_model = await client.models.resolve_chat()

        for attempt in range(max_retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=chat_model,
                    messages=[UserMessage(content="Count to 3")],
                    max_completion_tokens=50,
                )
                print(f"✅ Success on attempt {attempt + 1}")
                return response

            except RateLimitError:
                wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s, 8s
                print(f"⏱️ Rate limited on attempt {attempt + 1}, waiting {wait_time}s...")

                if attempt < max_retries:
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print("❌ Max retries exceeded")
                    raise

            except (APITimeoutError, APIConnectionError):
                wait_time = 1 + attempt  # Linear backoff for connection issues
                print(f"🌐 Connection issue on attempt {attempt + 1}, waiting {wait_time}s...")

                if attempt < max_retries:
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print("❌ Max retries exceeded")
                    raise

            except (AuthenticationError, InvalidRequestError) as e:
                # Don't retry these errors - they won't succeed
                print(f"🚫 Non-retryable error: {e}")
                raise

    async with VeniceClient() as client:
        try:
            response = await make_request_with_retry(client)
            if response:
                content = response.text or ""
                print(f"📝 Final response: {content}")
                return True
            return False
        except Exception as e:
            print(f"❌ Final failure: {type(e).__name__}: {e}")
            return False


async def graceful_degradation() -> bool:
    """Demonstrate graceful degradation patterns.

    Returns ``True`` once any model in the fallback chain answers, ``False`` if
    every candidate failed and the demo had to fall back to a default.
    """
    print("\n🎯 Graceful Degradation")
    print("-" * 30)

    async with VeniceClient() as client:
        # Get multiple models to try
        try:
            models_response = await client.models.list(type="chat")
            models_to_try = [m.id for m in models_response.data[:3]]
        except Exception:
            # Fallback to single model
            models_to_try = [await client.models.resolve_chat()]

        for model in models_to_try:
            try:
                print(f"🔄 Trying model: {model}")

                response = await client.chat.completions.create(
                    model=model,
                    messages=[UserMessage(content="What's 2+2?")],
                    max_completion_tokens=50,
                )

                print(f"✅ Success with {model}")
                content = str(response.text or "")
                if not content.strip():
                    continue
                print(f"📝 Response: {content}")
                return True  # Success, no need to try other models

            except NotFoundError:
                print(f"❌ Model {model} not available, trying next...")
                continue

            except RateLimitError:
                print(f"⏱️ Model {model} rate limited, trying next...")
                continue

            except Exception as e:
                print(f"❌ Error with {model}: {type(e).__name__}")
                continue

        print("❌ All models failed, implementing fallback...")
        print("💡 Fallback: Using cached response or default message")
        return False


async def error_context_handling() -> bool:
    """Demonstrate error handling with context preservation.

    Returns ``True`` if the request succeeded; ``False`` if it raised (the
    error-context block still runs to show the debugging pattern).
    """
    print("\n🎭 Error Context Handling")
    print("-" * 30)

    async with VeniceClient() as client:
        conversation_history = [
            UserMessage(content="What's your name?"),
        ]

        # Get available chat model dynamically
        chat_model = await client.models.resolve_chat()

        try:
            response = await client.chat.completions.create(
                model=chat_model, messages=conversation_history, max_completion_tokens=50
            )

            content = response.text or ""
            print(f"✅ Response: {content}")
            return True

        except Exception as e:
            print(f"❌ Error occurred: {type(e).__name__}: {e}")

            # Log error context for debugging
            error_context = {
                "model": chat_model,
                "message_count": len(conversation_history),
                "last_message": conversation_history[-1].content if conversation_history else None,
                "error_type": type(e).__name__,
                "error_message": str(e),
            }

            print("🔍 Error context for debugging:")
            for key, value in error_context.items():
                print(f"   {key}: {value}")

            # Provide helpful suggestions
            if isinstance(e, AuthenticationError):
                print("💡 Suggestion: Check your VENICE_API_KEY environment variable")
            elif isinstance(e, RateLimitError):
                print("💡 Suggestion: Reduce request frequency or upgrade your plan")
            elif isinstance(e, InvalidRequestError):
                print("💡 Suggestion: Check model name and parameters")

    return False


async def main() -> int:
    """Demonstrate comprehensive error handling patterns.

    Returns ``0`` only if every demo behaved as expected, ``1`` otherwise, so a
    real failure surfaces as a non-zero process exit instead of being masked by
    the success banner.
    """
    print("🚀 Venice AI Error Handling Examples")
    print("=" * 50)

    results: list[tuple[str, bool]] = [
        ("basic_error_handling", await basic_error_handling()),
        ("removed_model_lifecycle", await removed_model_lifecycle()),
        ("retry_with_backoff", await retry_with_backoff()),
        ("graceful_degradation", await graceful_degradation()),
        ("error_context_handling", await error_context_handling()),
    ]

    failed = [name for name, ok in results if not ok]

    if failed:
        print(
            f"\n⚠️ {len(failed)} of {len(results)} error-handling examples failed: {', '.join(failed)}"
        )
    else:
        print("\n✨ Error handling examples completed!")

    print("\n💡 Key takeaways:")
    print("   - Always use specific exception types for targeted handling")
    print("   - Implement retry logic for transient errors")
    print("   - Have fallback strategies for graceful degradation")
    print("   - Log error context for effective debugging")
    print("   - Provide helpful error messages to users")

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error in main: {e}", file=sys.stderr)
        sys.exit(1)
