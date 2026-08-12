#!/usr/bin/env python3
"""
Venice AI SDK - Advanced Error Recovery
========================================

This example demonstrates advanced error recovery patterns.
Learn how to build resilient applications with retry strategies and error handling patterns.
"""

import asyncio
import sys

from venice_ai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    HttpClientConfig,
    RateLimitError,
    RetryOptions,
    UserMessage,
    VeniceAIConfig,
    VeniceClient,
    VeniceError,
)


async def retry_configuration() -> bool:
    """Demonstrate retry strategy configuration.

    Pure informational output — always returns ``True``.
    """
    print("🔄 Retry Strategy Configuration")
    print("-" * 40)

    # RetryOptions configures the aiohttp middleware-level retry behavior
    retry_options = RetryOptions(
        max_attempts=3,
        base_delay=1.0,
        retry_status_codes={429, 500, 502, 503},
    )
    print(f"Retry options: max_attempts={retry_options.max_attempts}")
    print(f"Base delay: {retry_options.base_delay}")
    print(f"Retry on status codes: {retry_options.retry_status_codes}")

    # HttpClientConfig inside VeniceAIConfig controls SDK-level retries
    config = VeniceAIConfig(
        api_key="your-api-key",
        http_client=HttpClientConfig(
            max_retries=3,
            retry_backoff_factor=2.0,
        ),
    )
    print(f"HTTP client max_retries: {config.http_client.max_retries}")
    print(f"HTTP client retry_backoff_factor: {config.http_client.retry_backoff_factor}")
    return True


async def scoped_retry_override() -> bool:
    """Demonstrate `client.with_retries()` for per-block policy overrides.

    Don't write your own exponential/jitter retry loops — the SDK ships
    ``client.with_retries(RetryOptions(...))``, which swaps the retry
    policy for the duration of a context-managed block while leaving
    every other call on the same client unaffected. ``RetryOptions``
    handles exponential backoff, capping, and jitter for you.

    Returns ``True`` on success, ``False`` if the API calls failed.
    """
    print("\n🎯 Scoped Retry Policy Override (with_retries)")
    print("-" * 40)

    try:
        async with VeniceClient() as client:
            chat_model = await client.models.resolve_chat()

            print("Default policy applies to this call:")
            response = await client.chat.completions.create(
                model=chat_model,
                messages=[UserMessage(content="Reply with exactly the word OK.")],
                max_completion_tokens=5,
            )
            raw_default = response.text
            default_text = (raw_default if isinstance(raw_default, str) else "").strip()
            print(f"   ✓ default policy reply: {default_text!r}")

            print("\nOverridden policy (max_attempts=5, longer base_delay) for this block:")
            async with client.with_retries(
                RetryOptions(max_attempts=5, base_delay=2.0, jitter_factor=0.2)
            ):
                response = await client.chat.completions.create(
                    model=chat_model,
                    messages=[UserMessage(content="Reply with exactly the word OK.")],
                    max_completion_tokens=5,
                )
                raw_inside = response.text
                inside_text = (raw_inside if isinstance(raw_inside, str) else "").strip()
                print(f"   ✓ inside-block reply: {inside_text!r}")

            print("\nOutside the block, the original default is back in effect.")
    except Exception as e:
        print(f"❌ Error in scoped retry override: {e}")
        return False

    return True


async def error_recovery_best_practices() -> bool:
    """Demonstrate error recovery best practices.

    Pure informational output — always returns ``True``.
    """
    print("\n💡 Error Recovery Best Practices")
    print("-" * 40)

    print("✅ Best Practices:")
    print()

    print("1. 🎯 Classify Errors Correctly:")
    print("   Retryable:")
    print("   - RateLimitError (with backoff)")
    print("   - APITimeoutError (transient)")
    print("   - APIConnectionError (network issues)")
    print("   - InternalServerError (503, 500)")
    print()
    print("   Non-Retryable:")
    print("   - AuthenticationError (fix API key)")
    print("   - InvalidRequestError (fix parameters)")
    print("   - PermissionDeniedError (fix permissions)")
    print("   - NotFoundError (wrong endpoint/model)")
    print()

    print("2. 🔄 Use Appropriate Backoff:")
    print("   - Rate limits → Exponential with jitter")
    print("   - Network issues → Linear backoff")
    print("   - Server errors → Exponential backoff")
    print("   - Always cap maximum delay")
    print()

    print("3. 🔄 Configure Retry Strategies:")
    print("   - Set max retries per use case")
    print("   - Low retries (1-2): Critical services")
    print("   - Higher retries (3-5): Non-critical services")
    print("   - Balance retry count vs. latency")
    print()

    print("4. 📊 Monitor and Alert:")
    print("   - Track error rates by type")
    print("   - Monitor retry attempt counts")
    print("   - Alert on threshold breaches")
    print("   - Log recovery patterns")
    print()

    print("5. 🎭 Implement Fallbacks:")
    print("   - Cached responses")
    print("   - Degraded functionality")
    print("   - Alternative models")
    print("   - Default responses")
    print()

    print("6. ⏱️ Set Appropriate Timeouts:")
    print("   - Short timeout (5-10s): Interactive")
    print("   - Medium timeout (30-60s): Background")
    print("   - Long timeout (120s+): Batch processing")
    print("   - Consider user experience")
    return True


async def comprehensive_error_handling() -> bool:
    """Demonstrate comprehensive error handling pattern.

    Don't hand-roll a ``for attempt in range(...)`` + ``asyncio.sleep(2**n)``
    backoff loop — that duplicates (and usually gets wrong) what the SDK
    already does. Route the request through
    ``client.with_retries(RetryOptions(...))`` so the middleware owns the
    retry control flow (exponential backoff, capping, jitter) for you, and
    keep the granular ``except`` blocks purely for *classifying* the terminal
    error once the SDK has exhausted its retries. See ``scoped_retry_override``
    above for the same primitive applied as a per-block policy override.

    Returns ``True`` if the request ultimately succeeded, ``False`` if the SDK
    exhausted its retries and the error surfaced to the caller.
    """
    print("\n🛡️ Comprehensive Error Handling")
    print("-" * 40)

    async with VeniceClient() as client:
        chat_model = await client.models.resolve_chat()

        # The SDK owns the retry loop. RetryOptions handles exponential
        # backoff, capping (max_delay), and jitter — no manual sleep loop.
        retry_options = RetryOptions(
            max_attempts=3,
            base_delay=1.0,
            jitter_factor=0.2,
            retry_status_codes={429, 500, 502, 503, 504},
        )
        print(
            "Routing the robust request through "
            f"client.with_retries(RetryOptions(max_attempts={retry_options.max_attempts}, ...)) — "
            "the SDK handles backoff + jitter; no hand-rolled loop."
        )

        try:
            async with client.with_retries(retry_options):
                response = await client.chat.completions.create(
                    model=chat_model,
                    messages=[UserMessage(content="Hello, demonstrate error handling!")],
                    max_completion_tokens=50,
                )
            print("✅ Success!")
            content = response.text or ""
            print(f"\n📝 Response: {content[:100]}...")
            return True

        # The except blocks below no longer drive retries — they classify the
        # terminal error after the SDK has exhausted its configured attempts.
        except RateLimitError as e:
            print(f"⏱️ Rate limited (retries exhausted): {e}")
            return False
        except APITimeoutError as e:
            print(f"⏰ Timeout (retries exhausted): {e}")
            return False
        except APIConnectionError as e:
            print(f"🌐 Connection error (retries exhausted): {e}")
            return False
        except APIError as e:
            print(f"🚨 API Error (non-retryable): {e}")
            return False
        except VeniceError as e:
            print(f"⚠️ SDK Error: {e}")
            return False


async def resilience_patterns() -> bool:
    """Demonstrate combining multiple resilience patterns.

    Pure informational output — always returns ``True``.
    """
    print("\n🏗️ Combining Resilience Patterns")
    print("-" * 40)

    print("✅ Layered Defense Strategy:")
    print()

    print("Layer 1: Retry Strategy")
    print("   - Automatic retries for transient failures")
    print("   - Prevents resource exhaustion")
    print("   - Configurable backoff and limits")
    print()

    print("Layer 2: Retry Logic")
    print("   - Exponential backoff for transients")
    print("   - Linear backoff for network")
    print("   - Jitter to prevent thundering herd")
    print()

    print("Layer 3: Timeouts")
    print("   - Request-level timeouts")
    print("   - Overall operation timeouts")
    print("   - Prevent hanging requests")
    print()

    print("Layer 4: Fallbacks")
    print("   - Cached responses")
    print("   - Degraded mode")
    print("   - Alternative providers")
    print()

    print("Layer 5: Rate Limiting")
    print("   - Client-side rate limiting")
    print("   - Request queuing")
    print("   - Backpressure handling")
    print()

    print("💡 Configuration Example:")
    print("   ```python")
    print(
        "   from venice_ai import RetryOptions, VeniceAIConfig, HttpClientConfig, SchedulerConfig"
    )
    print("   # Configure SDK-level retries via http_client")
    print("   config = VeniceAIConfig(")
    print("       http_client=HttpClientConfig(")
    print("           max_retries=3,")
    print("           retry_backoff_factor=2.0,")
    print("           timeout=30.0,")
    print("           max_connections=100")
    print("       ),")
    print("       scheduler=SchedulerConfig(")
    print("           enable_rate_limiting=True,")
    print("           rate_limit_buffer_ratio=0.9")
    print("       )")
    print("   )")
    print("   # Or use RetryOptions for aiohttp middleware-level retries")
    print("   retry_options = RetryOptions(max_attempts=3, base_delay=1.0)")
    print("   ```")
    return True


async def main() -> int:
    """Run all error recovery examples.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    print("=" * 60)
    print("Venice AI SDK - Advanced Error Recovery Examples")
    print("=" * 60)

    results: list[tuple[str, bool]] = [
        ("retry_configuration", await retry_configuration()),
        ("scoped_retry_override", await scoped_retry_override()),
        ("error_recovery_best_practices", await error_recovery_best_practices()),
        ("comprehensive_error_handling", await comprehensive_error_handling()),
        ("resilience_patterns", await resilience_patterns()),
    ]

    failed = [name for name, ok in results if not ok]

    print("\n" + "=" * 60)
    if failed:
        print(f"⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("✅ All examples completed!")
    print("=" * 60)
    print()
    print("📚 Next Steps:")
    print("   - Review examples/basic/error_handling.py for basics")
    print("   - Implement retry strategies in production")
    print("   - Monitor error rates and recovery patterns")
    print("   - Adjust thresholds based on observed behavior")

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
