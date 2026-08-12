#!/usr/bin/env python3
"""
Venice AI SDK - API Key Management and Monitoring
=================================================

This example demonstrates how to manage API keys in the Venice AI SDK:
- Listing and retrieving existing API keys
- Understanding API key metadata and usage statistics
- Monitoring rate limits and consumption patterns
- Managing key lifecycle (creation should be done carefully in production)
"""

import asyncio
import sys
from datetime import UTC, datetime, timedelta

from venice_ai import VeniceClient
from venice_ai.types.api import CreateApiKeyRequest, UserMessage

# A clearly-named, throwaway label so the ephemeral key is unmistakable in the
# dashboard and trivially distinguishable from any real, pre-existing key.
EPHEMERAL_KEY_DESCRIPTION = "venice-sdk-example-ephemeral"


async def list_existing_keys() -> bool:
    """List and analyze existing API keys."""
    print("📋 API Key Inventory")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            # List all API keys
            keys = await client.api_keys.list()

            if keys:
                print(f"📊 Found {len(keys)} API key(s) in your account")

                for i, key in enumerate(keys, 1):
                    print(f"\n🔑 Key #{i}:")

                    key_id = key.id
                    description = key.description or "No description"
                    api_key_type = key.apiKeyType
                    created_at = key.createdAt
                    expires_at = key.expiresAt
                    last_used = key.lastUsedAt
                    usage = key.usage

                    print(f"   📛 ID: {key_id}")
                    print(f"   📝 Description: {description}")
                    print(f"   🏷️ Type: {api_key_type}")
                    print(f"   📅 Created: {created_at}")

                    # Show expiration if set
                    if expires_at:
                        print(f"   ⏰ Expires: {expires_at}")
                    else:
                        print("   ⏰ Expires: Never")

                    # Show last usage
                    if last_used:
                        print(f"   🕐 Last used: {last_used}")
                    else:
                        print("   🕐 Last used: Never")

                    # Show usage statistics if available
                    if usage:
                        trailing_days = usage.trailingSevenDays
                        if trailing_days:
                            usd_cost = trailing_days.usd or "0.00"
                            diem_usage = trailing_days.diem or "0"
                            print(f"   💰 Last 7 days: ${usd_cost} USD, {diem_usage} DIEM")
                        else:
                            print("   💰 Usage data available")
                    else:
                        print("   💰 No usage data available")

                    # Show consumption limits if available
                    if key.consumptionLimits:
                        print("   🚦 Consumption limits configured")

                    # Show partial key for identification
                    if key.last6Chars:
                        print(f"   🔒 Key ends with: ***{key.last6Chars}")

            else:
                print("ℹ️ No API keys found in your account")

            return True

        except Exception as e:
            print(f"❌ Error listing API keys: {e}")
            print("💡 Note: This requires a valid API key with appropriate permissions")
            return False


async def monitor_rate_limits() -> bool:
    """Monitor current rate limits and usage patterns."""
    print("\n📊 Rate Limit Monitoring")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            # Get current rate limits
            rate_limits_response = await client.api_keys.get_rate_limits()

            print("🚦 Current Rate Limit Status:")

            # Extract rate limit data
            if hasattr(rate_limits_response, "data"):
                rate_data = rate_limits_response.data

                # Show access status
                access_status = "✅ Permitted" if rate_data.accessPermitted else "❌ Denied"
                print(f"   🔓 API Access: {access_status}")

                # Show API tier info
                tier = rate_data.apiTier
                tier_status = "💳 Paid" if tier.isCharged else "🆓 Free"
                print(f"   🏷️ API Tier: {tier.id} ({tier_status})")

                # Show account balances
                balances = rate_data.balances
                print("   💰 Account Balances:")
                print(f"      💵 USD: ${balances.USD:.2f}")
                print(f"      🪙 DIEM: {balances.DIEM:.2f}")

                # Show next epoch time
                print(f"   ⏰ Next Rate Limit Reset: {rate_data.nextEpochBegins}")

                # Show rate limits by model
                print(f"   📊 Rate Limits by Model ({len(rate_data.rateLimits)} models):")

                for i, model_limit in enumerate(rate_data.rateLimits[:5]):  # Show first 5 models
                    model_name = model_limit.apiModelId or f"Model {i + 1}"
                    print(f"      🤖 {model_name}:")

                    for limit in model_limit.rateLimits:
                        limit_type = limit.type
                        limit_amount = limit.amount
                        print(f"         📈 {limit_type}: {int(limit_amount):,}")

                if len(rate_data.rateLimits) > 5:
                    remaining = len(rate_data.rateLimits) - 5
                    print(f"      ... and {remaining} more models")

            else:
                print("📊 Rate limit data format not recognized")
                print(f"Response: {rate_limits_response}")

            return True

        except Exception as e:
            print(f"❌ Error getting rate limits: {e}")
            print("💡 Note: Rate limit monitoring requires API access")
            return False


async def check_rate_limit_history() -> bool:
    """Check recent rate limit violations."""
    print("\n📈 Rate Limit Violation History")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            # Get rate limit logs
            logs_response = await client.api_keys.get_rate_limit_logs()

            if hasattr(logs_response, "data") and logs_response.data:
                violations = logs_response.data

                print(f"⚠️ Found {len(violations)} recent rate limit violations")

                # Show recent violations
                for i, violation in enumerate(violations[:10], 1):  # Show first 10
                    print(f"\n🚨 Violation #{i}:")

                    print(f"   🤖 Model: {violation.modelId}")
                    print(f"   📊 Type: {violation.rateLimitType}")
                    print(f"   🕐 Time: {violation.timestamp}")

                if len(violations) > 10:
                    print(f"\n... and {len(violations) - 10} more violations")

                # Analyze patterns
                print("\n📊 Violation Analysis:")

                # Count by model
                model_counts = {}
                type_counts = {}

                for violation in violations:
                    model = violation.modelId
                    v_type = violation.rateLimitType

                    model_counts[model] = model_counts.get(model, 0) + 1
                    type_counts[v_type] = type_counts.get(v_type, 0) + 1

                print("   🎯 Most affected models:")
                for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True)[
                    :5
                ]:
                    print(f"      {model}: {count} violations")

                print("   📈 Violation types:")
                for v_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"      {v_type}: {count} violations")

            else:
                print("✅ No recent rate limit violations found")
                print("👍 Your API usage is within limits!")

            return True

        except Exception as e:
            print(f"❌ Error getting rate limit logs: {e}")
            print("💡 Note: Rate limit history requires API access")
            return False


async def demonstrate_key_creation_workflow() -> bool:
    """Demonstrate the API key creation workflow (educational only)."""
    print("\n🔧 API Key Creation Workflow (Educational)")
    print("-" * 40)

    print("⚠️ This section demonstrates the workflow without actually creating keys")
    print("🔒 In production, create keys only when necessary and store them securely")

    # Show how to structure a key creation request
    print("\n📝 Example Key Creation Request:")

    # Show how to structure a key creation request (with optional parameters)
    print("Option 1 - Minimal request:")
    print("CreateApiKeyRequest(")
    print('    description="Example Production Key",')
    print('    apiKeyType="INFERENCE"')
    print(")")
    print("")
    print("Option 2 - With expiration and limits:")
    print("CreateApiKeyRequest(")
    print('    description="Temporary API Key",')
    print('    apiKeyType="INFERENCE",')
    print('    expiresAt="2024-12-31T23:59:59Z",  # Optional')
    print("    consumptionLimit={...}  # Optional usage limits")
    print(")")

    print("```python")
    print("from venice_ai.types.api import CreateApiKeyRequest")
    print("")
    print("# Create a production API key")
    print("request = CreateApiKeyRequest(")
    print('    description="Production Service Key",')
    print('    apiKeyType="INFERENCE"')
    print(")")
    print("")
    print("# Create the key (returns the secret only once!)")
    print("new_key = await client.api_keys.create(api_key_request=request)")
    print("secret_key = new_key.apiKey  # Store this securely!")
    print("```")

    print("\n💡 Key Management Best Practices:")
    print("   🔐 Store API keys in environment variables or secure vaults")
    print("   📅 Set expiration dates for temporary keys")
    print("   📝 Use descriptive names to identify key purposes")
    print("   🗑️ Delete unused keys immediately")
    print("   📊 Monitor usage regularly to detect anomalies")
    print("   🔄 Rotate keys periodically for security")
    print("   🚫 Never commit keys to version control")
    print("   📋 Document which services use which keys")
    return True


async def analyze_key_details() -> bool:
    """Analyze detailed information about a specific key."""
    print("\n🔍 Detailed Key Analysis")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            # First, get a list of keys to find one to analyze
            keys = await client.api_keys.list()

            if keys:
                # Analyze the first key in detail
                first_key = keys[0]

                key_id = first_key.id

                print(f"🔍 Analyzing key: {key_id}")

                # Get detailed information (returns bare ApiKey)
                key_data = await client.api_keys.retrieve(api_key_id=key_id)

                print("\n📊 Detailed Analysis:")

                print(f"   🆔 ID: {key_data.id}")
                print(f"   📝 Description: {key_data.description}")
                print(f"   🏷️ Type: {key_data.apiKeyType}")
                print(f"   📅 Created: {key_data.createdAt}")
                usage = key_data.usage
                if usage:
                    print("\n📈 Usage Analytics:")
                    trailing = usage.trailingSevenDays
                    if trailing:
                        usd = trailing.usd or "0.00"
                        diem = trailing.diem or "0"
                        print(f"   💰 7-day cost: ${usd}")
                        print(f"   ⚡ 7-day DIEM: {diem}")

                        # Calculate daily averages
                        try:
                            daily_usd = float(usd) / 7
                            daily_diem = float(diem) / 7
                            print(f"   📊 Avg daily cost: ${daily_usd:.3f}")
                            print(f"   📊 Avg daily DIEM: {daily_diem:.1f}")
                        except (ValueError, ZeroDivisionError):
                            pass

                # Show security information
                print("\n🔒 Security Information:")
                last_used = key_data.lastUsedAt
                if last_used:
                    print(f"   🕐 Last activity: {last_used}")

                    # Parse timestamp and show time since last use
                    try:
                        from dateutil import parser

                        last_time = parser.parse(last_used)
                        now = datetime.now(last_time.tzinfo)
                        time_since = now - last_time
                        print(f"   ⏱️ Time since last use: {time_since}")
                    except Exception:
                        pass
                else:
                    print("   ⚠️ Never used - consider removing if not needed")

                expires_at = key_data.expiresAt
                if expires_at:
                    print(f"   ⏰ Expires: {expires_at}")
                else:
                    print("   ♾️ No expiration set")

            else:
                print("ℹ️ No API keys available for detailed analysis")

            return True

        except Exception as e:
            print(f"❌ Error analyzing key details: {e}")
            return False


async def demonstrate_key_lifecycle() -> bool:
    """Run a SAFE create -> use -> delete lifecycle on an ephemeral key.

    This exercises the two mutating endpoints (``create`` and ``delete``) end to
    end against a single, clearly-named throwaway key so it never collides with
    real credentials. The key is always deleted in a ``finally`` block — even if
    the "use" step fails — so it can never leak. Defense in depth: the key is
    also created with a near-term expiry, so even a catastrophic cleanup
    failure leaves only a short-lived, self-expiring key.

    The PRIMARY targeted methods here are ``client.api_keys.create`` and
    ``client.api_keys.delete``; the chat "use" step is a best-effort demo that
    proves the freshly-minted secret actually works.
    """
    print("\n🔁 API Key Lifecycle: create → use → delete (live)")
    print("-" * 40)
    print("⚠️ This creates a REAL, ephemeral key and deletes it before exiting.")

    async with VeniceClient() as admin_client:
        created = None
        try:
            # --- CREATE -----------------------------------------------------
            # INFERENCE so the returned secret can actually be used below.
            # A near-term expiry acts as a safety net in the (guarded against)
            # event cleanup never runs. The API accepts a YYYY-MM-DD date here
            # and normalizes it to a full timestamp; tomorrow is the soonest
            # whole-day expiry we can request.
            expires_at = (datetime.now(UTC) + timedelta(days=1)).date().isoformat()
            request = CreateApiKeyRequest(
                apiKeyType="INFERENCE",
                description=EPHEMERAL_KEY_DESCRIPTION,
                expiresAt=expires_at,
            )
            print(f"\n🆕 Creating ephemeral key: '{EPHEMERAL_KEY_DESCRIPTION}'")
            created = await admin_client.api_keys.create(api_key_request=request)

            # The secret (``created.apiKey``) is returned ONCE and never again.
            # Never print a full secret — only enough to identify it.
            print(f"   ✅ Created key id: {created.id}")
            print(f"   🏷️ Type: {created.apiKeyType}")
            print(f"   📝 Description: {created.description}")
            print(f"   ⏰ Expires: {created.expiresAt}")
            secret = created.apiKey
            print(
                f"   🔒 Secret (preview): {secret[:6]}…{secret[-4:]} "
                f"(store securely — shown only at creation!)"
            )

            # --- USE --------------------------------------------------------
            # Authenticate a brand-new client with the freshly-minted secret and
            # make a tiny chat completion. New keys can take a moment to
            # propagate, so retry briefly before giving up.
            print("\n🔑 Using the new key for a tiny chat completion...")
            used_ok = False
            async with VeniceClient(api_key=secret) as new_client:
                # Resolver-based selection — never hardcode a model id.
                chat_model = await new_client.models.resolve_chat()
                last_err: Exception | None = None
                for attempt in range(1, 4):
                    try:
                        resp = await new_client.chat.completions.create(
                            model=chat_model,
                            messages=[UserMessage(content="Reply with a single word: ok")],
                            max_completion_tokens=8,
                        )
                        print(f"   🤖 Reply ({chat_model}): {resp.text!r}")
                        used_ok = True
                        break
                    except Exception as use_err:  # noqa: BLE001 - propagation/entitlement probe
                        last_err = use_err
                        print(
                            f"   ⏳ Attempt {attempt}/3 failed "
                            f"({type(use_err).__name__}); the key may still be "
                            f"propagating, retrying..."
                        )
                        await asyncio.sleep(3)

            if not used_ok:
                # The "use" step is best-effort: if the account/key can't run
                # inference yet, we still consider create+delete the primary win.
                # We do NOT mask this — it's reported plainly — but it does not
                # fail the section, since the targeted mutators both ran.
                print(
                    f"   ⏭️ Could not exercise the key for inference "
                    f"({type(last_err).__name__ if last_err else 'unknown'}). "
                    f"create+delete still validated below."
                )

            return True

        finally:
            # --- DELETE (always) -------------------------------------------
            # Guaranteed cleanup: if creation succeeded we MUST remove the key,
            # whatever happened above. If creation itself failed, ``created`` is
            # None and there is nothing to clean up (the error propagates).
            if created is not None:
                print(f"\n🗑️ Deleting ephemeral key id: {created.id}")
                result = await admin_client.api_keys.delete(api_key_id=created.id)
                if result.success:
                    print("   ✅ Deletion confirmed (success=True)")
                else:
                    # Loud, unmasked: surface a non-success deletion as an error.
                    print("   ❌ Deletion reported success=False — key may persist!")
                    raise RuntimeError(f"Failed to delete ephemeral API key {created.id}")


async def main() -> int:
    """Run all API key management examples.

    Returns ``0`` only if every sub-section succeeded, ``1`` otherwise, so a
    real API failure surfaces as a non-zero process exit.
    """
    print("🚀 Venice AI API Key Management Examples")
    print("=" * 60)

    results: list[tuple[str, bool]] = [
        ("list_existing_keys", await list_existing_keys()),
        ("monitor_rate_limits", await monitor_rate_limits()),
        ("check_rate_limit_history", await check_rate_limit_history()),
        ("demonstrate_key_creation_workflow", await demonstrate_key_creation_workflow()),
        ("analyze_key_details", await analyze_key_details()),
        ("demonstrate_key_lifecycle", await demonstrate_key_lifecycle()),
    ]

    failed = [name for name, ok in results if not ok]

    print("\n✨ API key management examples completed!")
    if failed:
        print(f"\n❌ {len(failed)} section(s) failed: {', '.join(failed)}")
    print("\n💡 Key concepts demonstrated:")
    print("   - Listing and inventorying API keys")
    print("   - Understanding key metadata and configuration")
    print("   - Monitoring rate limits and usage patterns")
    print("   - Analyzing rate limit violation history")
    print("   - Best practices for key lifecycle management")
    print("   - Security considerations and monitoring")
    print("   - Usage analytics and cost tracking")
    print("\n🔒 Security reminders:")
    print("   - Never share or commit API keys to version control")
    print("   - Rotate keys regularly and delete unused ones")
    print("   - Monitor usage for unexpected patterns")
    print("   - Use environment variables for key storage")

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("Check that your API key is valid and you have appropriate access.", file=sys.stderr)
        sys.exit(1)
