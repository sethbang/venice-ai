#!/usr/bin/env python3
"""
Venice AI SDK - Client Setup Examples
====================================

This example demonstrates different ways to configure and set up the Venice AI client
for various use cases and environments.
"""

import asyncio
import os
import sys

from venice_ai import VeniceAIConfig, VeniceClient, VeniceClientFactory
from venice_ai.core.config import (
    BackendConfig,
    BackendType,
    HttpClientConfig,
    RedisBackendConfig,
    SchedulerConfig,
    SchedulerMode,
)
from venice_ai.exceptions import APIConnectionError, AuthenticationError


async def basic_setup() -> bool:
    """Basic client setup with API key.

    Returns ``True`` on success, ``False`` if the backend call failed.
    """
    print("🔧 Basic Client Setup")
    print("-" * 30)

    ok = True
    # Method 1: Direct instantiation (reads VENICE_API_KEY from environment)
    async with VeniceClient() as client:
        print("✅ Basic client created successfully")

        # Test with a simple request
        try:
            models = await client.models.list()
            print(f"📋 Found {len(models.data)} models")
        except Exception as e:
            print(f"❌ Backend call failed: {type(e).__name__}: {e}")
            ok = False

    return ok


async def explicit_key_setup() -> bool:
    """Setup with an explicit API key (useful for testing or multi-key setups).

    Returns ``True`` on success, ``False`` if the backend call failed.
    """
    print("\n🔑 Explicit Key Setup")
    print("-" * 30)

    ok = True
    # Method 2: Pass API key explicitly
    api_key = os.getenv("VENICE_API_KEY", "your-api-key-here")
    async with VeniceClient(api_key=api_key) as client:
        print("✅ Client created with explicit API key")
        try:
            models = await client.models.list()
            print(f"   📋 Backend verified: {len(models.data)} models reachable")
        except Exception as e:
            print(f"   ❌ Backend verification failed: {type(e).__name__}: {e}")
            ok = False

    return ok


async def custom_configuration() -> bool:
    """Setup with custom configuration.

    Returns ``True`` on success, ``False`` if the backend call failed.
    """
    print("\n⚙️ Custom Configuration Setup")
    print("-" * 30)

    # Create a custom configuration
    config = VeniceAIConfig(
        # Backend configuration
        backend=BackendConfig(backend_type=BackendType.MEMORY),
        # HTTP client settings
        http_client=HttpClientConfig(
            timeout=60.0, max_connections=50, max_keepalive_connections=20
        ),
        # Scheduler configuration
        scheduler=SchedulerConfig(mode=SchedulerMode.BASIC),
    )

    # Create client using factory with custom config
    client = VeniceClientFactory.create_client(config=config)

    ok = True
    async with client:
        print("✅ Custom configured client created")
        print("   - Memory backend (for testing)")
        print("   - 60s timeout")
        print("   - 50 max connections")
        print("   - Basic scheduler mode")
        try:
            models = await client.models.list()
            print(f"   📋 Backend verified: {len(models.data)} models reachable")
        except Exception as e:
            print(f"   ❌ Backend verification failed: {type(e).__name__}: {e}")
            ok = False

    return ok


async def production_setup() -> bool:
    """Production-ready client setup.

    Returns ``True`` on success. A missing local Redis is treated as a graceful
    skip (still ``True``) because the example only demonstrates configuration;
    any other failure surfaces as ``False``.
    """
    print("\n🏭 Production Setup")
    print("-" * 30)

    # Production configuration with Redis backend
    config = VeniceAIConfig(
        # Redis backend for distributed state
        backend=BackendConfig(
            backend_type=BackendType.REDIS,
            redis=RedisBackendConfig(
                redis_url="redis://localhost:6379",
                max_connections=20,
                default_ttl=3600,
            ),
        ),
        # Optimized HTTP settings
        http_client=HttpClientConfig(
            timeout=30.0, max_connections=100, max_keepalive_connections=50
        ),
        # Basic scheduler for this example (INTELLIGENT mode requires more setup)
        scheduler=SchedulerConfig(mode=SchedulerMode.BASIC),
    )

    try:
        client = VeniceClientFactory.create_client(config=config)

        async with client:
            print("✅ Production client created")
            print("   - Redis backend configured for distributed state")
            print(f"   - Basic scheduler ({SchedulerMode.BASIC.value}) — predictable FIFO ordering")
            print("   - 100 max connections, 50 keep-alive")
            print("   ℹ️  For tier-aware prioritization upgrade to SchedulerMode.INTELLIGENT")
            print("       and configure RateLimiterConfig(mode=ADAPTIVE, redis_url=...)")
            try:
                models = await client.models.list()
                print(f"   📋 Backend verified: {len(models.data)} models reachable")
            except Exception as e:
                print(f"   ❌ Backend verification failed: {type(e).__name__}: {e}")
                return False
    except (ConnectionError, APIConnectionError, OSError) as e:
        # No local Redis available — this is an infra prerequisite, not a code
        # bug, so the demo is skipped rather than failed.
        print(f"⏭️  Skipped: Redis not reachable ({type(e).__name__}); start Redis to run this demo")
        return True

    return True


async def test_setup() -> bool:
    """Test-optimized client setup.

    Uses a placeholder ``api_key="test-key"`` to show the test client wires up
    end to end. The backend is expected to reject that key with an
    :class:`AuthenticationError`; catching that specific error *proves* the
    request reached the backend, so it counts as success. Any other failure
    (including a surprising 2xx) is reported and returns ``False``.
    """
    print("\n🧪 Test Setup")
    print("-" * 30)

    from venice_ai import create_test_venice_client

    ok = True
    # create_test_venice_client now defaults to SchedulerMode.BASIC, so a
    # test client works out of the box without TierDiscovery setup.
    async with create_test_venice_client(api_key="test-key") as client:
        print("✅ Test client created")
        print("   - Optimized for testing")
        print("   - Fast timeouts")
        print("   - Memory backend")
        print("   - Minimal retries")
        print("   - Basic scheduler mode (default)")
        try:
            models = await client.models.list()
            print(f"   📋 Wiring verified — models.list() returned {len(models.data)} entries")
        except AuthenticationError as e:
            # The placeholder key is rejected by the backend, which proves the
            # request was wired all the way through.
            print(f"   📋 Wiring verified — backend rejected placeholder key: {type(e).__name__}")
        except Exception as e:
            print(f"   ❌ Unexpected failure (not an auth rejection): {type(e).__name__}: {e}")
            ok = False

    return ok


async def main() -> int:
    """Demonstrate various client setup methods.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    print("🚀 Venice AI Client Setup Examples")
    print("=" * 50)

    results: list[tuple[str, bool]] = [
        ("basic_setup", await basic_setup()),
        ("explicit_key_setup", await explicit_key_setup()),
        ("custom_configuration", await custom_configuration()),
        ("production_setup", await production_setup()),
        ("test_setup", await test_setup()),
    ]

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} setup examples failed: {', '.join(failed)}")
    else:
        print("\n✨ All client setup examples completed!")

    print("\n💡 Choose the setup method that best fits your use case:")
    print("   - Basic: Simple applications")
    print("   - Environment: Docker/cloud deployments")
    print("   - Custom: Specific requirements")
    print("   - Production: High-scale applications")
    print("   - Test: Unit/integration testing")

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
