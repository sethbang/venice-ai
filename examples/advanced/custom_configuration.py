#!/usr/bin/env python3
"""
Venice AI SDK - Advanced Custom Configuration
==============================================

This example demonstrates advanced client configuration options.
Learn how to fine-tune Venice AI for specific use cases and environments.
"""

import asyncio
import sys

from venice_ai import (
    BackendConfig,
    BackendType,
    HttpClientConfig,
    SchedulerConfig,
    SchedulerMode,
    UserMessage,
    VeniceAIConfig,
    VeniceClient,
    VeniceClientFactory,
)


async def minimal_configuration():
    """Demonstrate minimal client configuration."""
    print("🎯 Minimal Configuration")
    print("-" * 40)

    # Simplest possible setup
    async with VeniceClient() as client:
        print("✅ Client created with defaults")
        print("   - Memory backend")
        print("   - Standard timeouts")
        print("   - Basic scheduler")

        # Test it works
        chat_model = await client.models.resolve_chat()

        response = await client.chat.completions.create(
            model=chat_model, messages=[UserMessage(content="Say hello!")], max_completion_tokens=20
        )

        print(f"\n🤖 Response: {response.text}")


async def custom_http_settings():
    """Configure custom HTTP client settings."""
    print("\n🌐 Custom HTTP Settings")
    print("-" * 40)

    # Create configuration with custom HTTP settings
    # Note: api_base_url is omitted - uses SDK default which handles URL correctly
    config = VeniceAIConfig(
        http_client=HttpClientConfig(
            timeout=60.0,  # Longer timeout for slow connections
            max_connections=100,  # Higher concurrency
            max_keepalive_connections=50,  # More persistent connections
        ),
        backend=BackendConfig(backend_type=BackendType.MEMORY),
        scheduler=SchedulerConfig(mode=SchedulerMode.BASIC),
    )

    client = VeniceClientFactory.create_client(config=config)

    async with client:
        print("✅ Custom HTTP client configured")
        print(f"   - Timeout: {config.http_client.timeout}s")
        print(f"   - Max connections: {config.http_client.max_connections}")
        print(f"   - Keepalive connections: {config.http_client.max_keepalive_connections}")

        # Test with a simple request
        chat_model = await client.models.resolve_chat()

        print(f"   📍 Selected model: {chat_model}")
        print("   🔍 Making API request...")

        response = await client.chat.completions.create(
            model=chat_model, messages=[UserMessage(content="Hello!")], max_completion_tokens=10
        )

        content = response.text or ""
        print(f"\n✅ Request successful: {content[:50]}...")


async def memory_backend_configuration():
    """Configure in-memory backend for development/testing."""
    print("\n💾 Memory Backend Configuration")
    print("-" * 40)

    # Memory backend configuration using factory
    # Note: api_base_url is omitted - uses SDK default which handles URL correctly
    config = VeniceAIConfig(
        backend=BackendConfig(
            backend_type=BackendType.MEMORY,
        ),
        http_client=HttpClientConfig(timeout=30.0),
        scheduler=SchedulerConfig(mode=SchedulerMode.BASIC),
    )

    client = VeniceClientFactory.create_client(config=config)

    async with client:
        print("✅ Memory backend configured")
        print("   - In-memory state management")
        print("   - No external dependencies required")
        print("   - Ideal for: Development, testing, single-server deployments")

        # Show it works
        chat_model = await client.models.resolve_chat()

        print(f"   📍 Selected model: {chat_model}")
        print("   🔍 Making API request...")

        response = await client.chat.completions.create(
            model=chat_model,
            messages=[UserMessage(content="Test memory backend")],
            max_completion_tokens=15,
        )

        content = response.text or ""
        print(f"\n✅ Backend operational: {content[:30]}...")


async def scheduler_configuration():
    """Configure different scheduler modes."""
    print("\n⚙️ Scheduler Configuration")
    print("-" * 40)

    # Demonstrate BASIC scheduler (recommended for most use cases)
    print("\n📋 BASIC Scheduler (Recommended):")
    print("   - Simple FIFO scheduling")
    print("   - Minimal overhead")
    print("   - Works out of the box")

    config = VeniceAIConfig(
        backend=BackendConfig(backend_type=BackendType.MEMORY),
        http_client=HttpClientConfig(timeout=30.0),
        scheduler=SchedulerConfig(
            mode=SchedulerMode.BASIC, max_concurrent_executions=10, max_queue_size=100
        ),
    )

    client = VeniceClientFactory.create_client(config=config)

    async with client:
        print("   ✅ BASIC scheduler active")
        print(f"   - Max concurrent: {config.scheduler.max_concurrent_executions}")
        print(f"   - Queue size: {config.scheduler.max_queue_size}")
        models = await client.models.list()
        print(f"   📋 Backend verified: {len(models.data)} models reachable")

    # Explain INTELLIGENT mode without trying to use it
    print("\n📋 INTELLIGENT Scheduler (Advanced):")
    print("   - Smart prioritization based on model tiers")
    print("   - Better for production at scale")
    print("   ⚠️  Requires additional setup:")
    print("      • Tier discovery configuration")
    print("      • Model capability mapping")
    print("      • Not needed for most use cases")
    print("   💡 Use BASIC mode unless you need tier-based prioritization")


async def production_optimized_config():
    """Show a production-optimized configuration."""
    print("\n🏭 Production-Optimized Configuration")
    print("-" * 40)

    # Production configuration
    config = VeniceAIConfig(
        # Note: api_base_url is omitted - uses SDK default which handles URL correctly
        # Memory backend (Redis would be used in real production)
        backend=BackendConfig(
            backend_type=BackendType.MEMORY,
        ),
        # Optimized HTTP settings
        http_client=HttpClientConfig(
            timeout=45.0,  # Reasonable timeout
            max_connections=200,  # High concurrency
            max_keepalive_connections=100,  # Many persistent connections
        ),
        # Basic scheduler (INTELLIGENT requires tier discovery setup)
        scheduler=SchedulerConfig(
            mode=SchedulerMode.BASIC,
            max_concurrent_executions=50,  # High throughput
            max_queue_size=1000,  # Large queue
        ),
    )

    client = VeniceClientFactory.create_client(config=config)

    async with client:
        print("✅ Production configuration loaded")
        print("\n📊 Configuration Summary:")
        print(f"   Backend: {config.backend.backend_type}")
        print(f"   HTTP Timeout: {config.http_client.timeout}s")
        print(f"   Max Connections: {config.http_client.max_connections}")
        print(f"   Scheduler Mode: {config.scheduler.mode}")
        print(f"   Max Concurrent: {config.scheduler.max_concurrent_executions}")
        print(f"   Queue Size: {config.scheduler.max_queue_size}")

        print("\n💡 Best For:")
        print("   - High-traffic applications")
        print("   - Production web services")
        print("   - Enterprise deployments")

        print("\n📝 Note:")
        print("   For INTELLIGENT scheduler mode, configure tier discovery")
        print("   Most production use cases work well with BASIC mode")

        chat_model = await client.models.resolve_chat()
        response = await client.chat.completions.create(
            model=chat_model,
            messages=[UserMessage(content="ping")],
            max_completion_tokens=5,
        )
        print(
            f"\n   ✅ Production config exercised — model={chat_model}, "
            f"tokens={response.usage.total_tokens if response.usage else 'n/a'}"
        )


async def development_config():
    """Show a development-optimized configuration."""
    print("\n🔧 Development Configuration")
    print("-" * 40)

    # Development configuration - optimized for debugging
    config = VeniceAIConfig(
        # Note: api_base_url is omitted - uses SDK default which handles URL correctly
        backend=BackendConfig(
            backend_type=BackendType.MEMORY,
        ),
        http_client=HttpClientConfig(
            timeout=120.0,  # Longer timeout for debugging
            max_connections=10,  # Lower concurrency for easier debugging
            max_keepalive_connections=5,
        ),
        scheduler=SchedulerConfig(
            mode=SchedulerMode.BASIC,  # Simpler for debugging
            max_concurrent_executions=5,
            max_queue_size=50,
        ),
    )

    client = VeniceClientFactory.create_client(config=config)

    async with client:
        print("✅ Development configuration loaded")
        print("\n📊 Configuration Summary:")
        print("   Backend: Memory (no external dependencies)")
        print(f"   HTTP Timeout: {config.http_client.timeout}s (longer for debugging)")
        print(f"   Max Connections: {config.http_client.max_connections} (lower for clarity)")
        print(f"   Scheduler: {config.scheduler.mode} (simpler)")

        print("\n💡 Best For:")
        print("   - Local development")
        print("   - Debugging and testing")
        print("   - Learning the SDK")

        models = await client.models.list()
        print(f"\n   ✅ Development config exercised — {len(models.data)} models reachable")


async def main():
    """Run all custom configuration examples."""
    print("🚀 Venice AI Advanced Configuration Examples")
    print("=" * 60)

    await minimal_configuration()
    await custom_http_settings()
    await memory_backend_configuration()
    await scheduler_configuration()
    await production_optimized_config()
    await development_config()

    print("\n✨ Configuration examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Minimal vs custom configurations")
    print("   - HTTP client tuning")
    print("   - Backend selection (Memory vs Redis)")
    print("   - Scheduler modes (Basic vs Intelligent)")
    print("   - Production-optimized settings")
    print("   - Development-friendly settings")
    print("\n📚 Next Steps:")
    print("   - Choose configuration based on your use case")
    print("   - Test different settings for your workload")
    print("   - Monitor performance and adjust as needed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
