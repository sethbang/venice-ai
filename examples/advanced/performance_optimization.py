#!/usr/bin/env python3
"""
Venice AI SDK - Performance Optimization
=========================================

This example demonstrates performance optimization techniques.
Learn how to maximize throughput and minimize latency for your applications.
"""

import asyncio
import sys
import time

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


async def connection_pooling_optimization():
    """Demonstrate connection pooling for better performance."""
    print("🔌 Connection Pooling Optimization")
    print("-" * 40)

    print("✅ HTTP Connection Pool Settings:")
    print()

    print("1. max_connections:")
    print("   - Total connections in pool")
    print("   - Higher = more concurrent requests")
    print("   - Recommended: 50-200 for production")
    print()

    print("2. max_keepalive_connections:")
    print("   - Persistent connections to reuse")
    print("   - Reduces handshake overhead")
    print("   - Recommended: 20-100 for production")
    print()

    print("3. Connection reuse benefits:")
    print("   ✓ Eliminates TCP handshake")
    print("   ✓ Reduces TLS negotiation")
    print("   ✓ Lower latency per request")
    print("   ✓ Better throughput")
    print()

    print("💡 Example Configuration:")
    print("   ```python")
    print("   HttpClientConfig(")
    print("       timeout=30.0,")
    print("       max_connections=100,")
    print("       max_keepalive_connections=50")
    print("   )")
    print("   ```")


async def concurrent_requests_pattern():
    """Demonstrate efficient concurrent request handling."""
    print("\n⚡ Concurrent Request Patterns")
    print("-" * 40)

    # High-performance configuration
    # Note: api_base_url is omitted - uses SDK default which handles URL correctly
    config = VeniceAIConfig(
        backend=BackendConfig(backend_type=BackendType.MEMORY),
        http_client=HttpClientConfig(
            timeout=30.0,
            max_connections=100,
            max_keepalive_connections=50,
        ),
        scheduler=SchedulerConfig(
            mode=SchedulerMode.BASIC,
            max_concurrent_executions=50,
            max_queue_size=1000,
        ),
    )

    client = VeniceClientFactory.create_client(config=config)

    async with client:
        chat_model = await client.models.resolve_chat()

        print("✅ Testing concurrent requests...")

        async def make_single_request(index: int):
            """Make a single request."""
            start = time.time()
            response = await client.chat.completions.create(
                model=chat_model,
                messages=[UserMessage(content=f"Count to {index}")],
                max_completion_tokens=10,
            )
            elapsed = time.time() - start
            return index, elapsed, response

        # Sequential vs Concurrent comparison
        print("\n📊 Performance Comparison:")

        # Sequential execution
        print("\n1. Sequential (one at a time):")
        start = time.time()
        sequential_results = []
        for i in range(3):
            try:
                result = await make_single_request(i + 1)
                sequential_results.append(result)
            except Exception as e:
                print(f"   Request {i + 1} failed: {e}")
        sequential_time = time.time() - start
        print(f"   Total time: {sequential_time:.2f}s")
        print(f"   Requests completed: {len(sequential_results)}")

        # Concurrent execution
        print("\n2. Concurrent (all at once):")
        start = time.time()
        tasks = [make_single_request(i + 1) for i in range(3)]
        concurrent_results = await asyncio.gather(*tasks, return_exceptions=True)
        concurrent_time = time.time() - start

        successful = sum(1 for r in concurrent_results if not isinstance(r, Exception))
        print(f"   Total time: {concurrent_time:.2f}s")
        print(f"   Requests completed: {successful}")

        if sequential_time > 0 and concurrent_time > 0:
            speedup = sequential_time / concurrent_time
            print(f"\n   ⚡ Speedup: {speedup:.1f}x faster")

        print("\n💡 Best Practice:")
        print("   - Use asyncio.gather() for concurrent requests")
        print("   - Configure appropriate max_concurrent_executions")
        print("   - Monitor rate limits to avoid throttling")


async def batching_strategies():
    """Demonstrate request batching for efficiency."""
    print("\n📦 Request Batching Strategies")
    print("-" * 40)

    print("✅ Batching Approaches:")
    print()

    print("1. Size-Based Batching:")
    print("   - Batch N requests together")
    print("   - Process when batch is full")
    print("   - Good for: Steady request rate")
    print()

    print("2. Time-Based Batching:")
    print("   - Collect requests for T seconds")
    print("   - Process accumulated batch")
    print("   - Good for: Variable request rate")
    print()

    print("3. Hybrid Batching:")
    print("   - Batch size OR time threshold")
    print("   - Whichever comes first")
    print("   - Good for: Production systems")
    print()

    print("💡 Example Implementation:")
    print("   ```python")
    print("   async def process_batch(requests, batch_size=10):")
    print("       batch = []")
    print("       for request in requests:")
    print("           batch.append(request)")
    print("           if len(batch) >= batch_size:")
    print("               await asyncio.gather(*batch)")
    print("               batch = []")
    print("       # Process remaining")
    print("       if batch:")
    print("           await asyncio.gather(*batch)")
    print("   ```")
    print()

    print("📊 Benefits:")
    print("   ✓ Better connection reuse")
    print("   ✓ Lower overhead per request")
    print("   ✓ More predictable performance")
    print("   ✓ Easier rate limit management")


async def streaming_for_large_responses():
    """Demonstrate streaming for better perceived performance."""
    print("\n🌊 Streaming for Large Responses")
    print("-" * 40)

    print("✅ Streaming Benefits:")
    print("   ✓ Lower time to first token")
    print("   ✓ Better user experience")
    print("   ✓ Memory efficient")
    print("   ✓ Progressive rendering")
    print()

    async with VeniceClient() as client:
        chat_model = await client.models.resolve_chat()
        prompt = (
            "Write a 5-paragraph short story about a robot learning to paint. "
            "Include vivid sensory detail."
        )
        max_tokens = 300

        # 1. Non-streaming: time to first byte == time to complete.
        print("1. Non-Streaming (wait for complete response):")
        start = time.time()
        try:
            response = await client.chat.completions.create(
                model=chat_model,
                messages=[UserMessage(content=prompt)],
                max_completion_tokens=max_tokens,
                stream=False,
            )
            ns_total = time.time() - start
            ns_chars = len(response.text or "")
            print(f"   Time to first token: {ns_total:.2f}s (== time to complete)")
            print(f"   Total time:          {ns_total:.2f}s")
            print(f"   Characters returned: {ns_chars}")
        except Exception as e:
            print(f"   Error: {e}")
            ns_total, ns_chars = None, 0

        print()

        # 2. Streaming: measure time to first token vs total.
        print("2. Streaming (progressive updates):")
        start = time.time()
        first_token_at = None
        s_chars = 0
        chunk_count = 0
        try:
            stream = await client.chat.completions.create(
                model=chat_model,
                messages=[UserMessage(content=prompt)],
                max_completion_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                chunk_count += 1
                if chunk.text:
                    if first_token_at is None:
                        first_token_at = time.time() - start
                    s_chars += len(chunk.text)
            s_total = time.time() - start
            ttft = first_token_at if first_token_at is not None else s_total
            print(f"   Time to first token: {ttft:.2f}s")
            print(f"   Total time:          {s_total:.2f}s")
            print(f"   Chunks received:     {chunk_count}")
            print(f"   Characters returned: {s_chars}")
        except Exception as e:
            print(f"   Error: {e}")
            s_total, ttft = None, None

        # 3. Compare.
        print()
        if ns_total is not None and ttft is not None:
            delta = ns_total - ttft
            pct = (delta / ns_total) * 100 if ns_total > 0 else 0.0
            print(
                f"⚡ Streaming time-to-first-token saved {delta:+.2f}s "
                f"({pct:+.1f}% of non-streaming wall clock)"
            )


async def memory_management():
    """Demonstrate memory-efficient patterns."""
    print("\n💾 Memory Management")
    print("-" * 40)

    print("✅ Memory Optimization Techniques:")
    print()

    print("1. Use Streaming for Large Responses:")
    print("   - Processes chunks as they arrive")
    print("   - Doesn't buffer entire response")
    print("   - Critical for long-form content")
    print()

    print("2. Limit max_queue_size:")
    print("   - Prevents unbounded memory growth")
    print("   - Provides backpressure")
    print("   - Recommended: 1000-5000")
    print()

    print("3. Close Clients Properly:")
    print("   ```python")
    print("   async with VeniceClient(api_key=key) as client:")
    print("       # Use client")
    print("       pass")
    print("   # Automatically cleaned up")
    print("   ```")
    print()

    print("4. Avoid Holding Large Response Objects:")
    print("   - Extract needed data immediately")
    print("   - Let garbage collector reclaim memory")
    print("   - Don't accumulate responses in lists")
    print()

    print("5. Monitor Connection Pool:")
    print("   - max_connections limits pool size")
    print("   - Each connection uses memory")
    print("   - Balance performance vs memory")


async def performance_monitoring():
    """Demonstrate performance monitoring approaches."""
    print("\n📊 Performance Monitoring")
    print("-" * 40)

    print("✅ Key Metrics to Track:")
    print()

    print("1. Request Latency:")
    print("   - Time from request to response")
    print("   - Track p50, p95, p99 percentiles")
    print("   - Identify slow endpoints/models")
    print()

    print("2. Throughput:")
    print("   - Requests per second")
    print("   - Tokens per second")
    print("   - Track over time for trends")
    print()

    print("3. Error Rates:")
    print("   - Percentage of failed requests")
    print("   - By error type")
    print("   - Correlate with performance")
    print()

    print("4. Rate Limit Usage:")
    print("   - Percentage of limit used")
    print("   - Requests remaining")
    print("   - Time to reset")
    print()

    print("5. Resource Utilization:")
    print("   - Connection pool usage")
    print("   - Queue depth")
    print("   - Memory consumption")
    print()

    print("💡 Example Monitoring:")
    print("   ```python")
    print("   start = time.time()")
    print("   response = await client.chat.completions.create(...)")
    print("   latency = time.time() - start")
    print("   ")
    print("   # Log metrics")
    print("   metrics.record_latency(latency)")
    print("   metrics.record_tokens(response.usage.total_tokens)")
    print("   ")
    print("   # Check rate limits")
    print("   if response.response_rate_limits:")
    print("       metrics.record_rate_limit_usage(")
    print("           response.response_rate_limits.remaining_requests")
    print("       )")
    print("   ```")


async def production_configuration():
    """Show production-optimized configuration."""
    print("\n🏭 Production Configuration")
    print("-" * 40)

    print("✅ High-Performance Production Setup:")
    print()
    print("```python")
    print("from venice_ai import (")
    print("    VeniceAIConfig, HttpClientConfig, SchedulerConfig, SchedulerMode,")
    print("    BackendConfig, BackendType, RedisBackendConfig,")
    print(")")
    print("")
    print("config = VeniceAIConfig(")
    print("    # HTTP Performance")
    print("    http_client=HttpClientConfig(")
    print("        timeout=45.0,              # Reasonable timeout")
    print("        max_connections=200,       # High concurrency")
    print("        max_keepalive_connections=100,  # Reuse connections")
    print("    ),")
    print("    ")
    print("    # Scheduler Optimization")
    print("    scheduler=SchedulerConfig(")
    print("        mode=SchedulerMode.BASIC,")
    print("        max_concurrent_executions=100,  # High throughput")
    print("        max_queue_size=5000,       # Large queue")
    print("        enable_rate_limiting=True, # Prevent violations")
    print("        rate_limit_buffer_ratio=0.9,  # Use 90% of limit")
    print("    ),")
    print("    ")
    print("    # Backend Choice")
    print("    backend=BackendConfig(")
    print("        backend_type=BackendType.REDIS,  # For distributed")
    print("        redis=RedisBackendConfig(")
    print("            max_connections=50,")
    print("            default_ttl=3600,")
    print("        )")
    print("    ),")
    print(")")
    print("```")
    print()

    print("📊 Expected Performance:")
    print("   - Throughput: 100+ requests/second")
    print("   - Latency: p50 < 500ms, p95 < 2s")
    print("   - Memory: Stable under load")
    print("   - Error rate: < 0.1%")


async def main():
    """Run all performance optimization examples."""
    print("=" * 60)
    print("Venice AI SDK - Performance Optimization Examples")
    print("=" * 60)

    await connection_pooling_optimization()
    await concurrent_requests_pattern()
    await batching_strategies()
    await streaming_for_large_responses()
    await memory_management()
    await performance_monitoring()
    await production_configuration()

    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
    print()
    print("📚 Key Takeaways:")
    print("   1. Use connection pooling for better performance")
    print("   2. Leverage asyncio.gather() for concurrent requests")
    print("   3. Implement streaming for large responses")
    print("   4. Monitor key metrics in production")
    print("   5. Configure appropriate queue sizes and timeouts")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
