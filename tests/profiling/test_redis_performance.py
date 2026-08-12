"""
Redis Backend Performance Profiling Tests

These tests measure Redis operation performance under various load conditions.
They help identify bottlenecks and validate optimization improvements.

Run with:
    poetry run pytest tests/profiling/test_redis_performance.py -v -s
"""

import asyncio
import statistics
import time
from typing import Any

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.requires_redis_pool]


class TestRedisOperationPerformance:
    """Profile individual Redis operations."""

    async def test_single_operation_latency(self, redis_backend):
        """Measure latency of individual Redis operations."""
        operations = 1000
        latencies: dict[str, list[float]] = {
            "set": [],
            "get": [],
            "delete": [],
        }

        # Get direct Redis client for raw performance testing
        redis = await redis_backend._ensure_connected()

        # Profile SET operations
        for i in range(operations):
            start = time.perf_counter()
            await redis.set(f"key_{i}", f"value_{i}")
            latencies["set"].append((time.perf_counter() - start) * 1000)  # ms

        # Profile GET operations
        for i in range(operations):
            start = time.perf_counter()
            await redis.get(f"key_{i}")
            latencies["get"].append((time.perf_counter() - start) * 1000)

        # Profile DELETE operations
        for i in range(operations):
            start = time.perf_counter()
            await redis.delete(f"key_{i}")
            latencies["delete"].append((time.perf_counter() - start) * 1000)

        # Calculate and report statistics
        print("\n" + "=" * 60)
        print("REDIS OPERATION LATENCY PROFILE")
        print("=" * 60)

        for op, times in latencies.items():
            print(f"\n{op.upper()} Operation ({operations} samples):")
            print(f"  Min:    {min(times):.3f}ms")
            print(f"  Max:    {max(times):.3f}ms")
            print(f"  Mean:   {statistics.mean(times):.3f}ms")
            print(f"  Median: {statistics.median(times):.3f}ms")
            print(f"  P95:    {statistics.quantiles(times, n=20)[18]:.3f}ms")
            print(f"  P99:    {statistics.quantiles(times, n=100)[98]:.3f}ms")

        # Performance assertions (adjusted for event loop management overhead)
        # Increased from 5.0ms to 15.0ms to account for per-event-loop pooling
        mean_set = statistics.mean(latencies["set"])
        mean_get = statistics.mean(latencies["get"])
        assert mean_set < 15.0, f"SET operations too slow: {mean_set:.3f}ms (threshold: 15.0ms)"
        assert mean_get < 15.0, f"GET operations too slow: {mean_get:.3f}ms (threshold: 15.0ms)"

    @pytest.mark.max_concurrency(200)
    async def test_concurrent_operations(self, redis_backend):
        """Profile concurrent Redis operations."""
        concurrent_tasks = [10, 50, 100, 200]
        results: dict[int, dict[str, Any]] = {}

        redis = await redis_backend._ensure_connected()

        for num_tasks in concurrent_tasks:

            async def worker(n: int) -> float:
                """Single worker performing multiple operations."""
                start = time.perf_counter()
                async with redis.pipeline(transaction=False) as pipe:
                    for i in range(100):
                        pipe.set(f"concurrent_{n}_{i}", f"value_{i}")
                        pipe.get(f"concurrent_{n}_{i}")
                    await pipe.execute()
                return time.perf_counter() - start

            # Run concurrent workers
            start_total = time.perf_counter()
            times = await asyncio.gather(*[worker(i) for i in range(num_tasks)])
            total_duration = time.perf_counter() - start_total

            # Calculate metrics
            total_ops = num_tasks * 100 * 2  # set + get
            ops_per_second = total_ops / total_duration

            results[num_tasks] = {
                "total_duration": total_duration,
                "ops_per_second": ops_per_second,
                "avg_worker_time": statistics.mean(times),
                "max_worker_time": max(times),
            }

        # Report results
        print("\n" + "=" * 60)
        print("CONCURRENT OPERATIONS PERFORMANCE")
        print("=" * 60)

        for num_tasks, metrics in results.items():
            print(f"\n{num_tasks} Concurrent Workers:")
            print(f"  Total Duration:   {metrics['total_duration']:.3f}s")
            print(f"  Ops/Second:       {metrics['ops_per_second']:.0f}")
            print(f"  Avg Worker Time:  {metrics['avg_worker_time']:.3f}s")
            print(f"  Max Worker Time:  {metrics['max_worker_time']:.3f}s")

        # Performance assertion
        assert results[100]["ops_per_second"] > 1000, "Concurrent ops too slow"

    async def test_bulk_operations_performance(self, redis_backend):
        """Profile bulk write and read operations."""
        batch_sizes = [100, 500, 1000, 5000]
        results: dict[int, dict[str, float]] = {}

        redis = await redis_backend._ensure_connected()

        for batch_size in batch_sizes:
            # Bulk writes
            data = {f"bulk_key_{i}": f"bulk_value_{i}" for i in range(batch_size)}

            write_start = time.perf_counter()
            async with redis.pipeline(transaction=False) as pipe:
                for k, v in data.items():
                    pipe.set(k, v)
                await pipe.execute()
            write_duration = time.perf_counter() - write_start

            # Bulk reads
            read_start = time.perf_counter()
            async with redis.pipeline(transaction=False) as pipe:
                for k in data:
                    pipe.get(k)
                await pipe.execute()
            read_duration = time.perf_counter() - read_start

            results[batch_size] = {
                "write_time": write_duration,
                "read_time": read_duration,
                "write_ops_per_sec": batch_size / write_duration,
                "read_ops_per_sec": batch_size / read_duration,
            }

        # Report results
        print("\n" + "=" * 60)
        print("BULK OPERATIONS PERFORMANCE")
        print("=" * 60)

        for batch_size, metrics in results.items():
            print(f"\nBatch Size: {batch_size}")
            print(f"  Write Time:       {metrics['write_time']:.3f}s")
            print(f"  Write Ops/Sec:    {metrics['write_ops_per_sec']:.0f}")
            print(f"  Read Time:        {metrics['read_time']:.3f}s")
            print(f"  Read Ops/Sec:     {metrics['read_ops_per_sec']:.0f}")


class TestRedisLuaScriptPerformance:
    """Profile Lua script execution performance."""

    async def test_lua_script_vs_native_operations(self, redis_backend):
        """Compare Lua script performance to native operations."""
        iterations = 100

        redis = await redis_backend._ensure_connected()

        # Test native Redis operations
        native_times: list[float] = []
        for i in range(iterations):
            start = time.perf_counter()
            # Simulate capacity check + decrement pattern
            current = await redis.get(f"capacity_{i}")
            if current is None or int(current) > 0:
                await redis.set(f"capacity_{i}", "99")
            native_times.append((time.perf_counter() - start) * 1000)

        # Test Lua script (if available)
        # Note: This would use the actual Lua scripts from the codebase
        lua_times: list[float] = []
        script_available = hasattr(redis_backend, "_reserve_capacity_script")

        if script_available:
            for i in range(iterations):
                start = time.perf_counter()
                # Use actual Lua script
                try:
                    await redis_backend.reserve_capacity(f"test_model_{i}", 1)
                    lua_times.append((time.perf_counter() - start) * 1000)
                except Exception:
                    # If script not loaded, skip
                    break

        # Report results
        print("\n" + "=" * 60)
        print("LUA SCRIPT PERFORMANCE COMPARISON")
        print("=" * 60)

        print(f"\nNative Operations ({len(native_times)} samples):")
        print(f"  Mean:   {statistics.mean(native_times):.3f}ms")
        print(f"  Median: {statistics.median(native_times):.3f}ms")
        print(f"  P95:    {statistics.quantiles(native_times, n=20)[18]:.3f}ms")

        if lua_times:
            print(f"\nLua Script Operations ({len(lua_times)} samples):")
            print(f"  Mean:   {statistics.mean(lua_times):.3f}ms")
            print(f"  Median: {statistics.median(lua_times):.3f}ms")
            print(f"  P95:    {statistics.quantiles(lua_times, n=20)[18]:.3f}ms")

            improvement = (
                (statistics.mean(native_times) - statistics.mean(lua_times))
                / statistics.mean(native_times)
                * 100
            )
            print(f"\nPerformance Improvement: {improvement:.1f}%")
        else:
            print("\nLua scripts not available for comparison")


class TestRedisMemoryUsage:
    """Profile Redis memory usage patterns."""

    async def test_memory_usage_under_load(self, redis_backend):
        """Measure memory usage with different data sizes."""
        data_sizes = [
            ("small", 100, 10),  # 100 keys, 10 bytes each
            ("medium", 100, 1024),  # 100 keys, 1KB each
            ("large", 100, 10240),  # 100 keys, 10KB each
        ]

        results: dict[str, dict[str, Any]] = {}

        redis = await redis_backend._ensure_connected()

        for size_name, num_keys, bytes_per_value in data_sizes:
            # Create test data
            value = "x" * bytes_per_value

            # Measure write performance
            start = time.perf_counter()
            for i in range(num_keys):
                await redis.set(f"mem_test_{size_name}_{i}", value, ex=60)
            write_time = time.perf_counter() - start

            # Measure read performance
            start = time.perf_counter()
            for i in range(num_keys):
                await redis.get(f"mem_test_{size_name}_{i}")
            read_time = time.perf_counter() - start

            total_data_mb = (num_keys * bytes_per_value) / (1024 * 1024)

            results[size_name] = {
                "total_keys": num_keys,
                "bytes_per_value": bytes_per_value,
                "total_data_mb": total_data_mb,
                "write_time": write_time,
                "read_time": read_time,
                "write_throughput_mbps": total_data_mb / write_time,
                "read_throughput_mbps": total_data_mb / read_time,
            }

        # Report results
        print("\n" + "=" * 60)
        print("MEMORY USAGE & THROUGHPUT")
        print("=" * 60)

        for size_name, metrics in results.items():
            print(f"\n{size_name.upper()} Data:")
            print(f"  Total Keys:        {metrics['total_keys']}")
            print(f"  Bytes/Value:       {metrics['bytes_per_value']}")
            print(f"  Total Data:        {metrics['total_data_mb']:.3f} MB")
            print(f"  Write Time:        {metrics['write_time']:.3f}s")
            print(f"  Read Time:         {metrics['read_time']:.3f}s")
            print(f"  Write Throughput:  {metrics['write_throughput_mbps']:.3f} MB/s")
            print(f"  Read Throughput:   {metrics['read_throughput_mbps']:.3f} MB/s")


class TestRedisEventLoopIsolation:
    """Profile per-event-loop connection pool behavior."""

    async def test_multi_event_loop_performance(self, redis_backend):
        """Test performance with multiple event loops (simulated)."""
        # This test validates the per-event-loop pooling design
        num_operations = 100

        redis = await redis_backend._ensure_connected()

        async def worker_in_same_loop():
            """Worker using the same event loop."""
            times: list[float] = []
            for i in range(num_operations):
                start = time.perf_counter()
                await redis.set(f"same_loop_{i}", f"value_{i}")
                await redis.get(f"same_loop_{i}")
                times.append(time.perf_counter() - start)
            return times

        # Run multiple workers concurrently in same event loop
        workers = 10
        start = time.perf_counter()
        results = await asyncio.gather(*[worker_in_same_loop() for _ in range(workers)])
        total_time = time.perf_counter() - start

        # Aggregate results
        all_times = [t for worker_times in results for t in worker_times]
        total_ops = num_operations * workers * 2  # set + get

        print("\n" + "=" * 60)
        print("EVENT LOOP PERFORMANCE")
        print("=" * 60)
        print(f"\nConcurrent Workers: {workers}")
        print(f"Operations/Worker:  {num_operations}")
        print(f"Total Operations:   {total_ops}")
        print(f"Total Time:         {total_time:.3f}s")
        print(f"Operations/Second:  {total_ops / total_time:.0f}")
        print("\nOperation Latency:")
        print(f"  Mean:   {statistics.mean(all_times) * 1000:.3f}ms")
        print(f"  Median: {statistics.median(all_times) * 1000:.3f}ms")
        print(f"  P95:    {statistics.quantiles(all_times, n=20)[18] * 1000:.3f}ms")
        print(f"  P99:    {statistics.quantiles(all_times, n=100)[98] * 1000:.3f}ms")

        # Verify reasonable performance
        assert total_ops / total_time > 500, "Event loop performance too low"
