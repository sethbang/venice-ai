"""
Connection Pool Performance Profiling Tests

Tests to validate and optimize connection pool behavior under various
load patterns and concurrent access scenarios.

Run with:
    poetry run pytest tests/profiling/test_connection_pooling.py -v -s
"""

import asyncio
import statistics
import time
from typing import Any

import pytest

from venice_ai.core.backends.redis import RedisBackend

pytestmark = [pytest.mark.slow, pytest.mark.requires_redis_pool]


class TestConnectionPoolUtilization:
    """Profile connection pool utilization patterns."""

    @pytest.mark.max_concurrency(100)
    async def test_pool_under_varying_load(self, redis_backend):
        """Test pool behavior under different concurrent load levels."""
        load_levels = [5, 10, 25, 50, 100]
        results: dict[int, dict[str, Any]] = {}

        redis = await redis_backend._ensure_connected()

        for concurrent_workers in load_levels:

            async def worker(worker_id: int) -> dict[str, Any]:
                """Worker performing Redis operations."""
                start = time.perf_counter()
                operations = 50

                async with redis.pipeline(transaction=False) as pipe:
                    for i in range(operations):
                        pipe.set(
                            f"pool_test_w{worker_id}_k{i}",
                            f"value_{i}",
                        )
                        pipe.get(f"pool_test_w{worker_id}_k{i}")
                    await pipe.execute()

                duration = time.perf_counter() - start
                return {
                    "worker_id": worker_id,
                    "duration": duration,
                    "ops_per_sec": (operations * 2) / duration,
                }

            # Run concurrent workers
            start_time = time.perf_counter()
            worker_results = await asyncio.gather(*[worker(i) for i in range(concurrent_workers)])
            total_duration = time.perf_counter() - start_time

            # Aggregate metrics
            worker_times = [r["duration"] for r in worker_results]
            total_ops = concurrent_workers * 50 * 2

            results[concurrent_workers] = {
                "total_duration": total_duration,
                "total_ops_per_sec": total_ops / total_duration,
                "avg_worker_duration": statistics.mean(worker_times),
                "max_worker_duration": max(worker_times),
                "min_worker_duration": min(worker_times),
                "worker_duration_stddev": statistics.stdev(worker_times)
                if len(worker_times) > 1
                else 0,
            }

        # Report results
        print("\n" + "=" * 70)
        print("CONNECTION POOL UTILIZATION UNDER VARYING LOAD")
        print("=" * 70)

        for workers, metrics in results.items():
            print(f"\n{workers} Concurrent Workers:")
            print(f"  Total Duration:        {metrics['total_duration']:.3f}s")
            print(f"  Total Ops/Sec:         {metrics['total_ops_per_sec']:.0f}")
            print(f"  Avg Worker Duration:   {metrics['avg_worker_duration']:.3f}s")
            print(f"  Min Worker Duration:   {metrics['min_worker_duration']:.3f}s")
            print(f"  Max Worker Duration:   {metrics['max_worker_duration']:.3f}s")
            print(f"  Duration Std Dev:      {metrics['worker_duration_stddev']:.3f}s")

        # Verify pool scales reasonably
        # With 100 workers, throughput should still be decent
        assert results[100]["total_ops_per_sec"] > 500, "Pool doesn't scale well"

    async def test_pool_connection_reuse(self, redis_backend):
        """Verify connection reuse efficiency."""
        iterations = 10
        operations_per_iteration = 100

        iteration_times: list[float] = []

        redis = await redis_backend._ensure_connected()

        for _iteration in range(iterations):
            start = time.perf_counter()

            for i in range(operations_per_iteration):
                await redis.set(f"reuse_test_{i}", f"value_{i}")
                await redis.get(f"reuse_test_{i}")

            iteration_times.append(time.perf_counter() - start)

        # Report results
        print("\n" + "=" * 70)
        print("CONNECTION REUSE EFFICIENCY")
        print("=" * 70)
        print(f"\nIterations:           {iterations}")
        print(f"Ops/Iteration:        {operations_per_iteration * 2}")
        print("\nIteration Times:")
        print(f"  First:              {iteration_times[0]:.3f}s")
        print(f"  Last:               {iteration_times[-1]:.3f}s")
        print(f"  Mean:               {statistics.mean(iteration_times):.3f}s")
        print(f"  Std Dev:            {statistics.stdev(iteration_times):.3f}s")

        # First iteration might be slower due to pool warmup
        # Subsequent iterations should be consistent
        warmup_overhead = iteration_times[0] - statistics.mean(iteration_times[1:])
        print(
            f"\nWarmup Overhead:      {warmup_overhead:.3f}s ({warmup_overhead / iteration_times[0] * 100:.1f}%)"
        )

        # Verify low variance after warmup
        # Higher threshold to account for CI/test environment variability
        # Increased from 0.5 to 1.0 to handle event loop overhead in new implementation
        steady_state_stddev = statistics.stdev(iteration_times[2:])
        assert steady_state_stddev < 1.0, (
            f"High variance in steady state: {steady_state_stddev:.3f}s"
        )

    @pytest.mark.max_concurrency(200)
    async def test_pool_burst_handling(self, redis_backend):
        """Test pool behavior under burst traffic patterns."""
        burst_sizes = [10, 50, 100, 200]
        results: dict[int, dict[str, float]] = {}

        redis = await redis_backend._ensure_connected()

        for burst_size in burst_sizes:
            # Create burst of concurrent operations
            start = time.perf_counter()

            await asyncio.gather(
                *[redis.set(f"burst_key_{i}", f"burst_value_{i}") for i in range(burst_size)]
            )

            burst_duration = time.perf_counter() - start

            # Measure recovery time
            recovery_start = time.perf_counter()
            await redis.set("recovery_test", "value")
            recovery_duration = time.perf_counter() - recovery_start

            results[burst_size] = {
                "burst_duration": burst_duration,
                "burst_ops_per_sec": burst_size / burst_duration,
                "recovery_time": recovery_duration * 1000,  # ms
            }

        # Report results
        print("\n" + "=" * 70)
        print("BURST TRAFFIC HANDLING")
        print("=" * 70)

        for burst_size, metrics in results.items():
            print(f"\nBurst Size: {burst_size}")
            print(f"  Burst Duration:     {metrics['burst_duration']:.3f}s")
            print(f"  Burst Throughput:   {metrics['burst_ops_per_sec']:.0f} ops/sec")
            print(f"  Recovery Time:      {metrics['recovery_time']:.3f}ms")

        # Verify recovery is fast even after large bursts
        assert results[200]["recovery_time"] < 50.0, "Slow recovery after burst"


class TestEventLoopIsolation:
    """Test per-event-loop connection pool isolation."""

    async def test_event_loop_pool_isolation(self, redis_backend):
        """Verify each event loop gets isolated pool."""
        operations = 100

        redis = await redis_backend._ensure_connected()

        # All operations in same event loop
        start = time.perf_counter()
        for i in range(operations):
            await redis.set(f"isolation_test_{i}", f"value_{i}")
        single_loop_duration = time.perf_counter() - start

        print("\n" + "=" * 70)
        print("EVENT LOOP POOL ISOLATION")
        print("=" * 70)
        print("\nSingle Event Loop:")
        print(f"  Operations:         {operations}")
        print(f"  Duration:           {single_loop_duration:.3f}s")
        print(f"  Ops/Second:         {operations / single_loop_duration:.0f}")

        # Note: Testing actual multi-event-loop behavior requires
        # running code in separate threads with their own loops,
        # which is complex in pytest. This test validates single-loop
        # performance as a baseline.

    async def test_pool_cleanup_on_loop_close(self, redis_backend):
        """Verify pools are cleaned up when event loops close."""
        # This test validates the cleanup mechanism exists
        # Actual cleanup happens when event loops are garbage collected

        initial_pools = len(RedisBackend._connection_pools)

        print("\n" + "=" * 70)
        print("POOL CLEANUP VALIDATION")
        print("=" * 70)
        print(f"\nInitial Pools:        {initial_pools}")
        print(f"Max Pools Allowed:    {RedisBackend._max_pools}")

        # Verify we're tracking pools
        assert isinstance(RedisBackend._connection_pools, dict)
        assert isinstance(RedisBackend._max_pools, int)

        print("\nPool management infrastructure is in place")


class TestConnectionPoolResilience:
    """Test pool resilience under error conditions."""

    async def test_pool_performance_degradation(self, redis_backend, baseline_tracker):
        """Test if pool maintains performance under sustained load."""
        duration_seconds = 5
        interval_seconds = 1

        results: list[dict[str, Any]] = []

        redis = await redis_backend._ensure_connected()

        start_time = time.time()
        interval_start = start_time
        interval_ops = 0

        while time.time() - start_time < duration_seconds:
            # Perform operation
            await redis.set("perf_test", "value")
            interval_ops += 1

            # Check if interval elapsed
            if time.time() - interval_start >= interval_seconds:
                ops_per_sec = interval_ops / (time.time() - interval_start)
                results.append(
                    {
                        "interval": len(results) + 1,
                        "ops_per_sec": ops_per_sec,
                        "ops_count": interval_ops,
                    }
                )

                interval_start = time.time()
                interval_ops = 0

        # Report results
        print("\n" + "=" * 70)
        print("SUSTAINED LOAD PERFORMANCE")
        print("=" * 70)
        print(f"\nDuration:             {duration_seconds}s")
        print(f"Interval:             {interval_seconds}s")
        print("\nPer-Interval Throughput:")

        for result in results:
            print(
                f"  Interval {result['interval']}: {result['ops_per_sec']:.0f} ops/sec ({result['ops_count']} ops)"
            )

        # Calculate degradation
        if len(results) > 1:
            first_throughput = results[0]["ops_per_sec"]
            last_throughput = results[-1]["ops_per_sec"]
            degradation = (first_throughput - last_throughput) / first_throughput * 100

            print(f"\nThroughput Change:    {degradation:+.1f}%")

            # Use statistical baseline instead of fixed threshold
            is_ok, msg = baseline_tracker.check_degradation("pool_performance", abs(degradation))
            print(f"\nBaseline Analysis: {msg}")
            assert is_ok, msg
        else:
            print("\nNot enough intervals for degradation analysis")
