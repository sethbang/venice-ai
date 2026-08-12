"""
Profiling tests configuration with dynamic connection pool sizing.

This conftest provides a redis_backend fixture that automatically sizes
the connection pool based on the max_concurrency marker on tests.
"""

import asyncio
from pathlib import Path

import pytest

from tests.profiling.performance_baseline import PerformanceBaseline
from venice_ai.core.backends.redis import RedisBackend


@pytest.fixture
async def redis_backend(request):
    """
    Create Redis backend with auto-sized connection pool.

    The pool size is calculated as max_concurrency * 1.2 for a 20% safety margin.
    Tests can declare their concurrency needs using the max_concurrency marker:

    @pytest.mark.max_concurrency(100)
    async def test_high_concurrency(redis_backend):
        # Automatically gets pool of 120 connections
        ...

    If no marker is present, defaults to max_concurrency=50 (pool size 60).
    """
    # Get test concurrency from marker or default
    marker = request.node.get_closest_marker("max_concurrency")
    max_concurrency = marker.args[0] if marker else 50

    # Size pool: (max_concurrency * 1.2) for safety margin
    pool_size = int(max_concurrency * 1.2)

    # Log pool sizing decision for validation
    test_name = request.node.name
    print(f"\n{'=' * 70}")
    print(f"DYNAMIC POOL SIZING for {test_name}")
    print(f"{'=' * 70}")
    print(f"Max Concurrency: {max_concurrency}")
    print(f"Pool Size:       {pool_size} (= {max_concurrency} * 1.2)")
    print(f"{'=' * 70}\n")

    backend = RedisBackend(
        redis_url="redis://localhost:6379/15",
        namespace="pool_profile",
        max_connections=pool_size,
        key_ttl=300,
    )

    await backend._ensure_connected()
    yield backend

    # Proper cleanup - ensure all Redis tasks complete
    if backend._redis:
        import contextlib

        with contextlib.suppress(Exception):
            # Close the connection
            await backend._redis.aclose()

        # Cancel and await any remaining background tasks
        try:
            current_tasks = asyncio.all_tasks()
            pending = [t for t in current_tasks if not t.done() and t != asyncio.current_task()]

            if pending:
                # Cancel pending tasks
                for task in pending:
                    task.cancel()

                # Wait for cancellation to complete
                await asyncio.gather(*pending, return_exceptions=True)
        except Exception:
            # Ignore cleanup errors
            pass


@pytest.fixture
def baseline_tracker():
    """
    Create performance baseline tracker for statistical analysis.

    Uses tests/profiling/.baselines.json to persist baseline data
    across test runs. Enables adaptive performance monitoring that
    accounts for environmental variance.
    """
    baseline_file = Path(__file__).parent / ".baselines.json"
    return PerformanceBaseline(baseline_file)
