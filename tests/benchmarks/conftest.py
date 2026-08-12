"""
Benchmark-specific pytest fixtures.

This module provides fixtures specifically designed for benchmarking
scheduler strategies. It includes configurations for various testing
scenarios and benchmark setup utilities.
"""

import asyncio
from datetime import UTC, datetime

import pytest

from venice_ai._queue_types import RequestMetadata, ResourceType


@pytest.fixture
def benchmark_fixture():
    """
    Setup fixture for benchmark tests.

    Provides utilities and configuration for conducting comprehensive
    benchmarks of scheduler strategies.

    Returns:
        Dictionary with benchmark utilities and configuration
    """

    class BenchmarkFixture:
        """Helper class for benchmark test setup and utilities."""

        def __init__(self):
            self.metrics = []
            self.start_time = None
            self.end_time = None

        async def create_test_requests(
            self,
            count: int,
            model_id: str = "test-model",
            resource_type: ResourceType = ResourceType.LLM,
            estimated_tokens: int = 100,
        ) -> list[RequestMetadata]:
            """Create a list of test request metadata objects."""
            requests = []
            for i in range(count):
                requests.append(
                    RequestMetadata(
                        request_id=f"bench_req_{i:06d}",
                        model_id=model_id,
                        resource_type=resource_type,
                        estimated_tokens=estimated_tokens,
                        priority=0,
                        submitted_at=datetime.now(UTC),
                        timeout=30.0,
                        endpoint="/chat/completions"
                        if resource_type == ResourceType.LLM
                        else "/embeddings",
                        requires_model=True,
                    )
                )
            return requests

        async def create_mixed_requests(self, count: int) -> list[RequestMetadata]:
            """Create a mix of different request types for comprehensive testing."""
            requests = []
            models = ["gpt-4", "llama-3.3-70b", "text-embedding-3-small"]
            resource_types = [
                ResourceType.LLM,
                ResourceType.LLM,
                ResourceType.EMBEDDING,
            ]
            token_estimates = [150, 200, 50]

            for i in range(count):
                model_idx = i % len(models)
                requests.append(
                    RequestMetadata(
                        request_id=f"mixed_req_{i:06d}",
                        model_id=models[model_idx],
                        resource_type=resource_types[model_idx],
                        estimated_tokens=token_estimates[model_idx],
                        priority=0,
                        submitted_at=datetime.now(UTC),
                        timeout=30.0,
                        endpoint="/chat/completions"
                        if resource_types[model_idx] == ResourceType.LLM
                        else "/embeddings",
                        requires_model=True,
                    )
                )
            return requests

        def start_timing(self):
            """Start timing a benchmark operation."""
            self.start_time = asyncio.get_event_loop().time()

        def stop_timing(self):
            """Stop timing and return elapsed time."""
            if self.start_time is None:
                raise RuntimeError("Timing not started")
            self.end_time = asyncio.get_event_loop().time()
            return self.end_time - self.start_time

        def record_metric(self, name: str, value: float, labels: dict[str, str] | None = None):
            """Record a benchmark metric."""
            self.metrics.append(
                {
                    "name": name,
                    "value": value,
                    "labels": labels or {},
                    "timestamp": datetime.now(UTC),
                }
            )

        def get_metrics(self, name: str | None = None) -> list[dict]:
            """Get recorded metrics, optionally filtered by name."""
            if name is None:
                return self.metrics
            return [m for m in self.metrics if m["name"] == name]

        def clear_metrics(self):
            """Clear all recorded metrics."""
            self.metrics.clear()
            self.start_time = None
            self.end_time = None

    return BenchmarkFixture()


@pytest.fixture(scope="module")
def benchmark_models() -> list[str]:
    """
    List of models to use for comprehensive benchmarking.

    Returns:
        List of model IDs representing different performance tiers
    """
    return [
        "gpt-4",  # Premium model with tight limits
        "claude-3-opus",  # Alternative premium model
        "llama-3.3-70b",  # High-performance open model
        "llama-3.2-3b",  # Fast, efficient model
        "text-embedding-3-small",  # Embedding model
        "text-embedding-3-large",  # Large embedding model
    ]


@pytest.fixture
def concurrent_load_generator():
    """
    Generator for creating concurrent load patterns.

    Provides utilities for generating various load patterns to test
    scheduler behavior under different concurrency scenarios.
    """

    class LoadGenerator:
        """Utility class for generating concurrent load patterns."""

        async def steady_load(
            self,
            rps: int,
            duration_seconds: int,
            create_request_func,
        ) -> list:
            """Generate steady load at specified RPS."""
            interval = 1.0 / rps
            requests = []
            end_time = asyncio.get_event_loop().time() + duration_seconds

            while asyncio.get_event_loop().time() < end_time:
                request = await create_request_func()
                requests.append(request)
                await asyncio.sleep(interval)

            return requests

        async def burst_load(
            self,
            burst_size: int,
            burst_interval: float,
            num_bursts: int,
            create_request_func,
        ) -> list:
            """Generate bursty load patterns."""
            requests = []

            for _ in range(num_bursts):
                # Create burst
                burst_requests = []
                for _ in range(burst_size):
                    request = await create_request_func()
                    burst_requests.append(request)

                requests.extend(burst_requests)

                # Wait between bursts
                await asyncio.sleep(burst_interval)

            return requests

        async def ramp_load(
            self,
            start_rps: int,
            end_rps: int,
            duration_seconds: int,
            create_request_func,
        ) -> list:
            """Generate ramping load from start_rps to end_rps."""
            requests = []
            start_time = asyncio.get_event_loop().time()

            while asyncio.get_event_loop().time() - start_time < duration_seconds:
                elapsed = asyncio.get_event_loop().time() - start_time
                progress = elapsed / duration_seconds
                current_rps = start_rps + (end_rps - start_rps) * progress

                interval = 1.0 / max(current_rps, 0.1)  # Minimum 0.1 RPS

                request = await create_request_func()
                requests.append(request)
                await asyncio.sleep(interval)

            return requests

    return LoadGenerator()


# Pytest markers for benchmark tests
pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.asyncio,
]
