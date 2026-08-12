"""
Benchmark Scenarios for Scheduler Strategy Testing
=================================================

This module provides benchmark scenarios and execution framework for testing
scheduler strategies under various load patterns. It supports multiple request
patterns, configurable duration and warmup periods, and comprehensive metrics
collection.

Key Components:
    * BenchmarkScenario: Configuration dataclass for test scenarios
    * ScenarioRunner: Executes benchmark scenarios with metrics collection
    * Load pattern generators: steady, burst, ramp, oscillating patterns
    * Optional mock API injection for cost-effective testing

Usage:
    >>> scenario = BenchmarkScenario(
    ...     name="steady_load_test",
    ...     duration_seconds=30,
    ...     request_pattern="steady",
    ...     target_rps=100
    ... )
    >>> runner = ScenarioRunner(mock_api)
    >>> results = await runner.run_scenario(strategy, scenario)
"""

import asyncio
import math
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

# Forward references for type hints
from typing import TYPE_CHECKING, Any

from venice_ai._queue_types import RequestMetadata, ResourceType

if TYPE_CHECKING:
    from tests.benchmarks.metrics import BenchmarkResults, MetricsCollector


class RequestPattern(Enum):
    """Available request load patterns for benchmark scenarios."""

    STEADY = "steady"
    BURST = "burst"
    RAMP = "ramp"
    OSCILLATING = "oscillating"


@dataclass
class BenchmarkScenario:
    """
    Configuration for a benchmark scenario.

    Defines the parameters for a complete benchmark test including duration,
    load patterns, model configuration, and concurrency settings.

    Attributes:
        name: Descriptive name for the scenario
        duration_seconds: How long to run the main benchmark (excluding warmup)
        warmup_seconds: Warmup period to stabilize before collecting metrics
        request_pattern: Type of load pattern to generate
        target_rps: Target requests per second (may vary based on pattern)
        models: List of model IDs to use (random selection if multiple)
        concurrency_limit: Maximum concurrent requests to allow
        burst_config: Configuration for burst patterns
        ramp_config: Configuration for ramp patterns
        oscillation_config: Configuration for oscillating patterns
    """

    name: str
    duration_seconds: int = 30
    warmup_seconds: int = 5
    request_pattern: RequestPattern | str = RequestPattern.STEADY
    target_rps: int = 100
    models: list[str] = field(default_factory=lambda: ["test-model"])
    concurrency_limit: int = 50

    # Pattern-specific configurations
    burst_config: dict[str, Any] = field(
        default_factory=lambda: {
            "burst_size": 20,
            "burst_interval": 2.0,
            "quiet_period": 1.0,
        }
    )

    ramp_config: dict[str, Any] = field(
        default_factory=lambda: {
            "start_rps": 10,
            "end_rps": 200,
            "ramp_type": "linear",  # linear, exponential
        }
    )

    oscillation_config: dict[str, Any] = field(
        default_factory=lambda: {
            "min_rps": 50,
            "max_rps": 150,
            "period_seconds": 10.0,
            "wave_type": "sine",  # sine, triangle, square
        }
    )

    def __post_init__(self):
        """Convert string pattern to enum if needed."""
        if isinstance(self.request_pattern, str):
            self.request_pattern = RequestPattern(self.request_pattern)

    @property
    def pattern_name(self) -> str:
        """Get the pattern name as a string."""
        if isinstance(self.request_pattern, RequestPattern):
            return self.request_pattern.value
        return str(self.request_pattern)


class LoadPatternGenerator:
    """
    Generates request timing patterns for different load scenarios.

    This class provides methods to generate various load patterns including
    steady state, burst, ramp, and oscillating patterns for comprehensive
    scheduler testing.
    """

    def __init__(self, scenario: BenchmarkScenario):
        """
        Initialize the load pattern generator.

        Args:
            scenario: Benchmark scenario configuration
        """
        self.scenario = scenario
        self._start_time: float | None = None

    async def generate_request_times(self) -> AsyncGenerator[float]:
        """
        Generate request timestamps based on the configured pattern.

        Yields:
            Timestamps (relative to start) when requests should be submitted
        """
        self._start_time = time.time()

        if self.scenario.request_pattern == RequestPattern.STEADY:
            async for timestamp in self._generate_steady_pattern():
                yield timestamp
        elif self.scenario.request_pattern == RequestPattern.BURST:
            async for timestamp in self._generate_burst_pattern():
                yield timestamp
        elif self.scenario.request_pattern == RequestPattern.RAMP:
            async for timestamp in self._generate_ramp_pattern():
                yield timestamp
        elif self.scenario.request_pattern == RequestPattern.OSCILLATING:
            async for timestamp in self._generate_oscillating_pattern():
                yield timestamp
        else:
            raise ValueError(f"Unknown request pattern: {self.scenario.request_pattern}")

    async def _generate_steady_pattern(self) -> AsyncGenerator[float]:
        """Generate steady-state load pattern."""
        interval = 1.0 / self.scenario.target_rps
        current_time = 0.0

        while current_time < self.scenario.duration_seconds:
            yield current_time
            current_time += interval
            await asyncio.sleep(0.001)  # Small yield to prevent blocking

    async def _generate_burst_pattern(self) -> AsyncGenerator[float]:
        """Generate bursty load pattern."""
        config = self.scenario.burst_config
        burst_size = config["burst_size"]
        burst_interval = config["burst_interval"]
        quiet_period = config["quiet_period"]

        current_time = 0.0

        while current_time < self.scenario.duration_seconds:
            # Generate burst
            for _ in range(burst_size):
                if current_time >= self.scenario.duration_seconds:
                    break
                yield current_time
                current_time += 0.01  # 10ms between requests in burst

            # Wait for next burst
            current_time += burst_interval + quiet_period
            await asyncio.sleep(0.001)

    async def _generate_ramp_pattern(self) -> AsyncGenerator[float]:
        """Generate ramping load pattern."""
        config = self.scenario.ramp_config
        start_rps = config["start_rps"]
        end_rps = config["end_rps"]
        ramp_type = config.get("ramp_type", "linear")

        current_time = 0.0

        while current_time < self.scenario.duration_seconds:
            # Calculate progress through ramp
            progress = current_time / self.scenario.duration_seconds

            if ramp_type == "exponential":
                # Exponential ramp
                current_rps = start_rps * ((end_rps / start_rps) ** progress)
            else:
                # Linear ramp (default)
                current_rps = start_rps + (end_rps - start_rps) * progress

            # Calculate interval for current RPS
            interval = 1.0 / max(current_rps, 0.1)  # Minimum 0.1 RPS

            yield current_time
            current_time += interval
            await asyncio.sleep(0.001)

    async def _generate_oscillating_pattern(self) -> AsyncGenerator[float]:
        """Generate oscillating load pattern."""
        config = self.scenario.oscillation_config
        min_rps = config["min_rps"]
        max_rps = config["max_rps"]
        period_seconds = config["period_seconds"]
        wave_type = config.get("wave_type", "sine")

        current_time = 0.0

        while current_time < self.scenario.duration_seconds:
            # Calculate phase in the oscillation
            phase = (current_time / period_seconds) * 2 * math.pi

            if wave_type == "triangle":
                # Triangle wave
                normalized = (
                    2 * abs(2 * (phase / (2 * math.pi) - math.floor(phase / (2 * math.pi) + 0.5)))
                    - 1
                )
            elif wave_type == "square":
                # Square wave
                normalized = 1 if math.sin(phase) >= 0 else -1
            else:
                # Sine wave (default)
                normalized = math.sin(phase)

            # Map to RPS range
            current_rps = min_rps + (max_rps - min_rps) * (normalized + 1) / 2

            # Calculate interval
            interval = 1.0 / max(current_rps, 0.1)

            yield current_time
            current_time += interval
            await asyncio.sleep(0.001)


class ScenarioRunner:
    """
    Executes benchmark scenarios and collects comprehensive metrics.

    This class orchestrates the execution of benchmark scenarios, including
    warmup periods, load generation, metrics collection, and result compilation.
    It supports optional mock API injection for testing and real schedulers for
    production validation.
    """

    def __init__(self, mock_api: Any | None = None):
        """
        Initialize the scenario runner.

        Args:
            mock_api: Optional mock API instance for testing
        """
        self.mock_api = mock_api
        self._active_requests: dict[str, float] = {}
        self._request_counter = 0

    async def run_scenario(
        self,
        strategy: Any,
        scenario: BenchmarkScenario,
        use_mock_api: bool = True,
    ) -> "BenchmarkResults":
        """
        Execute a complete benchmark scenario.

        Args:
            strategy: The scheduler strategy to test
            scenario: Benchmark scenario configuration
            use_mock_api: Whether to use mock API (default) or real API

        Returns:
            BenchmarkResults containing all collected metrics
        """
        from tests.benchmarks.metrics import MetricsCollector

        # Initialize metrics collector
        metrics = MetricsCollector()

        print(f"Starting benchmark scenario: {scenario.name}")
        print(f"Pattern: {scenario.pattern_name}, Duration: {scenario.duration_seconds}s")
        print(f"Target RPS: {scenario.target_rps}, Models: {scenario.models}")

        # Warmup phase
        if scenario.warmup_seconds > 0:
            print(f"Warmup phase: {scenario.warmup_seconds}s")
            await self._run_warmup(strategy, scenario)

        # Reset metrics after warmup
        await metrics.reset()

        # Main benchmark phase
        print("Starting main benchmark...")
        start_time = time.time()

        # Start metrics collection
        await metrics.start_collection()

        try:
            # Generate and submit requests according to pattern
            await self._execute_scenario(strategy, scenario, metrics)

            # Wait for all requests to complete
            await self._wait_for_completion(strategy, metrics, timeout=30.0)

        finally:
            # Stop metrics collection
            await metrics.stop_collection()

        end_time = time.time()
        actual_duration = end_time - start_time

        print(f"Benchmark completed in {actual_duration:.2f}s")

        # Compile results
        results = await self._compile_results(metrics, scenario, strategy, actual_duration)

        return results

    async def _run_warmup(self, strategy: Any, scenario: BenchmarkScenario) -> None:
        """
        Run warmup phase to stabilize scheduler state.

        Args:
            strategy: Scheduler strategy being tested
            scenario: Benchmark scenario configuration
        """
        warmup_scenario = BenchmarkScenario(
            name=f"{scenario.name}_warmup",
            duration_seconds=scenario.warmup_seconds,
            warmup_seconds=0,  # No nested warmup
            request_pattern=RequestPattern.STEADY,
            target_rps=min(scenario.target_rps // 2, 50),  # Gentle warmup
            models=scenario.models,
            concurrency_limit=scenario.concurrency_limit,
        )

        # Generate and submit warmup requests
        generator = LoadPatternGenerator(warmup_scenario)

        async for _request_time in generator.generate_request_times():
            # Create and submit request
            metadata = await self._create_request_metadata(warmup_scenario)
            await self._submit_request(strategy, metadata)

            # Don't wait for completion during warmup
            await asyncio.sleep(0.001)

    async def _execute_scenario(
        self,
        strategy: Any,
        scenario: BenchmarkScenario,
        metrics: "MetricsCollector",
    ) -> None:
        """
        Execute the main benchmark scenario.

        Args:
            strategy: Scheduler strategy being tested
            scenario: Benchmark scenario configuration
            metrics: Metrics collector for recording performance data
        """
        generator = LoadPatternGenerator(scenario)
        submitted_count = 0

        async for _request_time in generator.generate_request_times():
            # Create request metadata
            metadata = await self._create_request_metadata(scenario)

            # Record submission time
            submit_start = time.time()

            try:
                # Submit request to scheduler
                request_future = await self._submit_request(strategy, metadata)

                # Track active request
                self._active_requests[metadata.request_id] = submit_start

                # Record metrics
                await metrics.record_request_submitted(metadata, submit_start)

                submitted_count += 1

                # Create completion tracking task
                asyncio.create_task(
                    self._track_request_completion(metadata, request_future, submit_start, metrics)
                )

            except Exception as e:
                # Record submission failure
                await metrics.record_request_failed(metadata, submit_start, str(e))

            # Small yield to prevent blocking
            await asyncio.sleep(0.001)

        print(f"Submitted {submitted_count} requests")

    async def _track_request_completion(
        self,
        metadata: RequestMetadata,
        request_future: asyncio.Future,
        submit_time: float,
        metrics: "MetricsCollector",
    ) -> None:
        """
        Track individual request completion and record metrics.

        Args:
            metadata: Request metadata
            request_future: Future representing the request
            submit_time: Time when request was submitted
            metrics: Metrics collector
        """
        try:
            # Wait for request completion and capture result
            result = await request_future
            completion_time = time.time()

            # Extract headers if result is a tuple from mock API
            if isinstance(result, tuple) and len(result) == 2:
                response_data, headers = result
                # Record rate limit headers for efficiency calculation
                await metrics.record_rate_limit_headers(headers)

            # Calculate latency
            total_latency = completion_time - submit_time

            # Record successful completion
            await metrics.record_request_completed(metadata, completion_time, total_latency)

        except Exception as e:
            completion_time = time.time()

            # Record failure
            await metrics.record_request_failed(metadata, completion_time, str(e))

        finally:
            # Remove from active tracking
            self._active_requests.pop(metadata.request_id, None)

    async def _submit_request(self, strategy: Any, metadata: RequestMetadata) -> asyncio.Future:
        """
        Submit a request to the scheduler strategy.

        Args:
            strategy: Scheduler strategy
            metadata: Request metadata

        Returns:
            Future representing the request execution
        """

        # Create a mock request function that uses the mock API
        async def mock_request_func():
            if self.mock_api:
                return await self.mock_api.simulate_request(metadata)
            else:
                # For real API testing, would use actual client
                await asyncio.sleep(0.1)  # Simulate processing
                return {"mock": True}, {"x-request-id": metadata.request_id}

        # Submit to scheduler
        return await strategy.submit_request(metadata, mock_request_func)

    async def _create_request_metadata(self, scenario: BenchmarkScenario) -> RequestMetadata:
        """
        Create request metadata for the scenario.

        Args:
            scenario: Benchmark scenario configuration

        Returns:
            RequestMetadata object
        """
        self._request_counter += 1

        # Select model (round-robin if multiple)
        model_id = scenario.models[self._request_counter % len(scenario.models)]

        # Determine resource type based on model
        if "embedding" in model_id.lower():
            resource_type = ResourceType.EMBEDDING
            estimated_tokens = 50
            endpoint = "/embeddings"
        else:
            resource_type = ResourceType.LLM
            estimated_tokens = 150
            endpoint = "/chat/completions"

        return RequestMetadata(
            request_id=f"bench_{scenario.name}_{self._request_counter:06d}",
            model_id=model_id,
            resource_type=resource_type,
            estimated_tokens=estimated_tokens,
            priority=0,
            submitted_at=datetime.now(UTC),
            timeout=30.0,
            endpoint=endpoint,
            requires_model=True,
        )

    async def _wait_for_completion(
        self,
        strategy: Any,
        metrics: "MetricsCollector",
        timeout: float = 30.0,
    ) -> None:
        """
        Wait for all active requests to complete.

        Args:
            strategy: Scheduler strategy
            metrics: Metrics collector
            timeout: Maximum time to wait for completion
        """
        start_wait = time.time()

        while self._active_requests and (time.time() - start_wait) < timeout:
            active_count = len(self._active_requests)
            print(f"Waiting for {active_count} requests to complete...")

            # Record current queue state
            await metrics.record_queue_state(active_count)

            await asyncio.sleep(1.0)

        if self._active_requests:
            remaining = len(self._active_requests)
            print(f"Warning: {remaining} requests still pending after timeout")

    async def _compile_results(
        self,
        metrics: "MetricsCollector",
        scenario: BenchmarkScenario,
        strategy: Any,
        duration: float,
    ) -> "BenchmarkResults":
        """
        Compile benchmark results from collected metrics.

        Args:
            metrics: Metrics collector with recorded data
            scenario: Scenario configuration
            strategy: Strategy that was tested
            duration: Actual benchmark duration

        Returns:
            Compiled BenchmarkResults
        """

        return await metrics.compile_results(
            scenario_name=scenario.name,
            strategy_name=strategy.__class__.__name__,
            duration=duration,
        )


# Predefined scenarios for common testing patterns
STANDARD_SCENARIOS = [
    BenchmarkScenario(
        name="steady_load_low",
        duration_seconds=30,
        warmup_seconds=5,
        request_pattern=RequestPattern.STEADY,
        target_rps=50,
        models=["gpt-4"],
    ),
    BenchmarkScenario(
        name="steady_load_high",
        duration_seconds=30,
        warmup_seconds=5,
        request_pattern=RequestPattern.STEADY,
        target_rps=200,
        models=["llama-3.3-70b"],
    ),
    BenchmarkScenario(
        name="burst_pattern",
        duration_seconds=60,
        warmup_seconds=10,
        request_pattern=RequestPattern.BURST,
        target_rps=100,  # Not directly used in burst
        models=["gpt-4", "claude-3-opus"],
        burst_config={"burst_size": 30, "burst_interval": 5.0, "quiet_period": 2.0},
    ),
    BenchmarkScenario(
        name="ramp_up_load",
        duration_seconds=45,
        warmup_seconds=5,
        request_pattern=RequestPattern.RAMP,
        target_rps=100,  # Not directly used in ramp
        models=["llama-3.3-70b"],
        ramp_config={"start_rps": 10, "end_rps": 150, "ramp_type": "linear"},
    ),
    BenchmarkScenario(
        name="oscillating_load",
        duration_seconds=60,
        warmup_seconds=10,
        request_pattern=RequestPattern.OSCILLATING,
        target_rps=100,  # Not directly used in oscillation
        models=["gpt-4", "llama-3.3-70b", "text-embedding-3-small"],
        oscillation_config={
            "min_rps": 30,
            "max_rps": 120,
            "period_seconds": 15.0,
            "wave_type": "sine",
        },
    ),
    BenchmarkScenario(
        name="rate_limit_optimization",
        duration_seconds=120,
        warmup_seconds=10,
        request_pattern=RequestPattern.STEADY,
        target_rps=1000,  # Much higher than rate limits to test optimization
        models=["gpt-4"],  # Model with tight rate limits
        concurrency_limit=100,
    ),
    BenchmarkScenario(
        name="mixed_model_load",
        duration_seconds=60,
        warmup_seconds=10,
        request_pattern=RequestPattern.STEADY,
        target_rps=100,
        models=[
            "gpt-4",
            "claude-3-opus",
            "llama-3.3-70b",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ],
        concurrency_limit=75,
    ),
]
