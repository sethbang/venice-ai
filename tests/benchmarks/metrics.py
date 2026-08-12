"""
Performance Metrics Collection for Scheduler Benchmarking
=========================================================

This module provides comprehensive metrics collection and analysis capabilities
for benchmarking scheduler strategies. It tracks throughput, latency, queue
performance, rate limit compliance, and concurrency metrics.

Key Components:
    * MetricsCollector: Real-time metrics collection during benchmarks
    * BenchmarkResults: Structured results with calculated percentiles and stats
    * Performance analysis utilities for throughput and latency
    * Rate limit efficiency calculations

Usage:
    >>> collector = MetricsCollector()
    >>> await collector.start_collection()
    >>> await collector.record_request_completed(metadata, latency)
    >>> results = await collector.compile_results("test", "strategy", 30.0)
"""

import asyncio
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

from venice_ai._queue_types import RequestMetadata


@dataclass
class BenchmarkResults:
    """
    Comprehensive benchmark results with calculated metrics.

    Contains all performance metrics calculated from a benchmark run,
    including throughput stats, latency percentiles, rate limit efficiency,
    and concurrency analysis.
    """

    # Basic scenario info
    scenario_name: str
    strategy_name: str
    duration: float

    # Throughput metrics
    avg_throughput: float = 0.0  # Average RPS
    peak_throughput: float = 0.0  # Peak RPS in any 1-second window
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Latency metrics (in milliseconds)
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    max_latency: float = 0.0
    mean_latency: float = 0.0
    min_latency: float = 0.0

    # Rate limiting metrics
    rate_limit_efficiency: float = 0.0  # % of theoretical max achieved
    rate_limit_violations: int = 0
    theoretical_max_rps: float = 0.0
    achieved_percentage: float = 0.0

    # Concurrency metrics
    max_concurrent: int = 0
    avg_concurrent: float = 0.0
    configured_limit: int = 0

    # Queue metrics
    avg_queue_wait: float = 0.0  # Average time requests waited in queue (ms)
    max_queue_depth: int = 0

    # Error breakdown
    error_breakdown: dict[str, int] = field(default_factory=dict)

    # Timestamps
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime = field(default_factory=lambda: datetime.now(UTC))


class MetricsCollector:
    """
    Real-time metrics collector for benchmark scenarios.

    Collects detailed performance metrics during benchmark execution including
    request timing, throughput sampling, queue depths, and error tracking.
    Provides thread-safe collection with minimal performance overhead.
    """

    def __init__(self):
        """Initialize the metrics collector."""
        # Request tracking
        self.request_latencies: list[float] = []
        self.queue_wait_times: list[float] = []
        self.request_timestamps: list[float] = []
        self.completion_timestamps: list[float] = []

        # Throughput tracking (timestamp, count) pairs
        self.throughput_samples: list[tuple[float, int]] = []
        self.requests_per_second: deque = deque(maxlen=1000)  # Rolling window

        # Concurrency tracking
        self.concurrent_requests: list[tuple[float, int]] = []
        self.active_request_count = 0
        self.max_concurrent_seen = 0

        # Rate limiting
        self.rate_limit_violations = 0
        self.rate_limit_headers: list[dict[str, str]] = []

        # Success/failure tracking
        self.successful_requests = 0
        self.failed_requests = 0
        self.error_counts: dict[str, int] = defaultdict(int)

        # Queue metrics
        self.queue_depths: list[tuple[float, int]] = []
        self.queue_wait_measurements: list[float] = []

        # Collection state
        self._collecting = False
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._lock = asyncio.Lock()

        # Sampling for high-throughput scenarios
        self._sample_interval = 1.0  # 1 second
        self._last_sample_time = 0.0
        self._requests_since_sample = 0

    async def start_collection(self) -> None:
        """Start metrics collection."""
        async with self._lock:
            self._collecting = True
            self._start_time = time.time()
            self._last_sample_time = self._start_time

            # Start background sampling task
            asyncio.create_task(self._sample_metrics())

    async def stop_collection(self) -> None:
        """Stop metrics collection."""
        async with self._lock:
            self._collecting = False
            self._end_time = time.time()

    async def reset(self) -> None:
        """Reset all collected metrics."""
        async with self._lock:
            self.request_latencies.clear()
            self.queue_wait_times.clear()
            self.request_timestamps.clear()
            self.completion_timestamps.clear()
            self.throughput_samples.clear()
            self.requests_per_second.clear()
            self.concurrent_requests.clear()
            self.rate_limit_headers.clear()
            self.queue_depths.clear()
            self.queue_wait_measurements.clear()

            self.active_request_count = 0
            self.max_concurrent_seen = 0
            self.rate_limit_violations = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.error_counts.clear()

            self._start_time = None
            self._end_time = None
            self._requests_since_sample = 0

    async def record_request_submitted(self, metadata: RequestMetadata, submit_time: float) -> None:
        """
        Record a request submission.

        Args:
            metadata: Request metadata
            submit_time: Time when request was submitted
        """
        if not self._collecting:
            return

        async with self._lock:
            self.request_timestamps.append(submit_time)
            self.active_request_count += 1
            self.max_concurrent_seen = max(self.max_concurrent_seen, self.active_request_count)

            # Record concurrency sample
            self.concurrent_requests.append((submit_time, self.active_request_count))

    async def record_request_completed(
        self,
        metadata: RequestMetadata,
        completion_time: float,
        total_latency: float,
        queue_wait_time: float | None = None,
    ) -> None:
        """
        Record successful request completion.

        Args:
            metadata: Request metadata
            completion_time: Time when request completed
            total_latency: Total request latency in seconds
            queue_wait_time: Time spent waiting in queue (optional)
        """
        if not self._collecting:
            return

        async with self._lock:
            self.completion_timestamps.append(completion_time)
            self.request_latencies.append(total_latency * 1000)  # Convert to ms
            self.successful_requests += 1
            self.active_request_count = max(0, self.active_request_count - 1)
            self._requests_since_sample += 1

            if queue_wait_time is not None:
                self.queue_wait_times.append(queue_wait_time * 1000)  # Convert to ms

    async def record_request_failed(
        self, metadata: RequestMetadata, failure_time: float, error: str
    ) -> None:
        """
        Record request failure.

        Args:
            metadata: Request metadata
            failure_time: Time when request failed
            error: Error description
        """
        if not self._collecting:
            return

        async with self._lock:
            self.failed_requests += 1
            self.active_request_count = max(0, self.active_request_count - 1)
            self.error_counts[error] += 1

            # Check if it's a rate limit error
            if "rate limit" in error.lower() or "429" in error:
                self.rate_limit_violations += 1

    async def record_queue_state(self, queue_depth: int) -> None:
        """
        Record current queue state.

        Args:
            queue_depth: Current depth of the queue
        """
        if not self._collecting:
            return

        current_time = time.time()
        async with self._lock:
            self.queue_depths.append((current_time, queue_depth))

    async def record_rate_limit_headers(self, headers: dict[str, str]) -> None:
        """
        Record rate limit headers from API response.

        Args:
            headers: Response headers containing rate limit info
        """
        if not self._collecting:
            return

        async with self._lock:
            # Store relevant rate limit headers
            rate_limit_info = {
                k: v for k, v in headers.items() if k.lower().startswith("x-ratelimit")
            }
            if rate_limit_info:
                self.rate_limit_headers.append(rate_limit_info)

    async def _sample_metrics(self) -> None:
        """Background task to sample metrics at regular intervals."""
        while self._collecting:
            await asyncio.sleep(self._sample_interval)

            if not self._collecting:
                break

            current_time = time.time()

            async with self._lock:
                # Sample throughput
                elapsed = current_time - self._last_sample_time
                if elapsed >= self._sample_interval:
                    rps = self._requests_since_sample / elapsed
                    self.throughput_samples.append((current_time, self._requests_since_sample))
                    self.requests_per_second.append(rps)

                    self._last_sample_time = current_time
                    self._requests_since_sample = 0

    def calculate_percentile(self, values: list[float], percentile: int) -> float:
        """
        Calculate percentile value from a list of values.

        Args:
            values: List of numeric values
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value, or 0.0 if no values
        """
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def calculate_throughput_stats(self) -> tuple[float, float]:
        """
        Calculate average and peak throughput.

        Returns:
            Tuple of (average_rps, peak_rps)
        """
        if not self.throughput_samples:
            return 0.0, 0.0

        # Calculate average RPS over entire duration
        total_requests = sum(count for _, count in self.throughput_samples)
        total_duration = self.throughput_samples[-1][0] - self.throughput_samples[0][0]
        avg_rps = total_requests / total_duration if total_duration > 0 else 0.0

        # Peak RPS from samples
        peak_rps = max(self.requests_per_second) if self.requests_per_second else 0.0

        return avg_rps, peak_rps

    def calculate_concurrency_stats(self) -> tuple[int, float]:
        """
        Calculate concurrency statistics.

        Returns:
            Tuple of (max_concurrent, avg_concurrent)
        """
        if not self.concurrent_requests:
            return 0, 0.0

        max_concurrent = max(count for _, count in self.concurrent_requests)
        avg_concurrent = statistics.mean(count for _, count in self.concurrent_requests)

        return max_concurrent, avg_concurrent

    def calculate_rate_limit_efficiency(self) -> tuple[float, float]:
        """
        Calculate rate limit efficiency metrics.

        Returns:
            Tuple of (efficiency_percentage, theoretical_max_rps)
        """
        if not self.rate_limit_headers:
            return 0.0, 0.0

        # Analyze rate limit headers to determine theoretical maximum
        # This is a simplified calculation - in practice would need more sophisticated analysis
        latest_headers = self.rate_limit_headers[-1] if self.rate_limit_headers else {}

        limit_requests = latest_headers.get("x-ratelimit-limit-requests", "60")
        try:
            rpm_limit = int(limit_requests)
            theoretical_max_rps = rpm_limit / 60.0  # Convert RPM to RPS
        except (ValueError, TypeError):
            theoretical_max_rps = 1.0  # Default fallback

        # Calculate actual throughput
        avg_rps, _ = self.calculate_throughput_stats()
        efficiency = (avg_rps / theoretical_max_rps) * 100 if theoretical_max_rps > 0 else 0.0

        return efficiency, theoretical_max_rps

    async def compile_results(
        self, scenario_name: str, strategy_name: str, duration: float
    ) -> BenchmarkResults:
        """
        Compile all collected metrics into a BenchmarkResults object.

        Args:
            scenario_name: Name of the benchmark scenario
            strategy_name: Name of the strategy that was tested
            duration: Actual benchmark duration

        Returns:
            BenchmarkResults with all calculated metrics
        """
        async with self._lock:
            # Calculate throughput metrics
            avg_throughput, peak_throughput = self.calculate_throughput_stats()

            # Calculate latency percentiles
            p50_latency = self.calculate_percentile(self.request_latencies, 50)
            p95_latency = self.calculate_percentile(self.request_latencies, 95)
            p99_latency = self.calculate_percentile(self.request_latencies, 99)
            max_latency = max(self.request_latencies) if self.request_latencies else 0.0
            mean_latency = (
                statistics.mean(self.request_latencies) if self.request_latencies else 0.0
            )
            min_latency = min(self.request_latencies) if self.request_latencies else 0.0

            # Calculate concurrency stats
            max_concurrent, avg_concurrent = self.calculate_concurrency_stats()

            # Calculate rate limit efficiency
            efficiency, theoretical_max = self.calculate_rate_limit_efficiency()
            achieved_percentage = (
                (avg_throughput / theoretical_max) * 100 if theoretical_max > 0 else 0.0
            )

            # Calculate queue metrics
            avg_queue_wait = (
                statistics.mean(self.queue_wait_times) if self.queue_wait_times else 0.0
            )
            max_queue_depth = (
                max(depth for _, depth in self.queue_depths) if self.queue_depths else 0
            )

            return BenchmarkResults(
                scenario_name=scenario_name,
                strategy_name=strategy_name,
                duration=duration,
                avg_throughput=avg_throughput,
                peak_throughput=peak_throughput,
                total_requests=self.successful_requests + self.failed_requests,
                successful_requests=self.successful_requests,
                failed_requests=self.failed_requests,
                p50_latency=p50_latency,
                p95_latency=p95_latency,
                p99_latency=p99_latency,
                max_latency=max_latency,
                mean_latency=mean_latency,
                min_latency=min_latency,
                rate_limit_efficiency=efficiency,
                rate_limit_violations=self.rate_limit_violations,
                theoretical_max_rps=theoretical_max,
                achieved_percentage=achieved_percentage,
                max_concurrent=max_concurrent,
                avg_concurrent=avg_concurrent,
                configured_limit=0,  # Would be set by caller if known
                avg_queue_wait=avg_queue_wait,
                max_queue_depth=max_queue_depth,
                error_breakdown=dict(self.error_counts),
                start_time=datetime.fromtimestamp(self._start_time, tz=UTC)
                if self._start_time
                else datetime.now(UTC),
                end_time=datetime.fromtimestamp(self._end_time, tz=UTC)
                if self._end_time
                else datetime.now(UTC),
            )


class MetricsAnalyzer:
    """
    Advanced metrics analysis utilities.

    Provides additional analysis capabilities for benchmark results including
    trend analysis, performance regression detection, and comparative analysis
    between different strategies or configurations.
    """

    @staticmethod
    def analyze_latency_distribution(latencies: list[float]) -> dict[str, float]:
        """
        Analyze latency distribution characteristics.

        Args:
            latencies: List of latency values in milliseconds

        Returns:
            Dictionary with distribution statistics
        """
        if not latencies:
            return {}

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        return {
            "min": sorted_latencies[0],
            "max": sorted_latencies[-1],
            "mean": statistics.mean(sorted_latencies),
            "median": statistics.median(sorted_latencies),
            "std_dev": statistics.stdev(sorted_latencies) if n > 1 else 0.0,
            "p50": sorted_latencies[int(n * 0.5)],
            "p90": sorted_latencies[int(n * 0.9)],
            "p95": sorted_latencies[int(n * 0.95)],
            "p99": sorted_latencies[int(n * 0.99)],
            "p99_9": sorted_latencies[int(n * 0.999)] if n > 1000 else sorted_latencies[-1],
        }

    @staticmethod
    def compare_results(
        baseline: BenchmarkResults, comparison: BenchmarkResults
    ) -> dict[str, float]:
        """
        Compare two benchmark results and calculate performance differences.

        Args:
            baseline: Baseline benchmark results
            comparison: Comparison benchmark results

        Returns:
            Dictionary with percentage changes (positive = improvement)
        """

        def safe_percentage_change(baseline_val: float, comparison_val: float) -> float:
            if baseline_val == 0:
                return 0.0 if comparison_val == 0 else float("inf")
            return ((comparison_val - baseline_val) / baseline_val) * 100

        return {
            "throughput_change": safe_percentage_change(
                baseline.avg_throughput, comparison.avg_throughput
            ),
            "latency_p50_change": safe_percentage_change(
                baseline.p50_latency, comparison.p50_latency
            )
            * -1,  # Lower is better
            "latency_p95_change": safe_percentage_change(
                baseline.p95_latency, comparison.p95_latency
            )
            * -1,
            "latency_p99_change": safe_percentage_change(
                baseline.p99_latency, comparison.p99_latency
            )
            * -1,
            "efficiency_change": safe_percentage_change(
                baseline.rate_limit_efficiency, comparison.rate_limit_efficiency
            ),
            "error_rate_change": safe_percentage_change(
                baseline.failed_requests / max(baseline.total_requests, 1),
                comparison.failed_requests / max(comparison.total_requests, 1),
            )
            * -1,  # Lower error rate is better
        }

    @staticmethod
    def detect_performance_issues(results: BenchmarkResults) -> list[str]:
        """
        Detect potential performance issues from benchmark results.

        Args:
            results: Benchmark results to analyze

        Returns:
            List of detected issues/warnings
        """
        issues = []

        # High latency variance
        if results.p99_latency > results.p50_latency * 5:
            issues.append(
                f"High latency variance: P99 ({results.p99_latency:.1f}ms) is {results.p99_latency / results.p50_latency:.1f}x P50"
            )

        # Low throughput efficiency
        if results.rate_limit_efficiency < 50:
            issues.append(
                f"Low rate limit efficiency: Only {results.rate_limit_efficiency:.1f}% of theoretical maximum"
            )

        # High error rate
        error_rate = (results.failed_requests / max(results.total_requests, 1)) * 100
        if error_rate > 5:
            issues.append(f"High error rate: {error_rate:.1f}% of requests failed")

        # Rate limit violations
        if results.rate_limit_violations > 0:
            issues.append(
                f"Rate limit violations: {results.rate_limit_violations} requests exceeded limits"
            )

        # Low concurrency utilization
        if results.configured_limit > 0:
            utilization = (results.avg_concurrent / results.configured_limit) * 100
            if utilization < 30:
                issues.append(
                    f"Low concurrency utilization: Only {utilization:.1f}% of configured limit used"
                )

        return issues
