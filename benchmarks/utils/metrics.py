import statistics
import time
from dataclasses import dataclass, field


@dataclass
class RequestMetric:
    timestamp: float
    duration: float
    status_code: int
    endpoint: str
    queue_time: float = 0.0
    api_time: float = 0.0


@dataclass
class BenchmarkResult:
    scenario_name: str
    duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    rate_limited_requests: int
    throughput_rpm: float
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    efficiency: float
    metrics: list[RequestMetric] = field(default_factory=list)


class MetricsCollector:
    def __init__(self):
        self.metrics: list[RequestMetric] = []
        self.start_time = time.time()

    def record(self, metric: RequestMetric):
        self.metrics.append(metric)

    def calculate_results(self, scenario_name: str, duration: float) -> BenchmarkResult:
        total = len(self.metrics)
        if total == 0:
            return BenchmarkResult(
                scenario_name=scenario_name,
                duration=duration,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                rate_limited_requests=0,
                throughput_rpm=0.0,
                avg_latency=0.0,
                p50_latency=0.0,
                p95_latency=0.0,
                p99_latency=0.0,
                efficiency=0.0,
                metrics=[],
            )

        successful = sum(1 for m in self.metrics if 200 <= m.status_code < 300)
        failed = sum(1 for m in self.metrics if m.status_code >= 500)
        rate_limited = sum(1 for m in self.metrics if m.status_code == 429)

        latencies = [m.duration for m in self.metrics]
        avg_latency = statistics.mean(latencies)
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else p50
        p99 = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else p95

        throughput_rpm = (successful / duration) * 60

        # Efficiency: percentage of attempted requests that succeeded
        efficiency = (successful / total) * 100 if total > 0 else 0

        return BenchmarkResult(
            scenario_name=scenario_name,
            duration=duration,
            total_requests=total,
            successful_requests=successful,
            failed_requests=failed,
            rate_limited_requests=rate_limited,
            throughput_rpm=throughput_rpm,
            avg_latency=avg_latency,
            p50_latency=p50,
            p95_latency=p95,
            p99_latency=p99,
            efficiency=efficiency,
            metrics=self.metrics,
        )
