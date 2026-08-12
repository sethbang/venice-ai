# Venice AI Benchmark Suite

This benchmark suite validates the intelligent scheduler's rate limiting and multi-client coordination capabilities using the **DistributedRedisBackend**.

## Setup

1. **Start Redis** (required for shared state scenarios):
   ```bash
   docker run -d -p 6379:6379 redis:latest
   ```

2. **Install Dependencies**:
   ```bash
   poetry install
   ```

## Running Benchmarks

### Basic Usage

Run all scenarios:
```bash
PYTHONPATH=. poetry run python3 benchmarks/runner.py
```

Run specific scenario:
```bash
PYTHONPATH=. poetry run python3 benchmarks/runner.py --scenarios saturation
```

### Configuration Options

- `--scenarios`: Choose which scenarios to run (`all`, `saturation`, `shared_state`)
- `--duration`: Duration in seconds (default: 30)
- `--rate-limit`: Rate limit in RPM (default: 100)

### Example

```bash
PYTHONPATH=. poetry run python3 benchmarks/runner.py \
  --scenarios saturation \
  --duration 10 \
  --rate-limit 60
```

## Scenarios

### Saturation Scenario

Tests single-client throughput by sending traffic at 2x the rate limit to measure:
- Queue management
- Rate limit enforcement
- Latency under load

**Success Criteria**: Throughput ≈ configured rate limit, minimal 429 errors.

### Shared State Scenario (CRITICAL)

Tests multi-client coordination by spawning 4 worker processes, each attempting 50% of the rate limit:
- Distributed locking
- Global rate limit respect
- No cascading 429s

**Success Criteria**: Total throughput ≈ rate limit across all workers, fair distribution.

## Reports

Results are saved to `benchmarks/reports/latest.json` with metrics including:
- Throughput (RPM)
- Latency percentiles (p50, p95, p99)
- Success/failure counts
- Efficiency percentage

## Architecture

- **Mock Server**: `benchmarks/utils/mock_server.py` - Simulates Venice API with configurable rate limits
- **Scenarios**: `benchmarks/scenarios/*.py` - Test scenarios
- **Runner**: `benchmarks/runner.py` - Orchestrates execution and reporting
- **Metrics**: `benchmarks/utils/metrics.py` - Collects and aggregates performance data

## Notes

- The mock server simulates Venice API rate limit headers
- Redis is required for distributed state testing
- Scenarios use VeniceClientFactory for proper dependency injection