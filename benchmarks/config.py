import os

# Benchmark Configuration
MOCK_API_HOST = os.getenv("BENCHMARK_MOCK_API_HOST", "localhost")
MOCK_API_PORT = int(os.getenv("BENCHMARK_MOCK_API_PORT", "8080"))
MOCK_API_URL = f"http://{MOCK_API_HOST}:{MOCK_API_PORT}"

REDIS_URL = os.getenv("BENCHMARK_REDIS_URL", "redis://localhost:6379/0")

# Default limits for scenarios
DEFAULT_RATE_LIMIT = 100  # RPM
DEFAULT_DURATION = 30  # Seconds
