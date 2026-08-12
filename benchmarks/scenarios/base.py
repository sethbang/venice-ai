import abc
import asyncio
import time
from typing import Optional
from venice_ai import VeniceClient
from venice_ai.factory import VeniceClientFactory
from venice_ai.core.config import VeniceAIConfig, SchedulerConfig, SchedulerMode, BackendConfig, BackendType, RedisBackendConfig
from benchmarks.utils.metrics import MetricsCollector, RequestMetric, BenchmarkResult
from benchmarks.config import MOCK_API_URL, REDIS_URL

class BaseScenario(abc.ABC):
    def __init__(self, name: str, duration: int, rate_limit: int):
        self.name = name
        self.duration = duration
        self.rate_limit = rate_limit
        self.collector = MetricsCollector()
        self.should_stop = False

    def create_client(self) -> VeniceClient:
        config = VeniceAIConfig(
            api_base_url=MOCK_API_URL,
            scheduler=SchedulerConfig(
                mode=SchedulerMode.INTELLIGENT,
                max_concurrent_executions=50,
                max_queue_size=1000,
                enable_rate_limiting=True
            ),
            backend=BackendConfig(
                backend_type=BackendType.REDIS,
                redis=RedisBackendConfig(
                    redis_url=REDIS_URL,
                    key_prefix="benchmark:"
                )
            )
        )
        return VeniceClientFactory.create_client(config=config, api_key="benchmark-key", account_id="benchmark")

    async def run_request(self, client: VeniceClient, endpoint: str = "chat/completions"):
        start_time = time.time()
        status_code = 0
        try:
            # Use force_direct=False to ensure scheduler is used
            response = await client.post(
                endpoint,
                json_data={"model": "benchmark-model", "messages": [{"role": "user", "content": "test"}]},
                raw_response=True
            )
            status_code = response.status
            await response.release()
        except Exception as e:
            # Handle exceptions (e.g. timeouts, connection errors)
            # For benchmarking, we might want to log these as failures
            status_code = 500
            if hasattr(e, "status"):
                # Type ignore because we checked hasattr
                status_code = getattr(e, "status") # type: ignore
        finally:
            duration = time.time() - start_time
            self.collector.record(RequestMetric(
                timestamp=start_time,
                duration=duration,
                status_code=status_code,
                endpoint=endpoint
            ))

    @abc.abstractmethod
    async def execute(self) -> BenchmarkResult:
        pass