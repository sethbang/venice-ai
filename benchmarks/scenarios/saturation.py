import asyncio
import time
from benchmarks.scenarios.base import BaseScenario
from benchmarks.utils.metrics import BenchmarkResult

class SaturationScenario(BaseScenario):
    def __init__(self, duration: int = 30, rate_limit: int = 100):
        super().__init__("Saturation", duration, rate_limit)

    async def execute(self) -> BenchmarkResult:
        client = self.create_client()
        
        # Start time
        start_time = time.time()
        
        # Create a task to generate load
        tasks = []
        
        # We want to saturate the rate limit, so we'll try to send requests faster than the limit
        # E.g., 2x the rate limit
        target_rpm = self.rate_limit * 2
        interval = 60.0 / target_rpm
        
        async with client:
            while time.time() - start_time < self.duration:
                task = asyncio.create_task(self.run_request(client))
                tasks.append(task)
                
                # Wait for interval before next request
                await asyncio.sleep(interval)
            
            # Wait for all pending tasks to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                
        return self.collector.calculate_results(self.name, self.duration)