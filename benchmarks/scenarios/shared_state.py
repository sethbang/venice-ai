import asyncio
import multiprocessing
import time

from benchmarks.scenarios.base import BaseScenario
from benchmarks.utils.metrics import BenchmarkResult


def worker_process(duration: int, rate_limit: int, target_rpm: int, queue: multiprocessing.Queue):
    """
    Worker process that runs a client and reports metrics back via queue.
    """

    # We need to run an event loop in this process
    async def run_worker():
        # Create a concrete implementation of BaseScenario for the worker
        class WorkerScenario(BaseScenario):
            async def execute(self) -> BenchmarkResult:
                return self.collector.calculate_results(self.name, self.duration)

        scenario = WorkerScenario("SharedStateWorker", duration, rate_limit)
        client = scenario.create_client()

        start_time = time.time()
        interval = 60.0 / target_rpm
        tasks = []

        async with client:
            while time.time() - start_time < duration:
                task = asyncio.create_task(scenario.run_request(client))
                tasks.append(task)
                await asyncio.sleep(interval)

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        # Send metrics back to main process
        for metric in scenario.collector.metrics:
            queue.put(metric)

        # Signal completion
        queue.put(None)

    asyncio.run(run_worker())


class SharedStateScenario(BaseScenario):
    def __init__(self, duration: int = 30, rate_limit: int = 100, num_workers: int = 4):
        super().__init__("SharedState", duration, rate_limit)
        self.num_workers = num_workers

    async def execute(self) -> BenchmarkResult:
        # Calculate target RPM per worker
        # We want total attempts to be 2x the limit to test contention
        total_target_rpm = self.rate_limit * 2
        worker_target_rpm = total_target_rpm // self.num_workers

        queue = multiprocessing.Queue()
        processes = []

        for _ in range(self.num_workers):
            p = multiprocessing.Process(
                target=worker_process,
                args=(self.duration, self.rate_limit, worker_target_rpm, queue),
                daemon=True,
            )
            processes.append(p)
            p.start()

        # Collect metrics from queue until all workers are done
        # Use non-blocking get with sleep to allow mock server to handle requests
        import queue as queue_lib

        finished_workers = 0
        while finished_workers < self.num_workers:
            try:
                metric = queue.get_nowait()
                if metric is None:
                    finished_workers += 1
                else:
                    self.collector.record(metric)
            except queue_lib.Empty:
                await asyncio.sleep(0.01)

        # Wait for processes to finish
        for p in processes:
            p.join()

        return self.collector.calculate_results(self.name, self.duration)
