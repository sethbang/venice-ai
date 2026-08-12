import asyncio
import logging
import argparse
import json
import os
from typing import List
import redis.asyncio as redis
from benchmarks.utils.mock_server import MockVeniceServer
from benchmarks.scenarios.saturation import SaturationScenario
from benchmarks.scenarios.shared_state import SharedStateScenario
from benchmarks.config import MOCK_API_HOST, MOCK_API_PORT, DEFAULT_RATE_LIMIT, DEFAULT_DURATION, REDIS_URL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("BenchmarkRunner")

async def cleanup_redis_state():
    """Clean up Redis state from previous runs to prevent state pollution."""
    try:
        # Use async Redis client for cleanup
        r = await redis.from_url(REDIS_URL, decode_responses=True)
        
        # Cleanup patterns for DistributedRedisBackend (account_id="benchmark")
        # "benchmark" base64 encoded is "YmVuY2htYXJr"
        patterns = [
            "venice:unified:benchmark:*",       # Old backend keys
            "venice:{YmVuY2htYXJr|*",           # New backend hash-tagged keys
            "venice:YmVuY2htYXJr:*",            # New backend account-scoped keys
        ]
        
        all_keys = []
        for pattern in patterns:
            found = await r.keys(pattern) # type: ignore
            if found:
                all_keys.extend(found)
        
        if all_keys:
            logger.info(f"Found {len(all_keys)} keys to cleanup")
            deleted_count = await r.delete(*all_keys) # type: ignore
            logger.info(f"Cleaned up {deleted_count} Redis keys from previous benchmark runs")
            
            # Verify deletion
            remaining = []
            for pattern in patterns:
                found = await r.keys(pattern) # type: ignore
                if found:
                    remaining.extend(found)
            
            if remaining:
                logger.warning(f"⚠️ Failed to delete some keys: {remaining}")
            else:
                logger.info("✅ Verification successful: All benchmark keys deleted")
        else:
            logger.info("No previous benchmark keys found to cleanup")
        await r.aclose()
    except Exception as e:
        logger.warning(f"Failed to clean Redis state: {e}. Continuing anyway...")

async def run_benchmarks(scenarios: List[str], duration: int, rate_limit: int):
    # Clean up Redis state from previous runs
    await cleanup_redis_state()
    
    # Start Mock Server
    server = MockVeniceServer(host=MOCK_API_HOST, port=MOCK_API_PORT, rate_limit_rpm=rate_limit)
    await server.start()
    
    results = []
    
    try:
        if "saturation" in scenarios or "all" in scenarios:
            logger.info(f"Running Saturation Scenario (Duration: {duration}s, Limit: {rate_limit} RPM)")
            scenario = SaturationScenario(duration=duration, rate_limit=rate_limit)
            result = await scenario.execute()
            results.append(result)
            logger.info(f"Saturation Result: {result.throughput_rpm:.2f} RPM, Avg Latency: {result.avg_latency*1000:.2f}ms")

        if "shared_state" in scenarios or "all" in scenarios:
            logger.info(f"Running Shared State Scenario (Duration: {duration}s, Limit: {rate_limit} RPM)")
            scenario = SharedStateScenario(duration=duration, rate_limit=rate_limit)
            result = await scenario.execute()
            results.append(result)
            logger.info(f"Shared State Result: {result.throughput_rpm:.2f} RPM, Avg Latency: {result.avg_latency*1000:.2f}ms")
            
    finally:
        await server.stop()
        
    # Generate Report
    report_data = []
    for r in results:
        report_data.append({
            "scenario": r.scenario_name,
            "duration": r.duration,
            "total_requests": r.total_requests,
            "successful": r.successful_requests,
            "failed": r.failed_requests,
            "rate_limited": r.rate_limited_requests,
            "throughput_rpm": r.throughput_rpm,
            "avg_latency_ms": r.avg_latency * 1000,
            "p95_latency_ms": r.p95_latency * 1000,
            "efficiency_percent": r.efficiency
        })
        
    os.makedirs("benchmarks/reports", exist_ok=True)
    with open("benchmarks/reports/latest.json", "w") as f:
        json.dump(report_data, f, indent=2)
        
    logger.info("Benchmark complete. Report saved to benchmarks/reports/latest.json")

def main():
    parser = argparse.ArgumentParser(description="Venice AI Benchmark Runner")
    parser.add_argument("--scenarios", nargs="+", default=["all"], choices=["all", "saturation", "shared_state"], help="Scenarios to run")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION, help="Duration of each scenario in seconds")
    parser.add_argument("--rate-limit", type=int, default=DEFAULT_RATE_LIMIT, help="Rate limit in RPM")
    
    args = parser.parse_args()
    
    asyncio.run(run_benchmarks(args.scenarios, args.duration, args.rate_limit))

if __name__ == "__main__":
    main()