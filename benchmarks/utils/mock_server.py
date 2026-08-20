import asyncio
import json
import logging
import time
from typing import Any

from aiohttp import web

logger = logging.getLogger(__name__)


class MockVeniceServer:
    def __init__(self, host: str = "localhost", port: int = 8080, rate_limit_rpm: int = 100):
        self.host = host
        self.port = port
        self.rate_limit_rpm = rate_limit_rpm
        self.app = web.Application()
        self.setup_routes()
        self.runner: web.AppRunner | None = None

        # Rate limiting state
        self.window_size = 60.0  # 1 minute window

        # Bucket storage: key -> {count, window_start}
        # Keys can be model_id (for text) or tier_id (for shared)
        self.buckets: dict[str, dict[str, Any]] = {}

        # Configuration matching production tiers
        self.tiers = {
            "tier_high_capacity": {"rpm": 500, "tpm": 1000000},
            "tier_standard": {"rpm": 75, "tpm": 750000},
            "tier_medium": {"rpm": 50, "tpm": 750000},
            "tier_heavy": {"rpm": 20, "tpm": 500000},
            "tier_image": {"rpm": 20, "tpm": 0},  # Shared
            "tier_tts": {"rpm": 60, "tpm": 0},  # Shared
        }

        self.model_map = {
            # High Capacity
            "qwen3-4b": "tier_high_capacity",
            "llama-3.2-3b": "tier_high_capacity",
            # Standard
            "venice-uncensored": "tier_standard",
            "mistral-31-24b": "tier_standard",
            "benchmark-model": "tier_standard",  # For benchmark testing
            # Medium
            "llama-3.3-70b": "tier_medium",
            "qwen3-next-80b": "tier_medium",
            # Heavy
            "qwen3-235b": "tier_heavy",
            "hermes-3-llama-3.1-405b": "tier_heavy",
            "deepseek-ai-DeepSeek-R1": "tier_heavy",
            # Image / Special
            "venice-sd35": "tier_image",
            "tts-kokoro": "tier_tts",
        }

        # Load templates
        try:
            with open("benchmarks/data/response_templates.json") as f:
                self.templates = json.load(f)
        except FileNotFoundError:
            logger.warning("Response templates not found, using defaults")
            self.templates = {}

    def setup_routes(self):
        self.app.router.add_post("/api/v1/chat/completions", self.handle_chat)
        self.app.router.add_post("/api/v1/image/generate", self.handle_image)
        self.app.router.add_get("/api/v1/models", self.handle_models)
        self.app.router.add_get("/api/v1/api_keys/rate_limits", self.handle_rate_limits)

    def _get_bucket_key(self, model_id: str) -> str:
        tier_name = self.model_map.get(model_id, "tier_standard")
        tier_config = self.tiers.get(tier_name, self.tiers["tier_standard"])

        # Hybrid Logic:
        # If TPM > 0, it's a text model -> Independent Bucket (Key = model_id)
        # If TPM == 0, it's non-text -> Shared Bucket (Key = tier_name)
        if tier_config["tpm"] > 0:
            return model_id
        else:
            return tier_name

    def _check_rate_limit(self, model_id: str, tokens_used: int = 100) -> dict[str, str]:
        """
        Check rate limits and return all 6 rate limit headers.

        The distributed rate limiting system expects:
        - x-ratelimit-reset-requests: Absolute Unix timestamp
        - x-ratelimit-reset-tokens: Relative seconds (NOT timestamp!)

        Args:
            model_id: The model being used
            tokens_used: Tokens consumed by this request (default 100)

        Returns:
            Dict with all 6 rate limit headers
        """
        bucket_key = self._get_bucket_key(model_id)
        tier_name = self.model_map.get(model_id, "tier_standard")
        tier_config = self.tiers.get(tier_name, self.tiers["tier_standard"])
        rpm_limit = tier_config["rpm"]
        tpm_limit = tier_config["tpm"]

        now = time.time()

        if bucket_key not in self.buckets:
            self.buckets[bucket_key] = {"req_count": 0, "tok_count": 0, "window_start": now}

        bucket = self.buckets[bucket_key]

        # Reset window if expired
        if now - bucket["window_start"] > self.window_size:
            bucket["req_count"] = 0
            bucket["tok_count"] = 0
            bucket["window_start"] = now

        bucket["req_count"] += 1
        bucket["tok_count"] += tokens_used

        # Calculate remaining capacity
        remaining_requests = max(0, rpm_limit - bucket["req_count"])
        remaining_tokens = max(0, tpm_limit - bucket["tok_count"])

        # Calculate reset times
        # x-ratelimit-reset-requests: Absolute Unix timestamp
        reset_requests = int(bucket["window_start"] + self.window_size)

        # x-ratelimit-reset-tokens: Relative seconds until reset (NOT absolute!)
        # This is a critical difference from the request reset header
        reset_tokens_relative = max(0, int(self.window_size - (now - bucket["window_start"])))

        headers = {
            # Request-based limits
            "x-ratelimit-limit-requests": str(rpm_limit),
            "x-ratelimit-remaining-requests": str(remaining_requests),
            "x-ratelimit-reset-requests": str(reset_requests),  # Absolute timestamp
            # Token-based limits
            "x-ratelimit-limit-tokens": str(tpm_limit),
            "x-ratelimit-remaining-tokens": str(remaining_tokens),
            "x-ratelimit-reset-tokens": str(reset_tokens_relative),  # Relative seconds!
        }

        return headers

    async def handle_chat(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            model = data.get("model", "venice-uncensored")
            # Extract max_tokens for token accounting (default to 100 if not specified)
            max_tokens = data.get("max_tokens", 100)
        except Exception:
            model = "venice-uncensored"
            max_tokens = 100

        # Use max_tokens for token accounting (simulates actual token usage)
        # In reality, the response would have fewer tokens, but for benchmarking
        # we assume the full max_tokens is used
        tokens_used = max_tokens

        headers = self._check_rate_limit(model, tokens_used)

        if int(headers["x-ratelimit-remaining-requests"]) <= 0:
            return web.Response(
                status=429,
                headers=headers,
                text=json.dumps({"error": "Rate limit exceeded"}),
                content_type="application/json",
            )

        # Simulate processing delay
        await asyncio.sleep(0.05)

        # Build response with usage info (needed for streaming token accounting)
        template = self.templates.get("chat/completions", {}).get("body", {})
        if not template:
            template = {
                "id": "chatcmpl-benchmark",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Mock response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": tokens_used,
                    "total_tokens": 10 + tokens_used,
                },
            }
        return web.Response(
            status=200, headers=headers, text=json.dumps(template), content_type="application/json"
        )

    async def handle_image(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            model = data.get("model", "venice-sd35")
        except Exception:
            model = "venice-sd35"

        # Image generation typically consumes 0 tokens in the hybrid model
        # (it's purely request-limited)
        headers = self._check_rate_limit(model, tokens_used=0)

        if int(headers["x-ratelimit-remaining-requests"]) <= 0:
            return web.Response(
                status=429,
                headers=headers,
                text=json.dumps({"error": "Rate limit exceeded"}),
                content_type="application/json",
            )

        await asyncio.sleep(0.1)

        # Simple mock response for image
        return web.Response(
            status=200,
            headers=headers,
            text=json.dumps({"data": [{"url": "http://mock.url/image.png"}]}),
            content_type="application/json",
        )

    async def handle_models(self, request: web.Request) -> web.Response:
        # Models endpoint uses a default bucket
        headers = self._check_rate_limit("venice-uncensored")

        template = self.templates.get("models", {}).get("body", {})
        return web.Response(
            status=200, headers=headers, text=json.dumps(template), content_type="application/json"
        )

    async def handle_rate_limits(self, request: web.Request) -> web.Response:
        # Return rate limit info for TierDiscovery
        rate_limits_list = []

        for model_id, tier_name in self.model_map.items():
            tier_config = self.tiers.get(tier_name)
            if not tier_config:
                continue

            limits = [
                {"type": "RPM", "amount": tier_config["rpm"]},
                {"type": "RPD", "amount": tier_config["rpm"] * 1440},  # Mock RPD
            ]

            if tier_config["tpm"] > 0:
                limits.append({"type": "TPM", "amount": tier_config["tpm"]})

            rate_limits_list.append({"apiModelId": model_id, "rateLimits": limits})

        response_data = {"data": {"rateLimits": rate_limits_list}}
        return web.Response(
            status=200, text=json.dumps(response_data), content_type="application/json"
        )

    async def start(self):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        logger.info(f"Mock Venice API server started at http://{self.host}:{self.port}")

    async def stop(self):
        if self.runner:
            await self.runner.cleanup()
            logger.info("Mock Venice API server stopped")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    server = MockVeniceServer()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(server.start())
        loop.run_forever()
    except KeyboardInterrupt:
        loop.run_until_complete(server.stop())
