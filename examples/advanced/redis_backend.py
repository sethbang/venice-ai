#!/usr/bin/env python3
"""
Venice AI SDK - Redis Backend Configuration
============================================

Demonstrates the *correct* wiring for a Redis-backed Venice client and
self-verifies that Redis is actually contacted on the wire.

Key correctness note
--------------------
``BackendConfig(backend_type=BackendType.REDIS, ...)`` alone is **not** enough
to make the SDK use Redis at runtime: it also requires
``RateLimiterConfig(mode=RateLimiterMode.ADAPTIVE, redis_url=...)``. Without
ADAPTIVE mode the SDK falls back to the in-memory SimpleRateLimiter and Redis
is never contacted. The config validator now flags this misuse as an ERROR
(see ``venice_ai.validation.config_validator``); the production preset
already wires both pieces correctly and is the canonical entrypoint.

Run
---
Start Redis (any reachable instance works; localhost shown for demo)::

    docker run -d --name venice-redis -p 6379:6379 redis:7-alpine

Then::

    poetry run python examples/advanced/redis_backend.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, cast

import redis
from redis.exceptions import RedisError

# Line-buffer stdout so our prints show up in chronological order with any
# stderr output the package may emit (rather than appearing in a chunk at
# the end after stderr already flushed).
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

from venice_ai.factory import VeniceClientFactory  # noqa: E402
from venice_ai.presets import create_production_config  # noqa: E402
from venice_ai.types.api import UserMessage  # noqa: E402
from venice_ai.validation.config_validator import validate_config  # noqa: E402


def assert_redis_reachable(redis_url: str) -> redis.Redis:
    """Ping Redis up-front; exit with a clear message if unreachable.

    Returns a sync ``redis.Redis`` client we'll later use to verify the SDK
    actually wrote keys.
    """
    print(f"Checking Redis connectivity at {redis_url} ...")
    client = redis.Redis.from_url(redis_url, socket_connect_timeout=2.0)
    try:
        client.ping()
    except RedisError as exc:
        print(
            f"\nERROR: cannot reach Redis at {redis_url}: {exc}\n"
            "\nStart a local Redis (Docker):\n"
            "    docker run -d --name venice-redis -p 6379:6379 redis:7-alpine\n"
            "\nOr point VENICE_REDIS_URL / REDIS_URL at a reachable instance.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"   Redis reachable. Initial DBSIZE = {cast(int, client.dbsize())}")
    return client


def _redis_command_total(verifier: redis.Redis) -> int:
    """Sum non-INFO/PING commands processed by the server.

    DBSIZE doesn't catch the case where Redis was contacted heavily
    (HGETALL, EVAL, SCRIPT LOAD, ...) but no keys persisted (e.g. a
    failed HSET). ``INFO commandstats`` counts every command the server
    has executed and is the ground truth for "did anyone talk to Redis".
    """
    stats = cast(dict[str, Any], verifier.info("commandstats"))
    total = 0
    for cmd, info in stats.items():
        # We don't want to count our own ping/info/dbsize/scan probes
        # against the SDK; skip the commands the verifier itself uses.
        bare = cmd.replace("cmdstat_", "")
        if bare in {"info", "ping", "dbsize", "scan", "client|setinfo", "client"}:
            continue
        total += int(info.get("calls", 0))
    return total


async def run_redis_backed_request(redis_url: str) -> None:
    """Build a Redis-backed config, issue one request, then verify the SDK
    actually contacted Redis."""

    print("\nBuilding production config (BackendType.REDIS + RateLimiterMode.ADAPTIVE)")
    # The production preset is the canonical way to get this right. It wires:
    #   BackendConfig(backend_type=BackendType.REDIS, redis=RedisBackendConfig(...))
    # AND
    #   RateLimiterConfig(mode=RateLimiterMode.ADAPTIVE, redis_url=redis_url)
    # Both are required: the validator now errors if BackendType.REDIS is set
    # without ADAPTIVE rate limiting.
    config = create_production_config(
        redis_url=redis_url,
        redis_key_prefix="venice:example:",
        max_concurrent_executions=10,
        max_queue_size=100,
        # Localhost is fine for a local demo but rejected by default in
        # production mode.
        _allow_localhost_for_testing=True,
    )

    # Run the explicit config validator. With BOTH pieces wired we expect zero
    # errors. The validator rejects configs that set BackendType.REDIS without
    # RateLimiterMode.ADAPTIVE, since that combination silently falls back to the
    # in-memory rate limiter (Redis is never contacted).
    validation = validate_config(config)
    print(
        f"   Config validation: errors={len(validation.errors)} warnings={len(validation.warnings)}"
    )
    if validation.errors:
        for err in validation.errors:
            print(f"   ERROR: {err}", file=sys.stderr)
        sys.exit(1)

    # Snapshot Redis state BEFORE we exercise the SDK so we can diff after.
    verifier = redis.Redis.from_url(redis_url)
    pre_dbsize = cast(int, verifier.dbsize())
    pre_cmds = _redis_command_total(verifier)
    print(f"   Pre-request:  DBSIZE={pre_dbsize}, total SDK commands={pre_cmds}")

    # Build the SDK client. account_id matters for ADAPTIVE mode key scoping.
    client = VeniceClientFactory.create_client(
        config=config,
        api_key=os.environ["VENICE_API_KEY"],
        account_id="redis-example",
    )

    chat_response_text: str | None = None
    async with client:
        chat_model = await client.models.resolve_chat()
        print(f"\nIssuing chat completion via {chat_model} ...")
        response = await client.chat.completions.create(
            model=chat_model,
            messages=[UserMessage(content="Reply with exactly one word: OK")],
            max_completion_tokens=10,
        )
        raw_content = response.text
        chat_response_text = raw_content if isinstance(raw_content, str) else None
        print(f"   Response: {chat_response_text!r}")

    # Verification: did the SDK actually talk to Redis?
    post_dbsize = cast(int, verifier.dbsize())
    post_cmds = _redis_command_total(verifier)
    dbsize_delta = post_dbsize - pre_dbsize
    cmd_delta = post_cmds - pre_cmds
    print(
        f"\nPost-request: DBSIZE={post_dbsize} (delta {dbsize_delta:+d}), "
        f"SDK commands delta={cmd_delta:+d}"
    )

    # Sample up to 10 keys for visibility.
    sample_keys = sorted({k.decode() for k in verifier.scan_iter(count=100)})[:10]
    if sample_keys:
        print("   Sample keys in Redis:")
        for key in sample_keys:
            print(f"     - {key}")
    else:
        print("   (no keys present in Redis)")

    # Acceptance: prefer DBSIZE > 0 (most explicit), but accept "command
    # delta > 0" as proof Redis was contacted, since some SDK paths may
    # only read existing keys or fail to persist.
    if dbsize_delta > 0:
        print(f"\nSUCCESS: Redis was contacted (DBSIZE +{dbsize_delta}, commands +{cmd_delta}).")
        return
    if cmd_delta > 0:
        print(
            f"\nPARTIAL SUCCESS: Redis was contacted ({cmd_delta} commands "
            "executed by the SDK), but no keys persisted. This usually means "
            "the adaptive rate limiter's state writes failed; the wiring is "
            "correct (BackendType.REDIS + RateLimiterMode.ADAPTIVE)."
        )
        return

    print(
        "\nFAIL: Redis received zero commands from the SDK during the run.\n"
        "      The configuration is not actually using Redis — the same\n"
        "      silent fallback that BackendType.REDIS without\n"
        "      RateLimiterMode.ADAPTIVE produces. Confirm both are\n"
        "      set in your VeniceAIConfig.",
        file=sys.stderr,
    )
    sys.exit(1)


async def main() -> None:
    print("=" * 60)
    print("Venice AI SDK - Redis Backend Example")
    print("=" * 60)

    redis_url = os.getenv("VENICE_REDIS_URL") or os.getenv("REDIS_URL", "redis://localhost:6379")

    # 1. Defensive: confirm Redis is actually reachable before we configure
    #    the SDK against it. Otherwise failures get buried in async stacks.
    assert_redis_reachable(redis_url)

    # 2. Demonstrate the correct config + verify Redis is touched on the wire.
    await run_redis_backed_request(redis_url)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    if not os.getenv("VENICE_API_KEY"):
        print(
            "ERROR: VENICE_API_KEY is not set. Export it (e.g. via .env) and rerun.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
