"""Unit tests for VeniceClient.gather() — bounded-concurrency awaitable runner."""

import asyncio

import pytest

from venice_ai._client import VeniceClient


def _build_minimal_client() -> VeniceClient:
    """Build a VeniceClient bypassing __init__ — gather doesn't touch other state."""
    return VeniceClient.__new__(VeniceClient)


@pytest.mark.asyncio
async def test_gather_returns_results_in_order():
    client = _build_minimal_client()

    async def task(value: int, delay: float) -> int:
        await asyncio.sleep(delay)
        return value

    # Feed delays in reverse so order is preserved by gather, not finish time.
    results = await client.gather([task(0, 0.03), task(1, 0.02), task(2, 0.01)], max_concurrency=3)
    assert results == [0, 1, 2]


@pytest.mark.asyncio
async def test_gather_respects_max_concurrency():
    """At most max_concurrency tasks run simultaneously."""
    client = _build_minimal_client()

    in_flight = 0
    peak = 0

    async def task() -> int:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return 1

    await client.gather([task() for _ in range(10)], max_concurrency=3)
    assert peak <= 3
    assert peak >= 1


@pytest.mark.asyncio
async def test_gather_return_exceptions_default_collects():
    """With return_exceptions=True (default) failures land in result slots."""
    client = _build_minimal_client()

    async def good() -> str:
        return "ok"

    async def bad() -> str:
        raise ValueError("boom")

    results = await client.gather([good(), bad(), good()], max_concurrency=2)
    assert results[0] == "ok"
    assert isinstance(results[1], ValueError)
    assert results[2] == "ok"


@pytest.mark.asyncio
async def test_gather_return_exceptions_false_raises():
    """With return_exceptions=False the first failure propagates."""
    client = _build_minimal_client()

    async def good() -> str:
        await asyncio.sleep(0.01)
        return "ok"

    async def bad() -> str:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        await client.gather(
            [good(), bad(), good()],
            max_concurrency=2,
            return_exceptions=False,
        )


@pytest.mark.asyncio
async def test_gather_empty_iterable_returns_empty_list():
    client = _build_minimal_client()
    assert await client.gather([]) == []


@pytest.mark.asyncio
async def test_gather_rejects_zero_concurrency():
    client = _build_minimal_client()

    async def task() -> int:
        return 0

    coro = task()
    try:
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            await client.gather([coro], max_concurrency=0)
    finally:
        coro.close()


@pytest.mark.asyncio
async def test_gather_concurrency_one_serializes():
    """max_concurrency=1 runs everything sequentially."""
    client = _build_minimal_client()

    order: list[int] = []

    async def task(idx: int) -> int:
        order.append(idx)
        await asyncio.sleep(0.005)
        order.append(-idx)
        return idx

    await client.gather([task(i) for i in range(3)], max_concurrency=1)
    # Each task fully completes before the next one starts (interleaving == sequential).
    assert order == [0, 0, 1, -1, 2, -2]
