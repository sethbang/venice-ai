"""TDD: SimpleRateLimiter._parse_reset_time aligns to the canonical ms-epoch policy.

simple.py used a `>1e11` millisecond threshold, while the live-verified canonical
policy (utils.parsing.ms_epoch_to_seconds, mirrored by VeniceBaseModel._ms_to_seconds)
uses `>=1e12`. Align absolute-epoch handling to the canonical threshold while
preserving relative delta-seconds handling.
"""

import time

import pytest

from venice_ai.rate_limiting.simple import SimpleRateLimiter
from venice_ai.utils.parsing import ms_epoch_to_seconds


@pytest.fixture
def limiter() -> SimpleRateLimiter:
    return SimpleRateLimiter()


@pytest.mark.parametrize(
    "raw",
    ["1780580108941", "1700000000000", "1780580108", "1700000000"],
)
def test_absolute_epoch_matches_canonical(limiter, raw):
    assert limiter._parse_reset_time(raw) == ms_epoch_to_seconds(float(raw))


def test_boundary_zone_follows_canonical(limiter):
    # 5e11 is in [1e11, 1e12): the old heuristic divided by 1000 (treated as ms);
    # the canonical >=1e12 rule keeps it as seconds. Must follow canonical now.
    assert limiter._parse_reset_time("500000000000") == ms_epoch_to_seconds(500000000000.0)


def test_relative_delta_preserved(limiter):
    # Small values are relative deltas (reset in N seconds), not absolute epochs.
    assert abs(limiter._parse_reset_time("30") - (time.time() + 30.0)) < 2.0
