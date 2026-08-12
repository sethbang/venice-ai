"""
Unit tests to improve branch coverage for _queue_types.py.

This module targets specific uncovered branches in FailedRequestCounter
to improve overall branch coverage.
"""

from datetime import timedelta

from freezegun import freeze_time

from venice_ai._queue_types import FailedRequestCounter


class TestFailedRequestCounter:
    """Test FailedRequestCounter functionality."""

    def test_increment_within_window(self):
        """
        Test FailedRequestCounter increment behavior within the time window.

        Covers normal increment path where window hasn't expired.
        """
        counter = FailedRequestCounter(max_failures=3, window_seconds=30)

        count1 = counter.increment()
        assert count1 == 1
        assert not counter.is_limit_exceeded()

        count2 = counter.increment()
        assert count2 == 2
        assert not counter.is_limit_exceeded()

        count3 = counter.increment()
        assert count3 == 3
        assert counter.is_limit_exceeded()

        count4 = counter.increment()
        assert count4 == 4
        assert counter.is_limit_exceeded()

    def test_increment_resets_after_window_expires(self):
        """
        Test that increment resets the counter when window expires.

        Covers branch B234->235 where window expiration triggers reset.
        """
        with freeze_time("2024-01-01 00:00:00") as frozen:
            counter = FailedRequestCounter(max_failures=2, window_seconds=1)

            counter.increment()
            counter.increment()
            assert counter.count == 2
            assert counter.is_limit_exceeded()

            # Advance past the 1-second window
            frozen.tick(timedelta(seconds=1.1))

            # Increment again - should reset
            count = counter.increment()

            assert count == 1
            assert not counter.is_limit_exceeded()

    def test_is_limit_exceeded_within_window(self):
        """
        Test is_limit_exceeded when within the time window.

        Covers branch B246->247 where limit is checked against max_failures.
        """
        counter1 = FailedRequestCounter(max_failures=3)
        counter1.increment()
        counter1.increment()
        assert not counter1.is_limit_exceeded()

        counter2 = FailedRequestCounter(max_failures=2)
        counter2.increment()
        counter2.increment()
        assert counter2.is_limit_exceeded()

    def test_is_limit_exceeded_resets_after_window(self):
        """
        Test that is_limit_exceeded resets counter when window expires.

        Covers the reset logic in is_limit_exceeded method.
        """
        with freeze_time("2024-01-01 00:00:00") as frozen:
            counter = FailedRequestCounter(max_failures=2, window_seconds=1)
            counter.increment()
            counter.increment()
            assert counter.is_limit_exceeded()

            # Advance past the 1-second window
            frozen.tick(timedelta(seconds=1.1))

            assert not counter.is_limit_exceeded()
            assert counter.count == 0

    def test_custom_window_parameters(self):
        """
        Test FailedRequestCounter with custom window parameters.

        Ensures the counter properly uses custom max_failures and window_seconds.
        """
        counter = FailedRequestCounter(max_failures=5, window_seconds=60)

        for _ in range(4):
            counter.increment()

        assert counter.count == 4
        assert not counter.is_limit_exceeded()
        assert counter.max_failures == 5
        assert counter.window_seconds == 60

        counter.increment()
        assert counter.is_limit_exceeded()
