"""
Property-Based Tests for Backoff Delay Calculations

Tests exponential backoff algorithm invariants using Hypothesis to generate
a wide range of inputs and verify mathematical properties hold.
"""

from hypothesis import assume, given
from hypothesis import strategies as st

from venice_ai.middleware.retry import calculate_backoff_delay


class TestBackoffDelayProperties:
    """Property-based tests for exponential backoff calculations."""

    @given(
        attempt=st.integers(min_value=0, max_value=20),
        base_delay=st.floats(min_value=0.1, max_value=10.0),
        exponential_base=st.floats(min_value=1.1, max_value=5.0),
        max_delay=st.floats(min_value=10.0, max_value=300.0),
    )
    def test_delay_never_exceeds_maximum(
        self,
        attempt: int,
        base_delay: float,
        exponential_base: float,
        max_delay: float,
    ):
        """
        Property: Calculated delay should never exceed max_delay, regardless of
        attempt number or other parameters.

        This is a critical safety property - we never want unbounded delays.
        """
        delay = calculate_backoff_delay(
            attempt=attempt,
            base_delay=base_delay,
            exponential_base=exponential_base,
            max_delay=max_delay,
            jitter_factor=0.0,  # No jitter for this test
        )

        assert delay <= max_delay, (
            f"Delay {delay}s exceeded max_delay {max_delay}s for attempt {attempt}"
        )

    @given(
        attempt=st.integers(min_value=0, max_value=20),
        base_delay=st.floats(min_value=0.1, max_value=10.0),
        exponential_base=st.floats(min_value=1.1, max_value=5.0),
        max_delay=st.floats(min_value=10.0, max_value=300.0),
        jitter_factor=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_delay_is_never_negative(
        self,
        attempt: int,
        base_delay: float,
        exponential_base: float,
        max_delay: float,
        jitter_factor: float,
    ):
        """
        Property: Delay must always be non-negative, even with maximum jitter.

        Negative delays would cause errors in asyncio.sleep() and other timing
        mechanisms, so this is a critical correctness property.
        """
        delay = calculate_backoff_delay(
            attempt=attempt,
            base_delay=base_delay,
            exponential_base=exponential_base,
            max_delay=max_delay,
            jitter_factor=jitter_factor,
        )

        assert delay >= 0, f"Delay {delay}s is negative"

    @given(
        base_delay=st.floats(min_value=0.1, max_value=10.0),
        exponential_base=st.floats(min_value=1.1, max_value=5.0),
        max_delay=st.floats(min_value=10.0, max_value=300.0),
    )
    def test_first_attempt_equals_base_delay_without_jitter(
        self,
        base_delay: float,
        exponential_base: float,
        max_delay: float,
    ):
        """
        Property: First retry (attempt=0) should equal base_delay when jitter=0.

        This ensures the exponential function is correctly implemented:
        base_delay * (exponential_base^0) = base_delay * 1 = base_delay
        """
        # Ensure base_delay doesn't exceed max_delay
        assume(base_delay <= max_delay)

        delay = calculate_backoff_delay(
            attempt=0,
            base_delay=base_delay,
            exponential_base=exponential_base,
            max_delay=max_delay,
            jitter_factor=0.0,
        )

        assert abs(delay - base_delay) < 0.001, (
            f"First attempt delay {delay}s should equal base_delay {base_delay}s"
        )

    @given(
        base_delay=st.floats(min_value=0.1, max_value=5.0),
        exponential_base=st.floats(min_value=1.5, max_value=3.0),
        max_delay=st.floats(min_value=100.0, max_value=300.0),
    )
    def test_delays_increase_monotonically_without_jitter(
        self,
        base_delay: float,
        exponential_base: float,
        max_delay: float,
    ):
        """
        Property: Without jitter, delays should increase monotonically with
        each attempt until hitting max_delay.

        This verifies the exponential growth behavior.
        """
        delays = [
            calculate_backoff_delay(
                attempt=i,
                base_delay=base_delay,
                exponential_base=exponential_base,
                max_delay=max_delay,
                jitter_factor=0.0,
            )
            for i in range(10)
        ]

        for i in range(len(delays) - 1):
            # Each delay should be >= previous (monotonically non-decreasing)
            # They can be equal if we hit max_delay
            assert delays[i + 1] >= delays[i], (
                f"Delay decreased: delays[{i}]={delays[i]}, delays[{i + 1}]={delays[i + 1]}"
            )

    @given(
        attempt=st.integers(min_value=0, max_value=20),
        base_delay=st.floats(min_value=0.1, max_value=10.0),
        exponential_base=st.floats(min_value=1.1, max_value=5.0),
        max_delay=st.floats(min_value=10.0, max_value=300.0),
        jitter_factor=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_jitter_stays_within_bounds(
        self,
        attempt: int,
        base_delay: float,
        exponential_base: float,
        max_delay: float,
        jitter_factor: float,
    ):
        """
        Property: With jitter, delay should stay within reasonable bounds:
        [0, max_delay * (1 + jitter_factor)]

        The delay can exceed max_delay slightly due to jitter, but should be
        bounded by max_delay plus the maximum possible jitter.
        """
        delay = calculate_backoff_delay(
            attempt=attempt,
            base_delay=base_delay,
            exponential_base=exponential_base,
            max_delay=max_delay,
            jitter_factor=jitter_factor,
        )

        # Upper bound: max_delay + max possible jitter
        max_possible = max_delay * (1 + jitter_factor)

        assert 0 <= delay <= max_possible, (
            f"Delay {delay}s outside bounds [0, {max_possible}] with jitter_factor={jitter_factor}"
        )

    @given(
        attempt=st.integers(min_value=1, max_value=10),
        base_delay=st.floats(min_value=0.1, max_value=5.0),
        exponential_base=st.floats(min_value=2.0, max_value=2.0),  # Fixed at 2.0
        max_delay=st.floats(min_value=100.0, max_value=300.0),
    )
    def test_exponential_growth_with_base_2(
        self,
        attempt: int,
        base_delay: float,
        exponential_base: float,
        max_delay: float,
    ):
        """
        Property: With exponential_base=2.0, each attempt should roughly double
        the delay (until hitting max_delay).

        This tests the exponential formula: delay = base * (2^attempt)
        """
        delay_current = calculate_backoff_delay(
            attempt=attempt,
            base_delay=base_delay,
            exponential_base=2.0,
            max_delay=max_delay,
            jitter_factor=0.0,
        )

        delay_previous = calculate_backoff_delay(
            attempt=attempt - 1,
            base_delay=base_delay,
            exponential_base=2.0,
            max_delay=max_delay,
            jitter_factor=0.0,
        )

        # If neither hit the cap, current should be ~2x previous
        if delay_current < max_delay and delay_previous < max_delay:
            ratio = delay_current / delay_previous
            assert 1.9 <= ratio <= 2.1, (
                f"Expected ~2x increase, got {ratio}x ({delay_previous}s -> {delay_current}s)"
            )

    @given(
        attempt=st.integers(min_value=0, max_value=20),
        base_delay=st.floats(min_value=0.1, max_value=10.0),
        exponential_base=st.floats(min_value=1.1, max_value=5.0),
        max_delay=st.floats(min_value=10.0, max_value=300.0),
        jitter_factor=st.floats(min_value=0.1, max_value=0.5),
    )
    def test_jitter_provides_variation(
        self,
        attempt: int,
        base_delay: float,
        exponential_base: float,
        max_delay: float,
        jitter_factor: float,
    ):
        """
        Property: With jitter > 0, multiple calls with same parameters should
        produce different results (with very high probability).

        This verifies that jitter is actually being applied.
        """
        # Generate 10 delays with same parameters
        delays = [
            calculate_backoff_delay(
                attempt=attempt,
                base_delay=base_delay,
                exponential_base=exponential_base,
                max_delay=max_delay,
                jitter_factor=jitter_factor,
            )
            for _ in range(10)
        ]

        # With jitter, we should see variation
        # (very unlikely all 10 are identical)
        unique_delays = len(set(delays))

        # Allow for some duplicates, but expect most to be different
        assert unique_delays >= 5, (
            f"Expected variation with jitter_factor={jitter_factor}, "
            f"but only got {unique_delays} unique values out of 10"
        )

    @given(
        attempt=st.integers(min_value=0, max_value=20),
        base_delay=st.floats(min_value=0.1, max_value=10.0),
        exponential_base=st.floats(min_value=1.1, max_value=5.0),
        max_delay=st.floats(min_value=10.0, max_value=300.0),
    )
    def test_zero_jitter_is_deterministic(
        self,
        attempt: int,
        base_delay: float,
        exponential_base: float,
        max_delay: float,
    ):
        """
        Property: With jitter_factor=0, the function should be deterministic
        (same inputs always produce same output).
        """
        delay1 = calculate_backoff_delay(
            attempt=attempt,
            base_delay=base_delay,
            exponential_base=exponential_base,
            max_delay=max_delay,
            jitter_factor=0.0,
        )

        delay2 = calculate_backoff_delay(
            attempt=attempt,
            base_delay=base_delay,
            exponential_base=exponential_base,
            max_delay=max_delay,
            jitter_factor=0.0,
        )

        assert delay1 == delay2, "Zero jitter should be deterministic"

    @given(
        base_delay=st.floats(min_value=0.1, max_value=10.0),
        exponential_base=st.floats(min_value=1.1, max_value=5.0),
    )
    def test_eventual_capping_at_high_attempts(
        self,
        base_delay: float,
        exponential_base: float,
    ):
        """
        Property: For sufficiently high attempt numbers, delay should equal
        max_delay (exponential growth eventually hits the cap).
        """
        max_delay = 60.0

        # At very high attempt numbers, we should hit the cap
        delay = calculate_backoff_delay(
            attempt=100,  # Very high attempt number
            base_delay=base_delay,
            exponential_base=exponential_base,
            max_delay=max_delay,
            jitter_factor=0.0,
        )

        assert delay == max_delay, (
            f"Expected delay to be capped at {max_delay}s for high attempts, got {delay}s"
        )
