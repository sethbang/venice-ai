"""
Comprehensive test coverage for AccountBackend base class.

This test module targets specific missing lines and branches to achieve 90%+ coverage.
It focuses on:
- Lines 116-130: _parse_rate_limit_headers method
- Line 138: _calculate_wait_time empty rate_limits check
- Lines 147-148: RPM reset time parsing
- Lines 152-155: TPM reset time parsing
- Line 159: retry_after handling
- Lines 165-173: _parse_reset_time method
- All 11 partial branches for complete branch coverage
"""

import time
from typing import Any

from venice_ai.core.backends.base import AccountBackend, HealthCheckResult


class ConcreteBackend(AccountBackend):
    """
    Concrete implementation of AccountBackend for testing base class methods.

    All abstract methods are implemented as no-ops since we're testing
    the helper methods in the base class.
    """

    async def health_check(self) -> HealthCheckResult:
        return HealthCheckResult(healthy=True, backend_type="test", namespace=self.namespace)

    async def cleanup(self) -> None:
        pass

    async def get_all_stats(self) -> dict[str, Any]:
        return {}

    async def check_capacity(self, model: str, request_type: str = "default") -> tuple[bool, float]:
        return True, 0.0

    async def update_rate_limits(self, model: str, headers: dict[str, str]) -> None:
        pass

    async def record_request(self, model: str, tokens_used: int | None = None) -> None:
        pass

    async def record_failure(self, error_type: str, error_message: str = "") -> None:
        pass

    async def get_failure_count(self, window_seconds: int = 30) -> int:
        return 0

    async def is_circuit_broken(self) -> bool:
        return False

    async def clear_failures(self) -> None:
        pass

    async def force_circuit_break(self, duration: float) -> None:
        pass

    async def release_streaming_reservation(
        self,
        bucket_id: str,
        reservation_id: str,
        reserved_tokens: int,
        actual_tokens: int,
    ) -> bool:
        """No-op implementation for testing."""
        return True


class TestAccountBackendInitialization:
    """Test AccountBackend initialization."""

    def test_init_default_namespace(self):
        """Test initialization with default namespace."""
        backend = ConcreteBackend()
        assert backend.namespace == "venice_ai"

    def test_init_custom_namespace(self):
        """Test initialization with custom namespace."""
        backend = ConcreteBackend(namespace="custom_namespace")
        assert backend.namespace == "custom_namespace"


class TestParseRateLimitHeaders:
    """Test _parse_rate_limit_headers method (lines 116-130)."""

    def test_empty_headers(self):
        """Test parsing empty headers."""
        backend = ConcreteBackend()
        result = backend._parse_rate_limit_headers({})

        # Should only have timestamp
        assert "timestamp" in result
        assert "rpm_limit" not in result
        assert "tpm_limit" not in result

    def test_rpm_limit_header(self):
        """Test parsing x-ratelimit-limit-requests (line 116)."""
        backend = ConcreteBackend()
        headers = {"x-ratelimit-limit-requests": "100"}

        result = backend._parse_rate_limit_headers(headers)

        assert result["rpm_limit"] == 100

    def test_rpm_remaining_header(self):
        """Test parsing x-ratelimit-remaining-requests (line 117→118)."""
        backend = ConcreteBackend()
        headers = {"x-ratelimit-remaining-requests": "50"}

        result = backend._parse_rate_limit_headers(headers)

        assert result["rpm_remaining"] == 50

    def test_rpm_reset_header(self):
        """Test parsing x-ratelimit-reset-requests (line 119→120)."""
        backend = ConcreteBackend()
        headers = {"x-ratelimit-reset-requests": "60"}

        result = backend._parse_rate_limit_headers(headers)

        assert result["rpm_reset"] == 60

    def test_reset_header_float_epoch_ms_does_not_crash(self):
        """A float-valued reset header (an epoch-millis timestamp with a
        fractional part) must parse, not raise ValueError and drop the lot."""
        backend = ConcreteBackend()
        headers = {"x-ratelimit-reset-requests": "1750000000.123"}

        result = backend._parse_rate_limit_headers(headers)

        assert result["rpm_reset"] == 1750000000

    def test_malformed_header_is_skipped_not_fatal(self):
        """A non-numeric header value is skipped, and other valid headers in the
        same response are still parsed — one bad value must not drop them all."""
        backend = ConcreteBackend()
        headers = {
            "x-ratelimit-reset-requests": "n/a",
            "x-ratelimit-limit-requests": "100",
        }

        result = backend._parse_rate_limit_headers(headers)

        assert "rpm_reset" not in result  # bad value skipped, not fatal
        assert result["rpm_limit"] == 100  # good value still parsed

    def test_tpm_limit_header(self):
        """Test parsing x-ratelimit-limit-tokens (line 122→123)."""
        backend = ConcreteBackend()
        headers = {"x-ratelimit-limit-tokens": "10000"}

        result = backend._parse_rate_limit_headers(headers)

        assert result["tpm_limit"] == 10000

    def test_tpm_remaining_header(self):
        """Test parsing x-ratelimit-remaining-tokens (line 124→125)."""
        backend = ConcreteBackend()
        headers = {"x-ratelimit-remaining-tokens": "5000"}

        result = backend._parse_rate_limit_headers(headers)

        assert result["tpm_remaining"] == 5000

    def test_tpm_reset_header(self):
        """Test parsing x-ratelimit-reset-tokens (line 126→127)."""
        backend = ConcreteBackend()
        headers = {"x-ratelimit-reset-tokens": "120"}

        result = backend._parse_rate_limit_headers(headers)

        assert result["tpm_reset"] == 120

    def test_retry_after_header(self):
        """Test parsing retry-after (line 129→130)."""
        backend = ConcreteBackend()
        headers = {"retry-after": "30"}

        result = backend._parse_rate_limit_headers(headers)

        assert result["retry_after"] == 30

    def test_retry_after_float(self):
        """Test parsing retry-after with float value (line 130)."""
        backend = ConcreteBackend()
        headers = {"retry-after": "30.5"}

        result = backend._parse_rate_limit_headers(headers)

        # Should convert to int: int(float("30.5")) = 30
        assert result["retry_after"] == 30

    def test_all_headers(self):
        """Test parsing all headers at once (lines 116-130)."""
        backend = ConcreteBackend()
        headers = {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "50",
            "x-ratelimit-reset-requests": "60",
            "x-ratelimit-limit-tokens": "10000",
            "x-ratelimit-remaining-tokens": "5000",
            "x-ratelimit-reset-tokens": "120",
            "retry-after": "30",
        }

        result = backend._parse_rate_limit_headers(headers)

        assert result["rpm_limit"] == 100
        assert result["rpm_remaining"] == 50
        assert result["rpm_reset"] == 60
        assert result["tpm_limit"] == 10000
        assert result["tpm_remaining"] == 5000
        assert result["tpm_reset"] == 120
        assert result["retry_after"] == 30
        assert "timestamp" in result

    def test_timestamp_is_current(self):
        """Test that timestamp is set to current time."""
        backend = ConcreteBackend()
        before = int(time.time())

        result = backend._parse_rate_limit_headers({})

        after = int(time.time())
        assert before <= result["timestamp"] <= after

    def test_partial_rpm_headers(self):
        """Test parsing with only some RPM headers."""
        backend = ConcreteBackend()
        headers = {
            "x-ratelimit-limit-requests": "100",
            # No remaining or reset
        }

        result = backend._parse_rate_limit_headers(headers)

        assert result["rpm_limit"] == 100
        assert "rpm_remaining" not in result
        assert "rpm_reset" not in result

    def test_partial_tpm_headers(self):
        """Test parsing with only some TPM headers."""
        backend = ConcreteBackend()
        headers = {
            "x-ratelimit-remaining-tokens": "5000",
            # No limit or reset
        }

        result = backend._parse_rate_limit_headers(headers)

        assert result["tpm_remaining"] == 5000
        assert "tpm_limit" not in result
        assert "tpm_reset" not in result


class TestCalculateWaitTime:
    """Test _calculate_wait_time method (lines 135-161)."""

    def test_empty_rate_limits(self):
        """Test with empty rate_limits dict (line 138)."""
        backend = ConcreteBackend()

        wait_time = backend._calculate_wait_time({})

        assert wait_time == 0.0

    def test_none_rate_limits(self):
        """Test with None-ish rate_limits (line 137-138)."""
        backend = ConcreteBackend()

        # Empty dict evaluates to False in boolean context
        wait_time = backend._calculate_wait_time({})

        assert wait_time == 0.0

    def test_rpm_remaining_positive(self):
        """Test when rpm_remaining > 0 (line 144 false branch)."""
        backend = ConcreteBackend()
        rate_limits = {"rpm_remaining": 10}

        wait_time = backend._calculate_wait_time(rate_limits)

        assert wait_time == 0.0

    def test_rpm_remaining_zero_no_reset(self):
        """Test when rpm_remaining = 0 but no rpm_reset (line 146)."""
        backend = ConcreteBackend()
        rate_limits = {"rpm_remaining": 0}

        wait_time = backend._calculate_wait_time(rate_limits)

        # No reset time, so should be 0
        assert wait_time == 0.0

    def test_rpm_remaining_zero_with_reset(self):
        """Test when rpm_remaining = 0 with rpm_reset (lines 146-148)."""
        backend = ConcreteBackend()
        reset_time = int(time.time()) + 30
        rate_limits = {
            "rpm_remaining": 0,
            "rpm_reset": reset_time,
        }

        wait_time = backend._calculate_wait_time(rate_limits)

        # Should be approximately 30 seconds
        assert 25 <= wait_time <= 35

    def test_rpm_remaining_negative(self):
        """Test when rpm_remaining < 0 (line 144 true branch)."""
        backend = ConcreteBackend()
        reset_time = int(time.time()) + 20
        rate_limits = {
            "rpm_remaining": -5,
            "rpm_reset": reset_time,
        }

        wait_time = backend._calculate_wait_time(rate_limits)

        # Should wait since remaining < 0
        assert wait_time > 0

    def test_tpm_remaining_positive(self):
        """Test when tpm_remaining > 0 (line 151 false branch)."""
        backend = ConcreteBackend()
        rate_limits = {"tpm_remaining": 5000}

        wait_time = backend._calculate_wait_time(rate_limits)

        assert wait_time == 0.0

    def test_tpm_remaining_zero_no_reset(self):
        """Test when tpm_remaining = 0 but no tpm_reset (line 152)."""
        backend = ConcreteBackend()
        rate_limits = {
            "rpm_remaining": 10,  # RPM OK
            "tpm_remaining": 0,
        }

        wait_time = backend._calculate_wait_time(rate_limits)

        # No reset time, so should be 0
        assert wait_time == 0.0

    def test_tpm_remaining_zero_with_reset(self):
        """Test when tpm_remaining = 0 with tpm_reset (lines 152-155)."""
        backend = ConcreteBackend()
        reset_time = int(time.time()) + 45
        rate_limits = {
            "rpm_remaining": 10,  # RPM OK
            "tpm_remaining": 0,
            "tpm_reset": reset_time,
        }

        wait_time = backend._calculate_wait_time(rate_limits)

        # Should be approximately 45 seconds
        assert 40 <= wait_time <= 50

    def test_retry_after_zero(self):
        """Test when retry_after = 0 (line 158 false branch)."""
        backend = ConcreteBackend()
        rate_limits = {"retry_after": 0}

        wait_time = backend._calculate_wait_time(rate_limits)

        assert wait_time == 0.0

    def test_retry_after_positive(self):
        """Test when retry_after > 0 (line 158-159)."""
        backend = ConcreteBackend()
        rate_limits = {"retry_after": 25}

        wait_time = backend._calculate_wait_time(rate_limits)

        assert wait_time == 25.0

    def test_retry_after_takes_precedence(self):
        """Test that retry_after can be the max wait time (line 159)."""
        backend = ConcreteBackend()
        rate_limits = {
            "rpm_remaining": 10,  # RPM OK
            "tpm_remaining": 10,  # TPM OK
            "retry_after": 60,
        }

        wait_time = backend._calculate_wait_time(rate_limits)

        assert wait_time == 60.0

    def test_max_of_rpm_and_tpm_wait(self):
        """Test that max of RPM and TPM reset is used."""
        backend = ConcreteBackend()
        current_time = int(time.time())
        rate_limits = {
            "rpm_remaining": 0,
            "rpm_reset": current_time + 20,  # 20s wait
            "tpm_remaining": 0,
            "tpm_reset": current_time + 40,  # 40s wait
        }

        wait_time = backend._calculate_wait_time(rate_limits)

        # Should use the larger wait time
        assert 35 <= wait_time <= 45

    def test_retry_after_vs_reset_times(self):
        """Test retry_after compared against reset times."""
        backend = ConcreteBackend()
        current_time = int(time.time())
        rate_limits = {
            "rpm_remaining": 0,
            "rpm_reset": current_time + 10,  # 10s wait
            "retry_after": 30,  # 30s wait
        }

        wait_time = backend._calculate_wait_time(rate_limits)

        # retry_after (30) should be used if larger
        assert 25 <= wait_time <= 35

    def test_negative_result_returns_zero(self):
        """Test that negative calculated wait returns 0 (line 161)."""
        backend = ConcreteBackend()
        # Set reset time to past
        past_time = int(time.time()) - 60
        rate_limits = {
            "rpm_remaining": 0,
            "rpm_reset": past_time,
        }

        wait_time = backend._calculate_wait_time(rate_limits)

        # Should return 0, not negative
        assert wait_time == 0.0

    def test_missing_rpm_remaining_uses_default(self):
        """Test default rpm_remaining of 1 when not present."""
        backend = ConcreteBackend()
        rate_limits = {"rpm_reset": int(time.time()) + 30}

        wait_time = backend._calculate_wait_time(rate_limits)

        # Default rpm_remaining=1 means > 0, so no wait
        assert wait_time == 0.0

    def test_missing_tpm_remaining_uses_default(self):
        """Test default tpm_remaining of 1 when not present."""
        backend = ConcreteBackend()
        rate_limits = {"tpm_reset": int(time.time()) + 30}

        wait_time = backend._calculate_wait_time(rate_limits)

        # Default tpm_remaining=1 means > 0, so no wait
        assert wait_time == 0.0


class TestParseResetTime:
    """Test _parse_reset_time method (lines 163-173)."""

    def test_small_value_interpreted_as_relative(self):
        """Test values < 1e9 treated as relative seconds (line 171)."""
        backend = ConcreteBackend()
        current = time.time()

        result = backend._parse_reset_time("60")

        # Should be current time + 60 seconds
        expected = current + 60
        assert abs(result - expected) < 2

    def test_unix_timestamp_seconds(self):
        """Test Unix timestamp in seconds returned as-is (line 170)."""
        backend = ConcreteBackend()
        # A Unix timestamp in seconds (around year 2024)
        timestamp = 1704067200  # 2024-01-01 00:00:00 UTC

        result = backend._parse_reset_time(str(timestamp))

        assert result == timestamp

    def test_unix_timestamp_milliseconds(self):
        """Test Unix timestamp in milliseconds divided by 1000 (line 168)."""
        backend = ConcreteBackend()
        # A Unix timestamp in milliseconds (around year 2024)
        timestamp_ms = 1704067200000  # 2024-01-01 00:00:00 UTC in ms

        result = backend._parse_reset_time(str(timestamp_ms))

        # Should be divided by 1000
        assert result == timestamp_ms / 1000.0

    def test_invalid_value_returns_default(self):
        """Test invalid (non-numeric) value returns default (lines 172-173)."""
        backend = ConcreteBackend()
        current = time.time()

        result = backend._parse_reset_time("invalid")

        # Should return current time + 60 seconds
        expected = current + 60
        assert abs(result - expected) < 2

    def test_empty_string_returns_default(self):
        """Test empty string returns default."""
        backend = ConcreteBackend()
        current = time.time()

        result = backend._parse_reset_time("")

        expected = current + 60
        assert abs(result - expected) < 2

    def test_float_string(self):
        """Test float string is parsed correctly."""
        backend = ConcreteBackend()
        current = time.time()

        result = backend._parse_reset_time("30.5")

        expected = current + 30.5
        assert abs(result - expected) < 2

    def test_boundary_below_1e9(self):
        """Test value just below 1e9 threshold treated as relative."""
        backend = ConcreteBackend()
        current = time.time()
        # Small value - treated as relative seconds
        val = 3600  # 1 hour

        result = backend._parse_reset_time(str(val))

        # Should be treated as relative (current time + val)
        expected = current + val
        assert abs(result - expected) < 2

    def test_boundary_above_1e9(self):
        """Test value above 1e9 threshold (Unix timestamp)."""
        backend = ConcreteBackend()
        # Value well above 1e9 - Modern Unix timestamp
        val = 1700000000  # Nov 2023 timestamp

        result = backend._parse_reset_time(str(val))

        # Should be returned as-is (Unix timestamp in seconds)
        assert result == val

    def test_boundary_between_1e9_and_1e11(self):
        """Test value between 1e9 and 1e11."""
        backend = ConcreteBackend()
        # Modern Unix timestamp in seconds
        val = 1700000000  # Nov 2023

        result = backend._parse_reset_time(str(val))

        # Should be returned as-is
        assert result == val

    def test_boundary_just_above_1e11(self):
        """Test value just above 1e11 (milliseconds)."""
        backend = ConcreteBackend()
        # Value just above 1e11 - interpreted as milliseconds
        val = 100000000001  # Just above 1e11

        result = backend._parse_reset_time(str(val))

        # Should be divided by 1000
        assert result == val / 1000.0

    def test_negative_value(self):
        """Test negative value is treated as relative (past)."""
        backend = ConcreteBackend()
        current = time.time()

        result = backend._parse_reset_time("-30")

        # Negative value < 1e9, so treated as relative
        expected = current + (-30)
        assert abs(result - expected) < 2

    def test_zero_value(self):
        """Test zero value is treated as relative."""
        backend = ConcreteBackend()
        current = time.time()

        result = backend._parse_reset_time("0")

        # Zero < 1e9, so treated as relative (current time + 0)
        expected = current
        assert abs(result - expected) < 2


class TestHealthCheckResultDataclass:
    """Test HealthCheckResult dataclass."""

    def test_minimal_initialization(self):
        """Test initialization with required fields only."""
        result = HealthCheckResult(healthy=True, backend_type="test", namespace="test_ns")

        assert result.healthy is True
        assert result.backend_type == "test"
        assert result.namespace == "test_ns"
        assert result.error is None
        assert result.metadata is None

    def test_full_initialization(self):
        """Test initialization with all fields."""
        result = HealthCheckResult(
            healthy=False,
            backend_type="redis",
            namespace="prod",
            error="Connection failed",
            metadata={"host": "localhost", "port": 6379},
        )

        assert result.healthy is False
        assert result.backend_type == "redis"
        assert result.namespace == "prod"
        assert result.error == "Connection failed"
        assert result.metadata == {"host": "localhost", "port": 6379}

    def test_unhealthy_with_error(self):
        """Test unhealthy result with error message."""
        result = HealthCheckResult(
            healthy=False,
            backend_type="test",
            namespace="test",
            error="Backend unavailable",
        )

        assert result.healthy is False
        assert result.error == "Backend unavailable"


class TestBranchCoverage:
    """Additional tests specifically for branch coverage."""

    def test_parse_rate_limit_headers_no_rpm_limit(self):
        """Test when x-ratelimit-limit-requests is absent (line 115 false branch)."""
        backend = ConcreteBackend()
        headers = {"x-ratelimit-remaining-requests": "50"}

        result = backend._parse_rate_limit_headers(headers)

        assert "rpm_limit" not in result
        assert result["rpm_remaining"] == 50

    def test_parse_rate_limit_headers_no_rpm_remaining(self):
        """Test when x-ratelimit-remaining-requests is absent (line 117 false branch)."""
        backend = ConcreteBackend()
        headers = {"x-ratelimit-limit-requests": "100"}

        result = backend._parse_rate_limit_headers(headers)

        assert result["rpm_limit"] == 100
        assert "rpm_remaining" not in result

    def test_parse_rate_limit_headers_no_rpm_reset(self):
        """Test when x-ratelimit-reset-requests is absent (line 119 false branch)."""
        backend = ConcreteBackend()
        headers = {"x-ratelimit-limit-requests": "100"}

        result = backend._parse_rate_limit_headers(headers)

        assert "rpm_reset" not in result

    def test_parse_rate_limit_headers_no_tpm_limit(self):
        """Test when x-ratelimit-limit-tokens is absent (line 122 false branch)."""
        backend = ConcreteBackend()
        headers = {"x-ratelimit-remaining-tokens": "5000"}

        result = backend._parse_rate_limit_headers(headers)

        assert "tpm_limit" not in result
        assert result["tpm_remaining"] == 5000

    def test_parse_rate_limit_headers_no_tpm_remaining(self):
        """Test when x-ratelimit-remaining-tokens is absent (line 124 false branch)."""
        backend = ConcreteBackend()
        headers = {"x-ratelimit-limit-tokens": "10000"}

        result = backend._parse_rate_limit_headers(headers)

        assert result["tpm_limit"] == 10000
        assert "tpm_remaining" not in result

    def test_parse_rate_limit_headers_no_tpm_reset(self):
        """Test when x-ratelimit-reset-tokens is absent (line 126 false branch)."""
        backend = ConcreteBackend()
        headers = {"x-ratelimit-limit-tokens": "10000"}

        result = backend._parse_rate_limit_headers(headers)

        assert "tpm_reset" not in result

    def test_parse_rate_limit_headers_no_retry_after(self):
        """Test when retry-after is absent (line 129 false branch)."""
        backend = ConcreteBackend()
        headers = {"x-ratelimit-limit-requests": "100"}

        result = backend._parse_rate_limit_headers(headers)

        assert "retry_after" not in result

    def test_calculate_wait_time_rpm_reset_none(self):
        """Test _calculate_wait_time when rpm_reset is None (line 146 false branch)."""
        backend = ConcreteBackend()
        rate_limits = {
            "rpm_remaining": 0,
            # rpm_reset not present
        }

        wait_time = backend._calculate_wait_time(rate_limits)

        # No rpm_reset, so no wait from RPM
        assert wait_time == 0.0

    def test_calculate_wait_time_tpm_reset_none(self):
        """Test _calculate_wait_time when tpm_reset is None (line 152 false branch)."""
        backend = ConcreteBackend()
        rate_limits = {
            "tpm_remaining": 0,
            # tpm_reset not present
        }

        wait_time = backend._calculate_wait_time(rate_limits)

        # No tpm_reset, so no wait from TPM
        assert wait_time == 0.0

    def test_calculate_wait_time_both_reset_with_values(self):
        """Test when both rpm_reset and tpm_reset have values."""
        backend = ConcreteBackend()
        current = int(time.time())
        rate_limits = {
            "rpm_remaining": 0,
            "rpm_reset": current + 10,
            "tpm_remaining": 0,
            "tpm_reset": current + 20,
        }

        wait_time = backend._calculate_wait_time(rate_limits)

        # Should use max of both
        assert 15 <= wait_time <= 25

    def test_parse_reset_time_all_branches(self):
        """Test _parse_reset_time covers all conditional branches."""
        backend = ConcreteBackend()

        # Branch 1: val > 1e11 (milliseconds)
        result1 = backend._parse_reset_time("200000000000")
        assert result1 > 1e8  # Divided by 1000

        # Branch 2: val > 1e9 (Unix seconds)
        result2 = backend._parse_reset_time("1600000000")
        assert result2 == 1600000000

        # Branch 3: val <= 1e9 (relative seconds)
        result3 = backend._parse_reset_time("60")
        current = time.time()
        assert abs(result3 - (current + 60)) < 2

    def test_parse_reset_time_value_error_branch(self):
        """Test _parse_reset_time ValueError handling (line 172-173)."""
        backend = ConcreteBackend()
        current = time.time()

        # Various invalid inputs that cause ValueError (not NaN which is valid float)
        for invalid in ["abc", "not_a_number", "12.34.56", " "]:
            result = backend._parse_reset_time(invalid)
            expected = current + 60
            assert abs(result - expected) < 2, f"Failed for input: {invalid}"


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_parse_headers_from_real_api_response(self):
        """Test parsing headers similar to real API responses."""
        backend = ConcreteBackend()

        # Simulated real API response headers
        headers = {
            "x-ratelimit-limit-requests": "60",
            "x-ratelimit-remaining-requests": "45",
            "x-ratelimit-reset-requests": "1704067260",  # Unix timestamp
            "x-ratelimit-limit-tokens": "150000",
            "x-ratelimit-remaining-tokens": "120000",
            "x-ratelimit-reset-tokens": "1704067320",
        }

        result = backend._parse_rate_limit_headers(headers)

        assert result["rpm_limit"] == 60
        assert result["rpm_remaining"] == 45
        assert result["rpm_reset"] == 1704067260
        assert result["tpm_limit"] == 150000
        assert result["tpm_remaining"] == 120000
        assert result["tpm_reset"] == 1704067320

    def test_calculate_wait_when_exhausted(self):
        """Test wait time calculation when fully rate limited."""
        backend = ConcreteBackend()
        current = int(time.time())

        rate_limits = {
            "rpm_remaining": 0,
            "rpm_reset": current + 30,
            "tpm_remaining": 0,
            "tpm_reset": current + 60,
            "retry_after": 45,
        }

        wait_time = backend._calculate_wait_time(rate_limits)

        # Should use max of all (60 from tpm_reset)
        assert 55 <= wait_time <= 65

    def test_calculate_wait_when_partially_exhausted(self):
        """Test wait time when only one limit is exhausted."""
        backend = ConcreteBackend()
        current = int(time.time())

        rate_limits = {
            "rpm_remaining": 10,  # OK
            "rpm_reset": current + 30,
            "tpm_remaining": 0,  # Exhausted
            "tpm_reset": current + 45,
        }

        wait_time = backend._calculate_wait_time(rate_limits)

        # Only TPM is exhausted
        assert 40 <= wait_time <= 50

    def test_rate_limit_headers_case_sensitivity(self):
        """Test that header keys are case-sensitive (lowercase expected)."""
        backend = ConcreteBackend()

        # Wrong case - should not be parsed
        headers = {
            "X-Ratelimit-Limit-Requests": "100",
            "X-RATELIMIT-REMAINING-REQUESTS": "50",
        }

        result = backend._parse_rate_limit_headers(headers)

        # These should NOT be found (wrong case)
        assert "rpm_limit" not in result
        assert "rpm_remaining" not in result
