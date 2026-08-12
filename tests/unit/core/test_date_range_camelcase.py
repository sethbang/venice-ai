"""Tests for DateRangeParams in types/api/requests/common.py (camelCase variant).

The core models/common.py DateRangeParams uses snake_case and is already tested.
This covers the API-facing camelCase variant's validate_date_range validator.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from venice_ai.types.api.requests.common import DateRangeParams


class TestDateRangeParamsCamelCase:
    """Test the camelCase DateRangeParams validator."""

    def test_valid_range(self):
        params = DateRangeParams(
            startDate=datetime(2024, 1, 1, tzinfo=UTC),
            endDate=datetime(2024, 12, 31, tzinfo=UTC),
        )
        assert params.startDate is not None
        assert params.endDate is not None

    def test_end_before_start_raises(self):
        with pytest.raises(ValidationError, match="endDate must be after startDate"):
            DateRangeParams(
                startDate=datetime(2024, 12, 31, tzinfo=UTC),
                endDate=datetime(2024, 1, 1, tzinfo=UTC),
            )

    def test_end_equals_start_raises(self):
        dt = datetime(2024, 6, 15, tzinfo=UTC)
        with pytest.raises(ValidationError, match="endDate must be after startDate"):
            DateRangeParams(startDate=dt, endDate=dt)

    def test_no_start_date(self):
        params = DateRangeParams(startDate=None, endDate=datetime(2024, 12, 31, tzinfo=UTC))
        assert params.startDate is None
        assert params.endDate is not None

    def test_no_end_date(self):
        params = DateRangeParams(startDate=datetime(2024, 1, 1, tzinfo=UTC), endDate=None)
        assert params.startDate is not None
        assert params.endDate is None

    def test_both_none(self):
        params = DateRangeParams(startDate=None, endDate=None)
        assert params.startDate is None
        assert params.endDate is None
