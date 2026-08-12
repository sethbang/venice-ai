"""
Unit tests for API response validation module (_validation.py).

Tests cover:
- Successful response validation
- Validation failure scenarios
- Metrics collection and tracking
- Error context and logging
- Edge cases and error paths
"""

from unittest.mock import patch

import pytest
from pydantic import BaseModel, ValidationError, field_validator

from venice_ai._validation import (
    get_validation_metrics,
    reset_validation_metrics,
    validate_response_model,
)
from venice_ai.exceptions import APIResponseValidationError


class SimpleTestModel(BaseModel):
    """Simple test model for validation."""

    id: str
    value: int


class StrictTestModel(BaseModel):
    """Test model with validation rules."""

    email: str
    age: int

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Invalid email format")
        return v

    @field_validator("age")
    @classmethod
    def validate_age(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Age must be positive")
        return v


class TestValidateResponseModel:
    """Test validate_response_model function."""

    def setup_method(self):
        """Reset metrics before each test."""
        reset_validation_metrics()

    def test_successful_validation(self):
        """Test successful validation of valid response data."""
        response_data = {"id": "test-123", "value": 42}

        result = validate_response_model(
            response_data, SimpleTestModel, endpoint="/test", method="POST"
        )

        assert isinstance(result, SimpleTestModel)
        assert result.id == "test-123"
        assert result.value == 42

    def test_successful_validation_updates_metrics(self):
        """Test that successful validation updates metrics."""
        response_data = {"id": "test", "value": 1}

        initial_metrics = get_validation_metrics()
        initial_count = initial_metrics["successful_validations"]

        validate_response_model(response_data, SimpleTestModel)

        updated_metrics = get_validation_metrics()
        assert updated_metrics["successful_validations"] == initial_count + 1
        assert updated_metrics["failed_validations"] == 0

    def test_successful_validation_without_endpoint(self):
        """Test validation works without endpoint parameter."""
        response_data = {"id": "test", "value": 100}

        result = validate_response_model(response_data, SimpleTestModel)

        assert isinstance(result, SimpleTestModel)
        assert result.id == "test"
        assert result.value == 100

    def test_validation_failure_raises_error(self):
        """Test that validation failure raises APIResponseValidationError."""
        # Missing required field
        response_data = {"id": "test"}

        with pytest.raises(APIResponseValidationError) as exc_info:
            validate_response_model(response_data, SimpleTestModel, endpoint="/test", method="GET")

        error = exc_info.value
        assert "SimpleTestModel" in str(error)
        assert error.model_name == "SimpleTestModel"
        assert error.response_data == response_data
        assert isinstance(error.validation_error, ValidationError)

    def test_validation_failure_updates_metrics(self):
        """Test that validation failures update metrics correctly."""
        response_data = {"id": "test"}  # Missing 'value' field

        initial_metrics = get_validation_metrics()
        initial_failed = initial_metrics["failed_validations"]

        with pytest.raises(APIResponseValidationError):
            validate_response_model(response_data, SimpleTestModel)

        updated_metrics = get_validation_metrics()
        assert updated_metrics["failed_validations"] == initial_failed + 1

    def test_validation_failure_tracks_by_model(self):
        """Test that validation errors are tracked per model."""
        response_data = {"id": "test"}

        # Cause first failure
        with pytest.raises(APIResponseValidationError):
            validate_response_model(response_data, SimpleTestModel)

        metrics = get_validation_metrics()
        errors_by_model = metrics["validation_errors_by_model"]

        assert "SimpleTestModel" in errors_by_model
        assert errors_by_model["SimpleTestModel"] == 1

        # Cause second failure for same model
        with pytest.raises(APIResponseValidationError):
            validate_response_model(response_data, SimpleTestModel)

        updated_metrics = get_validation_metrics()
        assert updated_metrics["validation_errors_by_model"]["SimpleTestModel"] == 2

    def test_validation_error_with_custom_validator(self):
        """Test validation failure with custom field validators."""
        response_data = {
            "email": "invalid-email",  # Missing @
            "age": 25,
        }

        with pytest.raises(APIResponseValidationError) as exc_info:
            validate_response_model(response_data, StrictTestModel)

        error = exc_info.value
        assert error.model_name == "StrictTestModel"
        assert "Invalid email format" in str(error.validation_error)

    def test_validation_error_with_negative_value(self):
        """Test validation failure with invalid field value."""
        response_data = {
            "email": "test@example.com",
            "age": -5,  # Invalid negative age
        }

        with pytest.raises(APIResponseValidationError) as exc_info:
            validate_response_model(response_data, StrictTestModel)

        error = exc_info.value
        assert error.model_name == "StrictTestModel"
        assert "Age must be positive" in str(error.validation_error)

    def test_validation_error_preserves_context(self):
        """Test that validation errors preserve endpoint and method context."""
        response_data = {"id": "test"}
        endpoint = "/api/v1/test"
        method = "POST"

        with pytest.raises(APIResponseValidationError):
            validate_response_model(
                response_data, SimpleTestModel, endpoint=endpoint, method=method
            )

        # Error is logged with context (checked via logging mock in other test)

    def test_multiple_model_error_tracking(self):
        """Test that errors are tracked separately for different models."""
        # Fail validation for SimpleTestModel
        with pytest.raises(APIResponseValidationError):
            validate_response_model({"id": "test"}, SimpleTestModel)

        # Fail validation for StrictTestModel
        with pytest.raises(APIResponseValidationError):
            validate_response_model({"email": "bad"}, StrictTestModel)

        metrics = get_validation_metrics()
        errors_by_model = metrics["validation_errors_by_model"]

        assert "SimpleTestModel" in errors_by_model
        assert "StrictTestModel" in errors_by_model
        assert errors_by_model["SimpleTestModel"] == 1
        assert errors_by_model["StrictTestModel"] == 1

    @patch("venice_ai._validation.logger")
    def test_successful_validation_logs_debug(self, mock_logger):
        """Test that successful validation logs debug message."""
        response_data = {"id": "test", "value": 42}

        validate_response_model(response_data, SimpleTestModel, endpoint="/test", method="POST")

        # Check that debug was called
        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args
        assert "Successfully validated" in call_args[0][0]
        assert "SimpleTestModel" in call_args[0][0]

    @patch("venice_ai._validation.logger")
    def test_validation_failure_logs_error(self, mock_logger):
        """Test that validation failure logs error with context."""
        response_data = {"id": "test"}
        endpoint = "/api/test"
        method = "POST"

        with pytest.raises(APIResponseValidationError):
            validate_response_model(
                response_data, SimpleTestModel, endpoint=endpoint, method=method
            )

        # Check that error was logged
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args

        # Check message
        assert "Pydantic validation failed" in call_args[0][0]
        assert "SimpleTestModel" in call_args[0][0]

        # Check extra context
        extra = call_args[1]["extra"]
        assert extra["model_name"] == "SimpleTestModel"
        assert extra["response_data"] == response_data
        assert extra["endpoint"] == endpoint
        assert extra["method"] == method
        assert "validation_errors" in extra


class TestGetValidationMetrics:
    """Test get_validation_metrics function."""

    def setup_method(self):
        """Reset metrics before each test."""
        reset_validation_metrics()

    def test_get_metrics_returns_dict(self):
        """Test that get_validation_metrics returns a dictionary."""
        metrics = get_validation_metrics()

        assert isinstance(metrics, dict)
        assert "successful_validations" in metrics
        assert "failed_validations" in metrics
        assert "validation_errors_by_model" in metrics

    def test_get_metrics_returns_copy(self):
        """Test that get_validation_metrics returns a copy, not reference."""
        metrics1 = get_validation_metrics()
        metrics1["successful_validations"] = 999

        metrics2 = get_validation_metrics()
        assert metrics2["successful_validations"] != 999

    def test_get_metrics_initial_state(self):
        """Test initial state of metrics after reset."""
        metrics = get_validation_metrics()

        assert metrics["successful_validations"] == 0
        assert metrics["failed_validations"] == 0
        assert metrics["validation_errors_by_model"] == {}

    def test_get_metrics_after_operations(self):
        """Test metrics reflect actual validation operations."""
        # Perform successful validation
        validate_response_model({"id": "test", "value": 1}, SimpleTestModel)

        # Perform failed validation
        with pytest.raises(APIResponseValidationError):
            validate_response_model({"id": "test"}, SimpleTestModel)

        metrics = get_validation_metrics()
        assert metrics["successful_validations"] == 1
        assert metrics["failed_validations"] == 1
        assert metrics["validation_errors_by_model"]["SimpleTestModel"] == 1


class TestResetValidationMetrics:
    """Test reset_validation_metrics function."""

    def test_reset_clears_all_metrics(self):
        """Test that reset clears all metrics to initial state."""
        # Perform some operations to modify metrics
        validate_response_model({"id": "test", "value": 1}, SimpleTestModel)

        with pytest.raises(APIResponseValidationError):
            validate_response_model({"id": "test"}, SimpleTestModel)

        # Verify metrics are non-zero
        metrics = get_validation_metrics()
        assert metrics["successful_validations"] > 0
        assert metrics["failed_validations"] > 0

        # Reset
        reset_validation_metrics()

        # Verify all metrics are cleared
        metrics = get_validation_metrics()
        assert metrics["successful_validations"] == 0
        assert metrics["failed_validations"] == 0
        assert metrics["validation_errors_by_model"] == {}

    def test_reset_allows_fresh_tracking(self):
        """Test that tracking works correctly after reset."""
        # Do some operations
        validate_response_model({"id": "test", "value": 1}, SimpleTestModel)
        validate_response_model({"id": "test2", "value": 2}, SimpleTestModel)

        # Reset
        reset_validation_metrics()

        # Do new operation
        validate_response_model({"id": "test3", "value": 3}, SimpleTestModel)

        metrics = get_validation_metrics()
        assert metrics["successful_validations"] == 1  # Only counts after reset

    def test_reset_is_idempotent(self):
        """Test that calling reset multiple times is safe."""
        reset_validation_metrics()
        reset_validation_metrics()
        reset_validation_metrics()

        metrics = get_validation_metrics()
        assert metrics["successful_validations"] == 0
        assert metrics["failed_validations"] == 0
        assert metrics["validation_errors_by_model"] == {}


class TestMetricsIntegration:
    """Integration tests for metrics tracking."""

    def setup_method(self):
        """Reset metrics before each test."""
        reset_validation_metrics()

    def test_mixed_operations_tracking(self):
        """Test metrics tracking with mixed success and failure."""
        # 3 successful validations
        for i in range(3):
            validate_response_model({"id": f"test{i}", "value": i}, SimpleTestModel)

        # 2 failed validations for SimpleTestModel
        for _ in range(2):
            with pytest.raises(APIResponseValidationError):
                validate_response_model({"id": "test"}, SimpleTestModel)

        # 1 failed validation for StrictTestModel
        with pytest.raises(APIResponseValidationError):
            validate_response_model({"email": "bad"}, StrictTestModel)

        metrics = get_validation_metrics()
        assert metrics["successful_validations"] == 3
        assert metrics["failed_validations"] == 3
        assert metrics["validation_errors_by_model"]["SimpleTestModel"] == 2
        assert metrics["validation_errors_by_model"]["StrictTestModel"] == 1

    def test_high_volume_tracking(self):
        """Test metrics tracking with high volume of operations."""
        # Perform many successful validations
        for i in range(100):
            validate_response_model({"id": f"id-{i}", "value": i}, SimpleTestModel)

        metrics = get_validation_metrics()
        assert metrics["successful_validations"] == 100
        assert metrics["failed_validations"] == 0

    def test_error_tracking_multiple_models(self):
        """Test error tracking across multiple different models."""
        # Create failures for SimpleTestModel
        for _ in range(5):
            with pytest.raises(APIResponseValidationError):
                validate_response_model({"id": "test"}, SimpleTestModel)

        # Create failures for StrictTestModel
        for _ in range(3):
            with pytest.raises(APIResponseValidationError):
                validate_response_model({"email": "bad"}, StrictTestModel)

        metrics = get_validation_metrics()
        assert metrics["failed_validations"] == 8
        assert metrics["validation_errors_by_model"]["SimpleTestModel"] == 5
        assert metrics["validation_errors_by_model"]["StrictTestModel"] == 3


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    def setup_method(self):
        """Reset metrics before each test."""
        reset_validation_metrics()

    def test_validation_with_none_data(self):
        """Test validation with None response data."""
        with pytest.raises(APIResponseValidationError):
            validate_response_model(None, SimpleTestModel)

    def test_validation_with_empty_dict(self):
        """Test validation with empty dictionary."""
        with pytest.raises(APIResponseValidationError):
            validate_response_model({}, SimpleTestModel)

    def test_validation_with_wrong_type(self):
        """Test validation with completely wrong data type."""
        with pytest.raises(APIResponseValidationError):
            validate_response_model("not a dict", SimpleTestModel)

    def test_validation_with_list_data(self):
        """Test validation with list instead of dict."""
        with pytest.raises(APIResponseValidationError):
            validate_response_model([1, 2, 3], SimpleTestModel)

    def test_validation_with_extra_fields(self):
        """Test validation with extra fields (should succeed with Pydantic default)."""
        response_data = {"id": "test", "value": 42, "extra_field": "ignored"}

        # Pydantic v2 ignores extra fields by default
        result = validate_response_model(response_data, SimpleTestModel)
        assert result.id == "test"
        assert result.value == 42

    def test_validation_with_numeric_string(self):
        """Test validation with string that should be int."""
        response_data = {
            "id": "test",
            "value": "42",  # String instead of int
        }

        # Pydantic v2 will coerce this
        result = validate_response_model(response_data, SimpleTestModel)
        assert result.value == 42

    def test_validation_with_invalid_type_conversion(self):
        """Test validation when type conversion fails."""
        response_data = {"id": "test", "value": "not-a-number"}

        with pytest.raises(APIResponseValidationError):
            validate_response_model(response_data, SimpleTestModel)
