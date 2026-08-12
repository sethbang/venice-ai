"""Tests for VeniceBaseModel properties: pagination_info, request_id, content_safety_info, model_info."""

from unittest.mock import Mock

from venice_ai.core.models.base import VeniceBaseModel


def _model_with_headers(headers: dict[str, str]) -> VeniceBaseModel:
    """Create a VeniceBaseModel with mock response headers."""
    model = VeniceBaseModel()
    resp = Mock()
    resp.headers = headers
    model._response = resp
    return model


class TestPaginationInfo:
    """Test pagination_info property branch coverage."""

    def test_no_response(self):
        model = VeniceBaseModel()
        assert model.pagination_info is None

    def test_no_pagination_headers(self):
        model = _model_with_headers({"content-type": "application/json"})
        assert model.pagination_info is None

    def test_all_pagination_headers(self):
        model = _model_with_headers(
            {
                "x-pagination-limit": "50",
                "x-pagination-page": "2",
                "x-pagination-total": "200",
                "x-pagination-total-pages": "4",
            }
        )
        info = model.pagination_info
        assert info is not None
        assert info.limit == 50
        assert info.page == 2
        assert info.total == 200
        assert info.total_pages == 4

    def test_partial_pagination_headers_returns_none(self):
        """When some pagination headers are present but not all four, returns None."""
        model = _model_with_headers(
            {
                "x-pagination-limit": "50",
                "x-pagination-page": "2",
                # missing total and total_pages
            }
        )
        assert model.pagination_info is None

    def test_invalid_pagination_values_returns_none(self):
        """Non-numeric values parse to None, so not all fields present -> returns None."""
        model = _model_with_headers(
            {
                "x-pagination-limit": "bad",
                "x-pagination-page": "2",
                "x-pagination-total": "200",
                "x-pagination-total-pages": "4",
            }
        )
        assert model.pagination_info is None


class TestRequestId:
    """Test request_id property."""

    def test_no_response(self):
        model = VeniceBaseModel()
        assert model.request_id is None

    def test_with_cf_ray(self):
        model = _model_with_headers({"cf-ray": "abc123-IAD"})
        assert model.request_id == "abc123-IAD"

    def test_without_cf_ray(self):
        model = _model_with_headers({"content-type": "application/json"})
        assert model.request_id is None


class TestContentSafetyInfo:
    """Test content_safety_info property."""

    def test_no_response(self):
        model = VeniceBaseModel()
        assert model.content_safety_info is None

    def test_no_safety_headers(self):
        model = _model_with_headers({"content-type": "application/json"})
        assert model.content_safety_info is None

    def test_all_safety_headers(self):
        model = _model_with_headers(
            {
                "x-venice-is-blurred": "true",
                "x-venice-is-content-violation": "false",
                "x-venice-is-adult-model-content-violation": "true",
                "x-venice-contains-minor": "false",
            }
        )
        info = model.content_safety_info
        assert info is not None
        assert info.is_blurred is True
        assert info.is_content_violation is False
        assert info.is_adult_model_content_violation is True
        assert info.contains_minor is False

    def test_partial_safety_headers(self):
        model = _model_with_headers({"x-venice-is-blurred": "true"})
        info = model.content_safety_info
        assert info is not None
        assert info.is_blurred is True
        assert info.is_content_violation is None

    def test_parse_bool_none(self):
        """All safety headers absent -> all fields None -> returns None."""
        model = _model_with_headers({"x-unrelated": "value"})
        assert model.content_safety_info is None


class TestModelInfo:
    """Test model_info property."""

    def test_no_response(self):
        model = VeniceBaseModel()
        assert model.model_info is None

    def test_no_model_headers(self):
        model = _model_with_headers({"content-type": "application/json"})
        assert model.model_info is None

    def test_with_model_headers(self):
        model = _model_with_headers(
            {
                "x-venice-model-id": "llama-3.3-70b",
                "x-venice-model-name": "Llama 3.3 70B",
                "x-venice-model-router": "primary",
            }
        )
        info = model.model_info
        assert info is not None
        assert info.model_id == "llama-3.3-70b"
        assert info.model_name == "Llama 3.3 70B"
        assert info.model_router == "primary"

    def test_with_only_deprecation_headers_returns_none(self):
        """model_info only returns non-None when model_id, model_name, or model_router is set."""
        model = _model_with_headers(
            {
                "x-venice-model-deprecation-warning": "deprecated",
                "x-venice-model-deprecation-date": "2025-01-01",
            }
        )
        assert model.model_info is None

    def test_partial_model_headers(self):
        model = _model_with_headers({"x-venice-model-id": "some-model"})
        info = model.model_info
        assert info is not None
        assert info.model_id == "some-model"
        assert info.model_name is None
