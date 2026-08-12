# tests/unit/resources/test_models.py
"""Tests for venice_ai.resources.models.Models resource."""

from unittest.mock import AsyncMock

import pytest

from venice_ai.resources.models import Models
from venice_ai.types.api import (
    ModelCompatibilityResponse,
    ModelsListResponse,
    ModelTraitsResponse,
)


@pytest.mark.asyncio
async def test_models_list():
    """Test Models.list() calls correct endpoint with params."""
    mock_client = AsyncMock()
    expected = ModelsListResponse(object="list", type="text", data=[])
    mock_client.get = AsyncMock(return_value=expected)

    models = Models(mock_client)
    result = await models.list(type="text")

    mock_client.get.assert_called_once_with(
        "models", params={"type": "text"}, cast_to=ModelsListResponse, force_direct=True
    )
    assert result is expected
    assert result.object == "list"
    assert result.type == "text"
    assert result.data == []


@pytest.mark.asyncio
async def test_models_list_no_type_filter():
    """Bare ``list()`` auto-passes ``type="all"``.

    The server's own default is ``text``-only when ``type`` is omitted, which
    surprises callers who expect "no filter" to mean "every model". The SDK
    overrides that by sending ``type=all`` on the wire whenever the caller
    doesn't pass ``type=`` explicitly. Regression-guards against silently
    reverting to the empty-params behaviour.
    """
    mock_client = AsyncMock()
    expected = ModelsListResponse(object="list", type="all", data=[])
    mock_client.get = AsyncMock(return_value=expected)

    models = Models(mock_client)
    result = await models.list()

    mock_client.get.assert_called_once_with(
        "models", params={"type": "all"}, cast_to=ModelsListResponse, force_direct=True
    )
    assert result is expected


@pytest.mark.asyncio
async def test_models_list_traits():
    """Test Models.list_traits() calls correct endpoint with params."""
    mock_client = AsyncMock()
    traits_data = {"fastest": "model-a", "best": "model-b", "default": "model-c"}
    expected = ModelTraitsResponse(object="list", type="image", data=traits_data)
    mock_client.get = AsyncMock(return_value=expected)

    models = Models(mock_client)
    result = await models.list_traits(type="image")

    mock_client.get.assert_called_once_with(
        "models/traits", params={"type": "image"}, cast_to=ModelTraitsResponse, force_direct=True
    )
    assert result is expected
    assert result.data["fastest"] == "model-a"
    assert result.data["best"] == "model-b"


@pytest.mark.asyncio
async def test_models_list_traits_no_type():
    """Test Models.list_traits() without type filter."""
    mock_client = AsyncMock()
    expected = ModelTraitsResponse(object="list", type="all", data={})
    mock_client.get = AsyncMock(return_value=expected)

    models = Models(mock_client)
    result = await models.list_traits()

    mock_client.get.assert_called_once_with(
        "models/traits", params={}, cast_to=ModelTraitsResponse, force_direct=True
    )
    assert result.data == {}


@pytest.mark.asyncio
async def test_models_list_compatibility():
    """Test Models.list_compatibility() calls correct endpoint with params."""
    mock_client = AsyncMock()
    compat_data = {"gpt-4": "llama-3.3-70b", "gpt-3.5-turbo": "qwen3-4b"}
    expected = ModelCompatibilityResponse(object="list", type="text", data=compat_data)
    mock_client.get = AsyncMock(return_value=expected)

    models = Models(mock_client)
    result = await models.list_compatibility()

    mock_client.get.assert_called_once_with(
        "models/compatibility_mapping",
        params={},
        cast_to=ModelCompatibilityResponse,
        force_direct=True,
    )
    assert result is expected
    assert result.data["gpt-4"] == "llama-3.3-70b"


@pytest.mark.asyncio
async def test_models_list_compatibility_with_type():
    """Test Models.list_compatibility() with type filter."""
    mock_client = AsyncMock()
    expected = ModelCompatibilityResponse(object="list", type="image", data={})
    mock_client.get = AsyncMock(return_value=expected)

    models = Models(mock_client)
    result = await models.list_compatibility(type="image")

    mock_client.get.assert_called_once_with(
        "models/compatibility_mapping",
        params={"type": "image"},
        cast_to=ModelCompatibilityResponse,
        force_direct=True,
    )
    assert result.type == "image"
