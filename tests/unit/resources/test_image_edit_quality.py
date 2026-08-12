"""TDD: image edit `quality` param (audit MED #5).

edit.md prose documents `quality` (low/medium/high) for quality-aware edit
models (e.g. gpt-image-2-edit); multi_edit already supports it. Added as an
opt-in param (sent only when set) — model-dependent, mirroring multi_edit.
"""

from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from venice_ai.resources.image import Image
from venice_ai.types.api.requests.images import ImageEditRequest


def test_edit_request_accepts_quality():
    body = ImageEditRequest(prompt="p", image="https://e.com/a.png", quality="high").model_dump(
        exclude_none=True
    )
    assert body["quality"] == "high"


def test_edit_request_omits_quality_by_default():
    body = ImageEditRequest(prompt="p", image="https://e.com/a.png").model_dump(exclude_none=True)
    assert "quality" not in body


@pytest.fixture
def image_resource() -> Image:
    client = Mock()
    client._request = AsyncMock(return_value=b"img-bytes")
    return Image(client)


@pytest.mark.asyncio
async def test_edit_forwards_quality(image_resource: Image):
    await image_resource.edit(prompt="p", image="https://e.com/a.png", quality="medium")
    payload = cast(Any, image_resource._client._request).call_args.kwargs["json_data"]
    assert payload["quality"] == "medium"
