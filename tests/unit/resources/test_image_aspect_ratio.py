"""
Unit tests for ``aspect_ratio`` on image generate + edit.

The Venice API documents ``aspect_ratio`` as an optional body field on both
POST /image/generate and POST /image/edit (see
``api-reference/endpoint/image/{generate,edit}.md``). The SDK must accept
this kwarg and forward it verbatim in the JSON body.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.resources.image import Image
from venice_ai.types.api import ImageGenerationResponse
from venice_ai.types.api.base import TimingInfo


def _fake_response() -> ImageGenerationResponse:
    return ImageGenerationResponse(
        id="gen-ar",
        images=["ZmFrZQ=="],
        request=None,
        timing=TimingInfo(
            inferenceDuration=0, inferencePreprocessingTime=0, inferenceQueueTime=0, total=0
        ),
    )


@pytest.fixture
def image_resource() -> Image:
    client = MagicMock()
    client.post = AsyncMock(return_value=_fake_response())
    # For edit(), which calls _request directly, return raw bytes
    client._request = AsyncMock(return_value=b"\x89PNG")
    return Image(client)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_generate_forwards_aspect_ratio(image_resource: Image) -> None:
    await image_resource.create(
        model="some-image-model",
        prompt="A serene landscape",
        aspect_ratio="16:9",
    )

    kwargs = image_resource._client.post.call_args.kwargs  # type: ignore[attr-defined]
    body = kwargs["json_data"]
    assert body["aspect_ratio"] == "16:9"


@pytest.mark.asyncio
async def test_generate_omits_aspect_ratio_when_not_set(image_resource: Image) -> None:
    await image_resource.create(
        model="some-image-model",
        prompt="A quiet beach",
    )

    kwargs = image_resource._client.post.call_args.kwargs  # type: ignore[attr-defined]
    body = kwargs["json_data"]
    assert "aspect_ratio" not in body


@pytest.mark.asyncio
async def test_edit_forwards_aspect_ratio(image_resource: Image) -> None:
    await image_resource.edit(
        prompt="Make it cinematic",
        image="https://example.com/photo.jpg",
        aspect_ratio="21:9",
    )

    kwargs = image_resource._client._request.call_args.kwargs  # type: ignore[attr-defined]
    body = kwargs["json_data"]
    assert body["aspect_ratio"] == "21:9"


@pytest.mark.asyncio
async def test_edit_omits_aspect_ratio_when_not_set(image_resource: Image) -> None:
    await image_resource.edit(
        prompt="Make it cinematic",
        image="https://example.com/photo.jpg",
    )

    kwargs = image_resource._client._request.call_args.kwargs  # type: ignore[attr-defined]
    body = kwargs["json_data"]
    assert "aspect_ratio" not in body
