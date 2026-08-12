"""Unit tests for a per-call ``timeout`` on ``image.create``.

``image.upscale`` already accepts a per-call ``timeout`` but ``image.create``
did not, forcing callers to construct a whole new client with a wider timeout
for slow renders (e.g. ``quality='high'``). ``create`` should accept the same
``timeout`` and forward it to the underlying request on both dispatch paths
(JSON via ``post`` and binary via ``_request``).
"""

from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from venice_ai.resources.image import Image
from venice_ai.types.api import ImageGenerationResponse
from venice_ai.types.api.base import TimingInfo


def _fake_response() -> ImageGenerationResponse:
    return ImageGenerationResponse(
        id="gen-to",
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
    client._request = AsyncMock(return_value=b"\x89PNG")
    return Image(client)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_create_forwards_timeout_json_path(image_resource: Image) -> None:
    await image_resource.create(model="some-image-model", prompt="slow render", timeout=300.0)
    kwargs = image_resource._client.post.call_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["timeout"] == 300.0


@pytest.mark.asyncio
async def test_create_forwards_timeout_binary_path(image_resource: Image) -> None:
    ct = aiohttp.ClientTimeout(total=120.0)
    await image_resource.create(
        model="some-image-model", prompt="slow render", return_binary=True, timeout=ct
    )
    kwargs = image_resource._client._request.call_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["timeout"] is ct


@pytest.mark.asyncio
async def test_create_omits_timeout_when_not_set(image_resource: Image) -> None:
    await image_resource.create(model="some-image-model", prompt="quick")
    kwargs = image_resource._client.post.call_args.kwargs  # type: ignore[attr-defined]
    assert kwargs.get("timeout") is None
