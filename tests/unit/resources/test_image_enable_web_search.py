"""
Unit tests for ``Image.generate(enable_web_search=...)``.

The Venice API documents ``enable_web_search`` as an optional body field on
POST /image/generate (see ``api-reference/endpoint/image/generate.md``).
The SDK must accept this kwarg and forward it verbatim in the JSON body.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.resources.image import Image
from venice_ai.types.api import ImageGenerationResponse
from venice_ai.types.api.base import TimingInfo


def _fake_response() -> ImageGenerationResponse:
    return ImageGenerationResponse(
        id="gen-ws",
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
    return Image(client)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_generate_forwards_enable_web_search_true(image_resource: Image) -> None:
    await image_resource.create(
        model="some-image-model",
        prompt="A research lab at dusk",
        enable_web_search=True,
    )

    kwargs = image_resource._client.post.call_args.kwargs  # type: ignore[attr-defined]
    body = kwargs["json_data"]
    assert body["enable_web_search"] is True


@pytest.mark.asyncio
async def test_generate_forwards_enable_web_search_false(image_resource: Image) -> None:
    await image_resource.create(
        model="some-image-model",
        prompt="A quiet beach",
        enable_web_search=False,
    )

    kwargs = image_resource._client.post.call_args.kwargs  # type: ignore[attr-defined]
    body = kwargs["json_data"]
    assert body["enable_web_search"] is False


@pytest.mark.asyncio
async def test_generate_omits_enable_web_search_when_not_set(image_resource: Image) -> None:
    await image_resource.create(
        model="some-image-model",
        prompt="A quiet beach",
    )

    kwargs = image_resource._client.post.call_args.kwargs  # type: ignore[attr-defined]
    body = kwargs["json_data"]
    assert "enable_web_search" not in body
