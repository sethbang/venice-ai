"""Tests for ``Image.submit()`` and the ``ImageJob`` async context manager.

Image generation is synchronous on the server (no queue endpoint), so
``ImageJob.wait()`` does the actual HTTP call lazily and caches the result.
The shape mirrors ``MusicJob`` / ``VideoJob`` so callers can render images
in parallel with a uniform async-with idiom.
"""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock

import pytest

from venice_ai.resources.image import Image, ImageJob
from venice_ai.types.api import ImageGenerationResponse


@pytest.fixture
def image_resource() -> tuple[Image, AsyncMock]:
    """Build an ``Image`` whose ``create`` method is fully mocked out."""
    client = AsyncMock()
    client._api_key = "test-key"
    image = Image(client)
    return image, client


@pytest.fixture
def fake_response() -> ImageGenerationResponse:
    """A minimal response with one base64-encoded byte payload."""
    encoded = base64.b64encode(b"fake png bytes").decode("ascii")
    return ImageGenerationResponse(
        id="img-test",
        images=[encoded],
        request={},
        timing={
            "inferenceDuration": 0.0,
            "inferencePreprocessingTime": 0.0,
            "inferenceQueueTime": 0.0,
            "total": 0.0,
        },
    )


@pytest.mark.asyncio
async def test_submit_returns_image_job_without_firing_request(
    image_resource: tuple[Image, AsyncMock],
) -> None:
    image, client = image_resource
    client.image = AsyncMock()
    client.image.create = AsyncMock(side_effect=AssertionError("must not fire yet"))

    job = await image.submit(model="venice-sd35", prompt="test")
    assert isinstance(job, ImageJob)
    assert not job.is_complete


@pytest.mark.asyncio
async def test_wait_invokes_create_and_caches(
    image_resource: tuple[Image, AsyncMock],
    fake_response: ImageGenerationResponse,
) -> None:
    image, client = image_resource
    client.image = AsyncMock()
    client.image.create = AsyncMock(return_value=fake_response)

    job = await image.submit(model="venice-sd35", prompt="test")
    async with job:
        result_a = await job.wait()
        result_b = await job.wait()  # cached

    assert result_a is fake_response
    assert result_b is fake_response
    client.image.create.assert_awaited_once()
    assert job.is_complete


@pytest.mark.asyncio
async def test_download_writes_decoded_bytes(
    image_resource: tuple[Image, AsyncMock],
    fake_response: ImageGenerationResponse,
    tmp_path,
) -> None:
    image, client = image_resource
    client.image = AsyncMock()
    client.image.create = AsyncMock(return_value=fake_response)

    job = await image.submit(model="venice-sd35", prompt="test")
    out = tmp_path / "rendered.png"
    async with job:
        saved = await job.download(out)
    assert saved == out
    assert saved.read_bytes() == b"fake png bytes"


@pytest.mark.asyncio
async def test_download_handles_binary_response(
    image_resource: tuple[Image, AsyncMock],
    tmp_path,
) -> None:
    """When ``return_binary=True`` the response is raw bytes, not a model."""
    image, client = image_resource
    client.image = AsyncMock()
    client.image.create = AsyncMock(return_value=b"binary png stream")

    job = await image.submit(
        model="venice-sd35",
        prompt="test",
        return_binary=True,
    )
    out = tmp_path / "bin.png"
    async with job:
        await job.download(out)
    assert out.read_bytes() == b"binary png stream"


@pytest.mark.asyncio
async def test_drops_none_kwargs_so_create_sees_omitted_args(
    image_resource: tuple[Image, AsyncMock],
    fake_response: ImageGenerationResponse,
) -> None:
    """``submit(width=None)`` should be equivalent to omitting width entirely."""
    image, client = image_resource
    client.image = AsyncMock()
    client.image.create = AsyncMock(return_value=fake_response)

    job = await image.submit(
        model="venice-sd35",
        prompt="test",
        width=None,
        height=None,
    )
    async with job:
        await job.wait()

    call_kwargs = client.image.create.await_args.kwargs
    assert "width" not in call_kwargs
    assert "height" not in call_kwargs
    assert call_kwargs["model"] == "venice-sd35"
    assert call_kwargs["prompt"] == "test"


def test_image_job_is_re_exported_from_top_level() -> None:
    from venice_ai import ImageJob as TopLevelImageJob

    assert TopLevelImageJob is ImageJob
