"""
Unit tests for ``video.queue`` + ``video.quote`` accepting typed ``VideoElement``.

The Venice API documents the ``elements`` body field as an array of
``{frontal_image_url, reference_image_urls}`` objects (see
``api-reference/endpoint/video/queue.md``). The SDK already validates the
inbound payload via the ``VideoElement`` Pydantic model on the request side;
this test pins the public ``resources.video.Video.queue`` signature to accept
both typed ``VideoElement`` instances and raw dicts, and confirms both
serialize to the same wire payload.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.resources.video import Video
from venice_ai.types.api.requests.video import VideoElement
from venice_ai.types.api.video import VideoQueueResponse


def _fake_queue_response() -> VideoQueueResponse:
    return VideoQueueResponse(model="kling-o3-r2v", queue_id="q_fake")


@pytest.fixture
def video_resource() -> Video:
    client = MagicMock()
    client.post = AsyncMock(return_value=_fake_queue_response())
    return Video(client)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_queue_accepts_video_element_instances(video_resource: Video) -> None:
    element = VideoElement(
        frontal_image_url="https://example.com/a.png",
        reference_image_urls=["https://example.com/b.png"],
    )
    await video_resource.submit(
        model="kling-o3-r2v",
        prompt="A character walking",
        duration_seconds="5s",
        elements=[element],
    )

    body = video_resource._client.post.call_args.kwargs["json_data"]  # type: ignore[attr-defined]
    assert body["elements"] == [
        {
            "frontal_image_url": "https://example.com/a.png",
            "reference_image_urls": ["https://example.com/b.png"],
        }
    ]


@pytest.mark.asyncio
async def test_queue_accepts_raw_dicts_and_matches_typed(video_resource: Video) -> None:
    await video_resource.submit(
        model="kling-o3-r2v",
        prompt="A character walking",
        duration_seconds="5s",
        elements=[
            {
                "frontal_image_url": "https://example.com/a.png",
                "reference_image_urls": ["https://example.com/b.png"],
            }
        ],
    )

    body = video_resource._client.post.call_args.kwargs["json_data"]  # type: ignore[attr-defined]
    assert body["elements"] == [
        {
            "frontal_image_url": "https://example.com/a.png",
            "reference_image_urls": ["https://example.com/b.png"],
        }
    ]


@pytest.mark.asyncio
async def test_queue_mixes_typed_and_dict_elements(video_resource: Video) -> None:
    typed = VideoElement(
        frontal_image_url="https://example.com/a.png",
        reference_image_urls=None,
    )
    await video_resource.submit(
        model="kling-o3-r2v",
        prompt="Two characters",
        duration_seconds="5s",
        elements=[
            typed,
            {
                "frontal_image_url": "https://example.com/c.png",
                "reference_image_urls": ["https://example.com/d.png"],
            },
        ],
    )

    body = video_resource._client.post.call_args.kwargs["json_data"]  # type: ignore[attr-defined]
    assert len(body["elements"]) == 2
    assert body["elements"][0]["frontal_image_url"] == "https://example.com/a.png"
    assert body["elements"][1]["frontal_image_url"] == "https://example.com/c.png"
