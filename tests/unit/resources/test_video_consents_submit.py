"""TDD: ``submit()``/``run()`` forward Seedance consents into the /video/queue body.

Closes the user-facing half of the consents gap — the typed model exists, but
callers reach the queue through ``Video.submit``/``Video.run``, which build the
request from kwargs.
"""

from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from venice_ai.resources.video import Video
from venice_ai.types.api import VideoQueueResponse

CONSENTS = {
    "seedance": {
        "confirmed_terms_and_privacy": True,
        "confirmed_legal_right": True,
        "confirmed_screening_acknowledged": True,
    }
}


@pytest.fixture
def video_resource() -> Video:
    client = Mock()
    client.post = AsyncMock()
    return Video(client)


@pytest.mark.asyncio
async def test_submit_forwards_consents(video_resource: Video) -> None:
    post = cast(Any, video_resource._client.post)
    post.return_value = VideoQueueResponse(model="seedance-2-0-text-to-video", queue_id="q-consent")

    await video_resource.submit(
        model="seedance-2-0-text-to-video",
        prompt="a person waving at the camera",
        duration_seconds=5,
        consents=CONSENTS,
    )

    body = post.call_args.kwargs["json_data"]
    assert body["consents"]["seedance"] == CONSENTS["seedance"]


@pytest.mark.asyncio
async def test_submit_omits_consents_when_absent(video_resource: Video) -> None:
    post = cast(Any, video_resource._client.post)
    post.return_value = VideoQueueResponse(model="seedance-2-0-text-to-video", queue_id="q-none")

    await video_resource.submit(
        model="seedance-2-0-text-to-video",
        prompt="a calm ocean",
        duration_seconds=5,
    )

    assert "consents" not in post.call_args.kwargs["json_data"]
