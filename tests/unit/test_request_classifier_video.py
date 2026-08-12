"""TDD: async video-job endpoints must classify as VIDEO, not default to LLM.

Only ``video/transcriptions`` was mapped to ResourceType.VIDEO; the async
generation lifecycle (``video/queue|quote|retrieve|complete``) fell through to
the LLM default, mis-categorising those requests for queue/rate-limit routing.
"""

from unittest.mock import MagicMock

import pytest

from venice_ai._queue_types import ResourceType
from venice_ai._request_classifier import RequestClassifier


@pytest.fixture
def classifier():
    return RequestClassifier(MagicMock())


@pytest.mark.parametrize(
    "endpoint", ["video/queue", "video/quote", "video/retrieve", "video/complete"]
)
@pytest.mark.asyncio
async def test_async_video_job_endpoints_route_to_video(classifier, endpoint):
    md = await classifier.classify({"endpoint": endpoint, "model": "unknown"})
    assert md.resource_type == ResourceType.VIDEO, f"{endpoint} -> {md.resource_type}"


@pytest.mark.parametrize(
    "endpoint",
    ["video/queue", "video/quote", "video/retrieve", "video/complete", "video/transcriptions"],
)
def test_video_endpoints_determine_resource_type(classifier, endpoint):
    assert classifier._determine_resource_type(endpoint, "any-model") == ResourceType.VIDEO
