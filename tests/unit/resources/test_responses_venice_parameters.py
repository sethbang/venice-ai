"""TDD: /responses must not send chat-only venice_parameters keys (audit MED #4).

Swagger ResponsesRequest.venice_parameters documents 7 properties; the SDK
reuses the shared chat VeniceParameters model, which also carries chat-only
fields (strip_thinking_response, disable_thinking,
return_search_results_as_documents, enable_x_search). Those must be stripped
from the /responses body.
"""

from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from venice_ai.resources.responses import Responses

CHAT_ONLY = (
    "strip_thinking_response",
    "disable_thinking",
    "return_search_results_as_documents",
    "enable_x_search",
)


@pytest.fixture
def responses_resource() -> Responses:
    client = Mock()
    client.post = AsyncMock(return_value=Mock())
    return Responses(client)


@pytest.mark.asyncio
async def test_strips_chat_only_venice_parameters(responses_resource: Responses):
    await responses_resource.create(
        model="m",
        input="hi",
        venice_parameters={
            "enable_web_search": "auto",  # documented for /responses — keep
            "strip_thinking_response": True,
            "disable_thinking": True,
            "return_search_results_as_documents": True,
            "enable_x_search": True,
        },
    )
    body = cast(Any, responses_resource._client.post).call_args.kwargs["json_data"]
    vp = body.get("venice_parameters", {})
    assert vp.get("enable_web_search") == "auto"
    for k in CHAT_ONLY:
        assert k not in vp, f"{k} is chat-only and must be stripped for /responses"
