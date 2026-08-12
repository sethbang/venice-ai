"""Unit tests for ``Image.simple_generate()``.

The Venice API exposes an OpenAI-compatible ``POST /images/generations``
endpoint (operationId ``simpleGenerateImage``) distinct from the native
``POST /image/generate``. The SDK exposes it as :meth:`Image.simple_generate`.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.resources.image import Image
from venice_ai.types.api import (
    SimpleImageData,
    SimpleImageGenerationRequest,
    SimpleImageGenerationResponse,
)


def _fake_response() -> SimpleImageGenerationResponse:
    return SimpleImageGenerationResponse(
        created=1700000000,
        data=[SimpleImageData(b64_json="ZmFrZQ==", url=None)],
    )


@pytest.fixture
def image_resource() -> Image:
    client = MagicMock()
    client.post = AsyncMock(return_value=_fake_response())
    return Image(client)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_simple_generate_posts_to_openai_compat_path(image_resource: Image) -> None:
    """simple_generate() targets ``images/generations`` with cast_to set."""
    await image_resource.simple_generate(prompt="A red square", model="default")

    args = image_resource._client.post.call_args.args  # type: ignore[attr-defined]
    kwargs = image_resource._client.post.call_args.kwargs  # type: ignore[attr-defined]
    assert args[0] == "images/generations"
    assert kwargs["cast_to"] is SimpleImageGenerationResponse


@pytest.mark.asyncio
async def test_simple_generate_forwards_all_optional_params(image_resource: Image) -> None:
    """Optional OpenAI-compat params land in the request body verbatim."""
    await image_resource.simple_generate(
        prompt="A red square",
        model="default",
        n=1,
        size="1024x1024",
        response_format="url",
        output_format="png",
        quality="high",
        style="natural",
        background="transparent",
        moderation="auto",
        output_compression=80,
        user="user_123",
    )

    body = image_resource._client.post.call_args.kwargs["json_data"]  # type: ignore[attr-defined]
    assert body["prompt"] == "A red square"
    assert body["model"] == "default"
    assert body["n"] == 1
    assert body["size"] == "1024x1024"
    assert body["response_format"] == "url"
    assert body["output_format"] == "png"
    assert body["quality"] == "high"
    assert body["style"] == "natural"
    assert body["background"] == "transparent"
    assert body["moderation"] == "auto"
    assert body["output_compression"] == 80
    assert body["user"] == "user_123"


@pytest.mark.asyncio
async def test_simple_generate_omits_none_params(image_resource: Image) -> None:
    """None-valued kwargs are excluded from the body."""
    await image_resource.simple_generate(prompt="A red square")

    body = image_resource._client.post.call_args.kwargs["json_data"]  # type: ignore[attr-defined]
    # Only ``prompt`` is required and explicitly passed.
    assert body == {"prompt": "A red square"}


@pytest.mark.asyncio
async def test_simple_generate_returns_pydantic_response(image_resource: Image) -> None:
    """Return value is the SimpleImageGenerationResponse from the client."""
    result = await image_resource.simple_generate(prompt="A red square")
    assert isinstance(result, SimpleImageGenerationResponse)
    assert result.created == 1700000000
    assert result.data[0].b64_json == "ZmFrZQ=="


def test_simple_image_request_rejects_oversized_prompt() -> None:
    """The 1500-char ceiling per the OpenAPI spec is enforced."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SimpleImageGenerationRequest.model_validate({"prompt": "x" * 1501})
