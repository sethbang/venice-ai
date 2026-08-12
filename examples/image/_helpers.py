"""
Shared helpers for Venice AI image examples.

This module hosts ``generate_base_image`` — a convenience wrapper used by
the image editing and multi-edit example scripts to produce a self-contained
source image. Magic-byte format detection lives in the SDK proper as
``venice_ai.detect_image_format``.
"""

from venice_ai import VeniceClient
from venice_ai.types.api import ImageGenerationResponse


async def generate_base_image(
    client: "VeniceClient",
    prompt: str,
    *,
    width: int = 512,
    height: int = 512,
) -> bytes:
    """Generate a base image and return its raw bytes.

    This is a convenience wrapper that selects an image model automatically,
    generates a single image, and returns the decoded bytes — useful for
    creating self-contained examples that need a source image to edit.
    Use :func:`venice_ai.detect_image_format` to pick the file extension at
    write time, since the API may return PNG, WebP, or JPEG depending on
    the model.
    """
    image_model = await client.models.resolve_image()
    print(f"  📍 Using generation model: {image_model}")

    response: ImageGenerationResponse = await client.image.create(
        model=image_model,
        prompt=prompt,
        width=width,
        height=height,
        num_images=1,
        return_binary=False,
    )
    return response.bytes(0)
