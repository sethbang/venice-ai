"""Regression test: ``negative_prompt`` is no longer a parameter on image.create.

Venice disabled the field for image models in February 2026; v2.0.0
removed it entirely (see CHANGELOG). This test locks the regression: if
anyone re-adds the keyword, the signature inspection here flips it red.
Video generation is unaffected and still accepts ``negative_prompt``.
"""

from __future__ import annotations

import inspect

import pytest

from venice_ai.resources.image import Image
from venice_ai.types.api.requests.images import ImageGenerationRequest


def test_image_create_signature_has_no_negative_prompt() -> None:
    sig = inspect.signature(Image.create)
    assert "negative_prompt" not in sig.parameters, (
        "negative_prompt was removed in v2.0.0 — see CHANGELOG entry"
    )


def test_image_generation_request_model_has_no_negative_prompt() -> None:
    fields = ImageGenerationRequest.model_fields
    assert "negative_prompt" not in fields, (
        "ImageGenerationRequest.negative_prompt was removed in v2.0.0"
    )


def test_calling_image_create_with_negative_prompt_raises_type_error() -> None:
    """Behavioral check — TypeError fires before any HTTP code runs."""
    image = Image.__new__(Image)  # type: ignore[call-arg]  - skip __init__, we won't call methods
    with pytest.raises(TypeError, match="negative_prompt"):
        # Use a simple coroutine call that we never await; the binding
        # check happens at call site.
        coro = image.create(  # type: ignore[call-arg]
            model="m",
            prompt="p",
            negative_prompt="ugly",  # type: ignore[call-arg]
        )
        coro.close()
