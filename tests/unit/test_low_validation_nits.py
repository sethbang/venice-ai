"""TDD: LOW validation/wire nits (audit) — music duration integer, billing lookback
regex, embeddings dead default, augment text Accept header."""

from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from venice_ai.resources.augment import Augment
from venice_ai.resources.billing import Billing
from venice_ai.types.api.common import ErrorDetails
from venice_ai.types.api.requests.embeddings import EmbeddingsRequest


def test_embeddings_encoding_format_default_is_none():
    # default "float" was unreachable dead code (create() always passes it)
    assert EmbeddingsRequest(input="hi", model="m").encoding_format is None


def test_error_details_populates_errors_from_wire_underscore_key():
    # Wire shape is the Zod tree {"_errors": [...], "<field>": {"_errors": [...]}}.
    # `.errors` must populate from the documented `_errors` key.
    details = ErrorDetails.model_validate(
        {"_errors": ["top-level problem"], "model": {"_errors": ["Required"]}}
    )
    assert details.errors == ["top-level problem"]
    # The nested per-field tree is preserved via extra="allow".
    assert details.model_extra is not None
    assert details.model_extra.get("model") == {"_errors": ["Required"]}


def test_error_details_still_constructible_by_field_name():
    # populate_by_name keeps the existing `errors=` construction path working.
    details = ErrorDetails(errors=["manual"])
    assert details.errors == ["manual"]


@pytest.mark.asyncio
async def test_billing_lookback_rejects_leading_zero():
    b = Billing(Mock(get=AsyncMock()))
    # get_usage_analytics wraps a beta endpoint and always emits a FutureWarning;
    # assert it here so the beta signal stays pinned instead of leaking as noise.
    with pytest.warns(FutureWarning, match="beta API endpoint"), pytest.raises(ValueError):
        await b.get_usage_analytics(lookback="07d")  # swagger pattern ^[1-9]\d*d$


@pytest.mark.asyncio
async def test_augment_text_format_requests_text_accept():
    aug = Augment(Mock())
    aug._request_multipart = AsyncMock(return_value=b"plain text")  # type: ignore[method-assign]
    await aug.parse_text(file=b"hello world", response_format="text")
    headers = cast(Any, aug._request_multipart).call_args.kwargs.get("headers") or {}
    assert headers.get("Accept", "").startswith("text/")
