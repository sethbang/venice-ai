"""Additional tests for ``venice_ai.resources.augment``.

Exercises:

- ``_sniff_content_type`` ``UnicodeDecodeError`` branch
- the ``parse_text`` overload stub returns (``...``) — exercised only by
  static type checkers; covered with ``pragma: no cover`` clarifications via
  dedicated calls
- ``parse_text`` text-mode ``str``/``dict``/``TypeError`` paths
- ``parse_text`` json-mode ``bytes``/``TypeError`` paths
- ``_read_file`` ``BytesIO`` and file-like / async-read paths

All tests use mocked transport — no live API calls. The fixtures match
``test_augment.py`` style.
"""

from __future__ import annotations

import io
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.resources.augment import Augment, _sniff_content_type
from venice_ai.types.api.augment import AugmentTextParserResponse


@pytest.fixture
def augment() -> Augment:
    client = MagicMock()
    return Augment(client)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _sniff_content_type — 68->75, 71-72 (UnicodeDecodeError branch)
# ---------------------------------------------------------------------------


def test_sniff_content_type_invalid_utf8_returns_none() -> None:
    """Bytes that fail UTF-8 decode (lines 71-72) hit the ``return None`` exit."""
    # Bytes that aren't PDF, aren't ZIP, but trip UnicodeDecodeError on .decode("utf-8").
    # 0xFF 0xFE 0xFD is invalid UTF-8 leading bytes.
    invalid = b"\xff\xfe\xfd\xfc"
    assert _sniff_content_type(invalid) is None


def test_sniff_content_type_empty_bytes_returns_none() -> None:
    """Empty content (line 68 falsy guard → 75) returns None."""
    # When content is empty, `sample` is empty and we fall through to the final
    # ``return None``. This exercises the 68->75 branch.
    assert _sniff_content_type(b"") is None


def test_sniff_content_type_unprintable_text_returns_none() -> None:
    """Decodable but non-printable bytes still return None (75 fallthrough)."""
    # Decodes cleanly as ASCII but contains a control character (0x01, not in
    # \r\n\t), so the ``all(c.isprintable() ...)`` guard fails and we hit
    # the final ``return None``.
    assert _sniff_content_type(b"hello\x01world") is None


# ---------------------------------------------------------------------------
# parse_text response_format="text" — lines 242-246
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_text_text_mode_with_str_response(augment: Augment) -> None:
    """Line 242-243: text mode + ``str`` response returns the string verbatim."""
    augment._request_multipart = AsyncMock(return_value="already-a-string")  # type: ignore[method-assign]

    result = await augment.parse_text(file=b"PDF-bytes", response_format="text")

    assert result == "already-a-string"


@pytest.mark.asyncio
async def test_parse_text_text_mode_with_dict_response(augment: Augment) -> None:
    """Line 244-245: text mode + dict containing 'text' key returns dict['text']."""
    augment._request_multipart = AsyncMock(return_value={"text": "from-dict"})  # type: ignore[method-assign]

    result = await augment.parse_text(file=b"PDF-bytes", response_format="text")

    assert result == "from-dict"


@pytest.mark.asyncio
async def test_parse_text_text_mode_with_unexpected_type_raises(augment: Augment) -> None:
    """Line 246: text mode + unexpected response type raises TypeError."""
    augment._request_multipart = AsyncMock(return_value=12345)  # type: ignore[method-assign]

    with pytest.raises(TypeError, match="Unexpected text-parser response type"):
        await augment.parse_text(file=b"PDF-bytes", response_format="text")


# ---------------------------------------------------------------------------
# parse_text response_format="json" — lines 250-254
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_text_json_mode_with_bytes_response(augment: Augment) -> None:
    """Lines 250-253: JSON mode + raw bytes response decodes JSON and validates."""
    augment._request_multipart = AsyncMock(  # type: ignore[method-assign]
        return_value=b'{"text": "from-bytes", "tokens": 7}'
    )

    result = await augment.parse_text(file=b"PDF-bytes")

    assert isinstance(result, AugmentTextParserResponse)
    assert result.text == "from-bytes"
    assert result.tokens == 7


@pytest.mark.asyncio
async def test_parse_text_json_mode_with_unexpected_type_raises(augment: Augment) -> None:
    """Line 254: JSON mode + unexpected response type raises TypeError."""
    augment._request_multipart = AsyncMock(return_value=12345)  # type: ignore[method-assign]

    with pytest.raises(TypeError, match="Unexpected text-parser response type"):
        await augment.parse_text(file=b"PDF-bytes", response_format="json")


# ---------------------------------------------------------------------------
# _read_file — lines 268-283 (BytesIO + file-like / async-read)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_file_accepts_bytesio(augment: Augment) -> None:
    """Line 268-269: BytesIO is read via ``.read()`` and yields filename 'document'."""
    augment._request_multipart = AsyncMock(return_value={"text": "x", "tokens": 1})  # type: ignore[method-assign]
    buf = io.BytesIO(b"%PDF-1.4 in-memory")

    await augment.parse_text(file=buf)

    files = augment._request_multipart.call_args.kwargs["files"]  # type: ignore[attr-defined]
    filename, content, content_type = files["file"]
    # The default filename for BytesIO is "document" (no extension), so the
    # SDK falls through to magic-byte sniffing → "application/pdf".
    assert filename == "document"
    assert content == b"%PDF-1.4 in-memory"
    assert content_type == "application/pdf"


@pytest.mark.asyncio
async def test_read_file_accepts_filelike_with_name(augment: Augment) -> None:
    """Lines 270-282: arbitrary file-like with ``.name`` resolves filename."""

    class _FakeFile:
        name = "/tmp/uploads/quarterly.pdf"

        def read(self) -> bytes:
            return b"%PDF-1.4 fakey-fake"

    augment._request_multipart = AsyncMock(return_value={"text": "x", "tokens": 1})  # type: ignore[method-assign]

    await augment.parse_text(file=_FakeFile())  # type: ignore[arg-type]

    files = augment._request_multipart.call_args.kwargs["files"]  # type: ignore[attr-defined]
    filename, content, content_type = files["file"]
    assert filename == "quarterly.pdf"
    assert content == b"%PDF-1.4 fakey-fake"
    assert content_type == "application/pdf"


@pytest.mark.asyncio
async def test_read_file_accepts_filelike_with_async_read(augment: Augment) -> None:
    """Lines 273-274: file-like whose .read() returns a coroutine is awaited."""

    class _AsyncFile:
        name = "data.txt"

        async def read(self) -> bytes:
            return b"hello world"

    augment._request_multipart = AsyncMock(return_value={"text": "h", "tokens": 1})  # type: ignore[method-assign]

    await augment.parse_text(file=_AsyncFile())  # type: ignore[arg-type]

    files = augment._request_multipart.call_args.kwargs["files"]  # type: ignore[attr-defined]
    filename, content, content_type = files["file"]
    assert filename == "data.txt"
    assert content == b"hello world"
    assert content_type == "text/plain"


@pytest.mark.asyncio
async def test_read_file_filelike_returning_non_bytes_raises(augment: Augment) -> None:
    """Line 277-278: file-like returning a non-bytes value raises TypeError."""

    class _BadFile:
        def read(self) -> Any:
            return "this is a str, not bytes"

    with pytest.raises(TypeError, match="must return bytes"):
        await augment.parse_text(file=_BadFile())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_read_file_filelike_without_name_uses_document(augment: Augment) -> None:
    """Lines 279-281: file-like without ``.name`` defaults to 'document'."""

    class _NoNameFile:
        def read(self) -> bytes:
            return b"%PDF-1.4 anonymous"

    augment._request_multipart = AsyncMock(return_value={"text": "x", "tokens": 1})  # type: ignore[method-assign]

    await augment.parse_text(file=_NoNameFile())  # type: ignore[arg-type]

    files = augment._request_multipart.call_args.kwargs["files"]  # type: ignore[attr-defined]
    filename, _, content_type = files["file"]
    assert filename == "document"
    assert content_type == "application/pdf"


@pytest.mark.asyncio
async def test_read_file_unsupported_type_raises(augment: Augment) -> None:
    """Line 283: an unsupported file argument type raises TypeError."""
    with pytest.raises(TypeError, match="Unsupported file type"):
        await augment.parse_text(file=12345)  # type: ignore[arg-type]
