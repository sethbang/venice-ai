from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest
from pydantic import BaseModel

from venice_ai._client import VeniceClient
from venice_ai.exceptions import APIError, VeniceError


# Mock class for cast_to
class MockModel(BaseModel):
    id: str
    choices: list


def _make_streaming_response(lines: list[bytes]) -> AsyncMock:
    """Build a mock aiohttp.ClientResponse that streams ``lines`` via .content."""
    mock_response = AsyncMock(spec=aiohttp.ClientResponse)
    mock_response._body = None  # Real (non-VCR) response path
    mock_response.status = 200  # In-band error: HTTP itself succeeded
    mock_response.headers = {}
    mock_response.content = AsyncMock()
    mock_response.content.__aiter__.return_value = lines
    mock_response.content.read.return_value = b""
    mock_response.close = Mock()  # close() is synchronous in aiohttp
    return mock_response


@pytest.mark.asyncio
async def test_stream_request_uses_iteration():
    """
    Test that _stream_request iterates over response.content instead of reading it all at once.
    """
    client = VeniceClient(api_key="test")

    # Mock response
    mock_response = AsyncMock(spec=aiohttp.ClientResponse)
    mock_response._body = None  # Explicitly set to None to simulate real response
    mock_response.content = AsyncMock()
    mock_response.close = Mock()  # close() is synchronous in aiohttp

    # Setup async iterator for content
    # Note: aiohttp response.content yields bytes lines
    lines = [
        b'data: {"id": "1", "choices": [{"delta": {"content": "Hello"}}]}\n',
        b"\n",
        b'data: {"id": "2", "choices": [{"delta": {"content": " World"}}]}\n',
        b"\n",
        b"data: [DONE]\n",
    ]

    # AsyncMock.__aiter__ expects an iterable (list), not an async generator
    mock_response.content.__aiter__.return_value = lines

    # Ensure read() is NOT called (or we can check it later)
    # But for the current implementation, read() IS called.
    # We will configure read() to return empty bytes to simulate "streaming only"
    # if we wanted to fail the current implementation, but the current implementation
    # relies on read().

    # To verify the FIX, we want to ensure read() is NOT called, and iteration IS used.

    # Mock _prepare_and_send_request
    client._prepare_and_send_request = AsyncMock(return_value=mock_response)

    # Call _stream_request
    chunks = []
    async for chunk in client._stream_request("POST", "/chat/completions", cast_to=MockModel):
        chunks.append(chunk)

    # In the NEW implementation, this should work and produce 2 chunks.
    # In the OLD implementation, this might fail if we don't mock read(),
    # or if we mock read() to return nothing.

    # Let's mock read() to return nothing to prove the old implementation fails
    mock_response.content.read.return_value = b""

    assert len(chunks) == 2
    assert chunks[0].choices[0]["delta"]["content"] == "Hello"
    assert chunks[1].choices[0]["delta"]["content"] == " World"


@pytest.mark.asyncio
async def test_stream_request_vcr_fallback():
    """
    Test that VCR fallback still works if streaming yields nothing.
    """
    client = VeniceClient(api_key="test")

    mock_response = AsyncMock()
    mock_response.content = AsyncMock()
    mock_response.close = Mock()  # close() is synchronous

    # Empty iterator
    async def empty_iterator():
        if False:
            yield b""

    mock_response.content.__aiter__.return_value = empty_iterator()
    mock_response.content.read.return_value = b""

    # Setup VCR body
    vcr_content = b'data: {"id": "1", "choices": [{"delta": {"content": "VCR"}}]}\n\ndata: [DONE]\n'
    mock_response._body = vcr_content

    client._prepare_and_send_request = AsyncMock(return_value=mock_response)

    chunks = []
    async for chunk in client._stream_request("POST", "/chat/completions", cast_to=MockModel):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0].choices[0]["delta"]["content"] == "VCR"


# ---------------------------------------------------------------------------
# In-band SSE error frames must not be silently swallowed. A ``data:`` frame
# whose JSON fails to validate as a chat chunk (e.g. ``data: {"error": "..."}``)
# must RAISE rather than being dropped at DEBUG level. Otherwise the stream
# simply ends early, indistinguishable from a complete response. These tests
# assert the SDK RAISES instead.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_raises_on_inband_string_error_frame():
    """A mid-stream ``data: {"error": "<str>"}`` frame must raise, not end silently."""
    client = VeniceClient(api_key="test")

    lines = [
        b'data: {"id": "1", "choices": [{"delta": {"content": "Hello"}}]}\n',
        b"\n",
        b'data: {"id": "2", "choices": [{"delta": {"content": " World"}}]}\n',
        b"\n",
        b'data: {"error": "context_length_exceeded"}\n',
        b"\n",
        b"data: [DONE]\n",
    ]
    mock_response = _make_streaming_response(lines)
    client._prepare_and_send_request = AsyncMock(return_value=mock_response)

    chunks = []
    with pytest.raises(VeniceError) as exc_info:
        async for chunk in client._stream_request("POST", "/chat/completions", cast_to=MockModel):
            chunks.append(chunk)

    # Error message must surface the in-band error payload.
    assert "context_length_exceeded" in str(exc_info.value)
    # Chunks received before the error frame must still have been yielded
    # (so callers know the answer is truncated, not absent).
    assert len(chunks) == 2
    assert chunks[0].choices[0]["delta"]["content"] == "Hello"
    assert chunks[1].choices[0]["delta"]["content"] == " World"


@pytest.mark.asyncio
async def test_stream_raises_on_inband_nested_error_object():
    """A nested ``data: {"error": {"message": .., "code": ..}}`` frame must raise."""
    client = VeniceClient(api_key="test")

    lines = [
        b'data: {"id": "1", "choices": [{"delta": {"content": "Hi"}}]}\n',
        b"\n",
        b'data: {"error": {"message": "Inference failed mid-stream", '
        b'"code": "INFERENCE_FAILED"}}\n',
        b"\n",
        b"data: [DONE]\n",
    ]
    mock_response = _make_streaming_response(lines)
    client._prepare_and_send_request = AsyncMock(return_value=mock_response)

    chunks = []
    with pytest.raises(APIError) as exc_info:
        async for chunk in client._stream_request("POST", "/chat/completions", cast_to=MockModel):
            chunks.append(chunk)

    assert "Inference failed mid-stream" in str(exc_info.value)
    # The structured error code from the envelope should be preserved.
    assert exc_info.value.code == "INFERENCE_FAILED"
    assert len(chunks) == 1
    assert chunks[0].choices[0]["delta"]["content"] == "Hi"


@pytest.mark.asyncio
async def test_stream_ignores_benign_noise_and_done():
    """Keepalive comments, blank lines, non-JSON noise and [DONE] must NOT raise.

    Guards the happy path: only frames carrying an ``error`` payload should
    raise; genuinely-malformed/benign lines are still skipped silently.
    """
    client = VeniceClient(api_key="test")

    lines = [
        b": keepalive comment\n",  # SSE comment line (not a data: frame)
        b"\n",
        b'data: {"id": "1", "choices": [{"delta": {"content": "A"}}]}\n',
        b"data: not-json-keepalive\n",  # malformed data: noise -> skipped
        b'data: {"id": "2", "choices": [{"delta": {"content": "B"}}]}\n',
        b"data: [DONE]\n",
    ]
    mock_response = _make_streaming_response(lines)
    client._prepare_and_send_request = AsyncMock(return_value=mock_response)

    chunks = []
    async for chunk in client._stream_request("POST", "/chat/completions", cast_to=MockModel):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].choices[0]["delta"]["content"] == "A"
    assert chunks[1].choices[0]["delta"]["content"] == "B"
