"""
API-related fixtures for Venice AI testing.

This module provides fixtures for mocking API responses, headers, and
various API interaction scenarios.
"""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import pytest_asyncio


@pytest.fixture
def mock_api_response():
    """Create a mock API response with customizable attributes."""

    def _create_response(
        status_code: int = 200,
        json_data: dict[str, Any] | None = None,
        text: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> Mock:
        """
        Create a mock response object.

        Args:
            status_code: HTTP status code
            json_data: JSON response data
            text: Text response data
            headers: Response headers

        Returns:
            Mock response object
        """
        response = Mock()
        response.status = status_code
        response.status_code = status_code
        response.headers = headers or {}

        if json_data is not None:
            response.json = AsyncMock(return_value=json_data)
            response.text = AsyncMock(return_value=json.dumps(json_data))
        elif text is not None:
            response.text = AsyncMock(return_value=text)
            response.json = AsyncMock(side_effect=json.JSONDecodeError("", "", 0))
        else:
            response.json = AsyncMock(return_value={})
            response.text = AsyncMock(return_value="{}")

        return response

    return _create_response


@pytest.fixture
def mock_api_error():
    """Create mock API error responses."""

    def _create_error(
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        message: str = "Invalid request",
        code: str | None = None,
    ) -> Mock:
        """
        Create a mock error response.

        Args:
            status_code: HTTP error status code
            error_type: Type of error
            message: Error message
            code: Optional error code

        Returns:
            Mock error response
        """
        error_data = {"error": {"type": error_type, "message": message}}

        if code:
            error_data["error"]["code"] = code

        response = Mock()
        response.status = status_code
        response.status_code = status_code
        response.headers = {}
        response.json = AsyncMock(return_value=error_data)
        response.text = AsyncMock(return_value=json.dumps(error_data))

        return response

    return _create_error


@pytest.fixture
def mock_rate_limit_headers():
    """Create mock rate limit headers."""

    def _create_headers(
        rpm_limit: int = 60,
        rpm_remaining: int = 50,
        rpm_reset: int = 1234567890,
        tpm_limit: int = 10000,
        tpm_remaining: int = 8000,
        tpm_reset: int = 1234567890,
    ) -> dict[str, str]:
        """
        Create rate limit headers.

        Args:
            rpm_limit: Requests per minute limit
            rpm_remaining: Remaining requests this minute
            rpm_reset: Unix timestamp for RPM reset
            tpm_limit: Tokens per minute limit
            tpm_remaining: Remaining tokens this minute
            tpm_reset: Unix timestamp for TPM reset

        Returns:
            Dictionary of rate limit headers
        """
        headers = {
            "x-ratelimit-limit-requests": str(rpm_limit),
            "x-ratelimit-remaining-requests": str(rpm_remaining),
            "x-ratelimit-reset-requests": str(rpm_reset),
            "x-ratelimit-limit-tokens": str(tpm_limit),
            "x-ratelimit-remaining-tokens": str(tpm_remaining),
            "x-ratelimit-reset-tokens": str(tpm_reset),
        }

        if rpm_remaining == 0:
            headers["retry-after"] = "60"

        return headers

    return _create_headers


@pytest.fixture
def mock_streaming_response():
    """Create a mock streaming response for SSE testing."""

    def _create_streaming_response(chunks: list[str], delay: float = 0.1) -> Mock:
        """
        Create a mock streaming response.

        Args:
            chunks: List of data chunks to stream
            delay: Delay between chunks in seconds

        Returns:
            Mock streaming response
        """

        async def async_iter():
            for chunk in chunks:
                await asyncio.sleep(delay)
                yield chunk.encode("utf-8")

        response = Mock()
        response.status = 200
        response.headers = {"content-type": "text/event-stream"}
        response.content = MagicMock()
        response.content.iter_any = MagicMock(return_value=async_iter())
        response.__aiter__ = async_iter

        return response

    return _create_streaming_response


@pytest.fixture
def mock_chat_completion_response():
    """Create a mock chat completion response."""

    def _create_response(
        model: str = "llama-3.2-3b",
        content: str = "This is a test response.",
        role: str = "assistant",
        finish_reason: str = "stop",
        completion_tokens: int = 10,
        prompt_tokens: int = 5,
        response_id: str = "chatcmpl-test123",
    ) -> dict[str, Any]:
        """
        Create a mock chat completion response.

        Args:
            model: Model name
            content: Response content
            role: Message role
            finish_reason: Reason for completion
            completion_tokens: Number of completion tokens
            prompt_tokens: Number of prompt tokens
            response_id: Response ID

        Returns:
            Chat completion response dictionary
        """
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": 1234567890,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": role, "content": content},
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return _create_response


@pytest.fixture
def mock_image_generation_response():
    """Create a mock image generation response."""

    def _create_response(
        images: list[str] | None = None, model: str = "flux-schnell"
    ) -> dict[str, Any]:
        """
        Create a mock image generation response.

        Args:
            images: List of base64 encoded images or URLs
            model: Model name

        Returns:
            Image generation response dictionary
        """
        if images is None:
            images = [
                "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
            ]

        return {
            "created": 1234567890,
            "model": model,
            "data": [
                {
                    "b64_json": img if img.startswith("data:") else None,
                    "url": img if not img.startswith("data:") else None,
                }
                for img in images
            ],
        }

    return _create_response


@pytest.fixture
def mock_embedding_response():
    """Create a mock embedding response."""

    def _create_response(
        embeddings: list[list[float]] | None = None,
        model: str = "text-embedding-3-small",
        encoding_format: str = "float",
    ) -> dict[str, Any]:
        """
        Create a mock embedding response.

        Args:
            embeddings: List of embedding vectors
            model: Model name
            encoding_format: Format of embeddings

        Returns:
            Embedding response dictionary
        """
        if embeddings is None:
            # Create dummy embeddings
            embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(2)]

        return {
            "object": "list",
            "model": model,
            "data": [
                {"object": "embedding", "index": i, "embedding": emb}
                for i, emb in enumerate(embeddings)
            ],
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }

    return _create_response


@pytest.fixture
def mock_audio_response():
    """Create a mock audio/TTS response."""

    def _create_response(audio_data: bytes | None = None, format: str = "mp3") -> Mock:
        """
        Create a mock audio response.

        Args:
            audio_data: Raw audio bytes
            format: Audio format

        Returns:
            Mock audio response
        """
        if audio_data is None:
            # Create dummy audio data
            audio_data = b"RIFF\x00\x00\x00\x00WAVEfmt "

        response = Mock()
        response.status = 200
        response.headers = {
            "content-type": f"audio/{format}",
            "content-length": str(len(audio_data)),
        }
        response.content = AsyncMock(return_value=audio_data)
        response.read = AsyncMock(return_value=audio_data)

        return response

    return _create_response


@pytest_asyncio.fixture
async def mock_api_session():
    """Create a mock aiohttp ClientSession for API testing."""
    session = AsyncMock()

    # Mock common HTTP methods
    session.post = AsyncMock()
    session.get = AsyncMock()
    session.put = AsyncMock()
    session.delete = AsyncMock()
    session.patch = AsyncMock()

    # Mock session context manager
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)

    # Mock close method
    session.close = AsyncMock()

    yield session

    # Ensure session is closed
    await session.close()


@pytest.fixture
def api_response_sequence():
    """Create a sequence of API responses for testing retries and fallbacks."""

    class ResponseSequence:
        def __init__(self, responses: list[Mock]):
            self.responses = responses
            self.call_count = 0

        async def __call__(self, *args, **kwargs):
            if self.call_count < len(self.responses):
                response = self.responses[self.call_count]
                self.call_count += 1
                return response
            raise ValueError("No more responses in sequence")

        def reset(self):
            self.call_count = 0

    def _create_sequence(responses: list[Mock]) -> ResponseSequence:
        return ResponseSequence(responses)

    return _create_sequence
