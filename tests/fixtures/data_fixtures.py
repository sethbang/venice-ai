"""
Test data fixtures for Venice AI testing.

This module provides fixtures for common test data including messages,
prompts, and sample inputs for various API endpoints.
"""

import json
import random
import string
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def sample_chat_messages() -> list[dict[str, str]]:
    """
    Provide sample chat messages for testing.

    Returns:
        List of chat message dictionaries
    """
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {
            "role": "assistant",
            "content": "I'm doing well, thank you! How can I help you today?",
        },
        {"role": "user", "content": "Can you explain quantum computing?"},
    ]


@pytest.fixture
def sample_chat_messages_with_tools() -> list[dict[str, Any]]:
    """
    Provide chat messages with tool/function calling.

    Returns:
        List of chat messages with tool definitions
    """
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant with access to tools.",
        },
        {"role": "user", "content": "What's the weather in San Francisco?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco", "unit": "fahrenheit"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": '{"temperature": 72, "condition": "sunny"}',
        },
    ]


@pytest.fixture
def sample_image_prompt() -> str:
    """
    Provide a sample image generation prompt.

    Returns:
        Image generation prompt string
    """
    return "A serene landscape with mountains in the background, a crystal clear lake in the foreground, and a sunset painting the sky in vibrant orange and purple hues"


@pytest.fixture
def sample_image_prompts() -> list[str]:
    """
    Provide multiple image generation prompts.

    Returns:
        List of image generation prompts
    """
    return [
        "A futuristic city with flying cars and neon lights",
        "A medieval castle on a hilltop during a thunderstorm",
        "An underwater scene with colorful coral reefs and tropical fish",
        "A cozy cabin in a snowy forest with smoke from the chimney",
        "An abstract representation of artificial intelligence",
    ]


@pytest.fixture
def sample_audio_text() -> str:
    """
    Provide sample text for audio/TTS generation.

    Returns:
        Text string for audio generation
    """
    return "Welcome to Venice AI. This is a test of our text-to-speech capabilities. We hope you enjoy using our services."


@pytest.fixture
def sample_audio_texts() -> list[str]:
    """
    Provide multiple audio text samples.

    Returns:
        List of text strings for audio generation
    """
    return [
        "Hello, this is a test message.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Welcome to the future of AI-powered applications.",
        "Testing one, two, three. Can you hear me clearly?",
    ]


@pytest.fixture
def sample_embedding_input() -> list[str]:
    """
    Provide sample text for embedding generation.

    Returns:
        List of text strings for embedding
    """
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret visual information.",
    ]


@pytest.fixture
def test_models() -> dict[str, str]:
    """
    Provide a mapping of test model names (uses dynamic cache when available).

    Returns:
        Dictionary of model types to model names
    """
    from tests.fixtures.test_models import TEST_MODELS

    return {
        "chat": TEST_MODELS.SMALL_TEXT_MODEL,
        "chat_large": TEST_MODELS.DEFAULT_TEXT_MODEL,
        "vision": TEST_MODELS.VISION_MODEL,
        "embedding": TEST_MODELS.EMBEDDING_MODEL,
        "embedding_large": TEST_MODELS.EMBEDDING_MODEL,
        "tts": TEST_MODELS.TTS_MODEL,
        "tts_multilingual": TEST_MODELS.TTS_MODEL,
        "image": TEST_MODELS.DEFAULT_IMAGE_MODEL,
        "image_pro": TEST_MODELS.HIGHEST_QUALITY_IMAGE_MODEL,
    }


@pytest.fixture
def error_messages() -> dict[str, str]:
    """
    Provide common error messages for testing.

    Returns:
        Dictionary of error types to messages
    """
    return {
        "invalid_api_key": "Invalid API key provided",
        "rate_limit": "Rate limit exceeded. Please try again later.",
        "invalid_model": "The model specified does not exist",
        "context_length": "The request exceeds the maximum context length",
        "invalid_request": "The request format is invalid",
        "server_error": "An internal server error occurred",
        "timeout": "The request timed out",
        "insufficient_quota": "Insufficient quota for this operation",
    }


@pytest.fixture
def sample_request_ids() -> list[str]:
    """
    Provide sample request IDs for testing.

    Returns:
        List of request ID strings
    """
    return [
        "req_abc123def456",
        "req_789ghi012jkl",
        "req_mno345pqr678",
        "req_stu901vwx234",
        "req_yza567bcd890",
    ]


@pytest.fixture
def random_text_generator():
    """
    Provide a text generator for creating random test data.

    Returns:
        Function to generate random text
    """

    def generate_text(
        length: int = 100,
        include_punctuation: bool = True,
        include_numbers: bool = False,
    ) -> str:
        """
        Generate random text of specified length.

        Args:
            length: Number of characters to generate
            include_punctuation: Include punctuation marks
            include_numbers: Include numbers

        Returns:
            Random text string
        """
        chars = string.ascii_letters + " "
        if include_punctuation:
            chars += string.punctuation
        if include_numbers:
            chars += string.digits

        return "".join(random.choice(chars) for _ in range(length))

    return generate_text


@pytest.fixture
def conversation_generator():
    """
    Provide a conversation generator for testing.

    Returns:
        Function to generate conversations
    """

    def generate_conversation(
        num_turns: int = 5, system_prompt: str | None = None
    ) -> list[dict[str, str]]:
        """
        Generate a conversation with specified number of turns.

        Args:
            num_turns: Number of conversation turns
            system_prompt: Optional system prompt

        Returns:
            List of message dictionaries
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for i in range(num_turns):
            if i % 2 == 0:
                messages.append({"role": "user", "content": f"User message {i // 2 + 1}"})
            else:
                messages.append(
                    {"role": "assistant", "content": f"Assistant response {i // 2 + 1}"}
                )

        return messages

    return generate_conversation


@pytest.fixture
def sample_api_responses() -> dict[str, Any]:
    """
    Provide sample API responses for different endpoints.

    Returns:
        Dictionary of endpoint names to sample responses
    """
    return {
        "chat_completion": {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "llama-3.2-3b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        },
        "image_generation": {
            "created": 1677652288,
            "data": [
                {
                    "url": "https://example.com/image.png",
                    "revised_prompt": "A beautiful landscape",
                }
            ],
        },
        "embedding": {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                }
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        },
    }


@pytest.fixture
def load_test_json():
    """
    Provide a function to load JSON test data files.

    Returns:
        Function to load JSON files
    """

    def _load_json(filename: str) -> Any:
        """
        Load JSON data from test data directory.

        Args:
            filename: Name of JSON file in test data directory

        Returns:
            Parsed JSON data
        """
        data_dir = Path(__file__).parent / "data"
        file_path = data_dir / filename

        if not file_path.exists():
            # Return empty dict if file doesn't exist
            return {}

        with open(file_path) as f:
            return json.load(f)

    return _load_json


@pytest.fixture
def parametrized_models():
    """
    Provide parametrized model configurations for testing.

    Returns:
        List of model configuration tuples
    """
    from tests.fixtures.test_models import TEST_MODELS

    return [
        (TEST_MODELS.MEDIUM_BETA_MODEL, {"temperature": 0.7, "max_completion_tokens": 100}),
        (TEST_MODELS.SMALL_TEXT_MODEL, {"temperature": 0.5, "max_completion_tokens": 500}),
        (TEST_MODELS.LARGE_TEXT_MODEL, {"temperature": 0.9, "max_completion_tokens": 1000}),
        (TEST_MODELS.UNCENSORED_MODEL, {"temperature": 0.3, "max_completion_tokens": 2000}),
    ]


@pytest.fixture
def validation_test_cases():
    """
    Provide validation test cases for request validation.

    Returns:
        List of test case dictionaries
    """
    return [
        {
            "name": "valid_request",
            "input": {
                "model": "llama-3.2-3b",
                "messages": [{"role": "user", "content": "Hi"}],
            },
            "should_pass": True,
        },
        {
            "name": "missing_model",
            "input": {"messages": [{"role": "user", "content": "Hi"}]},
            "should_pass": False,
            "error": "model is required",
        },
        {
            "name": "empty_messages",
            "input": {"model": "llama-3.2-3b", "messages": []},
            "should_pass": False,
            "error": "messages cannot be empty",
        },
        {
            "name": "invalid_temperature",
            "input": {
                "model": "llama-3.2-3b",
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 2.5,
            },
            "should_pass": False,
            "error": "temperature must be between 0 and 2",
        },
    ]


@pytest.fixture
def performance_test_data():
    """
    Provide data for performance testing.

    Returns:
        Dictionary of performance test configurations
    """
    return {
        "small_payload": {"size": 100, "expected_latency_ms": 100},
        "medium_payload": {"size": 1000, "expected_latency_ms": 200},
        "large_payload": {"size": 10000, "expected_latency_ms": 500},
        "concurrent_requests": {
            "count": 100,
            "expected_throughput": 50,  # requests per second
        },
    }


@pytest.fixture
def mock_file_content():
    """
    Provide mock file content for file upload testing.

    Returns:
        Dictionary of file types to content
    """
    return {
        "text": b"This is a test text file content.",
        "json": b'{"key": "value", "number": 42}',
        "audio": b"RIFF\x00\x00\x00\x00WAVEfmt ",  # Minimal WAV header
        "image": b"\x89PNG\r\n\x1a\n",  # PNG header
    }
