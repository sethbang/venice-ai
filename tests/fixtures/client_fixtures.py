"""
Client-related fixtures for Venice AI testing.

This module provides fixtures for creating and configuring Venice AI clients
with various settings for different test scenarios.
"""

import contextlib
from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio

# Import Venice client (will be available when tests run)
from venice_ai import VeniceClient
from venice_ai.core.config import SchedulerMode, VeniceAIConfig
from venice_ai.middleware import RetryOptions


@pytest.fixture
def base_client() -> Generator[VeniceClient]:
    """
    Create a basic Venice client with minimal configuration.

    Yields:
        Venice client instance
    """
    client = VeniceClient(api_key="test-api-key", base_url="https://api.test.venice.ai/api/v1")
    yield client
    # Cleanup is handled by client's context manager


@pytest.fixture
def authenticated_client(test_api_key: str) -> Generator[VeniceClient]:
    """
    Create an authenticated Venice client with a valid test API key.

    Args:
        test_api_key: API key from session fixture

    Yields:
        Authenticated Venice client
    """
    client = VeniceClient(api_key=test_api_key)
    yield client


@pytest.fixture
def rate_limited_client() -> Generator[VeniceClient]:
    """
    Create a Venice client with rate limiting configured.

    Yields:
        Venice client with rate limiting
    """
    config = VeniceAIConfig.create_test_config(
        scheduler_mode=SchedulerMode.INTELLIGENT,
        enable_redis=True,
        test_rate_multiplier=10.0,
    )
    # Override for this specific test
    config.scheduler.max_concurrent_executions = 5
    config.scheduler.max_queue_size = 20

    client = VeniceClient(api_key="test-api-key", config=config)
    yield client


@pytest.fixture
def test_client_factory():
    """
    Factory fixture for creating Venice clients with custom configuration.

    Returns:
        Function to create configured clients
    """
    clients = []

    def _create_client(
        api_key: str = "test-key",
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        rate_limiter: dict[str, Any] | None = None,
        **kwargs,
    ) -> VeniceClient:
        """
        Create a Venice client with custom configuration.

        Args:
            api_key: API key
            base_url: Base URL for API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            rate_limiter: Rate limiter configuration
            **kwargs: Additional configuration options

        Returns:
            Configured Venice client
        """
        config = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }

        if base_url:
            config["base_url"] = base_url

        if rate_limiter:
            config["rate_limiter"] = rate_limiter

        config.update(kwargs)

        client = VeniceClient(**config)
        clients.append(client)
        return client

    yield _create_client

    # Cleanup all created clients
    for client in clients:
        with contextlib.suppress(AttributeError, RuntimeError, Exception):
            client.close()


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[VeniceClient]:
    """
    Create a Venice client for async testing.

    Yields:
        Venice client
    """
    client = VeniceClient(api_key="test-api-key", base_url="https://api.test.venice.ai/api/v1")

    # Initialize async session
    await client.__aenter__()

    try:
        yield client
    finally:
        await client.__aexit__(None, None, None)


@pytest.fixture
def mock_client() -> Generator[Mock]:
    """
    Create a fully mocked Venice client for unit testing.

    Yields:
        Mocked Venice client
    """
    client = Mock(spec=VeniceClient)

    # Mock chat completions
    client.chat = Mock()
    client.chat.completions = Mock()
    client.chat.completions.create = AsyncMock()
    client.chat.completions.stream = AsyncMock()

    # Mock images
    client.images = Mock()
    client.images.generate = AsyncMock()
    client.images.edit = AsyncMock()

    # Mock audio
    client.audio = Mock()
    client.audio.speech = Mock()
    client.audio.speech.create = AsyncMock()
    client.audio.transcriptions = Mock()
    client.audio.transcriptions.create = AsyncMock()

    # Mock embeddings
    client.embeddings = Mock()
    client.embeddings.create = AsyncMock()

    # Mock models
    client.models = Mock()
    client.models.list = AsyncMock()
    client.models.retrieve = AsyncMock()

    # Mock billing
    client.billing = Mock()
    client.billing.usage = AsyncMock()
    client.billing.limits = AsyncMock()

    # Mock API keys
    client.api_keys = Mock()
    client.api_keys.create = AsyncMock()
    client.api_keys.list = AsyncMock()
    client.api_keys.delete = AsyncMock()

    # Mock session management
    client.close = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock()

    yield client


@pytest.fixture
def client_with_mocked_session(
    base_client: VeniceClient,
) -> Generator[VeniceClient]:
    """
    Create a Venice client with a mocked HTTP session.

    Args:
        base_client: Base client fixture

    Yields:
        Venice client with mocked session
    """
    mock_session = AsyncMock()
    mock_session.post = AsyncMock()
    mock_session.get = AsyncMock()
    mock_session.delete = AsyncMock()
    mock_session.close = AsyncMock()

    with patch.object(base_client, "_session", mock_session):
        yield base_client


@pytest.fixture
def isolated_test_client() -> Generator[VeniceClient]:
    """
    Create an isolated Venice client for independent testing.

    This client uses isolated rate limiting and doesn't share state
    with other test clients.

    Yields:
        Isolated Venice client
    """
    config = VeniceAIConfig.create_test_config(
        scheduler_mode=SchedulerMode.INTELLIGENT,
        enable_redis=False,  # Use memory backend for isolation
        test_rate_multiplier=10.0,
    )
    # Override for isolated testing
    config.scheduler.max_concurrent_executions = 3
    config.scheduler.max_queue_size = 10
    config.state.namespace = "isolated_test"

    client = VeniceClient(api_key="isolated-test-key", config=config)

    yield client


@pytest.fixture
def streaming_client() -> Generator[VeniceClient]:
    """
    Create a Venice client configured for streaming responses.

    Yields:
        Venice client configured for streaming
    """
    client = VeniceClient(
        api_key="test-api-key",
        timeout=60.0,  # Longer timeout for streaming
    )

    yield client


@pytest.fixture
def client_with_retry() -> Generator[VeniceClient]:
    """
    Create a Venice client with custom retry configuration.

    Yields:
        Venice client with retry configuration
    """
    client = VeniceClient(
        api_key="test-api-key",
        max_retries=5,
        retry_options=RetryOptions(
            base_delay=1.0,
            exponential_base=2.0,
            jitter_factor=0.1,
        ),
    )

    yield client


@pytest.fixture
def multi_client_pool():
    """
    Create a pool of Venice clients for testing concurrent operations.

    Returns:
        Client pool manager
    """

    class ClientPool:
        def __init__(self):
            self.clients = []

        def create_client(self, **kwargs) -> VeniceClient:
            """Create and track a new client."""
            client = VeniceClient(api_key=f"test-key-{len(self.clients)}", **kwargs)
            self.clients.append(client)
            return client

        def get_clients(self, count: int) -> list[VeniceClient]:
            """Get a specific number of clients."""
            while len(self.clients) < count:
                self.create_client()
            return self.clients[:count]

        async def close_all(self):
            """Close all clients in the pool."""
            for client in self.clients:
                await client.close()

        def reset(self):
            """Reset the client pool."""
            self.clients.clear()

    pool = ClientPool()
    yield pool

    # Cleanup
    pool.reset()


@pytest.fixture
def client_with_custom_headers() -> Generator[VeniceClient]:
    """
    Create a Venice client with custom headers.

    Yields:
        Venice client with custom headers
    """
    custom_headers = {
        "X-Test-Header": "test-value",
        "X-Request-ID": "test-request-123",
        "User-Agent": "VeniceAI-Test/1.0",
    }

    client = VeniceClient(api_key="test-api-key", headers=custom_headers)

    yield client


@pytest.fixture
def debug_client() -> Generator[VeniceClient]:
    """
    Create a Venice client with debug mode enabled.

    Yields:
        Venice client in debug mode
    """
    client = VeniceClient(api_key="test-api-key")

    yield client
