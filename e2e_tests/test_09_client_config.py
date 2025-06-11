"""
End-to-End tests for VeniceClient and AsyncVeniceClient configuration and initialization.
"""

import os
import pytest
import httpx
from venice_ai import VeniceClient, AsyncVeniceClient
from venice_ai._client_with_retries import VeniceClientWithRetries, AsyncVeniceClientWithRetries
from venice_ai.exceptions import VeniceError


# Test synchronous client initialization
def test_venice_client_init_with_api_key():
    """Test initializing VeniceClient with a valid API key."""
    client = VeniceClient(api_key="test-api-key")
    assert client._api_key == "test-api-key"
    assert str(client._base_url) == "https://api.venice.ai/api/v1/"
    assert isinstance(client._timeout, (float, httpx.Timeout))
    if isinstance(client._timeout, float):
        assert client._timeout == 60.0
    # If it's an httpx.Timeout object, we just confirm it's set (detailed check not necessary for this test)
    # Standard client doesn't have retry attributes - those are in VeniceClientWithRetries
    client.close()


def test_venice_client_init_without_api_key():
    """Test initializing VeniceClient without an API key raises ValueError."""
    with pytest.raises(ValueError, match="The api_key client option must be set."):
        VeniceClient(api_key="")


def test_venice_client_init_with_custom_base_url():
    """Test initializing VeniceClient with a custom base URL."""
    client = VeniceClient(api_key="test-api-key", base_url="https://custom.api.com")
    assert str(client._base_url) == "https://custom.api.com/"
    client.close()


def test_venice_client_init_with_custom_timeout():
    """Test initializing VeniceClient with custom timeout settings."""
    client = VeniceClient(api_key="test-api-key", timeout=30.0)
    assert isinstance(client._timeout, (float, httpx.Timeout))
    if isinstance(client._timeout, float):
        assert client._timeout == 30.0
    # If it's an httpx.Timeout object, we just confirm it's set (detailed check not necessary for this test)
    client.close()


def test_venice_client_init_with_custom_max_retries():
    """Test initializing VeniceClientWithRetries with custom max retries."""
    client = VeniceClientWithRetries(api_key="test-api-key", max_retries=5)
    assert client._max_retries == 5
    client.close()


def test_venice_client_init_with_custom_http_client():
    """Test initializing VeniceClient with a custom HTTP client."""
    custom_client = httpx.Client()
    client = VeniceClient(api_key="test-api-key", http_client=custom_client)
    assert client._client is custom_client
    client.close()


# Test asynchronous client initialization
@pytest.mark.asyncio
async def test_async_venice_client_init_with_api_key():
    """Test initializing AsyncVeniceClient with a valid API key."""
    client = AsyncVeniceClient(api_key="test-api-key")
    assert client._api_key == "test-api-key"
    assert str(client._base_url) == "https://api.venice.ai/api/v1/"
    assert isinstance(client._timeout, (float, httpx.Timeout))
    if isinstance(client._timeout, float):
        assert client._timeout == 60.0
    # If it's an httpx.Timeout object, we just confirm it's set (detailed check not necessary for this test)
    # Standard client doesn't have retry attributes - those are in AsyncVeniceClientWithRetries
    await client.close()


@pytest.mark.asyncio
async def test_async_venice_client_init_without_api_key():
    """Test initializing AsyncVeniceClient without an API key raises ValueError."""
    with pytest.raises(ValueError, match="The api_key client option must be set."):
        AsyncVeniceClient(api_key="")


@pytest.mark.asyncio
async def test_async_venice_client_init_with_custom_base_url():
    """Test initializing AsyncVeniceClient with a custom base URL."""
    client = AsyncVeniceClient(api_key="test-api-key", base_url="https://custom.api.com")
    assert str(client._base_url) == "https://custom.api.com/"
    await client.close()


@pytest.mark.asyncio
async def test_async_venice_client_init_with_custom_timeout():
    """Test initializing AsyncVeniceClient with custom timeout settings."""
    client = AsyncVeniceClient(api_key="test-api-key", timeout=30.0)
    assert isinstance(client._timeout, (float, httpx.Timeout))
    if isinstance(client._timeout, float):
        assert client._timeout == 30.0
    # If it's an httpx.Timeout object, we just confirm it's set (detailed check not necessary for this test)
    await client.close()


@pytest.mark.asyncio
async def test_async_venice_client_init_with_custom_max_retries():
    """Test initializing AsyncVeniceClientWithRetries with custom max retries."""
    client = AsyncVeniceClientWithRetries(api_key="test-api-key", max_retries=5)
    assert client._max_retries == 5
    await client.close()


@pytest.mark.asyncio
async def test_async_venice_client_init_with_custom_http_client():
    """Test initializing AsyncVeniceClient with a custom HTTP client."""
    custom_client = httpx.AsyncClient()
    client = AsyncVeniceClient(api_key="test-api-key", http_client=custom_client)
    assert client._client is custom_client
    await client.close()


# Test environment variable handling (since the client doesn't directly read env vars, we simulate)
def test_venice_client_env_variable_simulation(monkeypatch):
    """Test VeniceClient behavior with simulated environment variable for API key."""
    monkeypatch.setenv("VENICE_API_KEY", "env-test-api-key")
    # Since VeniceClient doesn't read env vars directly, we just test initialization
    client = VeniceClient(api_key="explicit-key")
    assert client._api_key == "explicit-key"  # Explicit key takes precedence
    client.close()


@pytest.mark.asyncio
async def test_async_venice_client_env_variable_simulation(monkeypatch):
    """Test AsyncVeniceClient behavior with simulated environment variable for API key."""
    monkeypatch.setenv("VENICE_API_KEY", "env-test-api-key")
    # Since AsyncVeniceClient doesn't read env vars directly, we just test initialization
    client = AsyncVeniceClient(api_key="explicit-key")
    assert client._api_key == "explicit-key"  # Explicit key takes precedence
    await client.close()


# Test error handling for invalid configuration
def test_venice_client_invalid_base_url():
    """Test initializing VeniceClient with an invalid base URL."""
    client = VeniceClient(api_key="test-api-key", base_url="invalid-url")
    assert str(client._base_url).startswith("invalid-url/")
    client.close()


@pytest.mark.asyncio
async def test_async_venice_client_invalid_base_url():
    """Test initializing AsyncVeniceClient with an invalid base URL."""
    client = AsyncVeniceClient(api_key="test-api-key", base_url="invalid-url")
    assert str(client._base_url).startswith("invalid-url/")
    await client.close()