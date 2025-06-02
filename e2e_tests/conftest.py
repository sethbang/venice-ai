import os
import pytest
import pytest_asyncio
from typing import List, Tuple
from venice_ai import VeniceClient, AsyncVeniceClient
from venice_ai.exceptions import VeniceError

# Fixture to load the API key from environment variables
@pytest.fixture(scope="session")
def api_key():
    """Loads the API key from the VENICE_API_KEY environment variable."""
    key = os.environ.get("VENICE_API_KEY")
    if not key:
        pytest.fail("VENICE_API_KEY environment variable not set.")
    return key

# Fixture for the synchronous VeniceClient
@pytest.fixture(scope="session")
def venice_client(api_key):
    """Provides a synchronous VeniceClient instance."""
    client = VeniceClient(api_key=api_key)
    # Use internal attributes for logging, ensuring they exist
    print(f"Initializing VeniceClient with API key: {client._api_key[:4]}...{client._api_key[-4:]}")
    print(f"VeniceClient using base URL: {client._base_url}")
    yield client
    client.close()

# Fixture for the asynchronous VeniceClient
@pytest_asyncio.fixture(scope="function")
async def async_venice_client(api_key):
    """Provides an asynchronous AsyncVeniceClient instance."""
    client = AsyncVeniceClient(api_key=api_key)
    # Use internal attributes for logging, ensuring they exist
    print(f"Initializing AsyncVeniceClient with API key: {client._api_key[:4]}...{client._api_key[-4:]}")
    print(f"AsyncVeniceClient using base URL: {client._base_url}")
    yield client
    await client.close()

# Fixture for managing resources created during tests for cleanup
@pytest.fixture(scope="function")
def created_resources():
    """Fixture to track and clean up resources created during tests."""
    resources_to_clean: List[Tuple[str, str]] = []
    yield resources_to_clean
    # TODO: Implement cleanup logic based on resource type and ID
    # For example:
    # for resource_type, resource_id in resources_to_clean:
    #     if resource_type == "api_key":
    #         try:
    #             # Need a client instance here, maybe pass it to the fixture or use a global/session client
    #             # venice_client.api_keys.delete(api_key_id=resource_id)
    #             pass # Placeholder
    #         except VeniceError as e:
    #             print(f"Error cleaning up API key {resource_id}: {e}")
    pass # Placeholder for cleanup logic