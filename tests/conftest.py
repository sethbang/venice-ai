"""
Shared test fixtures and configuration for Venice AI client tests.
"""

import pytest
import pytest_asyncio
import httpx
from unittest.mock import MagicMock
import json


# Common test constants
API_KEY = "test-api-key"
BASE_URL = "https://api.venice.ai/api/v1"

def create_mock_response(status_code=400, json_data=None, text=None):
    """
    Create a mock httpx.Response object for testing exception handling.
    
    Args:
        status_code (int): HTTP status code for the response
        json_data (dict, optional): JSON data to include in the response
        text (str, optional): Text content for the response if not using JSON
        
    Returns:
        MagicMock: A mocked httpx.Response object
    """
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = status_code
    mock_response.request = MagicMock(spec=httpx.Request)
    mock_response.headers = {}  # Add headers attribute to prevent AttributeError
    
    # Set text content
    mock_response.text = text or json.dumps(json_data or {})
    
    # Configure json method if json_data is provided
    if json_data is not None:
        mock_response.json.return_value = json_data
    elif text:
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", text, 0)
    
    return mock_response

@pytest.fixture
def venice_client():
    """Create a synchronous Venice client for testing."""
    from venice_ai import VeniceClient
    client = VeniceClient(api_key=API_KEY, base_url=BASE_URL)
    yield client
    client.close()

@pytest_asyncio.fixture
async def async_venice_client():
    """Create an asynchronous Venice client for testing."""
    from venice_ai import AsyncVeniceClient
    client = AsyncVeniceClient(api_key=API_KEY, base_url=BASE_URL)
    yield client
    await client.close()