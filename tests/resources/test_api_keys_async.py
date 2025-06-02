"""
Tests for the asynchronous AsyncApiKeys resource.
"""

import pytest
import httpx
from typing import Dict, List, Literal, Optional, TypedDict, cast # Add cast
from unittest.mock import MagicMock, AsyncMock, patch, ANY

from venice_ai import AsyncVeniceClient
from venice_ai.types.api_keys import (
    ApiKeyList, ApiKeyCreateRequest, ApiKeyCreateResponse,
    RateLimitInfo, RateLimitLogList, ApiKey, ConsumptionLimit,
    ApiKeyGenerateWeb3KeyCreateRequest, ApiKeyGenerateWeb3KeyCreateResponse,
    RateLimitLog # Add RateLimitLog
)
from venice_ai.exceptions import APIError, AuthenticationError, InvalidRequestError
from venice_ai.resources.api_keys import AsyncApiKeys


# Define mock response structures for testing
class MockApiKeyUsage(TypedDict):
    trailingSevenDays: Dict[str, str]


class MockConsumptionLimit(TypedDict, total=False):
    usd: Optional[float]
    vcu: Optional[float]


class MockApiKey(TypedDict):
    apiKeyType: Literal["INFERENCE", "ADMIN"]
    consumptionLimits: MockConsumptionLimit
    createdAt: Optional[str]
    description: str
    expiresAt: Optional[str]
    id: str
    last6Chars: str
    lastUsedAt: Optional[str]
    usage: MockApiKeyUsage


class MockApiKeyList(TypedDict):
    data: List[MockApiKey]
    object: Literal["list"]


class MockApiKeyCreateResponse(TypedDict):
    data: Dict
    success: bool


class MockRateLimitInfo(TypedDict):
    accessPermitted: bool
    apiTier: Dict
    balances: Dict
    keyExpiration: Optional[str]
    nextEpochBegins: str
    rateLimits: List[Dict]


class MockRateLimitLogList(TypedDict):
    data: List[Dict]
    object: Literal["list"]


@pytest.mark.asyncio
async def test_list_success_async(httpx_mock):
    """Tests successful asynchronous retrieval of API keys list."""
    mock_response_data: MockApiKeyList = {
        "data": [
            {
                "apiKeyType": "INFERENCE",
                "consumptionLimits": {
                    "usd": 100.0,
                    "vcu": None
                },
                "createdAt": "2025-01-01T00:00:00Z",
                "description": "Test API Key",
                "expiresAt": "2026-01-01T00:00:00Z",
                "id": "key-123456",
                "last6Chars": "abcdef",
                "lastUsedAt": "2025-01-02T00:00:00Z",
                "usage": {
                    "trailingSevenDays": {
                        "usd": "10.50",
                        "vcu": "150.25"
                    }
                }
            }
        ],
        "object": "list"
    }

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/api_keys",
        json=mock_response_data,
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        response = await client.api_keys.list()

    assert isinstance(response, list)
    assert len(response) == 1
    assert response[0]["id"] == "key-123456"
    assert response[0]["apiKeyType"] == "INFERENCE"
    assert response[0]["consumptionLimits"].get("usd") == 100.0


@pytest.mark.asyncio
async def test_list_with_page_and_limit_async(httpx_mock):
    """Tests async list with both page and limit parameters."""
    mock_response_data: MockApiKeyList = {
        "data": [
            {
                "apiKeyType": "INFERENCE",
                "consumptionLimits": {
                    "usd": 100.0,
                    "vcu": None
                },
                "createdAt": "2024-01-15T10:30:00Z",
                "description": "Test API Key",
                "expiresAt": "2026-01-01T00:00:00Z",
                "id": "key-123456",
                "last6Chars": "abcdef",
                "lastUsedAt": "2024-01-16T10:30:00Z",
                "usage": {
                    "trailingSevenDays": {
                        "usd": "10.50",
                        "vcu": "150.25"
                    }
                }
            }
        ],
        "object": "list"
    }

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/api_keys?page=1&limit=10",
        json=mock_response_data,
        status_code=200,
        match_content=None,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        await client.api_keys.list(page=1, limit=10)
    
    # Verify the request contained the correct parameters
    request = httpx_mock.get_requests()[0]
    assert request.url.params.get("page") == "1"
    assert request.url.params.get("limit") == "10"


@pytest.mark.asyncio
async def test_list_with_page_only_async(httpx_mock):
    """Tests async list with only page parameter."""
    mock_response_data: MockApiKeyList = {
        "data": [
            {
                "apiKeyType": "INFERENCE",
                "consumptionLimits": {
                    "usd": 100.0,
                    "vcu": None
                },
                "createdAt": "2024-01-15T10:30:00Z",
                "description": "Test API Key",
                "expiresAt": "2026-01-01T00:00:00Z",
                "id": "key-123456",
                "last6Chars": "abcdef",
                "lastUsedAt": "2024-01-16T10:30:00Z",
                "usage": {
                    "trailingSevenDays": {
                        "usd": "10.50",
                        "vcu": "150.25"
                    }
                }
            }
        ],
        "object": "list"
    }

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/api_keys?page=2",
        json=mock_response_data,
        status_code=200,
        match_content=None,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        await client.api_keys.list(page=2)
    
    # Verify the request contained the correct parameter
    request = httpx_mock.get_requests()[0]
    assert request.url.params.get("page") == "2"
    assert "limit" not in request.url.params


@pytest.mark.asyncio
async def test_list_with_limit_only_async(httpx_mock):
    """Tests async list with only limit parameter."""
    mock_response_data: MockApiKeyList = {
        "data": [
            {
                "apiKeyType": "INFERENCE",
                "consumptionLimits": {
                    "usd": 100.0,
                    "vcu": None
                },
                "createdAt": "2024-01-15T10:30:00Z",
                "description": "Test API Key",
                "expiresAt": "2026-01-01T00:00:00Z",
                "id": "key-123456",
                "last6Chars": "abcdef",
                "lastUsedAt": "2024-01-16T10:30:00Z",
                "usage": {
                    "trailingSevenDays": {
                        "usd": "10.50",
                        "vcu": "150.25"
                    }
                }
            }
        ],
        "object": "list"
    }

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/api_keys?limit=5",
        json=mock_response_data,
        status_code=200,
        match_content=None,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        await client.api_keys.list(limit=5)
    
    # Verify the request contained the correct parameter
    request = httpx_mock.get_requests()[0]
    assert request.url.params.get("limit") == "5"
    assert "page" not in request.url.params


@pytest.mark.asyncio
async def test_async_api_keys_list_returns_list_directly(httpx_mock):
    """Tests async handling when the API returns a list directly instead of a dict with data."""
    # Mock response is a list directly without being wrapped in a dict
    mock_response_data = [
        {
            "apiKeyType": "INFERENCE",
            "id": "key-123456",
            "description": "Test API Key"
        }
    ]

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/api_keys",
        json=mock_response_data,
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        response = await client.api_keys.list()
    
    assert isinstance(response, list)
    assert len(response) == 1
    assert response[0]["id"] == "key-123456"


@pytest.mark.asyncio
async def test_async_api_keys_list_returns_empty_on_unexpected_response(httpx_mock):
    """Tests async handling when the API returns an unexpected response type."""
    # Mock response is a string, which is neither a dict with 'data' nor a list
    mock_response_data = "unexpected_string_response"

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/api_keys",
        json=mock_response_data,
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        response = await client.api_keys.list()
    
    assert isinstance(response, list)
    assert len(response) == 0


@pytest.mark.asyncio
async def test_async_api_keys_list_with_pagination_params():
    """Unit test for verifying pagination parameters are passed correctly to the async client."""
    # Create a mock async client
    mock_async_client = MagicMock(spec=AsyncVeniceClient)
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.json.return_value = {"data": [], "object": "list"}
    mock_async_client.get = AsyncMock(return_value=mock_httpx_response)

    # Create AsyncApiKeys resource with the mock client
    async_api_keys_resource = AsyncApiKeys(mock_async_client)

    # Define test pagination parameters
    test_limit = 5
    test_page = 3

    # Call the list method with both parameters
    await async_api_keys_resource.list(limit=test_limit, page=test_page)

    # Verify the client's get method was called with the correct parameters
    mock_async_client.get.assert_called_once()
    call_args = mock_async_client.get.call_args[0]
    call_kwargs = mock_async_client.get.call_args[1]
    
    assert call_args[0] == "api_keys"
    assert call_kwargs["params"] == {"limit": test_limit, "page": test_page}


@pytest.mark.asyncio
async def test_async_api_keys_list_with_limit_only_param():
    """Unit test for verifying only limit parameter is passed correctly to the async client."""
    # Create a mock async client
    mock_async_client = MagicMock(spec=AsyncVeniceClient)
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.json.return_value = {"data": [], "object": "list"}
    mock_async_client.get = AsyncMock(return_value=mock_httpx_response)

    # Create AsyncApiKeys resource with the mock client
    async_api_keys_resource = AsyncApiKeys(mock_async_client)

    # Define test parameter
    test_limit = 5

    # Call the list method with only limit parameter
    await async_api_keys_resource.list(limit=test_limit)

    # Verify the client's get method was called with the correct parameters
    mock_async_client.get.assert_called_once()
    call_args = mock_async_client.get.call_args[0]
    call_kwargs = mock_async_client.get.call_args[1]
    
    assert call_args[0] == "api_keys"
    assert call_kwargs["params"] == {"limit": test_limit}


@pytest.mark.asyncio
async def test_async_api_keys_list_with_page_only_param():
    """Unit test for verifying only page parameter is passed correctly to the async client."""
    # Create a mock async client
    mock_async_client = MagicMock(spec=AsyncVeniceClient)
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.json.return_value = {"data": [], "object": "list"}
    mock_async_client.get = AsyncMock(return_value=mock_httpx_response)

    # Create AsyncApiKeys resource with the mock client
    async_api_keys_resource = AsyncApiKeys(mock_async_client)

    # Define test parameter
    test_page = 3

    # Call the list method with only page parameter
    await async_api_keys_resource.list(page=test_page)

    # Verify the client's get method was called with the correct parameters
    mock_async_client.get.assert_called_once()
    call_args = mock_async_client.get.call_args[0]
    call_kwargs = mock_async_client.get.call_args[1]
    
    assert call_args[0] == "api_keys"
    assert call_kwargs["params"] == {"page": test_page}


@pytest.mark.asyncio
async def test_create_success_async(httpx_mock):
    """Tests successful asynchronous creation of an API key."""
    mock_request: ApiKeyCreateRequest = {
        "apiKeyType": "INFERENCE",
        "description": "New Test API Key",
        "consumptionLimit": {
            "usd": 200.0
        }
    }

    mock_response_data: MockApiKeyCreateResponse = {
        "data": {
            "apiKey": "sk-testapikey123456789",
            "apiKeyPrefix": "sk-testapi",
            "consumptionLimit": {
                "usd": 200.0,
                "vcu": None
            },
            "description": "New Test API Key",
            "expiresAt": None
        },
        "success": True
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/api_keys",
        json=mock_response_data,
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        response = await client.api_keys.create(api_key_request=mock_request)

    assert isinstance(response, dict)
    assert "description" in response
    assert response["description"] == "New Test API Key"
    assert response["consumptionLimit"]["usd"] == 200.0


@pytest.mark.asyncio
async def test_delete_success_async(httpx_mock):
    """Tests successful asynchronous deletion of an API key using query parameter."""
    api_key_id = "key-123456"

    mock_response_data = {
        "success": True,
        "message": "API key deleted successfully"
    }

    httpx_mock.add_response(
        method="DELETE",
        url=f"https://api.venice.ai/api/v1/api_keys/{api_key_id}", # Include ID in URL path
        json=mock_response_data,
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        response = await client.api_keys.delete(api_key_id=api_key_id)

    assert isinstance(response, dict)
    assert response["success"] is True
    assert response["message"] == "API key deleted successfully"


@pytest.mark.asyncio
async def test_get_web3_token_success_async(httpx_mock):
    """Tests successful asynchronous retrieval of a Web3 API key token."""
    mock_response_data = {
        "token": "web3-token-1234567890",
        "expiresAt": "2025-12-31T23:59:59Z"
    }

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/api_keys/generate_web3_key",
        json=mock_response_data,
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        response = await client.api_keys.get_web3_token() # Corrected method call

    assert isinstance(response, dict)
    assert "token" in response
    assert "expiresAt" in response
    assert response["token"] == "web3-token-1234567890"


@pytest.mark.asyncio
async def test_get_web3_token_api_error_async(httpx_mock):
    """Tests asynchronous API error handling for GET /api_keys/generate_web3_key."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/api_keys/generate_web3_key",
        status_code=401,
        json={"error": {"message": "Unauthorized", "type": "authentication_error"}},
    )

    async with AsyncVeniceClient(api_key="invalid-key") as client:
        with pytest.raises(AuthenticationError) as excinfo:
            await client.api_keys.get_web3_token() # Corrected method call

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 401
    assert "Unauthorized" in str(excinfo.value)


@pytest.mark.asyncio
async def test_create_web3_key_post_success_required_async(httpx_mock):
    """Tests successful asynchronous creation of a Web3 API key with required parameters."""
    mock_request: ApiKeyGenerateWeb3KeyCreateRequest = {
        "apiKeyType": "INFERENCE", # Changed for type compatibility with ApiKeyGenerateWeb3KeyCreateRequest
    }

    mock_response_data = {
        "apiKey": "sk-web3apikey123456789",
        "apiKeyPrefix": "sk-web3api",
        "consumptionLimit": None,
        "description": None,
        "expiresAt": None
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/api_keys/generate_web3_key",
        json=mock_response_data,
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        # Removed redundant type hint assignment
        response = await client.api_keys.create_web3_key(web3_key_request=mock_request) # Corrected method call and argument name

    assert isinstance(response, dict)
    assert "apiKey" in response
    assert response["apiKeyPrefix"] == "sk-web3api"


@pytest.mark.asyncio
async def test_create_web3_key_post_success_optional_async(httpx_mock):
    """Tests successful asynchronous creation of a Web3 API key with optional parameters."""
    mock_request: ApiKeyGenerateWeb3KeyCreateRequest = {
        "apiKeyType": "INFERENCE",
        "description": "My Web3 Key",
        "expiresAt": "2026-12-31T23:59:59Z",
        "consumptionLimit": {"usd": 50.0}
    }

    mock_response_data = {
        "apiKey": "sk-web3apikeyoptional123",
        "apiKeyPrefix": "sk-web3api",
        "consumptionLimit": {"usd": 50.0, "vcu": None},
        "description": "My Web3 Key",
        "expiresAt": "2026-12-31T23:59:59Z"
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/api_keys/generate_web3_key",
        json={"data": mock_response_data, "success": True}, # Wrap mock_response_data in 'data' key
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        # Removed redundant type hint assignment
        response = await client.api_keys.create_web3_key(web3_key_request=mock_request) # Corrected method call and argument name

    assert isinstance(response, dict)
    assert response["data"]["description"] == "My Web3 Key"
    assert response["data"]["expiresAt"] == "2026-12-31T23:59:59Z"
    assert cast(ConsumptionLimit, response["data"]["consumptionLimit"]).get("usd") == 50.0 # Use .get for optional key


@pytest.mark.asyncio
async def test_create_web3_key_post_validation_error_async(httpx_mock):
    """Tests asynchronous validation error handling for POST /api_keys/generate_web3_key."""
    mock_request: ApiKeyGenerateWeb3KeyCreateRequest = {
        "apiKeyType": "INFERENCE",
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/api_keys/generate_web3_key",
        status_code=400,
        json={"error": {"message": "Invalid apiKeyType", "type": "invalid_request_error"}},
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        with pytest.raises(InvalidRequestError) as excinfo:
            # Removed redundant type hint assignment
            await client.api_keys.create_web3_key(web3_key_request=mock_request) # Corrected method call and argument name

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 400
    assert "Invalid apiKeyType" in str(excinfo.value)


@pytest.mark.asyncio
async def test_get_rate_limits_success_async(httpx_mock):
    """Tests successful asynchronous retrieval of rate limit information."""
    mock_response_data = {
        "data": {
            "accessPermitted": True,
            "apiTier": {
                "id": "tier-standard",
                "isCharged": True
            },
            "balances": {
                "USD": 95.50,
                "VCU": 1000.0
            },
            "keyExpiration": "2026-01-01T00:00:00Z",
            "nextEpochBegins": "2025-05-01T00:00:00Z",
            "rateLimits": [
                {
                    "apiModelId": "model-1",
                    "rateLimits": [
                        {
                            "limit": 10.0,
                            "remaining": 8.5,
                            "unit": "RPM"
                        }
                    ]
                }
            ]
        }
    }

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/api_keys/rate_limits",
        json=mock_response_data,
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        response = await client.api_keys.get_rate_limits()

    assert isinstance(response, dict)
    assert "accessPermitted" in response
    assert response["accessPermitted"] is True
    assert response["apiTier"]["id"] == "tier-standard"
    assert response["balances"]["USD"] == 95.50
    assert response["nextEpochBegins"] == "2025-05-01T00:00:00Z"
    assert len(response["rateLimits"]) == 1
    assert response["rateLimits"][0]["apiModelId"] == "model-1"


@pytest.mark.asyncio
async def test_get_rate_limit_logs_success_async(httpx_mock):
    """Tests successful asynchronous retrieval of rate limit logs."""
    mock_response_data: MockRateLimitLogList = {
        "data": [
            {
                "apiKeyId": "key-123456",
                "modelId": "model-1",
                "rateLimitTier": "tier-standard",
                "rateLimitType": "RPM",
                "timestamp": "2025-05-01T12:30:45Z"
            },
            {
                "apiKeyId": "key-123456",
                "modelId": "model-2",
                "rateLimitTier": "tier-standard",
                "rateLimitType": "TPM",
                "timestamp": "2025-05-01T12:35:22Z"
            }
        ],
        "object": "list"
    }

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/api_keys/rate_limits/log",
        json=mock_response_data,
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        response: RateLimitLogList = await client.api_keys.get_rate_limit_logs()

    assert isinstance(response, dict) # Client returns a dict with data and object fields
    assert len(response["data"]) == 2 # Response data contains the list
    # Assert on the data items within the response dict
    assert response["data"][0]["apiKeyId"] == "key-123456"
    assert response["data"][0]["modelId"] == "model-1"
    assert response["data"][1]["rateLimitType"] == "TPM"


@pytest.mark.asyncio
async def test_api_key_error_async(httpx_mock):
    """Tests asynchronous API error handling for API keys operations."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/api_keys",
        status_code=401,
        json={"error": {"message": "Invalid API key", "type": "authentication_error"}},
    )

    async with AsyncVeniceClient(api_key="invalid-key") as client:
        with pytest.raises(AuthenticationError) as excinfo:
            await client.api_keys.list()

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 401
    assert "Invalid API key" in str(excinfo.value)