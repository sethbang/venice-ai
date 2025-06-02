"""
Tests for the synchronous ApiKeys resource.
"""

import pytest
import httpx
from typing import Dict, List, Literal, Optional, TypedDict, cast # Add cast
from unittest.mock import MagicMock, patch, ANY

from venice_ai import VeniceClient
from venice_ai.types.api_keys import (
    ApiKeyList, ApiKeyCreateRequest, ApiKeyCreateResponse,
    RateLimitInfo, RateLimitLogList, ApiKey, ConsumptionLimit,
    ApiKeyGenerateWeb3KeyCreateRequest, ApiKeyGenerateWeb3KeyCreateResponse, # Add these
    RateLimitLog # Add this
)
from venice_ai.exceptions import APIError, AuthenticationError, InvalidRequestError
from venice_ai.resources.api_keys import ApiKeys


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


def test_list_success(httpx_mock):
    """Tests successful retrieval of API keys list."""
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

    client = VeniceClient(api_key="test-key")
    response = client.api_keys.list()

    assert isinstance(response, list)
    assert len(response) == 1
    assert response[0]["id"] == "key-123456"
    assert response[0]["apiKeyType"] == "INFERENCE"
    assert response[0]["consumptionLimits"].get("usd") == 100.0


def test_list_with_page_and_limit(httpx_mock):
    """Tests list with both page and limit parameters."""
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

    client = VeniceClient(api_key="test-key")
    client.api_keys.list(page=1, limit=10)
    
    # Verify the request contained the correct parameters
    request = httpx_mock.get_requests()[0]
    assert request.url.params.get("page") == "1"
    assert request.url.params.get("limit") == "10"


def test_list_with_page_only(httpx_mock):
    """Tests list with only page parameter."""
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

    client = VeniceClient(api_key="test-key")
    client.api_keys.list(page=2)
    
    # Verify the request contained the correct parameter
    request = httpx_mock.get_requests()[0]
    assert request.url.params.get("page") == "2"
    assert "limit" not in request.url.params


def test_list_with_limit_only(httpx_mock):
    """Tests list with only limit parameter."""
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

    client = VeniceClient(api_key="test-key")
    client.api_keys.list(limit=5)
    
    # Verify the request contained the correct parameter
    request = httpx_mock.get_requests()[0]
    assert request.url.params.get("limit") == "5"
    assert "page" not in request.url.params


def test_api_keys_list_returns_list_directly(httpx_mock):
    """Tests handling when the API returns a list directly instead of a dict with data."""
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

    client = VeniceClient(api_key="test-key")
    response = client.api_keys.list()
    
    assert isinstance(response, list)
    assert len(response) == 1
    assert response[0]["id"] == "key-123456"


def test_api_keys_list_returns_empty_on_unexpected_response(httpx_mock):
    """Tests handling when the API returns an unexpected response type."""
    # Mock response is a string, which is neither a dict with 'data' nor a list
    mock_response_data = "unexpected_string_response"

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/api_keys",
        json=mock_response_data,
        status_code=200,
    )

    client = VeniceClient(api_key="test-key")
    response = client.api_keys.list()
    
    assert isinstance(response, list)
    assert len(response) == 0


def test_api_keys_list_with_pagination_params():
    """Unit test for verifying pagination parameters are passed correctly to the client."""
    # Create a mock client
    mock_client = MagicMock(spec=VeniceClient)
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.json.return_value = {"data": [], "object": "list"}
    mock_client.get.return_value = mock_httpx_response

    # Create ApiKeys resource with the mock client
    api_keys_resource = ApiKeys(mock_client)

    # Define test pagination parameters
    test_limit = 10
    test_page = 2

    # Call the list method with both parameters
    api_keys_resource.list(limit=test_limit, page=test_page)

    # Verify the client's get method was called with the correct parameters
    mock_client.get.assert_called_once()
    call_args = mock_client.get.call_args[0]
    call_kwargs = mock_client.get.call_args[1]
    
    assert call_args[0] == "api_keys"
    assert call_kwargs["params"] == {"limit": test_limit, "page": test_page}


def test_api_keys_list_with_limit_only_param():
    """Unit test for verifying only limit parameter is passed correctly to the client."""
    # Create a mock client
    mock_client = MagicMock(spec=VeniceClient)
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.json.return_value = {"data": [], "object": "list"}
    mock_client.get.return_value = mock_httpx_response

    # Create ApiKeys resource with the mock client
    api_keys_resource = ApiKeys(mock_client)

    # Define test parameter
    test_limit = 10

    # Call the list method with only limit parameter
    api_keys_resource.list(limit=test_limit)

    # Verify the client's get method was called with the correct parameters
    mock_client.get.assert_called_once()
    call_args = mock_client.get.call_args[0]
    call_kwargs = mock_client.get.call_args[1]
    
    assert call_args[0] == "api_keys"
    assert call_kwargs["params"] == {"limit": test_limit}


def test_api_keys_list_with_page_only_param():
    """Unit test for verifying only page parameter is passed correctly to the client."""
    # Create a mock client
    mock_client = MagicMock(spec=VeniceClient)
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.json.return_value = {"data": [], "object": "list"}
    mock_client.get.return_value = mock_httpx_response

    # Create ApiKeys resource with the mock client
    api_keys_resource = ApiKeys(mock_client)

    # Define test parameter
    test_page = 2

    # Call the list method with only page parameter
    api_keys_resource.list(page=test_page)

    # Verify the client's get method was called with the correct parameters
    mock_client.get.assert_called_once()
    call_args = mock_client.get.call_args[0]
    call_kwargs = mock_client.get.call_args[1]
    
    assert call_args[0] == "api_keys"
    assert call_kwargs["params"] == {"page": test_page}


def test_create_success(httpx_mock):
    """Tests successful creation of an API key."""
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

    client = VeniceClient(api_key="test-key")
    response = client.api_keys.create(api_key_request=mock_request)

    assert isinstance(response, dict)
    assert "description" in response
    assert response["description"] == "New Test API Key"
    assert response["consumptionLimit"]["usd"] == 200.0


def test_delete_success(httpx_mock):
    """Tests successful deletion of an API key using query parameter."""
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

    client = VeniceClient(api_key="test-key")
    response = client.api_keys.delete(api_key_id=api_key_id)

    assert isinstance(response, dict)
    assert response["success"] is True
    assert response["message"] == "API key deleted successfully"


def test_get_web3_token_success(httpx_mock):
    """Tests successful retrieval of a Web3 API key token."""
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

    client = VeniceClient(api_key="test-key")
    response = client.api_keys.get_web3_token() # Corrected method call

    assert isinstance(response, dict)
    assert "token" in response
    assert "expiresAt" in response
    assert response["token"] == "web3-token-1234567890"


def test_get_web3_token_api_error(httpx_mock):
    """Tests API error handling for GET /api_keys/generate_web3_key."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/api_keys/generate_web3_key",
        status_code=401,
        json={"error": {"message": "Unauthorized", "type": "authentication_error"}},
    )

    client = VeniceClient(api_key="invalid-key")

    with pytest.raises(AuthenticationError) as excinfo:
        client.api_keys.get_web3_token() # Corrected method call

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 401
    assert "Unauthorized" in str(excinfo.value)


def test_create_web3_key_post_success_required(httpx_mock):
    """Tests successful creation of a Web3 API key with required parameters."""
    mock_request = {
        "apiKeyType": "WEB3",
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

    client = VeniceClient(api_key="test-key")
    response = client.api_keys.create_web3_key(web3_key_request=cast(ApiKeyGenerateWeb3KeyCreateRequest, mock_request)) # Corrected method call and argument name

    assert isinstance(response, dict)
    assert "apiKey" in response
    assert response["apiKeyPrefix"] == "sk-web3api"


def test_create_web3_key_post_success_optional(httpx_mock):
    """Tests successful creation of a Web3 API key with optional parameters."""
    mock_request = {
        "apiKeyType": "INFERENCE",
        "description": "My Web3 Key",
        "expiresAt": "2026-12-31T23:59:59Z",
        "consumptionLimit": {"usd": 50.0}
    }

    mock_response_data = {
        "data": {
            "apiKey": "sk-web3apikeyoptional123",
            "apiKeyPrefix": "sk-web3api",
            "consumptionLimit": {"usd": 50.0, "vcu": None},
            "description": "My Web3 Key",
            "expiresAt": "2026-12-31T23:59:59Z"
        }
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/api_keys/generate_web3_key",
        json=mock_response_data,
        status_code=200,
    )

    client = VeniceClient(api_key="test-key")
    response = client.api_keys.create_web3_key(web3_key_request=cast(ApiKeyGenerateWeb3KeyCreateRequest, mock_request)) # Corrected method call and argument name

    assert isinstance(response, dict)
    assert response["data"]["description"] == "My Web3 Key"
    assert response["data"]["expiresAt"] == "2026-12-31T23:59:59Z"
    assert cast(ConsumptionLimit, response["data"]["consumptionLimit"]).get("usd") == 50.0


def test_create_web3_key_post_validation_error(httpx_mock):
    """Tests validation error handling for POST /api_keys/generate_web3_key."""
    mock_request = {
        "apiKeyType": "INVALID_TYPE", # Invalid type
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/api_keys/generate_web3_key",
        status_code=400,
        json={"error": {"message": "Invalid apiKeyType", "type": "invalid_request_error"}},
    )

    client = VeniceClient(api_key="test-key")

    with pytest.raises(InvalidRequestError) as excinfo:
        client.api_keys.create_web3_key(web3_key_request=cast(ApiKeyGenerateWeb3KeyCreateRequest, mock_request)) # Corrected method call and argument name

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 400
    assert "Invalid apiKeyType" in str(excinfo.value)


def test_get_rate_limits_success(httpx_mock):
    """Tests successful retrieval of rate limit information."""
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

    client = VeniceClient(api_key="test-key")
    response = client.api_keys.get_rate_limits()

    assert isinstance(response, dict)
    assert "accessPermitted" in response
    assert response["accessPermitted"] is True
    assert response["apiTier"]["id"] == "tier-standard"
    assert response["balances"]["USD"] == 95.50
    assert response["nextEpochBegins"] == "2025-05-01T00:00:00Z"
    assert len(response["rateLimits"]) == 1
    assert response["rateLimits"][0]["apiModelId"] == "model-1"


def test_get_rate_limit_logs_success(httpx_mock):
    """Tests successful retrieval of rate limit logs."""
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

    client = VeniceClient(api_key="test-key")
    response: RateLimitLogList = client.api_keys.get_rate_limit_logs()

    assert isinstance(response, dict)
    assert len(response["data"]) == 2
    logged_data = cast(List[RateLimitLog], response["data"])
    assert logged_data[0]["apiKeyId"] == "key-123456"
    assert logged_data[0]["modelId"] == "model-1"
    assert logged_data[1]["rateLimitType"] == "TPM"


def test_api_key_error(httpx_mock):
    """Tests API error handling for API keys operations."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/api_keys",
        status_code=401,
        json={"error": {"message": "Invalid API key", "type": "authentication_error"}},
    )

    client = VeniceClient(api_key="invalid-key")

    with pytest.raises(AuthenticationError) as excinfo:
        client.api_keys.list()

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 401
    assert "Invalid API key" in str(excinfo.value)


class TestApiKeysMissedLines:
    """Test class for covering missed lines in ApiKeys synchronous methods."""

    def test_create_request_with_asdict(self):
        """Test Case 1.1: Cover line 109 - request with _asdict method."""
        import collections
        from unittest.mock import Mock
        from venice_ai._client import VeniceClient
        from venice_ai.resources.api_keys import ApiKeys

        # Create a mock request object with _asdict method
        ApiKeyRequestAsdict = collections.namedtuple("ApiKeyRequestAsdict", ["name", "expires_at"])
        mock_request = ApiKeyRequestAsdict(name="test_key_asdict", expires_at=None)
        
        # Create mock client
        mock_venice_client = Mock(spec=VeniceClient)
        mock_venice_client.post.return_value = {"data": {"id": "key_123", "key": "secret_..."}}
        
        # Create ApiKeys instance
        api_keys_instance = ApiKeys(mock_venice_client)
        
        # Call create method
        api_keys_instance.create(api_key_request=cast(ApiKeyCreateRequest, mock_request))
        
        # Verify post was called
        mock_venice_client.post.assert_called_once()
        
        # Inspect the json_data argument passed to post
        call_args = mock_venice_client.post.call_args
        json_data = call_args[1]["json_data"]
        assert json_data == {"name": "test_key_asdict"}  # expires_at=None should be filtered out

    def test_create_request_with_dunder_dict(self):
        """Test Case 1.2: Cover line 111 - request with __dict__ attribute."""
        from unittest.mock import Mock
        from venice_ai._client import VeniceClient
        from venice_ai.resources.api_keys import ApiKeys

        # Create a mock request object with __dict__
        class ApiKeyRequestDunderDict:
            def __init__(self, name, expires_at=None):
                self.name = name
                self.expires_at = expires_at

        mock_request = ApiKeyRequestDunderDict(name="test_key_dict", expires_at="2025-01-01T00:00:00Z")
        
        # Create mock client
        mock_venice_client = Mock(spec=VeniceClient)
        mock_venice_client.post.return_value = {"data": {"id": "key_123", "key": "secret_..."}}
        
        # Create ApiKeys instance
        api_keys_instance = ApiKeys(mock_venice_client)
        
        # Call create method
        api_keys_instance.create(api_key_request=cast(ApiKeyCreateRequest, mock_request))
        
        # Verify post was called
        mock_venice_client.post.assert_called_once()
        
        # Assert json_data passed to post
        call_args = mock_venice_client.post.call_args
        json_data = call_args[1]["json_data"]
        assert json_data == {"name": "test_key_dict", "expires_at": "2025-01-01T00:00:00Z"}

    def test_create_response_no_data_key(self):
        """Test Case 1.3: Cover line 121 - response without 'data' key."""
        from unittest.mock import Mock
        from venice_ai._client import VeniceClient
        from venice_ai.resources.api_keys import ApiKeys

        # Create mock client that returns response without 'data' key
        mock_venice_client = Mock(spec=VeniceClient)
        mock_venice_client.post.return_value = {"id": "key_123", "key": "secret_...", "name": "test_key_direct"}
        
        # Create ApiKeys instance
        api_keys_instance = ApiKeys(mock_venice_client)
        
        # Call create method
        response = api_keys_instance.create(api_key_request=cast(ApiKeyCreateRequest, {"name": "test_key_direct"}))
        
        # Assert response matches direct return from post
        assert response == {"id": "key_123", "key": "secret_...", "name": "test_key_direct"}

    def test_retrieve_api_key(self):
        """Test Case 2.1: Cover lines 172-173 - retrieve method."""
        from unittest.mock import Mock
        from venice_ai._client import VeniceClient
        from venice_ai.resources.api_keys import ApiKeys

        # Create mock client
        mock_venice_client = Mock(spec=VeniceClient)
        mock_venice_client.get.return_value = {"id": "key_abc", "name": "retrieved_key"}
        
        # Create ApiKeys instance
        api_keys_instance = ApiKeys(mock_venice_client)
        
        # Call retrieve method
        response = api_keys_instance.retrieve(api_key_id="key_abc")
        
        # Verify get was called with correct parameters
        mock_venice_client.get.assert_called_once_with("api_keys/key_abc")
        
        # Assert response
        assert response == {"id": "key_abc", "name": "retrieved_key"}


class TestApiKeysCreateWeb3KeyMissedLines:
    """Test class for covering missed lines in ApiKeys.create_web3_key method."""

    def test_create_web3_key_request_with_asdict(self):
        """Test Case 3.1: Cover line 212 - web3 request with _asdict method."""
        import collections
        from unittest.mock import Mock
        from venice_ai._client import VeniceClient
        from venice_ai.resources.api_keys import ApiKeys

        # Create a mock request object with _asdict method
        Web3KeyRequestAsdict = collections.namedtuple("Web3KeyRequestAsdict", ["web3_network_id", "web3_address"])
        mock_request = Web3KeyRequestAsdict(web3_network_id="1", web3_address="0x123")
        
        # Create mock client
        mock_venice_client = Mock(spec=VeniceClient)
        mock_venice_client.post.return_value = {"id": "web3_key_123", "key": "secret_web3"}
        
        # Create ApiKeys instance
        api_keys_instance = ApiKeys(mock_venice_client)
        
        # Call create_web3_key method
        api_keys_instance.create_web3_key(web3_key_request=cast(ApiKeyGenerateWeb3KeyCreateRequest, mock_request))
        
        # Verify post was called
        mock_venice_client.post.assert_called_once()
        
        # Verify json_data passed to post
        call_args = mock_venice_client.post.call_args
        json_data = call_args[1]["json_data"]
        assert json_data == {"web3_network_id": "1", "web3_address": "0x123"}

    def test_create_web3_key_request_with_dunder_dict(self):
        """Test Case 3.2: Cover line 214 - web3 request with __dict__ attribute."""
        from unittest.mock import Mock
        from venice_ai._client import VeniceClient
        from venice_ai.resources.api_keys import ApiKeys

        # Create a mock request object with __dict__
        class Web3KeyRequestDunderDict:
            def __init__(self, web3_network_id, web3_address):
                self.web3_network_id = web3_network_id
                self.web3_address = web3_address

        mock_request = Web3KeyRequestDunderDict(web3_network_id="1", web3_address="0x456")
        
        # Create mock client
        mock_venice_client = Mock(spec=VeniceClient)
        mock_venice_client.post.return_value = {"id": "web3_key_456", "key": "secret_web3_456"}
        
        # Create ApiKeys instance
        api_keys_instance = ApiKeys(mock_venice_client)
        
        # Call create_web3_key method
        api_keys_instance.create_web3_key(web3_key_request=cast(ApiKeyGenerateWeb3KeyCreateRequest, mock_request))
        
        # Verify post was called
        mock_venice_client.post.assert_called_once()
        
        # Verify json_data passed to post
        call_args = mock_venice_client.post.call_args
        json_data = call_args[1]["json_data"]
        assert json_data == {"web3_network_id": "1", "web3_address": "0x456"}

    def test_get_rate_limits_response_no_data_key(self):
        """Test Case 4.1: Cover line 241 - get_rate_limits response without 'data' key."""
        from unittest.mock import Mock
        from venice_ai._client import VeniceClient
        from venice_ai.resources.api_keys import ApiKeys

        # Create mock client that returns response without 'data' key
        mock_venice_client = Mock(spec=VeniceClient)
        mock_venice_client.get.return_value = {"limit": 100, "remaining": 50}
        
        # Create ApiKeys instance
        api_keys_instance = ApiKeys(mock_venice_client)
        
        # Call get_rate_limits method
        response = api_keys_instance.get_rate_limits()
        
        # Assert response matches direct mock return
        assert response == {"limit": 100, "remaining": 50}

    def test_get_rate_limit_logs_direct_list_response(self):
        """Test Case 5.1: Cover lines 262-263 - get_rate_limit_logs returns direct list."""
        from unittest.mock import Mock
        from venice_ai._client import VeniceClient
        from venice_ai.resources.api_keys import ApiKeys

        # Create mock client that returns a direct list
        mock_venice_client = Mock(spec=VeniceClient)
        mock_venice_client.get.return_value = [{"event": "reset", "timestamp": "2025-01-01T00:00:00Z"}]
        
        # Create ApiKeys instance
        api_keys_instance = ApiKeys(mock_venice_client)
        
        # Call get_rate_limit_logs method
        response = api_keys_instance.get_rate_limit_logs()
        
        # Assert response matches the direct list
        assert response == [{"event": "reset", "timestamp": "2025-01-01T00:00:00Z"}]

    def test_get_rate_limit_logs_unexpected_response_fallback(self):
        """Test Case 5.2: Cover line 264 - get_rate_limit_logs unexpected response fallback."""
        from unittest.mock import Mock
        from venice_ai._client import VeniceClient
        from venice_ai.resources.api_keys import ApiKeys

        # Create mock client that returns an empty dictionary
        mock_venice_client = Mock(spec=VeniceClient)
        mock_venice_client.get.return_value = {}
        
        # Create ApiKeys instance
        api_keys_instance = ApiKeys(mock_venice_client)
        
        # Call get_rate_limit_logs method
        response = api_keys_instance.get_rate_limit_logs()
        
        # Assert response is empty list (fallback)
        assert response == []