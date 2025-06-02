"""
Tests for missed lines coverage in the asynchronous AsyncApiKeys resource.
"""

import pytest
import collections
from typing import cast
from unittest.mock import Mock, AsyncMock
from venice_ai._async_client import AsyncVeniceClient
from venice_ai.resources.api_keys import AsyncApiKeys
from venice_ai.types.api_keys import ApiKeyCreateRequest, ApiKeyGenerateWeb3KeyCreateRequest


class TestAsyncApiKeysCreateMissedLines:
    """Test class for covering missed lines in AsyncApiKeys.create method."""

    @pytest.mark.asyncio
    async def test_async_create_request_with_asdict(self):
        """Test Case 6.1: Cover line 346 - async create request with _asdict method."""
        # Create a mock request object with _asdict method
        ApiKeyRequestAsdict = collections.namedtuple("ApiKeyRequestAsdict", ["name", "expires_at"])
        mock_request = ApiKeyRequestAsdict(name="test_key_asdict_async", expires_at=None)
        
        # Create mock async client
        mock_async_venice_client = AsyncMock(spec=AsyncVeniceClient)
        mock_async_venice_client.post.return_value = {"data": {"id": "key_123", "key": "secret_async"}}
        
        # Create AsyncApiKeys instance
        async_api_keys_instance = AsyncApiKeys(mock_async_venice_client)
        
        # Call create method
        await async_api_keys_instance.create(api_key_request=cast(ApiKeyCreateRequest, mock_request))
        
        # Verify post was called
        mock_async_venice_client.post.assert_called_once()
        
        # Verify json_data passed to post
        call_args = mock_async_venice_client.post.call_args
        json_data = call_args[1]["json_data"]
        assert json_data == {"name": "test_key_asdict_async"}  # expires_at=None should be filtered out

    @pytest.mark.asyncio
    async def test_async_create_request_with_dunder_dict(self):
        """Test Case 6.2: Cover line 348 - async create request with __dict__ attribute."""
        # Create a mock request object with __dict__
        class ApiKeyRequestDunderDict:
            def __init__(self, name, expires_at=None):
                self.name = name
                self.expires_at = expires_at

        mock_request = ApiKeyRequestDunderDict(name="test_key_dict_async", expires_at="2025-01-01T00:00:00Z")
        
        # Create mock async client
        mock_async_venice_client = AsyncMock(spec=AsyncVeniceClient)
        mock_async_venice_client.post.return_value = {"data": {"id": "key_456", "key": "secret_async_456"}}
        
        # Create AsyncApiKeys instance
        async_api_keys_instance = AsyncApiKeys(mock_async_venice_client)
        
        # Call create method
        await async_api_keys_instance.create(api_key_request=cast(ApiKeyCreateRequest, mock_request))
        
        # Verify post was called
        mock_async_venice_client.post.assert_called_once()
        
        # Verify json_data passed to post
        call_args = mock_async_venice_client.post.call_args
        json_data = call_args[1]["json_data"]
        assert json_data == {"name": "test_key_dict_async", "expires_at": "2025-01-01T00:00:00Z"}

    @pytest.mark.asyncio
    async def test_async_create_response_no_data_key(self):
        """Test Case 6.3: Cover line 358 - async create response without 'data' key."""
        # Create mock async client that returns response without 'data' key
        mock_async_venice_client = AsyncMock(spec=AsyncVeniceClient)
        mock_async_venice_client.post.return_value = {"id": "key_789", "key": "secret_direct_async", "name": "test_key_direct_async"}
        
        # Create AsyncApiKeys instance
        async_api_keys_instance = AsyncApiKeys(mock_async_venice_client)
        
        # Call create method
        api_key_request_obj = ApiKeyCreateRequest(description="test_key_direct_async")
        response = await async_api_keys_instance.create(api_key_request=api_key_request_obj)
        
        # Assert response matches direct return from post
        assert response == {"id": "key_789", "key": "secret_direct_async", "name": "test_key_direct_async"}

    @pytest.mark.asyncio
    async def test_async_retrieve_api_key(self):
        """Test Case 7.1: Cover lines 409-410 - async retrieve method."""
        # Create mock async client
        mock_async_venice_client = AsyncMock(spec=AsyncVeniceClient)
        mock_async_venice_client.get.return_value = {"id": "key_xyz", "name": "retrieved_key_async"}
        
        # Create AsyncApiKeys instance
        async_api_keys_instance = AsyncApiKeys(mock_async_venice_client)
        
        # Call retrieve method
        response = await async_api_keys_instance.retrieve(api_key_id="key_xyz")
        
        # Verify get was called with correct parameters
        mock_async_venice_client.get.assert_called_once_with("api_keys/key_xyz")
        
        # Assert response
        assert response == {"id": "key_xyz", "name": "retrieved_key_async"}


class TestAsyncApiKeysCreateWeb3KeyMissedLines:
    """Test class for covering missed lines in AsyncApiKeys.create_web3_key method."""

    @pytest.mark.asyncio
    async def test_async_create_web3_key_request_with_asdict(self):
        """Test Case 8.1: Cover line 449 - async web3 request with _asdict method."""
        # Create a mock request object with _asdict method
        Web3KeyRequestAsdict = collections.namedtuple("Web3KeyRequestAsdict", ["web3_network_id", "web3_address"])
        mock_request = Web3KeyRequestAsdict(web3_network_id="1", web3_address="0x123")
        
        # Create mock async client
        mock_async_venice_client = AsyncMock(spec=AsyncVeniceClient)
        mock_async_venice_client.post.return_value = {"id": "web3_key_async_123", "key": "secret_web3_async"}
        
        # Create AsyncApiKeys instance
        async_api_keys_instance = AsyncApiKeys(mock_async_venice_client)
        
        # Call create_web3_key method
        await async_api_keys_instance.create_web3_key(web3_key_request=cast(ApiKeyGenerateWeb3KeyCreateRequest, mock_request))
        
        # Verify post was called
        mock_async_venice_client.post.assert_called_once()
        
        # Verify json_data passed to post
        call_args = mock_async_venice_client.post.call_args
        json_data = call_args[1]["json_data"]
        assert json_data == {"web3_network_id": "1", "web3_address": "0x123"}

    @pytest.mark.asyncio
    async def test_async_create_web3_key_request_with_dunder_dict(self):
        """Test Case 8.2: Cover line 451 - async web3 request with __dict__ attribute."""
        # Create a mock request object with __dict__
        class Web3KeyRequestDunderDict:
            def __init__(self, web3_network_id, web3_address):
                self.web3_network_id = web3_network_id
                self.web3_address = web3_address

        mock_request = Web3KeyRequestDunderDict(web3_network_id="1", web3_address="0x456")
        
        # Create mock async client
        mock_async_venice_client = AsyncMock(spec=AsyncVeniceClient)
        mock_async_venice_client.post.return_value = {"id": "web3_key_async_456", "key": "secret_web3_async_456"}
        
        # Create AsyncApiKeys instance
        async_api_keys_instance = AsyncApiKeys(mock_async_venice_client)
        
        # Call create_web3_key method
        await async_api_keys_instance.create_web3_key(web3_key_request=cast(ApiKeyGenerateWeb3KeyCreateRequest, mock_request))
        
        # Verify post was called
        mock_async_venice_client.post.assert_called_once()
        
        # Verify json_data passed to post
        call_args = mock_async_venice_client.post.call_args
        json_data = call_args[1]["json_data"]
        assert json_data == {"web3_network_id": "1", "web3_address": "0x456"}

    @pytest.mark.asyncio
    async def test_async_get_rate_limits_response_no_data_key(self):
        """Test Case 9.1: Cover line 478 - async get_rate_limits response without 'data' key."""
        # Create mock async client that returns response without 'data' key
        mock_async_venice_client = AsyncMock(spec=AsyncVeniceClient)
        mock_async_venice_client.get.return_value = {"limit": 200, "remaining": 100}
        
        # Create AsyncApiKeys instance
        async_api_keys_instance = AsyncApiKeys(mock_async_venice_client)
        
        # Call get_rate_limits method
        response = await async_api_keys_instance.get_rate_limits()
        
        # Assert response matches direct mock return
        assert response == {"limit": 200, "remaining": 100}

    @pytest.mark.asyncio
    async def test_async_get_rate_limit_logs_direct_list_response(self):
        """Test Case 10.1: Cover lines 498-499 - async get_rate_limit_logs returns direct list."""
        # Create mock async client that returns a direct list
        mock_async_venice_client = AsyncMock(spec=AsyncVeniceClient)
        mock_async_venice_client.get.return_value = [{"event": "reset", "timestamp": "2025-01-01T00:00:00Z"}]
        
        # Create AsyncApiKeys instance
        async_api_keys_instance = AsyncApiKeys(mock_async_venice_client)
        
        # Call get_rate_limit_logs method
        response = await async_api_keys_instance.get_rate_limit_logs()
        
        # Assert response matches the direct list
        assert response == [{"event": "reset", "timestamp": "2025-01-01T00:00:00Z"}]

    @pytest.mark.asyncio
    async def test_async_get_rate_limit_logs_unexpected_response_fallback(self):
        """Test Case 10.2: Cover line 500 - async get_rate_limit_logs unexpected response fallback."""
        # Create mock async client that returns an empty dictionary
        mock_async_venice_client = AsyncMock(spec=AsyncVeniceClient)
        mock_async_venice_client.get.return_value = {}
        
        # Create AsyncApiKeys instance
        async_api_keys_instance = AsyncApiKeys(mock_async_venice_client)
        
        # Call get_rate_limit_logs method
        response = await async_api_keys_instance.get_rate_limit_logs()
        
        # Assert response is empty list (fallback)
        assert response == []