"""
Tests to improve coverage for API keys functionality.

This module focuses on testing the specific lines and branches that are missing
coverage in src/venice_ai/resources/api_keys.py, including:
- Fallback conversions for unexpected request types
- Response processing variations (missing fields, renamed fields)
- Parameter variation in get_rate_limit_logs
"""

import pytest
from typing import Dict, Any, List, Optional, cast
from unittest.mock import MagicMock, AsyncMock
from collections.abc import Mapping

from venice_ai import VeniceClient, AsyncVeniceClient
from venice_ai.resources.api_keys import ApiKeys, AsyncApiKeys
from venice_ai.types.api_keys import (
    ApiKeyCreateRequest, ApiKeyGenerateWeb3KeyCreateRequest,
    ApiKey, RateLimitLogList
)
from venice_ai.exceptions import APIResponseProcessingError


class UnexpectedRequestType:
    """A request type that doesn't have __dict__, isn't a Mapping, and doesn't have _asdict."""
    
    def __init__(self, description: str, apiKeyType: str):
        # Store data in a way that's not accessible via __dict__
        # Use __slots__ to prevent __dict__ from being created
        self.description = description
        self.apiKeyType = apiKeyType
    
    __slots__ = ['description', 'apiKeyType']
    
    def __iter__(self):
        # This will be called by dict() as a fallback
        # Return key-value pairs for the attributes
        for slot in self.__slots__:
            if hasattr(self, slot):
                yield (slot, getattr(self, slot))


class TestApiKeysFallbackConversions:
    """Test fallback conversions for unexpected request types."""

    def test_create_with_unexpected_request_type_sync(self):
        """Test create method with unexpected request type (sync)."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.post.return_value = {
            "data": {
                "id": "key_123",
                "description": "test_key",
                "apiKeyType": "INFERENCE",
                "consumptionLimits": {}
            }
        }
        
        api_keys = ApiKeys(mock_client)
        
        # Use an unexpected request type that will trigger the fallback
        unexpected_request = UnexpectedRequestType("test_key", "INFERENCE")
        
        result = api_keys.create(api_key_request=cast(ApiKeyCreateRequest, unexpected_request))
        
        # Verify the fallback conversion worked
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "api_keys"
        assert call_args[1]["json_data"]["description"] == "test_key"
        assert call_args[1]["json_data"]["apiKeyType"] == "INFERENCE"
        
        assert result["id"] == "key_123"

    @pytest.mark.asyncio
    async def test_create_with_unexpected_request_type_async(self):
        """Test create method with unexpected request type (async)."""
        mock_client = AsyncMock(spec=AsyncVeniceClient)
        mock_client.post.return_value = {
            "data": {
                "id": "key_123",
                "description": "test_key",
                "apiKeyType": "INFERENCE",
                "consumptionLimits": {}
            }
        }
        
        api_keys = AsyncApiKeys(mock_client)
        
        # Use an unexpected request type that will trigger the fallback
        unexpected_request = UnexpectedRequestType("test_key", "INFERENCE")
        
        result = await api_keys.create(api_key_request=cast(ApiKeyCreateRequest, unexpected_request))
        
        # Verify the fallback conversion worked
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "api_keys"
        assert call_args[1]["json_data"]["description"] == "test_key"
        assert call_args[1]["json_data"]["apiKeyType"] == "INFERENCE"
        
        assert result["id"] == "key_123"

    def test_create_web3_key_with_unexpected_request_type_sync(self):
        """Test create_web3_key method with unexpected request type (sync)."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.post.return_value = {
            "data": {
                "apiKey": "sk-web3key123",
                "description": "web3_key"
            }
        }
        
        api_keys = ApiKeys(mock_client)
        
        # Use an unexpected request type that will trigger the fallback
        unexpected_request = UnexpectedRequestType("web3_key", "INFERENCE")
        
        result = api_keys.create_web3_key(web3_key_request=cast(ApiKeyGenerateWeb3KeyCreateRequest, unexpected_request))
        
        # Verify the fallback conversion worked
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "api_keys/generate_web3_key"
        assert call_args[1]["json_data"]["description"] == "web3_key"

    @pytest.mark.asyncio
    async def test_create_web3_key_with_unexpected_request_type_async(self):
        """Test create_web3_key method with unexpected request type (async)."""
        mock_client = AsyncMock(spec=AsyncVeniceClient)
        mock_client.post.return_value = {
            "data": {
                "apiKey": "sk-web3key123",
                "description": "web3_key"
            }
        }
        
        api_keys = AsyncApiKeys(mock_client)
        
        # Use an unexpected request type that will trigger the fallback
        unexpected_request = UnexpectedRequestType("web3_key", "INFERENCE")
        
        result = await api_keys.create_web3_key(web3_key_request=cast(ApiKeyGenerateWeb3KeyCreateRequest, unexpected_request))
        
        # Verify the fallback conversion worked
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "api_keys/generate_web3_key"
        assert call_args[1]["json_data"]["description"] == "web3_key"


class TestApiKeysResponseProcessing:
    """Test response processing variations."""

    def test_create_response_with_consumption_limit_field_sync(self):
        """Test create response with consumptionLimit field that needs renaming (sync)."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.post.return_value = {
            "data": {
                "id": "key_123",
                "description": "test_key",
                "apiKeyType": "INFERENCE",
                "consumptionLimit": {"usd": 100.0}  # This should be renamed to consumptionLimits
            }
        }
        
        api_keys = ApiKeys(mock_client)
        
        result = api_keys.create(api_key_request={"description": "test_key", "apiKeyType": "INFERENCE"})
        
        # Verify the field was renamed
        assert "consumptionLimits" in result
        assert cast(Dict[str, Any], result["consumptionLimits"])["usd"] == 100.0
        assert "consumptionLimit" not in result

    def test_create_response_missing_consumption_limits_sync(self):
        """Test create response missing consumptionLimits field (sync)."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.post.return_value = {
            "data": {
                "id": "key_123",
                "description": "test_key",
                "apiKeyType": "INFERENCE"
                # Missing consumptionLimits field
            }
        }
        
        api_keys = ApiKeys(mock_client)
        
        result = api_keys.create(api_key_request={"description": "test_key", "apiKeyType": "INFERENCE"})
        
        # Verify default consumptionLimits was added
        assert "consumptionLimits" in result
        assert result["consumptionLimits"] == {}

    def test_create_response_without_data_key_sync(self):
        """Test create response without data key (sync)."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.post.return_value = {
            "id": "key_123",
            "description": "test_key",
            "apiKeyType": "INFERENCE",
            "consumptionLimits": {}
        }
        
        api_keys = ApiKeys(mock_client)
        
        result = api_keys.create(api_key_request={"description": "test_key", "apiKeyType": "INFERENCE"})
        
        # Verify the response was returned directly
        assert result["id"] == "key_123"
        assert result["description"] == "test_key"

    @pytest.mark.asyncio
    async def test_create_response_with_consumption_limit_field_async(self):
        """Test create response with consumptionLimit field that needs renaming (async)."""
        mock_client = AsyncMock(spec=AsyncVeniceClient)
        mock_client.post.return_value = {
            "data": {
                "id": "key_123",
                "description": "test_key",
                "apiKeyType": "INFERENCE",
                "consumptionLimit": {"usd": 100.0}  # This should be renamed to consumptionLimits
            }
        }
        
        api_keys = AsyncApiKeys(mock_client)
        
        result = await api_keys.create(api_key_request={"description": "test_key", "apiKeyType": "INFERENCE"})
        
        # Verify the field was renamed
        assert "consumptionLimits" in result
        assert cast(Dict[str, Any], result["consumptionLimits"])["usd"] == 100.0
        assert "consumptionLimit" not in result

    @pytest.mark.asyncio
    async def test_create_response_missing_consumption_limits_async(self):
        """Test create response missing consumptionLimits field (async)."""
        mock_client = AsyncMock(spec=AsyncVeniceClient)
        mock_client.post.return_value = {
            "data": {
                "id": "key_123",
                "description": "test_key",
                "apiKeyType": "INFERENCE"
                # Missing consumptionLimits field
            }
        }
        
        api_keys = AsyncApiKeys(mock_client)
        
        result = await api_keys.create(api_key_request={"description": "test_key", "apiKeyType": "INFERENCE"})
        
        # Verify default consumptionLimits was added
        assert "consumptionLimits" in result
        assert result["consumptionLimits"] == {}

    @pytest.mark.asyncio
    async def test_create_response_without_data_key_async(self):
        """Test create response without data key (async)."""
        mock_client = AsyncMock(spec=AsyncVeniceClient)
        mock_client.post.return_value = {
            "id": "key_123",
            "description": "test_key",
            "apiKeyType": "INFERENCE",
            "consumptionLimits": {}
        }
        
        api_keys = AsyncApiKeys(mock_client)
        
        result = await api_keys.create(api_key_request={"description": "test_key", "apiKeyType": "INFERENCE"})
        
        # Verify the response was returned directly
        assert result["id"] == "key_123"
        assert result["description"] == "test_key"


class TestApiKeysParameterVariation:
    """Test parameter variation in get_rate_limit_logs."""

    def test_get_rate_limit_logs_with_all_parameters_sync(self):
        """Test get_rate_limit_logs with all parameters (sync)."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.get.return_value = {"data": []}
        
        api_keys = ApiKeys(mock_client)
        
        api_keys.get_rate_limit_logs(
            api_key_id="key_123",
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-31T23:59:59Z",
            limit=50,
            page=2
        )
        
        # Verify all parameters were passed
        call_args = mock_client.get.call_args
        assert call_args[0][0] == "api_keys/rate_limits/log"
        params = call_args[1]["params"]
        assert params["api_key_id"] == "key_123"
        assert params["start_date"] == "2024-01-01T00:00:00Z"
        assert params["end_date"] == "2024-01-31T23:59:59Z"
        assert params["limit"] == 50
        assert params["page"] == 2

    def test_get_rate_limit_logs_with_some_parameters_sync(self):
        """Test get_rate_limit_logs with some parameters (sync)."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.get.return_value = {"data": []}
        
        api_keys = ApiKeys(mock_client)
        
        api_keys.get_rate_limit_logs(
            api_key_id="key_123",
            limit=25
            # start_date, end_date, page are None
        )
        
        # Verify only non-None parameters were passed
        call_args = mock_client.get.call_args
        params = call_args[1]["params"]
        assert params["api_key_id"] == "key_123"
        assert params["limit"] == 25
        assert "start_date" not in params
        assert "end_date" not in params
        assert "page" not in params

    def test_get_rate_limit_logs_with_no_parameters_sync(self):
        """Test get_rate_limit_logs with no parameters (sync)."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.get.return_value = {"data": []}
        
        api_keys = ApiKeys(mock_client)
        
        api_keys.get_rate_limit_logs()
        
        # Verify no params were passed (params should be None)
        call_args = mock_client.get.call_args
        assert call_args[1]["params"] is None

    @pytest.mark.asyncio
    async def test_get_rate_limit_logs_with_all_parameters_async(self):
        """Test get_rate_limit_logs with all parameters (async)."""
        mock_client = AsyncMock(spec=AsyncVeniceClient)
        mock_client.get.return_value = {"data": []}
        
        api_keys = AsyncApiKeys(mock_client)
        
        await api_keys.get_rate_limit_logs(
            api_key_id="key_123",
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-31T23:59:59Z",
            limit=50,
            page=2
        )
        
        # Verify all parameters were passed
        call_args = mock_client.get.call_args
        assert call_args[0][0] == "api_keys/rate_limits/log"
        params = call_args[1]["params"]
        assert params["api_key_id"] == "key_123"
        assert params["start_date"] == "2024-01-01T00:00:00Z"
        assert params["end_date"] == "2024-01-31T23:59:59Z"
        assert params["limit"] == 50
        assert params["page"] == 2

    @pytest.mark.asyncio
    async def test_get_rate_limit_logs_with_some_parameters_async(self):
        """Test get_rate_limit_logs with some parameters (async)."""
        mock_client = AsyncMock(spec=AsyncVeniceClient)
        mock_client.get.return_value = {"data": []}
        
        api_keys = AsyncApiKeys(mock_client)
        
        await api_keys.get_rate_limit_logs(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-31T23:59:59Z"
            # api_key_id, limit, page are None
        )
        
        # Verify only non-None parameters were passed
        call_args = mock_client.get.call_args
        params = call_args[1]["params"]
        assert params["start_date"] == "2024-01-01T00:00:00Z"
        assert params["end_date"] == "2024-01-31T23:59:59Z"
        assert "api_key_id" not in params
        assert "limit" not in params
        assert "page" not in params

    @pytest.mark.asyncio
    async def test_get_rate_limit_logs_with_no_parameters_async(self):
        """Test get_rate_limit_logs with no parameters (async)."""
        mock_client = AsyncMock(spec=AsyncVeniceClient)
        mock_client.get.return_value = {"data": []}
        
        api_keys = AsyncApiKeys(mock_client)
        
        await api_keys.get_rate_limit_logs()
        
        # Verify no params were passed (params should be None)
        call_args = mock_client.get.call_args
        assert call_args[1]["params"] is None


class TestApiKeysAdditionalCoverage:
    """Test additional coverage scenarios."""

    def test_get_rate_limits_with_data_key_sync(self):
        """Test get_rate_limits with data key in response (sync)."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.get.return_value = {
            "data": {
                "limits": {"requests_per_minute": 100}
            }
        }
        
        api_keys = ApiKeys(mock_client)
        
        result = api_keys.get_rate_limits()
        
        # Verify data was extracted from data key
        assert cast(Dict[str, Any], result)["limits"]["requests_per_minute"] == 100

    def test_get_rate_limits_without_data_key_sync(self):
        """Test get_rate_limits without data key in response (sync)."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.get.return_value = {
            "limits": {"requests_per_minute": 100}
        }
        
        api_keys = ApiKeys(mock_client)
        
        result = api_keys.get_rate_limits()
        
        # Verify response was returned directly
        assert cast(Dict[str, Any], result)["limits"]["requests_per_minute"] == 100

    @pytest.mark.asyncio
    async def test_get_rate_limits_with_data_key_async(self):
        """Test get_rate_limits with data key in response (async)."""
        mock_client = AsyncMock(spec=AsyncVeniceClient)
        mock_client.get.return_value = {
            "data": {
                "limits": {"requests_per_minute": 100}
            }
        }
        
        api_keys = AsyncApiKeys(mock_client)
        
        result = await api_keys.get_rate_limits()
        
        # Verify data was extracted from data key
        assert cast(Dict[str, Any], result)["limits"]["requests_per_minute"] == 100

    @pytest.mark.asyncio
    async def test_get_rate_limits_without_data_key_async(self):
        """Test get_rate_limits without data key in response (async)."""
        mock_client = AsyncMock(spec=AsyncVeniceClient)
        mock_client.get.return_value = {
            "limits": {"requests_per_minute": 100}
        }
        
        api_keys = AsyncApiKeys(mock_client)
        
        result = await api_keys.get_rate_limits()
        
        # Verify response was returned directly
        assert cast(Dict[str, Any], result)["limits"]["requests_per_minute"] == 100

    def test_create_web3_key_with_data_key_sync(self):
        """Test create_web3_key with data key in response (sync)."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.post.return_value = {
            "data": {
                "apiKey": "sk-web3key123",
                "description": "web3_key"
            }
        }
        
        api_keys = ApiKeys(mock_client)
        
        result = api_keys.create_web3_key(web3_key_request={"description": "web3_key"})
        
        # Verify response with data key was returned
        assert "data" in result
        assert result["data"]["apiKey"] == "sk-web3key123"

    def test_create_web3_key_without_data_key_sync(self):
        """Test create_web3_key without data key in response (sync)."""
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.post.return_value = {
            "apiKey": "sk-web3key123",
            "description": "web3_key"
        }
        
        api_keys = ApiKeys(mock_client)
        
        result = api_keys.create_web3_key(web3_key_request={"description": "web3_key"})
        
        # Verify response was returned directly
        assert cast(Dict[str, Any], result)["apiKey"] == "sk-web3key123"
        assert cast(Dict[str, Any], result)["description"] == "web3_key"

    @pytest.mark.asyncio
    async def test_create_web3_key_with_data_key_async(self):
        """Test create_web3_key with data key in response (async)."""
        mock_client = AsyncMock(spec=AsyncVeniceClient)
        mock_client.post.return_value = {
            "data": {
                "apiKey": "sk-web3key123",
                "description": "web3_key"
            }
        }
        
        api_keys = AsyncApiKeys(mock_client)
        
        result = await api_keys.create_web3_key(web3_key_request={"description": "web3_key"})
        
        # Verify response with data key was returned
        assert "data" in result
        assert result["data"]["apiKey"] == "sk-web3key123"

    @pytest.mark.asyncio
    async def test_create_web3_key_without_data_key_async(self):
        """Test create_web3_key without data key in response (async)."""
        mock_client = AsyncMock(spec=AsyncVeniceClient)
        mock_client.post.return_value = {
            "apiKey": "sk-web3key123",
            "description": "web3_key"
        }
        
        api_keys = AsyncApiKeys(mock_client)
        
        result = await api_keys.create_web3_key(web3_key_request={"description": "web3_key"})
        
        # Verify response was returned directly
        assert cast(Dict[str, Any], result)["apiKey"] == "sk-web3key123"
        assert cast(Dict[str, Any], result)["description"] == "web3_key"