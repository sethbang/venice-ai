"""
Resource module for interacting with the Venice API keys endpoints.

This module provides both synchronous and asynchronous resource classes for managing API keys.
API keys are used for authentication and authorization when making requests to the Venice API.
They control access to various Venice API features and endpoints, and are subject to rate limits
that govern the number of requests that can be made within a specific time period.

Classes:
    ApiKeys: Synchronous client for API key management
    AsyncApiKeys: Asynchronous client for API key management
"""

from typing import Optional, Dict, Any, Union, cast, TYPE_CHECKING, List
from collections.abc import Mapping # Added for isinstance(Mapping) check

from .._resource import APIResource, AsyncAPIResource
from ..exceptions import APIResponseProcessingError
from ..types.api_keys import (
    ApiKey, ApiKeyList, ApiKeyCreateRequest, ApiKeyCreateResponse,
    RateLimitInfo, RateLimitLog, RateLimitLogList, # Added RateLimitLog
    ApiKeyGenerateWeb3KeyGetResponse, ApiKeyGenerateWeb3KeyCreateRequest,
    ApiKeyGenerateWeb3KeyCreateResponse
)

if TYPE_CHECKING:
    from .._client import VeniceClient
    from .._async_client import AsyncVeniceClient


class ApiKeys(APIResource):
    """
    Provides access to API key management operations.
    
    This class implements the synchronous interface for API key management,
    including creating, listing, deleting API keys, and managing rate limits.
    It inherits from :class:`~venice_ai._resource.APIResource` which handles
    the underlying HTTP requests.
    
    :param _client: The Venice client instance used for making API requests.
    :type _client: :class:`~venice_ai._client.VeniceClient`
    
    Example:
        .. code-block:: python
        
            from venice_ai import VeniceClient
            from venice_ai.types.api_keys import ApiKeyCreateRequest
            
            client = VeniceClient()
            
            # List existing API keys
            keys = client.api_keys.list(limit=10)
            for key in keys:
                print(f"Key ID: {key.id}, Description: {key.description}")
            
            # Create a new API key
            create_request = ApiKeyCreateRequest(
                description="My Test Key",
                apiKeyType="INFERENCE"
            )
            new_key = client.api_keys.create(api_key_request=create_request)
            print(f"Created key: {new_key.apiKey}")  # Only shown on creation
    """
    
    def list(self, *, page: Optional[int] = None, limit: Optional[int] = None) -> List[ApiKey]:
        """
        Lists API keys for the authenticated account, with optional pagination.
        
        Retrieves a list of API keys associated with the current account.
        This includes active and inactive API keys. Supports pagination for
        managing large numbers of API keys.
        
        :param page: Page number to retrieve (1-based indexing). If not provided,
            returns the first page.
        :type page: Optional[int]
        :param limit: Maximum number of API keys to return per page. If not provided,
            uses the server's default limit.
        :type limit: Optional[int]

        :return: A list of API key objects containing metadata such as ID, description,
            creation date, and status. Note that the actual API key values are not
            included in the response for security reasons.
        :rtype: List[:class:`~venice_ai.types.api_keys.ApiKey`]

        :raises venice_ai.exceptions.AuthenticationError: If authentication fails.
        :raises venice_ai.exceptions.APIError: If the API returns an error.
        :raises venice_ai.exceptions.APIConnectionError: If there's an issue connecting to the API.
        
        Example:
            .. code-block:: python
            
                # List all API keys
                all_keys = client.api_keys.list()
                
                # List with pagination
                page_keys = client.api_keys.list(page=1, limit=5)
                for key in page_keys:
                    print(f"Key ID: {key.id}, Description: {key.description}")
        """
        params: Dict[str, Any] = {}
        if page is not None:
            params["page"] = page
        if limit is not None:
            params["limit"] = limit
        
        response_data = self._client.get("api_keys", params=params if params else None)

        # Case 1: API returns a list of ApiKey objects directly
        if isinstance(response_data, list):
            return cast(List[ApiKey], response_data)

        # Case 2: API returns a dictionary with a 'data' key containing the list
        elif isinstance(response_data, dict) and "data" in response_data and isinstance(response_data["data"], list):
            return cast(List[ApiKey], response_data["data"])

        # Case 3: Unexpected response format
        else:
            return []
    
    def create(
        self,
        *,
        api_key_request: ApiKeyCreateRequest
    ) -> ApiKey:
        """
        Creates a new API key.
        
        Creates a new API key with the specified parameters. The created API key
        will be returned only once in the response and cannot be retrieved later,
        so it should be securely stored immediately.
        
        :param api_key_request: Request object containing API key configuration.
            Must include at minimum a description and apiKeyType. The request can contain:
            
            - ``description`` (str): Human-readable description of the API key
            - ``apiKeyType`` (str): Type of API key (e.g., "INFERENCE", "ADMIN")
            - ``expiresAt`` (Optional[str]): ISO 8601 timestamp when key expires
            - ``consumptionLimit`` (Optional[int]): Maximum usage limit for the key
            
        :type api_key_request: :class:`~venice_ai.types.api_keys.ApiKeyCreateRequest`

        :return: Response containing the newly created API key details, including
            the secret key value (only returned once), key ID, creation timestamp,
            and other metadata.
        :rtype: :class:`~venice_ai.types.api_keys.ApiKey`

        :raises venice_ai.exceptions.AuthenticationError: If authentication fails.
        :raises venice_ai.exceptions.APIError: If the API returns an error, such as when
            maximum API key limit is reached or invalid parameters are provided.
        :raises venice_ai.exceptions.APIConnectionError: If there's an issue connecting to the API.
        
        Example:
            .. code-block:: python
            
                from venice_ai.types.api_keys import ApiKeyCreateRequest
                
                # Create a basic API key
                create_request = ApiKeyCreateRequest(
                    description="My Test Key",
                    apiKeyType="INFERENCE"
                )
                new_key = client.api_keys.create(api_key_request=create_request)
                print(f"Created key ID: {new_key.id}")
                print(f"API Key: {new_key.apiKey}")  # Store this securely!
                
                # Create a key with expiration and limits
                advanced_request = ApiKeyCreateRequest(
                    description="Limited Production Key",
                    apiKeyType="INFERENCE",
                    expiresAt="2024-12-31T23:59:59Z",
                    consumptionLimit=10000
                )
                limited_key = client.api_keys.create(api_key_request=advanced_request)
        """
        # Convert the request object to a dictionary for JSON serialization
        # Robustly convert the request object to a dictionary for JSON serialization
        if hasattr(api_key_request, '__dict__'): # for simple objects
            data = vars(api_key_request)
        # Check for Mapping (dict-like) first, as TypedDict objects are Mapping instances
        elif isinstance(api_key_request, Mapping): # for dict/TypedDict/Pydantic models
            data = dict(api_key_request.items())
        # Handle NamedTuple objects that have _asdict method
        elif hasattr(api_key_request, '_asdict'):
            data = api_key_request._asdict()
        else:
            # Fallback for unexpected types, may raise TypeError
            data = dict(api_key_request)
             # Refine conversion to filter None values
        data = {k: v for k, v in data.items() if v is not None}
        
        response = self._client.post("api_keys", json_data=data)
        if isinstance(response, dict) and "data" in response:
            api_key_data = response["data"]
            if isinstance(api_key_data, dict):
                api_key_data = dict(api_key_data)  # Make a copy to avoid modifying original
                # Handle field name mapping: consumptionLimit -> consumptionLimits
                if "consumptionLimit" in api_key_data and "consumptionLimits" not in api_key_data:
                    api_key_data["consumptionLimits"] = api_key_data.pop("consumptionLimit")
                # Handle missing consumptionLimits field by providing default
                elif "consumptionLimits" not in api_key_data:
                    api_key_data["consumptionLimits"] = {}
                # Filter to only include valid ApiKey fields
                valid_fields = {"apiKey", "apiKeyType", "consumptionLimits", "createdAt", "description", "expiresAt", "id", "last6Chars", "lastUsedAt", "usage"}
                api_key_data = {k: v for k, v in api_key_data.items() if k in valid_fields}
            return cast(ApiKey, api_key_data)
        elif isinstance(response, dict):
            # Handle response without 'data' key - return response directly without processing
            return cast(ApiKey, response)
        raise APIResponseProcessingError("Unexpected response format from API key creation endpoint. Expected a 'data' field.")
    
    def delete(
        self,
        *,
        api_key_id: str
    ) -> Dict[str, Any]:
        """
        Deletes an API key.
        
        Permanently deletes the specified API key. Once deleted, the API key
        can no longer be used to authenticate requests and this action cannot be undone.
        Use with caution in production environments.
        
        :param api_key_id: Unique identifier of the API key to delete. This is the key's
            ID (not the secret key value) as returned by the create or list operations.
        :type api_key_id: str

        :return: Response indicating the result of the deletion operation,
            typically containing a success flag and deletion confirmation message.
        :rtype: Dict[str, Any]

        :raises venice_ai.exceptions.AuthenticationError: If authentication fails.
        :raises venice_ai.exceptions.APIError: If the API returns an error, such as when
            the API key ID does not exist or belongs to another account.
        :raises venice_ai.exceptions.APIConnectionError: If there's an issue connecting to the API.
        
        Example:
            .. code-block:: python
            
                # Delete an API key (use with caution)
                result = client.api_keys.delete(api_key_id="key_123456789")
                print(f"Deletion result: {result}")
                
                # Safe deletion pattern
                keys = client.api_keys.list()
                test_keys = [k for k in keys if "test" in k.description.lower()]
                for test_key in test_keys:
                    client.api_keys.delete(api_key_id=test_key.id)
                    print(f"Deleted test key: {test_key.id}")
        """
        # Construct the URL with the API key ID as a query parameter
        path = "api_keys"
        params = {"id": api_key_id}
        return cast(Dict[str, Any], self._client.delete(path, params=params))

    def retrieve(
        self,
        *,
        api_key_id: str
    ) -> Dict[str, Any]:
        """
        Retrieves a specific API key by ID.
        
        Fetches the details of a specific API key using its unique identifier.
        Note that the actual API key value is not included in the response for security reasons.
        
        :param api_key_id: Unique identifier of the API key to retrieve. This is the key's
            ID (not the secret key value) as returned by the create or list operations.
        :type api_key_id: str

        :return: API key details including metadata such as description, creation date,
            expiration, usage statistics, and other configuration information.
        :rtype: Dict[str, Any]

        :raises venice_ai.exceptions.AuthenticationError: If authentication fails.
        :raises venice_ai.exceptions.NotFoundError: If the API key ID does not exist.
        :raises venice_ai.exceptions.APIError: If the API returns an error.
        :raises venice_ai.exceptions.APIConnectionError: If there's an issue connecting to the API.
        
        Example:
            .. code-block:: python
            
                # Retrieve a specific API key
                key_details = client.api_keys.retrieve(api_key_id="key_123456789")
                print(f"Key description: {key_details['description']}")
                print(f"Created at: {key_details['createdAt']}")
        """
        path = "api_keys"
        params = {"id": api_key_id}
        response = self._client.get(path, params=params)
        if isinstance(response, dict) and "data" in response and isinstance(response["data"], list) and len(response["data"]) > 0:
            return cast(Dict[str, Any], response["data"][0])
        # If the structure is not as expected, or data is empty,
        # this will either raise an error in _client.get or return an unexpected structure.
        # For now, we assume _client.get handles 404s by raising NotFoundError.
        # If data is empty, it implies not found or an issue.
        # Consider raising NotFoundError explicitly if response["data"] is empty.
        return cast(Dict[str, Any], response) # Fallback, though ideally an error or specific handling.
        return cast(Dict[str, Any], self._client.get(path, params=params))
        return cast(Dict[str, Any], self._client.get(path, params=params))

    def get_web3_token(self) -> ApiKeyGenerateWeb3KeyGetResponse:
        """
        Retrieves a token for Web3 API key generation.

        This token is required for the subsequent POST request to create a Web3 API key.

        :return: Response containing the token required for Web3 key generation.
        :rtype: :class:`~venice_ai.types.api_keys.ApiKeyGenerateWeb3KeyGetResponse`

        :raises venice_ai.exceptions.APIError: If the API returns an error.
        :raises venice_ai.exceptions.APIConnectionError: If there's an issue connecting to the API.
        """
        return cast(ApiKeyGenerateWeb3KeyGetResponse, self._client.get("api_keys/generate_web3_key"))

    def create_web3_key(
        self,
        *,
        web3_key_request: ApiKeyGenerateWeb3KeyCreateRequest
    ) -> ApiKeyGenerateWeb3KeyCreateResponse:
        """
        Creates a new Web3 API key.

        Creates a new API key authenticated via a Web3 signature.

        :param web3_key_request: Request body containing Web3 authentication details
            (such as ``web3_network_id``, ``web3_address``, and signature) and API key parameters.
        :type web3_key_request: :class:`~venice_ai.types.api_keys.ApiKeyGenerateWeb3KeyCreateRequest`

        :return: Response containing the newly created API key details.
        :rtype: :class:`~venice_ai.types.api_keys.ApiKeyGenerateWeb3KeyCreateResponse`

        :raises venice_ai.exceptions.APIError: If the API returns an error.
        :raises venice_ai.exceptions.APIConnectionError: If there's an issue connecting to the API.
        """
        # Convert the request object to a dictionary for JSON serialization
        # Robustly convert the request object to a dictionary for JSON serialization
        if hasattr(web3_key_request, '__dict__'): # for simple objects
            data = vars(web3_key_request)
        # Check for Mapping (dict-like) first, as TypedDict objects are Mapping instances
        elif isinstance(web3_key_request, Mapping): # for dict/TypedDict/Pydantic models
            data = dict(web3_key_request.items())
        # Handle NamedTuple objects that have _asdict method
        elif hasattr(web3_key_request, '_asdict'):
            data = web3_key_request._asdict()
        else:
            # Fallback for unexpected types, may raise TypeError
            data = dict(web3_key_request)
            # Refine conversion to filter None values
        data = {k: v for k, v in data.items() if v is not None}
        
        response = self._client.post("api_keys/generate_web3_key", json_data=data)
        if isinstance(response, dict) and "data" in response:
            return cast(ApiKeyGenerateWeb3KeyCreateResponse, response)
        return cast(ApiKeyGenerateWeb3KeyCreateResponse, response) # Fallback if no 'data' key

    def get_rate_limits(self) -> RateLimitInfo:
        """
        Retrieves rate limit information for the current API key.
        
        Returns information about the rate limits applied to the current API key,
        including the limits per minute, hour, day, and month, as well as the
        current usage against those limits.
        
        :return: Rate limit information, including limits and current usage.
        :rtype: :class:`~venice_ai.types.api_keys.RateLimitInfo`

        :raises venice_ai.exceptions.AuthenticationError: If authentication fails.
        :raises venice_ai.exceptions.APIError: If the API returns an error.
        :raises venice_ai.exceptions.APIConnectionError: If there's an issue connecting to the API.
        """
        response = self._client.get("api_keys/rate_limits")
        if isinstance(response, dict) and "data" in response:
            return cast(RateLimitInfo, response["data"])
        return cast(RateLimitInfo, response) # Fallback
    
    def get_rate_limit_logs(
        self,
        *,
        api_key_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None
    ) -> RateLimitLogList:
        """
        Retrieves rate limit logs for API keys.
        
        Returns a history of rate limit events, such as when rate limits were
        reset, exceeded, or modified. This can be useful for understanding API usage
        patterns, diagnosing rate limit issues, and optimizing request timing.
        
        :param api_key_id: Specific API key ID to get logs for. If not provided,
            returns logs for the current API key.
        :type api_key_id: Optional[str]
        :param start_date: Start date for log retrieval in ISO 8601 format
            (e.g., "2024-01-01T00:00:00Z"). If not provided, uses a default lookback period.
        :type start_date: Optional[str]
        :param end_date: End date for log retrieval in ISO 8601 format
            (e.g., "2024-01-31T23:59:59Z"). If not provided, uses current time.
        :type end_date: Optional[str]
        :param limit: Maximum number of log entries to return per page.
        :type limit: Optional[int]
        :param page: Page number for pagination (1-based indexing).
        :type page: Optional[int]
        
        :return: A list of rate limit log entries with timestamps, event types,
            and related metadata.
        :rtype: :class:`~venice_ai.types.api_keys.RateLimitLogList`

        :raises venice_ai.exceptions.AuthenticationError: If authentication fails.
        :raises venice_ai.exceptions.APIError: If the API returns an error.
        :raises venice_ai.exceptions.APIConnectionError: If there's an issue connecting to the API.
        
        Example:
            .. code-block:: python
            
                # Get recent rate limit logs
                logs = client.api_keys.get_rate_limit_logs(limit=10)
                for log_entry in logs:
                    print(f"Event: {log_entry.event_type} at {log_entry.timestamp}")
                
                # Get logs for a specific date range
                logs = client.api_keys.get_rate_limit_logs(
                    start_date="2024-01-01T00:00:00Z",
                    end_date="2024-01-31T23:59:59Z",
                    limit=50
                )
        """
        params: Dict[str, Any] = {}
        if api_key_id is not None:
            params["api_key_id"] = api_key_id
        if start_date is not None:
            params["start_date"] = start_date
        if end_date is not None:
            params["end_date"] = end_date
        if limit is not None:
            params["limit"] = limit
        if page is not None:
            params["page"] = page
            
        response = self._client.get("api_keys/rate_limits/log", params=params if params else None)
        if isinstance(response, dict) and "data" in response:
            if isinstance(response["data"], list):
                return cast(RateLimitLogList, response)
            else:
                # If 'data' exists but is not a list, as per test expectation, return empty list
                return cast(RateLimitLogList, [])
        elif isinstance(response, list): # If API directly returns a list
            return cast(RateLimitLogList, response)
        # Fallback for any other unexpected response format
        return cast(RateLimitLogList, [])


class AsyncApiKeys(AsyncAPIResource):
    """
    Provides access to API key management operations asynchronously.
    
    This class implements the asynchronous interface for API key management,
    including creating, listing, deleting API keys, and managing rate limits.
    It inherits from :class:`~venice_ai._resource.AsyncAPIResource` which handles
    the underlying HTTP requests. All methods return awaitable coroutines that
    should be awaited by the caller.
    
    :param _client: The AsyncVeniceClient instance used for making asynchronous API requests.
    :type _client: :class:`~venice_ai._async_client.AsyncVeniceClient`
    
    Example:
        .. code-block:: python
        
            import asyncio
            from venice_ai import AsyncVeniceClient
            from venice_ai.types.api_keys import ApiKeyCreateRequest
            
            async def manage_api_keys():
                client = AsyncVeniceClient()
                
                # List existing API keys
                keys = await client.api_keys.list(limit=10)
                for key in keys:
                    print(f"Key ID: {key.id}, Description: {key.description}")
                
                # Create a new API key
                create_request = ApiKeyCreateRequest(
                    description="My Async Test Key",
                    apiKeyType="INFERENCE"
                )
                new_key = await client.api_keys.create(api_key_request=create_request)
                print(f"Created key: {new_key.apiKey}")  # Only shown on creation
            
            asyncio.run(manage_api_keys())
    """
    
    async def list(self, *, page: Optional[int] = None, limit: Optional[int] = None) -> List[ApiKey]:
        """
        Lists API keys for the authenticated account asynchronously, with optional pagination.
        
        Retrieves a list of API keys associated with the current account.
        This includes active and inactive API keys. Supports pagination.

        :param page: Page number to retrieve (1-based indexing). If not provided,
            returns the first page.
        :type page: Optional[int]
        :param limit: Maximum number of API keys to return per page. If not provided,
            uses the server's default limit.
        :type limit: Optional[int]

        :return: A list of API key objects containing metadata such as ID, description,
            creation date, and status. Note that the actual API key values are not
            included in the response for security reasons.
        :rtype: List[:class:`~venice_ai.types.api_keys.ApiKey`]

        :raises venice_ai.exceptions.AuthenticationError: If authentication fails.
        :raises venice_ai.exceptions.APIError: If the API returns an error.
        :raises venice_ai.exceptions.APIConnectionError: If there's an issue connecting to the API.
        
        Example:
            .. code-block:: python
            
                # List all API keys asynchronously
                all_keys = await client.api_keys.list()
                
                # List with pagination
                page_keys = await client.api_keys.list(page=1, limit=5)
                for key in page_keys:
                    print(f"Key ID: {key.id}, Description: {key.description}")
        """
        params: Dict[str, Any] = {}
        if page is not None:
            params["page"] = page
        if limit is not None:
            params["limit"] = limit
            
        response_data = await self._client.get("api_keys", params=params if params else None)

        # Case 1: API returns a list of ApiKey objects directly
        if isinstance(response_data, list):
            return cast(List[ApiKey], response_data)

        # Case 2: API returns a dictionary with a 'data' key containing the list
        elif isinstance(response_data, dict) and "data" in response_data and isinstance(response_data["data"], list):
            return cast(List[ApiKey], response_data["data"])

        # Case 3: Unexpected response format
        else:
            return []
    
    async def create(
        self,
        *,
        api_key_request: ApiKeyCreateRequest
    ) -> ApiKey:
        """
        Creates a new API key asynchronously.
        
        Creates a new API key with the specified parameters. The created API key
        will be returned only once in the response and cannot be retrieved later,
        so it should be securely stored immediately.
        
        :param api_key_request: Request object containing API key configuration.
            Must include at minimum a description and apiKeyType. The request can contain:
            
            - ``description`` (str): Human-readable description of the API key
            - ``apiKeyType`` (str): Type of API key (e.g., "INFERENCE", "ADMIN")
            - ``expiresAt`` (Optional[str]): ISO 8601 timestamp when key expires
            - ``consumptionLimit`` (Optional[ConsumptionLimit]): Usage limits for the key
            
        :type api_key_request: :class:`~venice_ai.types.api_keys.ApiKeyCreateRequest`

        :return: Response containing the newly created API key details, including
            the secret key value (only returned once), key ID, creation timestamp,
            and other metadata.
        :rtype: :class:`~venice_ai.types.api_keys.ApiKey`

        :raises venice_ai.exceptions.AuthenticationError: If authentication fails.
        :raises venice_ai.exceptions.APIError: If the API returns an error, such as when
            maximum API key limit is reached or invalid parameters are provided.
        :raises venice_ai.exceptions.APIConnectionError: If there's an issue connecting to the API.
        
        Example:
            .. code-block:: python
            
                from venice_ai.types.api_keys import ApiKeyCreateRequest
                
                # Create a basic API key asynchronously
                create_request = ApiKeyCreateRequest(
                    description="My Async Test Key",
                    apiKeyType="INFERENCE"
                )
                new_key = await client.api_keys.create(api_key_request=create_request)
                print(f"Created key ID: {new_key.id}")
                print(f"API Key: {new_key.apiKey}")  # Store this securely!
        """
        # Convert the request object to a dictionary for JSON serialization
        # Robustly convert the request object to a dictionary for JSON serialization
        if hasattr(api_key_request, '__dict__'): # for simple objects
            data = vars(api_key_request)
        # Check for Mapping (dict-like) first, as TypedDict objects are Mapping instances
        elif isinstance(api_key_request, Mapping): # for dict/TypedDict/Pydantic models
            data = dict(api_key_request.items())
        # Handle NamedTuple objects that have _asdict method
        elif hasattr(api_key_request, '_asdict'):
            data = api_key_request._asdict()
        else:
            # Fallback for unexpected types, may raise TypeError
            data = dict(api_key_request)
            # Refine conversion to filter None values
        data = {k: v for k, v in data.items() if v is not None}
        
        response = await self._client.post("api_keys", json_data=data)
        if isinstance(response, dict) and "data" in response:
            api_key_data = response["data"]
            if isinstance(api_key_data, dict):
                api_key_data = dict(api_key_data)  # Make a copy to avoid modifying original
                # Handle field name mapping: consumptionLimit -> consumptionLimits
                if "consumptionLimit" in api_key_data and "consumptionLimits" not in api_key_data:
                    api_key_data["consumptionLimits"] = api_key_data.pop("consumptionLimit")
                # Handle missing consumptionLimits field by providing default
                elif "consumptionLimits" not in api_key_data:
                    api_key_data["consumptionLimits"] = {}
                # Filter to only include valid ApiKey fields
                valid_fields = {"apiKey", "apiKeyType", "consumptionLimits", "createdAt", "description", "expiresAt", "id", "last6Chars", "lastUsedAt", "usage"}
                api_key_data = {k: v for k, v in api_key_data.items() if k in valid_fields}
            return cast(ApiKey, api_key_data)
        elif isinstance(response, dict):
            # Handle response without 'data' key - return response directly without processing
            return cast(ApiKey, response)
        raise APIResponseProcessingError("Unexpected response format from API key creation endpoint. Expected a 'data' field.")
    
    async def delete(
        self,
        *,
        api_key_id: str
    ) -> Dict[str, Any]:
        """
        Deletes an API key asynchronously.
        
        Permanently deletes the specified API key. Once deleted, the API key
        can no longer be used to authenticate requests and this action cannot be undone.
        
        :param api_key_id: ID of the API key to delete. This is the key's
            unique identifier, not the secret key value.
        :type api_key_id: str

        :return: Response indicating the result of the operation,
            typically containing a success flag and deletion confirmation.
        :rtype: Dict[str, Any]

        :raises venice_ai.exceptions.AuthenticationError: If authentication fails.
        :raises venice_ai.exceptions.APIError: If the API returns an error, such as when
            the API key ID does not exist or belongs to another account.
        :raises venice_ai.exceptions.APIConnectionError: If there's an issue connecting to the API.
        
        Example:
            .. code-block:: python
            
                # Delete an API key asynchronously (use with caution)
                result = await client.api_keys.delete(api_key_id="key_123456789")
                print(f"Deletion result: {result}")
                
                # Safe deletion pattern
                keys = await client.api_keys.list()
                test_keys = [k for k in keys if "test" in k.description.lower()]
                for test_key in test_keys:
                    await client.api_keys.delete(api_key_id=test_key.id)
                    print(f"Deleted test key: {test_key.id}")
        """
        # Construct the URL with the API key ID as a query parameter
        path = "api_keys"
        params = {"id": api_key_id}
        return cast(Dict[str, Any], await self._client.delete(path, params=params))

    async def retrieve(
        self,
        *,
        api_key_id: str
    ) -> Dict[str, Any]:
        """
        Retrieves a specific API key by ID asynchronously.
        
        Fetches the details of a specific API key using its unique identifier.
        Note that the actual API key value is not included in the response for security reasons.
        
        :param api_key_id: Unique identifier of the API key to retrieve. This is the key's
            ID (not the secret key value) as returned by the create or list operations.
        :type api_key_id: str

        :return: API key details including metadata such as description, creation date,
            expiration, usage statistics, and other configuration information.
        :rtype: Dict[str, Any]

        :raises venice_ai.exceptions.AuthenticationError: If authentication fails.
        :raises venice_ai.exceptions.NotFoundError: If the API key ID does not exist.
        :raises venice_ai.exceptions.APIError: If the API returns an error.
        :raises venice_ai.exceptions.APIConnectionError: If there's an issue connecting to the API.
        
        Example:
            .. code-block:: python
            
                # Retrieve a specific API key asynchronously
                key_details = await client.api_keys.retrieve(api_key_id="key_123456789")
                print(f"Key description: {key_details['description']}")
                print(f"Created at: {key_details['createdAt']}")
        """
        path = "api_keys"
        params = {"id": api_key_id}
        response = await self._client.get(path, params=params)
        if isinstance(response, dict) and "data" in response and isinstance(response["data"], list) and len(response["data"]) > 0:
            return cast(Dict[str, Any], response["data"][0])
        # Similar to sync: handle cases where data might be empty or response malformed.
        return cast(Dict[str, Any], response) # Fallback
        path = "api_keys"
        params = {"id": api_key_id}
        return cast(Dict[str, Any], await self._client.get(path, params=params))

    async def get_web3_token(self) -> ApiKeyGenerateWeb3KeyGetResponse:
        """
        Retrieves a token for Web3 API key generation asynchronously.

        This token is required for the subsequent POST request to create a Web3 API key.

        :return: Response containing the token required for Web3 key generation.
        :rtype: :class:`~venice_ai.types.api_keys.ApiKeyGenerateWeb3KeyGetResponse`

        :raises venice_ai.exceptions.APIError: If the API returns an error.
        :raises venice_ai.exceptions.APIConnectionError: If there's an issue connecting to the API.
        """
        return cast(ApiKeyGenerateWeb3KeyGetResponse, await self._client.get("api_keys/generate_web3_key"))

    async def create_web3_key(
        self,
        *,
        web3_key_request: ApiKeyGenerateWeb3KeyCreateRequest
    ) -> ApiKeyGenerateWeb3KeyCreateResponse:
        """
        Creates a new Web3 API key asynchronously.

        Creates a new API key authenticated via a Web3 signature.

        :param web3_key_request: Request body containing Web3 authentication details
            (such as ``web3_network_id``, ``web3_address``, and signature) and API key parameters.
        :type web3_key_request: :class:`~venice_ai.types.api_keys.ApiKeyGenerateWeb3KeyCreateRequest`

        :return: Response containing the newly created API key details.
        :rtype: :class:`~venice_ai.types.api_keys.ApiKeyGenerateWeb3KeyCreateResponse`

        :raises venice_ai.exceptions.APIError: If the API returns an error.
        :raises venice_ai.exceptions.APIConnectionError: If there's an issue connecting to the API.
        """
        # Convert the request object to a dictionary for JSON serialization
        # Robustly convert the request object to a dictionary for JSON serialization
        if hasattr(web3_key_request, '__dict__'): # for simple objects
            data = vars(web3_key_request)
        # Check for Mapping (dict-like) first, as TypedDict objects are Mapping instances
        elif isinstance(web3_key_request, Mapping): # for dict/TypedDict/Pydantic models
            data = dict(web3_key_request.items())
        # Handle NamedTuple objects that have _asdict method
        elif hasattr(web3_key_request, '_asdict'):
            data = web3_key_request._asdict()
        else:
            # Fallback for unexpected types, may raise TypeError
            data = dict(web3_key_request)
            # Refine conversion to filter None values
        data = {k: v for k, v in data.items() if v is not None}
        
        response = await self._client.post("api_keys/generate_web3_key", json_data=data)
        if isinstance(response, dict) and "data" in response:
            return cast(ApiKeyGenerateWeb3KeyCreateResponse, response)
        return cast(ApiKeyGenerateWeb3KeyCreateResponse, response) # Fallback if no 'data' key

    async def get_rate_limits(self) -> RateLimitInfo:
        """
        Retrieves rate limit information for the current API key asynchronously.
        
        Returns information about the rate limits applied to the current API key,
        including the limits per minute, hour, day, and month, as well as the
        current usage against those limits.
        
        :return: Rate limit information, including limits and current usage.
        :rtype: :class:`~venice_ai.types.api_keys.RateLimitInfo`

        :raises venice_ai.exceptions.AuthenticationError: If authentication fails.
        :raises venice_ai.exceptions.APIError: If the API returns an error.
        :raises venice_ai.exceptions.APIConnectionError: If there's an issue connecting to the API.
        """
        response = await self._client.get("api_keys/rate_limits")
        if isinstance(response, dict) and "data" in response:
            return cast(RateLimitInfo, response["data"])
        return cast(RateLimitInfo, response) # Fallback
    
    async def get_rate_limit_logs(
        self,
        *,
        api_key_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None
    ) -> RateLimitLogList:
        """
        Retrieves rate limit logs for API keys asynchronously.
        
        Returns a history of rate limit events, such as when rate limits were
        reset, exceeded, or modified. This can be useful for understanding API usage
        patterns, diagnosing rate limit issues, and optimizing request timing.
        
        :param api_key_id: Specific API key ID to get logs for. If not provided,
            returns logs for the current API key.
        :type api_key_id: Optional[str]
        :param start_date: Start date for log retrieval in ISO 8601 format
            (e.g., "2024-01-01T00:00:00Z"). If not provided, uses a default lookback period.
        :type start_date: Optional[str]
        :param end_date: End date for log retrieval in ISO 8601 format
            (e.g., "2024-01-31T23:59:59Z"). If not provided, uses current time.
        :type end_date: Optional[str]
        :param limit: Maximum number of log entries to return per page.
        :type limit: Optional[int]
        :param page: Page number for pagination (1-based indexing).
        :type page: Optional[int]
        
        :return: A list of rate limit log entries with timestamps, event types,
            and related metadata.
        :rtype: :class:`~venice_ai.types.api_keys.RateLimitLogList`

        :raises venice_ai.exceptions.AuthenticationError: If authentication fails.
        :raises venice_ai.exceptions.APIError: If the API returns an error.
        :raises venice_ai.exceptions.APIConnectionError: If there's an issue connecting to the API.
        
        Example:
            .. code-block:: python
            
                # Get recent rate limit logs asynchronously
                logs = await client.api_keys.get_rate_limit_logs(limit=10)
                for log_entry in logs:
                    print(f"Event: {log_entry.event_type} at {log_entry.timestamp}")
                
                # Get logs for a specific date range
                logs = await client.api_keys.get_rate_limit_logs(
                    start_date="2024-01-01T00:00:00Z",
                    end_date="2024-01-31T23:59:59Z",
                    limit=50
                )
        """
        params: Dict[str, Any] = {}
        if api_key_id is not None:
            params["api_key_id"] = api_key_id
        if start_date is not None:
            params["start_date"] = start_date
        if end_date is not None:
            params["end_date"] = end_date
        if limit is not None:
            params["limit"] = limit
        if page is not None:
            params["page"] = page
            
        response = await self._client.get("api_keys/rate_limits/log", params=params if params else None)
        if isinstance(response, dict) and "data" in response:
            if isinstance(response["data"], list):
                return cast(RateLimitLogList, response)
            else:
                # If 'data' exists but is not a list, as per test expectation, return empty list
                return cast(RateLimitLogList, [])
        elif isinstance(response, list): # If API directly returns a list
            return cast(RateLimitLogList, response)
        # Fallback for any other unexpected response format
        return cast(RateLimitLogList, [])
