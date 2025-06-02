"""
Type definitions for Venice AI API Keys functionality.
"""

from typing import Dict, List, Literal, Optional, TypedDict, Union


class ConsumptionLimit(TypedDict, total=False):
    """Defines consumption limits for API keys within each billing epoch.
    
    This type represents the spending and usage constraints that can be applied
    to API keys to control resource consumption. Limits can be specified in both
    USD currency and Venice Compute Units (VCU).
    """

    usd: Optional[float]
    """Optional spending limit in US Dollars for the billing period."""
    vcu: Optional[float]
    """Optional usage limit in Venice Compute Units for the billing period."""


class ApiKeyUsage(TypedDict):
    """Represents usage statistics and metrics for an API key.
    
    This type encapsulates usage information for an API key, providing insights
    into consumption patterns over specific time periods. Used to track and
    monitor API key activity for billing and rate limiting purposes.
    """

    trailingSevenDays: Dict[str, str]
    """Usage statistics for the trailing 7-day period, containing 'usd' and 'vcu' consumption values."""


class ApiKey(TypedDict):
    """Represents a complete API key object in the Venice AI system.

    This type defines the structure of an API key as returned by the Venice AI
    API key management endpoints. Contains all metadata, configuration, and
    usage information associated with an API key, including its type, limits,
    creation details, and current usage statistics.

    Retrieved from /api_keys endpoint.
    """

    apiKeyType: Literal["INFERENCE", "ADMIN"]
    """Type of the API key, determining its access permissions and capabilities."""
    consumptionLimits: ConsumptionLimit
    """Consumption limits and spending constraints associated with this API key."""
    createdAt: Optional[str]
    """ISO 8601 timestamp indicating when the API key was created."""
    description: str
    """Human-readable description or name assigned to the API key."""
    expiresAt: Optional[str]
    """ISO 8601 timestamp when the API key expires, or None if it never expires."""
    id: str
    """Unique identifier for the API key used in management operations."""
    last6Chars: str
    """Last 6 characters of the actual API key value for identification purposes."""
    lastUsedAt: Optional[str]
    """ISO 8601 timestamp of the most recent API request made with this key."""
    usage: ApiKeyUsage
    """Current usage statistics and consumption metrics for this API key."""


class ApiKeyCreateRequest(TypedDict, total=False):
    """Request payload for creating a new API key.

    This type defines the structure of the request body used when creating
    a new API key through the Venice AI API. Includes all configurable
    parameters such as key type, consumption limits, expiration settings,
    and optional Web3 integration parameters.

    Used with POST /api_keys endpoint.
    """

    apiKeyType: Literal["INFERENCE", "ADMIN"]
    """Type of API key to create, determining access permissions and capabilities."""
    consumptionLimit: ConsumptionLimit
    """Spending and usage limits to apply to the new API key."""
    description: str
    """Human-readable description or name for the new API key."""
    expiresAt: Optional[str]
    """Optional expiration date in ISO 8601 format, or empty string for no expiration."""
    web3_network_id: Optional[str]
    """Optional Web3 network identifier for blockchain-authenticated API keys."""
    web3_address: Optional[str]
    """Optional Web3 wallet address for blockchain-authenticated API keys."""


class ApiKeyCreateResponse(TypedDict):
    """Response payload returned after successful API key creation.

    This type represents the response structure returned by the Venice AI API
    when a new API key is successfully created. Contains the complete details
    of the newly created key, including the secret key value which is only
    shown once during creation for security purposes.

    Contains the newly created API key details.
    """

    data: Dict[str, Union[str, ConsumptionLimit, None]]
    """Dictionary containing the created API key details, including the secret key value (shown only once)."""
    success: bool
    """Boolean flag indicating whether the API key creation operation was successful."""


class ApiKeyList(TypedDict):
    """Response payload containing a collection of API key objects.

    This type represents the response structure returned by the Venice AI API
    when retrieving a list of API keys. Provides a standardized container
    for multiple API key objects with metadata indicating the response type.

    Retrieved from GET /api_keys endpoint.
    """

    data: List[ApiKey]
    """Array of API key objects containing metadata and configuration details."""
    object: Literal["list"]
    """Response type identifier, always "list" for collection responses."""


class ApiKeyRateLimitItem(TypedDict):
    """Represents rate limit configuration for a specific API model.
    
    This type defines the rate limiting rules applied to a particular model
    when accessed through an API key. Contains the model identifier and
    associated rate limit policies that govern request frequency and volume.
    """

    apiModelId: str
    """Unique identifier of the API model to which these rate limits apply."""
    rateLimits: List[Dict[str, Union[float, str]]]
    """Array of rate limiting rules and policies governing usage of this model."""


class ApiTier(TypedDict):
    """Represents API tier information and billing configuration.
    
    This type defines the characteristics of an API tier, including its
    identifier and billing status. API tiers determine access levels,
    rate limits, and whether usage is subject to charges.
    """

    id: str
    """Unique identifier of the API tier level."""
    isCharged: bool
    """Boolean flag indicating whether usage under this tier incurs billing charges."""


class Balances(TypedDict):
    """Represents current account balances in supported currencies.
    
    This type contains the available balances for an account across
    different currency types supported by the Venice AI platform,
    including traditional USD and Venice Compute Units (VCU).
    """

    USD: float
    """Current account balance in US Dollars."""
    VCU: float
    """Current account balance in Venice Compute Units."""


class RateLimitInfo(TypedDict):
    """Comprehensive rate limit and access information for an API key.

    This type represents the complete rate limiting context for an API key,
    including current access permissions, tier information, account balances,
    key expiration details, and specific rate limit configurations. Used
    to determine whether requests can be processed and what limits apply.

    Retrieved from /api_keys/rate_limits endpoint.
    """

    accessPermitted: bool
    """Boolean flag indicating whether API access is currently permitted based on rate limits and account status."""
    apiTier: ApiTier
    """API tier configuration and billing information associated with this key."""
    balances: Balances
    """Current account balances across supported currency types."""
    keyExpiration: Optional[str]
    """ISO 8601 timestamp when the API key expires, or None if it never expires."""
    nextEpochBegins: str
    """ISO 8601 timestamp indicating when the next rate limiting epoch period begins."""
    rateLimits: List[ApiKeyRateLimitItem]
    """Array of model-specific rate limiting rules and configurations applied to this key."""


class RateLimitLog(TypedDict):
    """Represents a single rate limit event log entry.

    This type defines the structure of individual rate limit log entries
    that track rate limiting events for API keys. Contains details about
    the key, model, tier, event type, and timing information for auditing
    and monitoring rate limit enforcement.

    Retrieved from /api_keys/rate_limits/log endpoint.
    """

    apiKeyId: str
    """Unique identifier of the API key that triggered this rate limit event."""
    modelId: str
    """Identifier of the API model involved in the rate limiting event."""
    rateLimitTier: str
    """Rate limit tier that was active when this event occurred."""
    rateLimitType: str
    """Type of rate limit event (e.g., "exceeded", "reset", "warning")."""
    timestamp: str
    """ISO 8601 timestamp when this rate limit event was recorded."""


class RateLimitLogList(TypedDict):
    """Response payload containing a collection of rate limit log entries.

    This type represents the response structure returned by the Venice AI API
    when retrieving rate limit logs. Provides a standardized container for
    multiple rate limit log entries with metadata indicating the response type.

    Retrieved from GET /api_keys/rate_limits/log endpoint.
    """

    data: List[RateLimitLog]
    """Array of rate limit log entries containing event details and timestamps."""
    object: Literal["list"]
    """Response type identifier, always "list" for collection responses."""


class ApiKeyGenerateWeb3KeyGetResponse(TypedDict):
    """Response payload for Web3 key generation token retrieval.

    This type represents the response structure returned when requesting
    a token for Web3 API key generation. The token is required as part
    of the Web3 key creation process to ensure secure authentication
    through wallet signature verification.

    Response from the GET /api_keys/generate_web3_key endpoint.

    Contains token needed for Web3 key generation.
    """
    data: Dict[str, str]
    """Dictionary containing the authentication token required for subsequent Web3 key creation."""
    success: bool
    """Boolean flag indicating whether the token retrieval operation was successful."""


class ApiKeyGenerateWeb3KeyCreateRequest(TypedDict, total=False):
    """Request payload for creating a new Web3-authenticated API key.

    This type defines the structure of the request body used when creating
    a Web3 API key through wallet signature verification. Includes all
    standard API key parameters plus Web3-specific fields for address
    verification, signature proof, and the authentication token.

    Used with POST /api_keys/generate_web3_key endpoint.
    """
    apiKeyType: Literal["INFERENCE", "ADMIN"]
    """Type of API key to create, determining access permissions and capabilities."""
    address: str
    """Web3 wallet address used for blockchain-based authentication."""
    signature: str
    """Cryptographic signature proving ownership of the specified wallet address."""
    token: str
    """Authentication token obtained from the preliminary GET request."""
    description: str
    """Human-readable description or name for the new Web3 API key."""
    expiresAt: Optional[str]
    """Optional expiration date in ISO 8601 format, or None for no expiration."""
    consumptionLimit: ConsumptionLimit
    """Spending and usage limits to apply to the new Web3 API key."""


class ApiKeyGenerateWeb3KeyCreateResponse(TypedDict):
    """Response payload returned after successful Web3 API key creation.

    This type represents the response structure returned by the Venice AI API
    when a new Web3 API key is successfully created through wallet signature
    verification. Contains the complete details of the newly created key,
    including the secret key value which is only shown once during creation.

    Response format for Web3 API key creation.

    Contains the newly created API key details.
    """
    data: Dict[str, Union[str, ConsumptionLimit, None]]
    """Dictionary containing the created Web3 API key details, including the secret key value (shown only once)."""
    success: bool
    """Boolean flag indicating whether the Web3 API key creation operation was successful."""