"""
Type definitions for Venice AI Billing API.

This module contains TypedDict definitions for request parameters and response objects
in the Venice AI Billing API, covering the billing usage endpoint.
"""

from enum import Enum
from typing import Dict, List, Literal, Optional, TypedDict, Union


class InferenceDetails(TypedDict, total=False):
    """
    Represents detailed information about an inference request for billing purposes.
    
    This model contains metadata about LLM inference requests, including token counts
    and execution metrics. Used within billing usage entries to provide granular
    details about API usage costs and performance.
    
    .. note::
        These details are only present for LLM usage entries and may be absent
        for other types of API usage.
    """
    
    completionTokens: Optional[float]
    """Number of tokens generated in the completion response (present only for LLM inference requests)."""
    
    promptTokens: Optional[float]
    """Number of tokens in the input prompt (present only for LLM inference requests)."""
    
    inferenceExecutionTime: Optional[float]
    """Total execution time for the inference request in milliseconds."""
    
    requestId: Optional[str]
    """Unique identifier for the specific inference request."""


class BillingUsageEntry(TypedDict):
    """
    Represents a single billing usage record from the Venice AI API.
    
    This model defines the structure of individual usage entries returned by the
    billing usage endpoint. Each entry represents a billable event with associated
    costs, units consumed, and metadata about the API usage.
    
    Used as the primary data structure for tracking and reporting API usage costs
    across different services and time periods.
    """
    
    amount: float
    """Total amount charged for this usage entry."""
    
    currency: Literal["USD", "VCU"]
    """Currency denomination for the charge (either USD or Venice Compute Units)."""
    
    inferenceDetails: Optional[InferenceDetails]
    """Detailed inference metadata, present only for LLM-related usage entries."""
    
    notes: str
    """Additional notes or description associated with this billing entry."""
    
    pricePerUnitUsd: float
    """Price per unit in USD for this specific usage type."""
    
    sku: str
    """Stock Keeping Unit (SKU) identifier for the product or service used."""
    
    timestamp: str
    """ISO 8601 formatted timestamp indicating when this usage occurred."""
    
    units: float
    """Quantity of units consumed for this billing entry."""


class BillingUsagePagination(TypedDict):
    """
    Represents pagination metadata for billing usage API responses.
    
    This model contains information about the pagination state of billing usage
    queries, including current page position, total available records, and
    pagination limits. Used in conjunction with billing usage responses to
    enable efficient navigation through large datasets.
    
    Essential for handling paginated billing data retrieval from the Venice AI API.
    """
    
    limit: float
    """Maximum number of items returned per page in the current request."""
    
    page: float
    """Current page number in the paginated result set (1-based)."""
    
    total: float
    """Total number of billing usage entries available across all pages."""
    
    totalPages: float
    """Total number of pages available for the current query parameters."""


class BillingUsageResponse(TypedDict):
    """
    Represents the complete response structure from the billing usage endpoint.
    
    This model serves as the top-level container for billing usage data returned
    by the Venice AI API. It combines the actual usage records with pagination
    metadata, providing a comprehensive view of billing information for a given
    query.
    
    Used as the primary response type for all billing usage API calls.
    """
    
    data: List[BillingUsageEntry]
    """Array of billing usage records for the requested time period and filters."""
    
    pagination: BillingUsagePagination
    """Pagination metadata including current page, total items, and page limits."""


class BillingUsageRequestParams(TypedDict, total=False):
    """
    Represents query parameters for filtering and paginating billing usage data.
    
    This model defines the optional parameters that can be used to customize
    billing usage queries to the Venice AI API. Supports filtering by date range,
    currency type, and pagination controls to retrieve specific subsets of
    billing data.
    
    All parameters are optional and will use API defaults when not specified.
    Used to construct targeted billing usage requests based on specific criteria.
    """
    
    currency: Optional[Literal["USD", "VCU"]]
    """Filter results by currency type (USD for US Dollars, VCU for Venice Compute Units)."""
    
    startDate: Optional[str]
    """Start date for the billing period filter in ISO 8601 format (e.g., "2025-01-01T00:00:00Z")."""
    
    endDate: Optional[str]
    """End date for the billing period filter in ISO 8601 format (e.g., "2025-05-01T00:00:00Z")."""
    
    limit: Optional[int]
    """Maximum number of items to return per page (valid range: 1-500, default: 200)."""
    
    page: Optional[int]
    """Page number for pagination, starting from 1 (default: 1)."""
    
    sortOrder: Optional[Literal["asc", "desc"]]
    """Sort order for results by timestamp (ascending or descending, default: 'desc')."""


class BillingFormatEnum(str, Enum):
    """
    Defines available output formats for billing usage data responses.
    
    This enumeration specifies the supported data formats that can be requested
    when retrieving billing usage information from the Venice AI API. Different
    formats may be suitable for different use cases, such as programmatic
    processing or data export.
    
    Used to specify the desired response format in billing usage API requests.
    """
    
    JSON = "json"
    """JSON format - returns structured data as BillingUsageResponse."""
    
    CSV = "csv"
    """CSV format - returns raw CSV data as bytes for export purposes."""