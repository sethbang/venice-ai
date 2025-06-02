"""
Venice AI Billing API resources.

This module provides classes for interacting with the Venice AI Billing API,
allowing clients to retrieve usage information in different formats (JSON or CSV).
The module offers both synchronous and asynchronous interfaces for retrieving
billing data, designed to integrate smoothly with the respective client types.
"""

from typing import Dict, Any, Optional, Union, cast, TYPE_CHECKING

from .._resource import APIResource, AsyncAPIResource
from venice_ai.types.billing import (
    BillingFormatEnum,
    BillingUsageRequestParams,
    BillingUsageResponse,
)

if TYPE_CHECKING:
    from .._client import VeniceClient
    from .._async_client import AsyncVeniceClient


class Billing(APIResource):
    """
    Provides access to billing and usage data operations.
    
    Manages synchronous billing operations, providing methods to retrieve billing usage data 
    in either JSON or CSV format. It handles API requests to the Venice AI Billing API endpoints, 
    managing request parameters, headers, and response formats. When initialized with a 
    :class:`~venice_ai.VeniceClient` instance, it inherits the client's configuration including 
    API key authentication.
    
    :param client: The Venice AI client instance used for making API requests.
    :type client: venice_ai.VeniceClient
    """
    
    def get_usage(
        self,
        *,
        format: BillingFormatEnum = BillingFormatEnum.JSON,
        currency: Optional[str] = None,
        startDate: Optional[str] = None,
        endDate: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None,
        sortOrder: Optional[str] = None,
    ) -> Union[BillingUsageResponse, bytes]:
        """
        Retrieves billing usage information.
        
        Fetches usage data from the Venice AI Billing API with various filtering options.
        The response format is determined by the 'format' parameter and corresponding
        'Accept' header: 'application/json' for JSON format or 'text/csv' for CSV format.
        
        :param format: Response format (JSON or CSV). Defaults to :attr:`~venice_ai.types.billing.BillingFormatEnum.JSON`.
        :type format: venice_ai.types.billing.BillingFormatEnum
        :param currency: Optional currency filter (USD or VCU).
        :type currency: Optional[str]
        :param startDate: Optional start date (ISO 8601 format, e.g., ``"2025-01-01T00:00:00Z"``).
        :type startDate: Optional[str]
        :param endDate: Optional end date (ISO 8601 format, e.g., ``"2025-05-01T00:00:00Z"``).
        :type endDate: Optional[str]
        :param limit: Optional number of items per page (1-500, default ``200``).
        :type limit: Optional[int]
        :param page: Optional page number for pagination (default ``1``).
        :type page: Optional[int]
        :param sortOrder: Optional sort order for timestamp (asc/desc, default ``'desc'``).
        :type sortOrder: Optional[str]

        :return: Billing usage data as :class:`~venice_ai.types.billing.BillingUsageResponse` for JSON, or ``bytes`` for CSV.
        :rtype: Union[venice_ai.types.billing.BillingUsageResponse, bytes]

        :raises venice_ai.exceptions.InvalidRequestError: If parameter values are invalid.
        :raises venice_ai.exceptions.AuthenticationError: If the API key is invalid.
        :raises venice_ai.exceptions.PermissionDeniedError: If access is denied.
        :raises venice_ai.exceptions.RateLimitError: If rate limits are exceeded.
        :raises venice_ai.exceptions.APIError: For other API-related errors.
            
        Example:

            .. code-block:: python

               from venice_ai import VeniceClient
               from venice_ai.types.billing import BillingFormatEnum
               
               client = VeniceClient(api_key="your-api-key")
               
               # Get JSON usage data
               usage_response = client.billing.get_usage(
                   startDate="2025-01-01T00:00:00Z",
                   endDate="2025-05-01T00:00:00Z",
                   limit=10,
                   page=1
               )
               
               # Access usage records
               for usage_record in usage_response['data']:
                   print(f"Date: {usage_record['timestamp']}, Cost: {usage_record['amount']}")
               
               # Get CSV usage data
               usage_csv = client.billing.get_usage(
                   format=BillingFormatEnum.CSV,
                   startDate="2025-01-01T00:00:00Z",
                   endDate="2025-05-01T00:00:00Z"
               )
               
               # Write CSV to file
               with open("usage.csv", "wb") as f:
                   f.write(usage_csv)
            
        """
        # Build the query parameters
        params: Dict[str, Any] = {}
        
        if currency is not None:
            params["currency"] = currency
            
        if startDate is not None:
            params["startDate"] = startDate
            
        if endDate is not None:
            params["endDate"] = endDate
            
        if limit is not None:
            params["limit"] = limit
            
        if page is not None:
            params["page"] = page
            
        if sortOrder is not None:
            params["sortOrder"] = sortOrder
            
        # Set headers based on requested format
        headers = {}
        raw_response = False
        
        if format == BillingFormatEnum.CSV:
            # Use lowercase header name to ensure it replaces any default header
            headers["accept"] = "text/csv"
            raw_response = True
        else:  # JSON format
            headers["accept"] = "application/json"
            
        # Make the API request
        result = self._client._request(
            "GET",
            "billing/usage",
            params=params,
            headers=headers,
            raw_response=raw_response,
        )
        
        # For JSON responses, the result is already parsed as a dict
        # For CSV responses, the result is returned as raw bytes
        if format == BillingFormatEnum.JSON:
            return cast(BillingUsageResponse, result)
        else:
            return cast(bytes, result)
            
    def export(
        self,
        *,
        start_date: str,
        end_date: str,
    ) -> bytes:
        """
        Exports billing data as a CSV file for a specified date range.
        
        This method retrieves billing export data from the Venice AI Billing API
        for the given date range and returns it as CSV content. The method always
        returns data in CSV format and sets the appropriate 'Accept' header in the
        request to ensure this.
        
        :param start_date: Start date for the billing export (YYYY-MM-DD format, e.g., ``"2025-01-01"``).
        :type start_date: str
        :param end_date: End date for the billing export (YYYY-MM-DD format, e.g., ``"2025-05-01"``).
        :type end_date: str

        :return: Raw ``bytes`` of the CSV file content.
        :rtype: bytes

        :raises venice_ai.exceptions.InvalidRequestError: If parameter values are invalid or date format is incorrect.
        :raises venice_ai.exceptions.AuthenticationError: If the API key is invalid.
        :raises venice_ai.exceptions.PermissionDeniedError: If access is denied.
        :raises venice_ai.exceptions.RateLimitError: If rate limits are exceeded.
        :raises venice_ai.exceptions.APIError: For other API-related errors.
            
        Example:
        
            .. code-block:: python
            
               from venice_ai import VeniceClient
               
               client = VeniceClient(api_key="your-api-key")
               
               # Export billing data for Q1 2025
               csv_data = client.billing.export(
                   start_date="2025-01-01",
                   end_date="2025-03-31"
               )
               
               # Write CSV to file
               with open("billing_export.csv", "wb") as f:
                   f.write(csv_data)
        """
        # Build the query parameters
        params: Dict[str, Any] = {
            "start_date": start_date,
            "end_date": end_date,
        }
        
        # Set headers for CSV response
        headers: Dict[str, str] = {"Accept": "text/csv"}
        
        # Make the API request with raw_response=True to get bytes
        result = self._client._request(
            "GET",
            "billing/export",
            params=params,
            headers=headers,
            raw_response=True,
        )
        
        return cast(bytes, result)


class AsyncBilling(AsyncAPIResource):
    """
    Provides access to billing and usage data operations using asynchronous requests.
    
    Manages asynchronous billing operations, providing methods to retrieve billing usage data
    in either JSON or CSV format using asynchronous requests. It's designed to work with
    :class:`~venice_ai.AsyncVeniceClient`, allowing for non-blocking API calls in asynchronous
    applications. The class handles request formatting, response parsing, and proper type
    conversions based on the requested format.
    
    :param client: The asynchronous Venice AI client instance used for making API requests.
    :type client: venice_ai.AsyncVeniceClient
    """
    
    async def get_usage(
        self,
        *,
        format: BillingFormatEnum = BillingFormatEnum.JSON,
        currency: Optional[str] = None,
        startDate: Optional[str] = None,
        endDate: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None,
        sortOrder: Optional[str] = None,
    ) -> Union[BillingUsageResponse, bytes]:
        """
        Retrieves billing usage information asynchronously.
        
        Fetches usage data from the Venice AI Billing API with various filtering options,
        using asynchronous HTTP requests. The method sets the appropriate 'Accept' header
        (``'application/json'`` or ``'text/csv'``) based on the requested format, which determines
        how the API processes and returns the data.
        
        :param format: Response format (JSON or CSV). Defaults to :attr:`~venice_ai.types.billing.BillingFormatEnum.JSON`.
        :type format: venice_ai.types.billing.BillingFormatEnum
        :param currency: Optional currency filter (USD or VCU).
        :type currency: Optional[str]
        :param startDate: Optional start date (ISO 8601 format, e.g., ``"2025-01-01T00:00:00Z"``).
        :type startDate: Optional[str]
        :param endDate: Optional end date (ISO 8601 format, e.g., ``"2025-05-01T00:00:00Z"``).
        :type endDate: Optional[str]
        :param limit: Optional number of items per page (1-500, default ``200``).
        :type limit: Optional[int]
        :param page: Optional page number for pagination (default ``1``).
        :type page: Optional[int]
        :param sortOrder: Optional sort order for timestamp (asc/desc, default ``'desc'``).
        :type sortOrder: Optional[str]

        :return: Billing usage data as :class:`~venice_ai.types.billing.BillingUsageResponse` for JSON, or ``bytes`` for CSV.
        :rtype: Union[venice_ai.types.billing.BillingUsageResponse, bytes]

        :raises venice_ai.exceptions.InvalidRequestError: If parameter values are invalid.
        :raises venice_ai.exceptions.AuthenticationError: If the API key is invalid.
        :raises venice_ai.exceptions.PermissionDeniedError: If access is denied.
        :raises venice_ai.exceptions.RateLimitError: If rate limits are exceeded.
        :raises venice_ai.exceptions.APIError: For other API-related errors.
            
        Example:

            .. code-block:: python

               import asyncio
               from venice_ai import AsyncVeniceClient
               from venice_ai.types.billing import BillingFormatEnum
               
               async def get_usage_example():
                   async with AsyncVeniceClient(api_key="your-api-key") as client:
                       # Get JSON usage data
                       usage_response = await client.billing.get_usage(
                           startDate="2025-01-01T00:00:00Z",
                           endDate="2025-05-01T00:00:00Z",
                           limit=10,
                           page=1
                       )
                       
                       # Access usage records
                       for usage_record in usage_response['data']:
                           print(f"Date: {usage_record['timestamp']}, Cost: {usage_record['amount']}")
                       
                       # Get CSV usage data
                       usage_csv = await client.billing.get_usage(
                           format=BillingFormatEnum.CSV,
                           startDate="2025-01-01T00:00:00Z",
                           endDate="2025-05-01T00:00:00Z"
                       )
                       
                       # Write CSV to file
                       with open("usage.csv", "wb") as f:
                           f.write(usage_csv)
               
               asyncio.run(get_usage_example())
            
        """
        # Build the query parameters
        params: Dict[str, Any] = {}
        
        if currency is not None:
            params["currency"] = currency
            
        if startDate is not None:
            params["startDate"] = startDate
            
        if endDate is not None:
            params["endDate"] = endDate
            
        if limit is not None:
            params["limit"] = limit
            
        if page is not None:
            params["page"] = page
            
        if sortOrder is not None:
            params["sortOrder"] = sortOrder
            
        # Set headers based on requested format
        headers = {}
        raw_response = False
        
        if format == BillingFormatEnum.CSV:
            # Use lowercase header name to ensure it replaces any default header
            headers["accept"] = "text/csv"
            raw_response = True
        else:  # JSON format
            headers["accept"] = "application/json"
            
        # Make the API request
        result = await self._client._request(
            "GET",
            "billing/usage",
            params=params,
            headers=headers,
            raw_response=raw_response,
        )
        
        # For JSON responses, the result is already parsed as a dict
        # For CSV responses, the result is returned as raw bytes
        if format == BillingFormatEnum.JSON:
            return cast(BillingUsageResponse, result)
        else:
            return cast(bytes, result)
            
    async def export(
        self,
        *,
        start_date: str,
        end_date: str,
    ) -> bytes:
        """
        Exports billing data as a CSV file for a specified date range asynchronously.
        
        This method asynchronously retrieves billing export data from the Venice AI Billing API
        for the given date range and returns it as CSV content. The method always
        returns data in CSV format and sets the appropriate 'Accept' header in the
        request to ensure this.
        
        :param start_date: Start date for the billing export (YYYY-MM-DD format, e.g., ``"2025-01-01"``).
        :type start_date: str
        :param end_date: End date for the billing export (YYYY-MM-DD format, e.g., ``"2025-05-01"``).
        :type end_date: str

        :return: Raw ``bytes`` of the CSV file content.
        :rtype: bytes

        :raises venice_ai.exceptions.InvalidRequestError: If parameter values are invalid or date format is incorrect.
        :raises venice_ai.exceptions.AuthenticationError: If the API key is invalid.
        :raises venice_ai.exceptions.PermissionDeniedError: If access is denied.
        :raises venice_ai.exceptions.RateLimitError: If rate limits are exceeded.
        :raises venice_ai.exceptions.APIError: For other API-related errors.
            
        Example:
        
            .. code-block:: python
            
               import asyncio
               from venice_ai import AsyncVeniceClient
               
               async def export_billing_example():
                   async with AsyncVeniceClient(api_key="your-api-key") as client:
                       # Export billing data for Q1 2025
                       csv_data = await client.billing.export(
                           start_date="2025-01-01",
                           end_date="2025-03-31"
                       )
                       
                       # Write CSV to file
                       with open("billing_export.csv", "wb") as f:
                           f.write(csv_data)
               
               asyncio.run(export_billing_example())
        """
        # Build the query parameters
        params: Dict[str, Any] = {
            "start_date": start_date,
            "end_date": end_date,
        }
        
        # Set headers for CSV response
        headers: Dict[str, str] = {"Accept": "text/csv"}
        
        # Make the API request with raw_response=True to get bytes
        result = await self._client._request(
            "GET",
            "billing/export",
            params=params,
            headers=headers,
            raw_response=True,
        )
        
        return cast(bytes, result)