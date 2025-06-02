"""
Tests for the export methods of the Billing and AsyncBilling resources.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from venice_ai._client import VeniceClient
from venice_ai._async_client import AsyncVeniceClient
from venice_ai.resources.billing import Billing, AsyncBilling


def test_billing_export():
    """Tests that Billing.export makes the correct API call and returns the expected result."""
    # Mock the VeniceClient
    mock_client = MagicMock(spec=VeniceClient)
    
    # Create a mock response
    mock_response = b"date,amount,currency,description\n2023-01-15,10.50,USD,API usage"
    mock_client._request.return_value = mock_response
    
    # Create a Billing instance with the mock client
    billing_resource = Billing(mock_client)
    
    # Define test parameters
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    
    # Call the export method
    result = billing_resource.export(start_date=start_date, end_date=end_date)
    
    # Verify the client method was called with the correct arguments
    mock_client._request.assert_called_once_with(
        "GET",
        "billing/export",
        params={"start_date": start_date, "end_date": end_date},
        headers={"Accept": "text/csv"},
        raw_response=True,
    )
    
    # Verify the result
    assert result == mock_response
    assert isinstance(result, bytes)


@pytest.mark.asyncio
async def test_async_billing_export():
    """Tests that AsyncBilling.export makes the correct API call and returns the expected result."""
    # Mock the AsyncVeniceClient
    mock_client = MagicMock(spec=AsyncVeniceClient)
    
    # Create a mock response
    mock_response = b"date,amount,currency,description\n2023-01-15,10.50,USD,API usage"
    
    # Configure the mock client to return the mock response
    mock_client._request = AsyncMock(return_value=mock_response)
    
    # Create an AsyncBilling instance with the mock client
    async_billing_resource = AsyncBilling(mock_client)
    
    # Define test parameters
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    
    # Call the export method
    result = await async_billing_resource.export(start_date=start_date, end_date=end_date)
    
    # Verify the client method was called with the correct arguments
    mock_client._request.assert_called_once_with(
        "GET",
        "billing/export",
        params={"start_date": start_date, "end_date": end_date},
        headers={"Accept": "text/csv"},
        raw_response=True,
    )
    
    # Verify the result
    assert result == mock_response
    assert isinstance(result, bytes)