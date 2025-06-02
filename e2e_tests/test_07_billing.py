import pytest
import pytest_asyncio
from datetime import datetime, timedelta, UTC
from venice_ai import VeniceClient, AsyncVeniceClient
from venice_ai.exceptions import VeniceError, InvalidRequestError
from venice_ai.types.billing import BillingFormatEnum

# Functional Tests for Billing API

def test_get_usage_json_sync(venice_client: VeniceClient):
    """Tests synchronous retrieval of billing usage data in JSON format."""
    # Get usage data without filters
    usage_data = venice_client.billing.get_usage(
        format=BillingFormatEnum.JSON
    )
    
    # Check that we got a structured response
    assert isinstance(usage_data, dict)
    assert "data" in usage_data or "usage" in usage_data  # Adjust based on actual response structure
    
def test_get_usage_csv_sync(venice_client: VeniceClient):
    """Tests synchronous retrieval of billing usage data in CSV format."""
    # Get usage data in CSV format
    usage_data = venice_client.billing.get_usage(
        format=BillingFormatEnum.CSV
    )
    
    # Check that we got binary data back (CSV as bytes)
    assert isinstance(usage_data, bytes)
    assert len(usage_data) > 0
    
    # Optionally decode and check CSV content
    csv_content = usage_data.decode('utf-8')
    assert len(csv_content) > 0
    assert ',' in csv_content  # Basic check for CSV format

def test_get_usage_with_date_range_sync(venice_client: VeniceClient):
    """Tests synchronous retrieval of billing usage with a date range filter."""
    # Set date range for the last 30 days
    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=30)
    
    usage_data = venice_client.billing.get_usage(
        format=BillingFormatEnum.JSON,
        startDate=start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        endDate=end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    
    # Check that we got a structured response
    assert isinstance(usage_data, dict)
    assert "data" in usage_data or "usage" in usage_data

def test_get_usage_with_pagination_sync(venice_client: VeniceClient):
    """Tests synchronous retrieval of billing usage with pagination parameters."""
    usage_data = venice_client.billing.get_usage(
        format=BillingFormatEnum.JSON,
        limit=10,
        page=1,
        sortOrder="desc"
    )
    
    # Check that we got a structured response
    assert isinstance(usage_data, dict)
    assert "data" in usage_data or "usage" in usage_data

def test_get_usage_invalid_date_format_sync(venice_client: VeniceClient):
    """Tests error handling for invalid date format in synchronous request."""
    with pytest.raises(InvalidRequestError) as excinfo:
        venice_client.billing.get_usage(
            format=BillingFormatEnum.JSON,
            startDate="invalid-date-format"
        )
    
    assert excinfo.value is not None

@pytest.mark.asyncio
async def test_get_usage_json_async(async_venice_client: AsyncVeniceClient):
    """Tests asynchronous retrieval of billing usage data in JSON format."""
    # Get usage data without filters
    usage_data = await async_venice_client.billing.get_usage(
        format=BillingFormatEnum.JSON
    )
    
    # Check that we got a structured response
    assert isinstance(usage_data, dict)
    assert "data" in usage_data or "usage" in usage_data  # Adjust based on actual response structure

@pytest.mark.asyncio
async def test_get_usage_csv_async(async_venice_client: AsyncVeniceClient):
    """Tests asynchronous retrieval of billing usage data in CSV format."""
    # Get usage data in CSV format
    usage_data = await async_venice_client.billing.get_usage(
        format=BillingFormatEnum.CSV
    )
    
    # Check that we got binary data back (CSV as bytes)
    assert isinstance(usage_data, bytes)
    assert len(usage_data) > 0
    
    # Optionally decode and check CSV content
    csv_content = usage_data.decode('utf-8')
    assert len(csv_content) > 0
    assert ',' in csv_content  # Basic check for CSV format

@pytest.mark.asyncio
async def test_get_usage_with_date_range_async(async_venice_client: AsyncVeniceClient):
    """Tests asynchronous retrieval of billing usage with a date range filter."""
    # Set date range for the last 30 days
    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=30)
    
    usage_data = await async_venice_client.billing.get_usage(
        format=BillingFormatEnum.JSON,
        startDate=start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        endDate=end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    
    # Check that we got a structured response
    assert isinstance(usage_data, dict)
    assert "data" in usage_data or "usage" in usage_data

@pytest.mark.asyncio
async def test_get_usage_with_pagination_async(async_venice_client: AsyncVeniceClient):
    """Tests asynchronous retrieval of billing usage with pagination parameters."""
    usage_data = await async_venice_client.billing.get_usage(
        format=BillingFormatEnum.JSON,
        limit=10,
        page=1,
        sortOrder="desc"
    )
    
    # Check that we got a structured response
    assert isinstance(usage_data, dict)
    assert "data" in usage_data or "usage" in usage_data

@pytest.mark.asyncio
async def test_get_usage_invalid_date_format_async(async_venice_client: AsyncVeniceClient):
    """Tests error handling for invalid date format in asynchronous request."""
    with pytest.raises(InvalidRequestError) as excinfo:
        await async_venice_client.billing.get_usage(
            format=BillingFormatEnum.JSON,
            startDate="invalid-date-format"
        )
    
    assert excinfo.value is not None