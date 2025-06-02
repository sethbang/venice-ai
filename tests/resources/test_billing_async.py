"""
Tests for the asynchronous AsyncBilling resource.
"""

import pytest
import httpx
from typing import Dict, Any

from venice_ai import AsyncVeniceClient
from venice_ai.types.billing import BillingFormatEnum, BillingUsageResponse
from venice_ai.exceptions import APIError, AuthenticationError


@pytest.mark.asyncio
async def test_get_usage_json_format_async(httpx_mock):
    """Tests successful asynchronous retrieval of usage data in JSON format."""
    # Mock JSON response
    mock_response = {
        "data": [
            {
                "amount": 0.12,
                "currency": "USD",
                "inferenceDetails": {
                    "completionTokens": 100.0,
                    "promptTokens": 200.0,
                    "inferenceExecutionTime": 150.5,
                    "requestId": "req_123456789"
                },
                "notes": "GPT-4 generation",
                "pricePerUnitUsd": 0.01,
                "sku": "gpt-4-8k",
                "timestamp": "2025-01-15T12:30:45Z",
                "units": 12.0
            }
        ],
        "pagination": {
            "limit": 100.0,
            "page": 1.0,
            "total": 1.0,
            "totalPages": 1.0
        }
    }

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/billing/usage",
        json=mock_response,
        headers={"Content-Type": "application/json"},
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        response = await client.billing.get_usage()

    assert isinstance(response, dict)
    assert "data" in response
    assert "pagination" in response
    assert len(response["data"]) == 1
    assert response["data"][0]["amount"] == 0.12
    assert response["data"][0]["currency"] == "USD"
    assert response["pagination"]["page"] == 1.0


@pytest.mark.asyncio
async def test_get_usage_csv_format_async(httpx_mock):
    """Tests successful asynchronous retrieval of usage data in CSV format."""
    # Mock CSV response
    mock_csv_data = b"timestamp,amount,currency,units,pricePerUnitUsd,sku,notes\n2025-01-15T12:30:45Z,0.12,USD,12.0,0.01,gpt-4-8k,GPT-4 generation"

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/billing/usage",
        content=mock_csv_data,
        headers={"Content-Type": "text/csv"},
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        response = await client.billing.get_usage(format=BillingFormatEnum.CSV)

    assert isinstance(response, bytes)
    assert response == mock_csv_data


@pytest.mark.asyncio
async def test_get_usage_with_parameters_async(httpx_mock):
    """Tests asynchronous usage retrieval with query parameters."""
    # Mock response
    mock_response = {
        "data": [],
        "pagination": {
            "limit": 50.0,
            "page": 2.0,
            "total": 0.0,
            "totalPages": 0.0
        }
    }

    # Expected URL with query parameters
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/billing/usage?currency=USD&startDate=2025-01-01T00%3A00%3A00Z&endDate=2025-02-01T00%3A00%3A00Z&limit=50&page=2&sortOrder=asc",
        json=mock_response,
        headers={"Content-Type": "application/json"},
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        response = await client.billing.get_usage(
            currency="USD",
            startDate="2025-01-01T00:00:00Z",
            endDate="2025-02-01T00:00:00Z",
            limit=50,
            page=2,
            sortOrder="asc"
        )

    assert isinstance(response, dict)
    assert response["pagination"]["limit"] == 50.0
    assert response["pagination"]["page"] == 2.0


@pytest.mark.asyncio
async def test_get_usage_error_async(httpx_mock):
    """Tests asynchronous error handling for usage retrieval."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/billing/usage",
        status_code=401,
        json={"error": {"message": "Invalid API key", "type": "authentication_error"}},
    )

    async with AsyncVeniceClient(api_key="invalid-key") as client:
        with pytest.raises(AuthenticationError) as excinfo:
            await client.billing.get_usage()

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 401
    assert "Invalid API key" in str(excinfo.value)