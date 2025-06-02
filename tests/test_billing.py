"""
Tests for the Billing and AsyncBilling resources.
"""

import pytest
import httpx
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from venice_ai import VeniceClient, AsyncVeniceClient
from venice_ai.types.billing import BillingFormatEnum, BillingUsageResponse
from venice_ai.exceptions import (
    InvalidRequestError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,  # Added NotFoundError
    RateLimitError,
    APIError,
)


# Synchronous Tests
def test_get_usage_json_format(httpx_mock):
    """Tests successful retrieval of usage data in JSON format."""
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

    with VeniceClient(api_key="test-key") as client:
        response = client.billing.get_usage()

    assert isinstance(response, dict)
    assert "data" in response
    assert "pagination" in response
    assert len(response["data"]) == 1
    assert response["data"][0]["amount"] == 0.12
    assert response["data"][0]["currency"] == "USD"
    assert response["pagination"]["page"] == 1.0


def test_get_usage_csv_format(httpx_mock):
    """Tests successful retrieval of usage data in CSV format."""
    # Mock CSV response
    mock_csv_data = b"timestamp,amount,currency,units,pricePerUnitUsd,sku,notes\n2025-01-15T12:30:45Z,0.12,USD,12.0,0.01,gpt-4-8k,GPT-4 generation"

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/billing/usage",
        content=mock_csv_data,
        headers={"Content-Type": "text/csv"},
        status_code=200,
    )

    with VeniceClient(api_key="test-key") as client:
        response = client.billing.get_usage(format=BillingFormatEnum.CSV)

    assert isinstance(response, bytes)
    assert response == mock_csv_data


def test_get_usage_with_parameters(httpx_mock):
    """Tests usage retrieval with query parameters."""
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

    with VeniceClient(api_key="test-key") as client:
        response = client.billing.get_usage(
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


def test_get_usage_accept_header_json(httpx_mock):
    """Tests if the correct Accept header is sent for JSON format."""
    mock_response = {"data": [], "pagination": {"limit": 100.0, "page": 1.0, "total": 0.0, "totalPages": 0.0}}

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/billing/usage",
        json=mock_response,
        status_code=200,
    )

    with VeniceClient(api_key="test-key") as client:
        client.billing.get_usage(format=BillingFormatEnum.JSON)

    request = httpx_mock.get_request()
    assert request.headers.get("Accept") == "application/json"


def test_get_usage_accept_header_csv(httpx_mock):
    """Tests if the correct Accept header is sent for CSV format."""
    mock_csv_data = b"timestamp,amount,currency,units,pricePerUnitUsd,sku,notes\n"

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/billing/usage",
        content=mock_csv_data,
        status_code=200,
    )

    with VeniceClient(api_key="test-key") as client:
        client.billing.get_usage(format=BillingFormatEnum.CSV)

    request = httpx_mock.get_request()
    assert request.headers.get("Accept") == "text/csv"


def test_get_usage_error_400(httpx_mock):
    """Tests error handling for invalid request (400)."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/billing/usage",
        status_code=400,
        json={"error": {"message": "Invalid parameters", "type": "invalid_request_error"}},
    )

    with VeniceClient(api_key="test-key") as client:
        with pytest.raises(InvalidRequestError) as excinfo:
            client.billing.get_usage()

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 400
    assert "Invalid parameters" in str(excinfo.value)


def test_get_usage_error_401(httpx_mock):
    """Tests error handling for authentication error (401)."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/billing/usage",
        status_code=401,
        json={"error": {"message": "Invalid API key", "type": "authentication_error"}},
    )

    with VeniceClient(api_key="invalid-key") as client:
        with pytest.raises(AuthenticationError) as excinfo:
            client.billing.get_usage()

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 401
    assert "Invalid API key" in str(excinfo.value)


def test_get_usage_error_403(httpx_mock):
    """Tests error handling for permission denied (403)."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/billing/usage",
        status_code=403,
        json={"error": {"message": "Access denied", "type": "permission_denied_error"}},
    )

    with VeniceClient(api_key="test-key") as client:
        with pytest.raises(PermissionDeniedError) as excinfo:
            client.billing.get_usage()

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 403
    assert "Access denied" in str(excinfo.value)


def test_get_usage_error_404(httpx_mock):
    """Tests error handling for not found (404)."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/billing/usage",
        status_code=404,
        json={"error": {"message": "Endpoint not found", "type": "not_found_error"}},
    )

    with VeniceClient(api_key="test-key") as client:
        with pytest.raises(NotFoundError) as excinfo: # Changed APIError to NotFoundError
            client.billing.get_usage()

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 404
    assert "Endpoint not found" in str(excinfo.value)


def test_get_usage_error_429(httpx_mock):
    """Tests error handling for rate limit exceeded (429)."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/billing/usage",
        status_code=429,
        json={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
    )

    with VeniceClient(api_key="test-key") as client:
        with pytest.raises(RateLimitError) as excinfo:
            client.billing.get_usage()

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 429
    assert "Rate limit exceeded" in str(excinfo.value)


# Asynchronous Tests
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
async def test_get_usage_accept_header_json_async(httpx_mock):
    """Tests if the correct Accept header is sent for JSON format in async request."""
    mock_response = {"data": [], "pagination": {"limit": 100.0, "page": 1.0, "total": 0.0, "totalPages": 0.0}}

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/billing/usage",
        json=mock_response,
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        await client.billing.get_usage(format=BillingFormatEnum.JSON)

    request = httpx_mock.get_request()
    assert request.headers.get("Accept") == "application/json"


@pytest.mark.asyncio
async def test_get_usage_accept_header_csv_async(httpx_mock):
    """Tests if the correct Accept header is sent for CSV format in async request."""
    mock_csv_data = b"timestamp,amount,currency,units,pricePerUnitUsd,sku,notes\n"

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/billing/usage",
        content=mock_csv_data,
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        await client.billing.get_usage(format=BillingFormatEnum.CSV)

    request = httpx_mock.get_request()
    assert request.headers.get("Accept") == "text/csv"


@pytest.mark.asyncio
async def test_get_usage_error_400_async(httpx_mock):
    """Tests asynchronous error handling for invalid request (400)."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/billing/usage",
        status_code=400,
        json={"error": {"message": "Invalid parameters", "type": "invalid_request_error"}},
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        with pytest.raises(InvalidRequestError) as excinfo:
            await client.billing.get_usage()

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 400
    assert "Invalid parameters" in str(excinfo.value)


@pytest.mark.asyncio
async def test_get_usage_error_401_async(httpx_mock):
    """Tests asynchronous error handling for authentication error (401)."""
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


@pytest.mark.asyncio
async def test_get_usage_error_403_async(httpx_mock):
    """Tests asynchronous error handling for permission denied (403)."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/billing/usage",
        status_code=403,
        json={"error": {"message": "Access denied", "type": "permission_denied_error"}},
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        with pytest.raises(PermissionDeniedError) as excinfo:
            await client.billing.get_usage()

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 403
    assert "Access denied" in str(excinfo.value)


@pytest.mark.asyncio
async def test_get_usage_error_404_async(httpx_mock):
    """Tests asynchronous error handling for not found (404)."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/billing/usage",
        status_code=404,
        json={"error": {"message": "Endpoint not found", "type": "not_found_error"}},
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        with pytest.raises(NotFoundError) as excinfo: # Changed APIError to NotFoundError
            await client.billing.get_usage()

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 404
    assert "Endpoint not found" in str(excinfo.value)


@pytest.mark.asyncio
async def test_get_usage_error_429_async(httpx_mock):
    """Tests asynchronous error handling for rate limit exceeded (429)."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/billing/usage",
        status_code=429,
        json={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        with pytest.raises(RateLimitError) as excinfo:
            await client.billing.get_usage()

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 429
    assert "Rate limit exceeded" in str(excinfo.value)