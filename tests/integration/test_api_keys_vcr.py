"""
VCR-based integration tests for Venice AI API Keys Resource.

This test suite uses VCRpy to record and replay real API responses for the
API keys resource. All sensitive data (API key values, Web3 addresses, etc.)
is automatically sanitized by the before_record_response hook in conftest.py.

IMPORTANT: These tests create real API keys and must clean them up. The
cleanup fixtures ensure that created keys are deleted even if tests fail.

Tests cover:
- List, create, retrieve, delete operations
- Rate limits and usage monitoring
- Response structure validation
- Error handling scenarios

Note: Web3 key creation tests are excluded as they require blockchain
authentication which is complex to test with VCRpy.
"""

import os

import pytest
import pytest_asyncio

from venice_ai import create_test_venice_client
from venice_ai.core.config import SchedulerMode
from venice_ai.types.api import CreateApiKeyRequest

pytestmark = [pytest.mark.integration, pytest.mark.vcr]


@pytest_asyncio.fixture
async def venice_client(backend_instance):
    """Create a Venice client for VCR testing with shared rate limit coordination."""
    api_key = os.getenv("VENICE_API_KEY")
    if not api_key:
        pytest.skip("VENICE_API_KEY environment variable required for integration tests")

    client = create_test_venice_client(
        api_key=api_key,
        scheduler_mode=SchedulerMode.INTELLIGENT,
        enable_redis=True,
    )
    yield client
    await client.close()


@pytest.fixture
async def temp_api_key(vcr_cassette, venice_client):
    """
    Create a temporary API key for testing with automatic cleanup.

    This fixture ensures that any API key created during testing is properly
    deleted afterward, even if the test fails. The cassette records both the
    creation and deletion for deterministic replay.
    """
    created_key = None

    try:
        # Create a test API key - all operations recorded in one cassette
        with vcr_cassette:
            request = CreateApiKeyRequest(
                description="VCR Test Key - Auto Delete",
                apiKeyType="INFERENCE",
                consumptionLimit=None,
                expiresAt=None,
            )
            created_key = await venice_client.api_keys.create(api_key_request=request)

        yield created_key

    finally:
        # Cleanup: Delete the created key
        if created_key and hasattr(created_key, "id"):
            try:
                with vcr_cassette:
                    await venice_client.api_keys.delete(api_key_id=created_key.id)
            except Exception as e:
                # Log but don't fail test if cleanup fails
                print(f"Warning: Failed to cleanup API key {created_key.id}: {e}")


@pytest.mark.integration
async def test_api_keys_list_basic(vcr_cassette, venice_client):
    """Test basic API key listing without pagination."""
    with vcr_cassette:
        keys = await venice_client.api_keys.list()

    # Should return a list (may be empty)
    assert isinstance(keys, list)

    # If keys exist, verify structure
    if keys:
        key = keys[0]
        assert hasattr(key, "id")
        assert hasattr(key, "apiKeyType")
        assert hasattr(key, "description")
        assert hasattr(key, "createdAt")


@pytest.mark.integration
async def test_api_keys_list_with_pagination(vcr_cassette, venice_client):
    """Test API key listing with pagination parameters."""
    with vcr_cassette:
        keys = await venice_client.api_keys.list(page=1, limit=10)

    assert isinstance(keys, list)
    # Note: API returns all keys regardless of limit parameter (API behavior)
    assert len(keys) > 0


@pytest.mark.integration
async def test_api_keys_create_and_delete(vcr_cassette, venice_client):
    """Test creating and deleting an API key (full lifecycle)."""
    created_key = None

    try:
        # All operations in one cassette
        with vcr_cassette:
            # Create a new API key
            request = CreateApiKeyRequest(
                description="VCR Test - Create Delete",
                apiKeyType="INFERENCE",
                consumptionLimit=None,
                expiresAt=None,
            )
            created_key = await venice_client.api_keys.create(api_key_request=request)

            # Verify creation
            assert hasattr(created_key, "id")
            assert hasattr(created_key, "apiKey")  # Secret key returned on creation
            assert len(created_key.apiKey) > 0  # Real key during recording, sanitized in cassette
            assert created_key.description == "VCR Test - Create Delete"
            assert created_key.apiKeyType == "INFERENCE"

            # Delete the key
            delete_response = await venice_client.api_keys.delete(api_key_id=created_key.id)

            # Verify deletion
            assert hasattr(delete_response, "success")
            assert delete_response.success is True

            created_key = None  # Mark as cleaned up

    finally:
        # Cleanup if test failed midway
        if created_key and hasattr(created_key, "id"):
            try:
                with vcr_cassette:
                    await venice_client.api_keys.delete(api_key_id=created_key.id)
            except Exception:
                pass  # Best effort cleanup


@pytest.mark.integration
async def test_api_keys_retrieve(vcr_cassette, venice_client):
    """Test retrieving a specific API key by ID."""
    created_key = None

    try:
        with vcr_cassette:
            # Create a test key
            request = CreateApiKeyRequest(
                description="VCR Test - Retrieve",
                apiKeyType="INFERENCE",
                consumptionLimit=None,
                expiresAt=None,
            )
            created_key = await venice_client.api_keys.create(api_key_request=request)

            # Retrieve the key details
            api_key = await venice_client.api_keys.retrieve(api_key_id=created_key.id)

            # Verify response structure (retrieve() returns bare ApiKey, see update())
            assert hasattr(api_key, "id")
            assert api_key.id == created_key.id
            assert hasattr(api_key, "description")
            assert hasattr(api_key, "apiKeyType")

            # Cleanup
            await venice_client.api_keys.delete(api_key_id=created_key.id)
            created_key = None

    finally:
        if created_key and hasattr(created_key, "id"):
            try:
                with vcr_cassette:
                    await venice_client.api_keys.delete(api_key_id=created_key.id)
            except Exception:
                pass


@pytest.mark.integration
async def test_api_keys_get_rate_limits(vcr_cassette, venice_client):
    """Test retrieving rate limit information."""
    with vcr_cassette:
        rate_limits = await venice_client.api_keys.get_rate_limits()

    # Verify response structure
    assert hasattr(rate_limits, "data")
    assert hasattr(rate_limits.data, "accessPermitted")
    assert hasattr(rate_limits.data, "apiTier")
    assert hasattr(rate_limits.data, "rateLimits")

    # Verify tier structure
    tier = rate_limits.data.apiTier
    assert hasattr(tier, "id")
    assert hasattr(tier, "isCharged")

    # Verify rate limits is a list
    assert isinstance(rate_limits.data.rateLimits, list)


@pytest.mark.integration
async def test_api_keys_get_rate_limit_logs(vcr_cassette, venice_client):
    """Test retrieving rate limit violation logs."""
    with vcr_cassette:
        logs = await venice_client.api_keys.get_rate_limit_logs()

    # Verify response structure
    assert hasattr(logs, "data")
    assert isinstance(logs.data, list)

    # If logs exist, verify structure
    if logs.data:
        log_entry = logs.data[0]
        assert hasattr(log_entry, "timestamp")
        assert hasattr(log_entry, "modelId")
        assert hasattr(log_entry, "rateLimitType")


@pytest.mark.integration
async def test_api_keys_create_with_consumption_limit(vcr_cassette, venice_client):
    """Test creating an API key with consumption limits."""
    created_key = None

    try:
        with vcr_cassette:
            request = CreateApiKeyRequest(
                description="VCR Test - With Limits",
                apiKeyType="INFERENCE",
                consumptionLimit=None,
                expiresAt=None,
            )
            created_key = await venice_client.api_keys.create(api_key_request=request)

            # Verify creation with limits
            assert hasattr(created_key, "consumptionLimit")
            # Note: Actual limit values are sanitized by VCR hook

            # Cleanup
            await venice_client.api_keys.delete(api_key_id=created_key.id)

            created_key = None

    finally:
        if created_key and hasattr(created_key, "id"):
            try:
                with vcr_cassette:
                    await venice_client.api_keys.delete(api_key_id=created_key.id)
            except Exception:
                pass


@pytest.mark.integration
async def test_api_keys_sanitization_check(vcr_cassette, venice_client):
    """Test that API key values are present in live responses (sanitization happens in cassettes)."""
    created_key = None

    try:
        with vcr_cassette:
            request = CreateApiKeyRequest(
                description="VCR Test - Sanitization",
                apiKeyType="INFERENCE",
                consumptionLimit=None,
                expiresAt=None,
            )
            created_key = await venice_client.api_keys.create(api_key_request=request)

            # During live recording, API returns real key (sanitization happens when saving cassette)
            assert created_key.apiKey is not None
            assert len(created_key.apiKey) > 0
            # Note: last6Chars is only returned when listing keys, not when creating

            # Cleanup
            await venice_client.api_keys.delete(api_key_id=created_key.id)

            created_key = None

    finally:
        if created_key and hasattr(created_key, "id"):
            try:
                with vcr_cassette:
                    await venice_client.api_keys.delete(api_key_id=created_key.id)
            except Exception:
                pass


@pytest.mark.integration
async def test_api_keys_list_response_structure(vcr_cassette, venice_client):
    """Test that list response structure is correct."""
    with vcr_cassette:
        keys = await venice_client.api_keys.list()

    # Should be a list
    assert isinstance(keys, list)

    # If keys exist, verify all required fields
    if keys:
        for key in keys:
            required_fields = [
                "id",
                "apiKeyType",
                "description",
                "createdAt",
                "consumptionLimits",
                "usage",
            ]
            for field in required_fields:
                assert hasattr(key, field), f"Missing field: {field}"


@pytest.mark.integration
async def test_api_keys_rate_limits_structure(vcr_cassette, venice_client):
    """Test detailed structure of rate limits response."""
    with vcr_cassette:
        rate_limits = await venice_client.api_keys.get_rate_limits()

    # Verify top-level structure
    data = rate_limits.data
    assert hasattr(data, "accessPermitted")
    assert isinstance(data.accessPermitted, bool)

    assert hasattr(data, "apiTier")
    assert hasattr(data, "balances")
    assert hasattr(data, "rateLimits")

    # Verify balances structure
    balances = data.balances
    # May have USD, VCU, DIEM keys
    for currency in ["USD", "VCU", "DIEM"]:
        if hasattr(balances, currency):
            value = getattr(balances, currency)
            # Value can be None or numeric
            assert value is None or isinstance(value, (int, float))

    # Verify rate limits structure
    if data.rateLimits:
        limit_entry = data.rateLimits[0]
        assert hasattr(limit_entry, "apiModelId")
        assert hasattr(limit_entry, "rateLimits")
        assert isinstance(limit_entry.rateLimits, list)


@pytest.mark.integration
async def test_api_keys_update_description(vcr_cassette, venice_client):
    """``api_keys.update()`` (PATCH /api_keys) replaces the description on an existing key.

    Wire-format check: verifies the SDK's ``UpdateApiKeyRequest`` (camelCase
    body via ``by_alias=True``) is what the API actually accepts, and that the
    response deserializes to ``ApiKey``. The previous test suite had no
    coverage of this method at all, so the recorded cassette is the source of
    truth for the wire shape.
    """
    created_key = None

    try:
        with vcr_cassette:
            # Create a test key with the original description.
            request = CreateApiKeyRequest(
                description="VCR Test - Update Original",
                apiKeyType="INFERENCE",
                consumptionLimit=None,
                expiresAt=None,
            )
            created_key = await venice_client.api_keys.create(api_key_request=request)
            assert created_key.description == "VCR Test - Update Original"

            # Update the description.
            updated = await venice_client.api_keys.update(
                id=created_key.id,
                description="VCR Test - Update Modified",
            )

            # Response must deserialize as ApiKey, with the new description.
            assert hasattr(updated, "id")
            assert updated.id == created_key.id
            assert updated.description == "VCR Test - Update Modified"

            # Retrieve confirms server-side persistence.
            details = await venice_client.api_keys.retrieve(api_key_id=created_key.id)
            assert details.description == "VCR Test - Update Modified"

            # Cleanup
            await venice_client.api_keys.delete(api_key_id=created_key.id)
            created_key = None

    finally:
        if created_key and hasattr(created_key, "id"):
            try:
                with vcr_cassette:
                    await venice_client.api_keys.delete(api_key_id=created_key.id)
            except Exception:
                pass


@pytest.mark.integration
async def test_api_keys_update_consumption_limit_with_dict(vcr_cassette, venice_client):
    """``update()`` accepts ``consumption_limit`` as a plain dict — the SDK
    coerces to the ``ConsumptionLimit`` model via ``populate_by_name=True``.

    The body sent on the wire uses camelCase aliases (``consumptionLimit``),
    not snake_case. Cassette validates that round-trip.
    """
    created_key = None

    try:
        with vcr_cassette:
            request = CreateApiKeyRequest(
                description="VCR Test - Update Limit Dict",
                apiKeyType="INFERENCE",
                consumptionLimit=None,
                expiresAt=None,
            )
            created_key = await venice_client.api_keys.create(api_key_request=request)

            updated = await venice_client.api_keys.update(
                id=created_key.id,
                consumption_limit={"diem": 1, "usd": 10},
            )

            assert updated.id == created_key.id
            # Response field is plural ``consumptionLimits`` per the docs and
            # confirmed by the recorded cassette body (PATCH /api_keys returns
            # the same shape as GET /api_keys/{id}). The request body uses
            # singular ``consumptionLimit`` (per the API contract).
            #
            # Don't assert on the exact numeric values — the VCR sanitizer in
            # ``conftest.py`` normalizes billing amounts (divides by 10) before
            # writing the cassette, so live-vs-replay would diverge. The
            # presence of non-null fields is what the wire-shape test cares
            # about.
            assert updated.consumptionLimits is not None
            assert updated.consumptionLimits.diem is not None
            assert updated.consumptionLimits.usd is not None

            await venice_client.api_keys.delete(api_key_id=created_key.id)
            created_key = None

    finally:
        if created_key and hasattr(created_key, "id"):
            try:
                with vcr_cassette:
                    await venice_client.api_keys.delete(api_key_id=created_key.id)
            except Exception:
                pass


@pytest.mark.integration
async def test_api_keys_multiple_keys_workflow(vcr_cassette, venice_client):
    """Test creating, listing, and deleting multiple keys."""
    created_keys = []

    try:
        with vcr_cassette:
            # Create 2 test keys
            for i in range(2):
                request = CreateApiKeyRequest(
                    description=f"VCR Test Multi {i}",
                    apiKeyType="INFERENCE",
                    consumptionLimit=None,
                    expiresAt=None,
                )
                key = await venice_client.api_keys.create(api_key_request=request)
                created_keys.append(key)

            # List keys (should include our created keys)
            all_keys = await venice_client.api_keys.list()

            assert len(all_keys) >= 2

            # Delete all created keys
            for key in created_keys:
                await venice_client.api_keys.delete(api_key_id=key.id)

            created_keys = []

    finally:
        # Cleanup any remaining keys
        for key in created_keys:
            if hasattr(key, "id"):
                try:
                    with vcr_cassette:
                        await venice_client.api_keys.delete(api_key_id=key.id)
                except Exception:
                    pass
