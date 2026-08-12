"""
Comprehensive tests for venice_ai/provider/venice_provider.py

This module provides thorough test coverage for VeniceProvider implementation,
targeting 85%+ coverage including:
- VeniceProvider initialization with/without client and discovery
- discover_limits: async limit discovery with tier conversion
- parse_rate_limit_response: Venice-specific header parsing
- _parse_int, _parse_timestamp: Helper methods
- get_bucket_for_model: Model to bucket mapping
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.core.rate_limit_discovery import RateLimitBucket, RateLimitDiscovery
from venice_ai.provider.venice_provider import VeniceProvider


class TestVeniceProviderInit:
    """Tests for VeniceProvider initialization."""

    def test_init_without_client_or_discovery(self):
        """VeniceProvider can be initialized without client or discovery."""
        provider = VeniceProvider()
        assert provider._client is None
        assert provider._discovery is None
        assert provider.name == "venice"

    def test_init_with_client_creates_discovery(self):
        """VeniceProvider creates RateLimitDiscovery when client is provided."""
        mock_client = MagicMock()
        provider = VeniceProvider(client=mock_client, account_id="test-account")

        assert provider._client is mock_client
        assert provider._discovery is not None
        assert isinstance(provider._discovery, RateLimitDiscovery)

    def test_init_with_custom_discovery(self):
        """VeniceProvider uses provided discovery instance."""
        mock_client = MagicMock()
        mock_discovery = MagicMock(spec=RateLimitDiscovery)

        provider = VeniceProvider(client=mock_client, rate_limit_discovery=mock_discovery)

        assert provider._client is mock_client
        assert provider._discovery is mock_discovery

    def test_init_with_custom_cache_duration(self):
        """VeniceProvider passes cache_duration to discovery."""
        mock_client = MagicMock()
        provider = VeniceProvider(client=mock_client, cache_duration=600)

        assert provider._discovery is not None
        assert provider._discovery.cache_duration == 600


class TestVeniceProviderName:
    """Tests for VeniceProvider.name property."""

    def test_name_returns_venice(self):
        """Provider name is 'venice'."""
        provider = VeniceProvider()
        assert provider.name == "venice"


class TestDiscoverLimits:
    """Tests for VeniceProvider.discover_limits method."""

    @pytest.mark.asyncio
    async def test_discover_limits_without_discovery_returns_empty(self):
        """discover_limits returns empty dict when no discovery is configured."""
        provider = VeniceProvider()
        assert provider._discovery is None

        result = await provider.discover_limits()

        assert result == {}

    @pytest.mark.asyncio
    async def test_discover_limits_with_empty_tiers(self):
        """discover_limits returns empty dict when discovery returns no tiers."""
        mock_discovery = AsyncMock(spec=RateLimitDiscovery)
        mock_discovery.discover_tiers = AsyncMock(return_value={})

        provider = VeniceProvider(rate_limit_discovery=mock_discovery)

        result = await provider.discover_limits()

        assert result == {}
        mock_discovery.discover_tiers.assert_called_once_with(force_refresh=False)

    @pytest.mark.asyncio
    async def test_discover_limits_converts_venice_buckets_to_core(self):
        """discover_limits converts Venice RateLimitBucket to core format."""
        # Create Venice-style buckets
        venice_bucket = RateLimitBucket(
            bucket_id="test-model",
            rpm_limit=100,
            tpm_limit=10000,
            rpd_limit=1000,
            models={"test-model"},
        )

        mock_discovery = AsyncMock(spec=RateLimitDiscovery)
        mock_discovery.discover_tiers = AsyncMock(return_value={"test-model": venice_bucket})

        provider = VeniceProvider(rate_limit_discovery=mock_discovery)

        result = await provider.discover_limits()

        assert "test-model" in result
        core_bucket = result["test-model"]
        assert core_bucket.bucket_id == "test-model"
        assert core_bucket.rpm_limit == 100
        assert core_bucket.tpm_limit == 10000

    @pytest.mark.asyncio
    async def test_discover_limits_with_multiple_buckets(self):
        """discover_limits handles multiple buckets correctly."""
        buckets = {
            "model-a": RateLimitBucket(bucket_id="model-a", rpm_limit=50, models={"model-a"}),
            "model-b": RateLimitBucket(
                bucket_id="model-b", rpm_limit=200, tpm_limit=50000, models={"model-b"}
            ),
        }

        mock_discovery = AsyncMock(spec=RateLimitDiscovery)
        mock_discovery.discover_tiers = AsyncMock(return_value=buckets)

        provider = VeniceProvider(rate_limit_discovery=mock_discovery)

        result = await provider.discover_limits()

        assert len(result) == 2
        assert result["model-a"].rpm_limit == 50
        assert result["model-b"].rpm_limit == 200
        assert result["model-b"].tpm_limit == 50000

    @pytest.mark.asyncio
    async def test_discover_limits_force_refresh(self):
        """discover_limits passes force_refresh to discovery."""
        mock_discovery = AsyncMock(spec=RateLimitDiscovery)
        mock_discovery.discover_tiers = AsyncMock(return_value={})

        provider = VeniceProvider(rate_limit_discovery=mock_discovery)

        await provider.discover_limits(force_refresh=True)

        mock_discovery.discover_tiers.assert_called_once_with(force_refresh=True)


class TestParseRateLimitResponse:
    """Tests for VeniceProvider.parse_rate_limit_response method."""

    def test_parse_all_headers(self):
        """parse_rate_limit_response extracts all rate limit headers.

        NEW-CORE-A: both reset headers are absolute Unix ms-epochs and must be
        normalized to absolute seconds (not now+value).
        """
        provider = VeniceProvider()
        headers = {
            "x-ratelimit-remaining-requests": "50",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-reset-requests": "1780580108941",
            "x-ratelimit-remaining-tokens": "10000",
            "x-ratelimit-limit-tokens": "100000",
            "x-ratelimit-reset-tokens": "1780580108942",
            "retry-after": "30",
        }

        info = provider.parse_rate_limit_response(headers, status_code=200)

        assert info.rpm_remaining == 50
        assert info.rpm_limit == 100
        # ms-epoch normalized to absolute seconds, not the raw ms value.
        assert info.rpm_reset == 1780580108.941
        assert info.tpm_remaining == 10000
        assert info.tpm_limit == 100000
        # tpm_reset must be the absolute seconds-epoch, NOT now + huge ms.
        assert info.tpm_reset == 1780580108.942
        # Both reset values live in the absolute-seconds band.
        assert 1e9 <= info.rpm_reset < 1e11
        assert 1e9 <= info.tpm_reset < 1e11
        assert info.retry_after == 30
        assert info.is_rate_limited is False

    def test_parse_rate_limited_response(self):
        """parse_rate_limit_response detects 429 status correctly."""
        provider = VeniceProvider()
        headers = {"retry-after": "30"}

        info = provider.parse_rate_limit_response(headers, status_code=429)

        assert info.is_rate_limited is True
        assert info.retry_after == 30

    def test_parse_case_insensitive_headers(self):
        """parse_rate_limit_response normalizes header keys to lowercase."""
        provider = VeniceProvider()
        headers = {
            "X-RateLimit-Remaining-Requests": "25",
            "X-RATELIMIT-LIMIT-REQUESTS": "50",
        }

        info = provider.parse_rate_limit_response(headers, status_code=200)

        assert info.rpm_remaining == 25
        assert info.rpm_limit == 50

    def test_parse_missing_headers(self):
        """parse_rate_limit_response handles missing headers gracefully."""
        provider = VeniceProvider()
        headers = {}

        info = provider.parse_rate_limit_response(headers, status_code=200)

        assert info.rpm_remaining is None
        assert info.rpm_limit is None
        assert info.tpm_remaining is None
        assert info.tpm_limit is None
        assert info.retry_after is None
        assert info.is_rate_limited is False

    def test_parse_with_body_parameter(self):
        """parse_rate_limit_response accepts body parameter."""
        provider = VeniceProvider()
        headers = {"x-ratelimit-limit-requests": "100"}
        body = {"error": "rate limited"}

        info = provider.parse_rate_limit_response(headers, body=body, status_code=200)

        assert info.rpm_limit == 100


class TestParseInt:
    """Tests for safe_int (formerly VeniceProvider._parse_int)."""

    def test_parse_int_valid_integer_string(self):
        """safe_int parses valid integer strings."""
        from venice_ai.utils.parsing import safe_int

        assert safe_int("50") == 50
        assert safe_int("0") == 0
        assert safe_int("999999") == 999999

    def test_parse_int_float_string(self):
        """safe_int handles float-style strings like '50.0'."""
        from venice_ai.utils.parsing import safe_int

        assert safe_int("50.0") == 50
        assert safe_int("100.5") == 100

    def test_parse_int_none_returns_none(self):
        """safe_int returns None for None input."""
        from venice_ai.utils.parsing import safe_int

        assert safe_int(None) is None

    def test_parse_int_invalid_string_returns_none(self):
        """safe_int returns None for invalid strings (ValueError path)."""
        from venice_ai.utils.parsing import safe_int

        assert safe_int("not_a_number") is None
        assert safe_int("abc") is None
        assert safe_int("") is None

    def test_parse_int_type_error_path(self):
        """safe_int handles TypeError cases."""
        from venice_ai.utils.parsing import safe_int

        # float("NaN") works but int(float("NaN")) raises ValueError
        assert safe_int("NaN") is None


class TestParseTimestamp:
    """Tests for VeniceProvider._parse_timestamp method."""

    def test_parse_timestamp_valid(self):
        """_parse_timestamp parses valid Unix timestamps (seconds pass through)."""
        provider = VeniceProvider()
        assert provider._parse_timestamp("1704067200") == 1704067200.0
        assert provider._parse_timestamp("1704067200.5") == 1704067200.5

    def test_parse_timestamp_ms_epoch_normalized(self):
        """NEW-CORE-A: 13-digit ms-epoch values are normalized to seconds."""
        provider = VeniceProvider()
        assert provider._parse_timestamp("1780580108941") == 1780580108.941

    def test_parse_timestamp_none_returns_none(self):
        """_parse_timestamp returns None for None input."""
        provider = VeniceProvider()
        assert provider._parse_timestamp(None) is None

    def test_parse_timestamp_invalid_string_returns_none(self):
        """_parse_timestamp returns None for invalid strings (ValueError path)."""
        provider = VeniceProvider()
        assert provider._parse_timestamp("not_a_timestamp") is None
        assert provider._parse_timestamp("abc") is None
        assert provider._parse_timestamp("") is None


class TestGetBucketForModel:
    """Tests for VeniceProvider.get_bucket_for_model method."""

    @pytest.mark.asyncio
    async def test_get_bucket_without_discovery_returns_model_id(self):
        """get_bucket_for_model returns model_id as fallback when no discovery."""
        provider = VeniceProvider()
        assert provider._discovery is None

        result = await provider.get_bucket_for_model("test-model")

        assert result == "test-model"

    @pytest.mark.asyncio
    async def test_get_bucket_with_discovery_found(self):
        """get_bucket_for_model returns bucket from discovery when found."""
        mock_discovery = AsyncMock(spec=RateLimitDiscovery)
        mock_discovery.get_tier_for_model = AsyncMock(return_value="discovered-bucket")

        provider = VeniceProvider(rate_limit_discovery=mock_discovery)

        result = await provider.get_bucket_for_model("test-model")

        assert result == "discovered-bucket"
        mock_discovery.get_tier_for_model.assert_called_once_with("test-model", None)

    @pytest.mark.asyncio
    async def test_get_bucket_with_resource_type(self):
        """get_bucket_for_model passes resource_type to discovery."""
        mock_discovery = AsyncMock(spec=RateLimitDiscovery)
        mock_discovery.get_tier_for_model = AsyncMock(return_value="image-bucket")

        provider = VeniceProvider(rate_limit_discovery=mock_discovery)

        result = await provider.get_bucket_for_model("test-model", resource_type="image")

        assert result == "image-bucket"
        mock_discovery.get_tier_for_model.assert_called_once_with("test-model", "image")

    @pytest.mark.asyncio
    async def test_get_bucket_discovery_returns_none_fallbacks_to_model_id(self):
        """get_bucket_for_model falls back to model_id when discovery returns None."""
        mock_discovery = AsyncMock(spec=RateLimitDiscovery)
        mock_discovery.get_tier_for_model = AsyncMock(return_value=None)

        provider = VeniceProvider(rate_limit_discovery=mock_discovery)

        result = await provider.get_bucket_for_model("unknown-model")

        assert result == "unknown-model"


class TestEdgeCases:
    """Edge cases and integration tests for VeniceProvider."""

    @pytest.mark.asyncio
    async def test_discover_limits_with_none_tpm_limit(self):
        """discover_limits handles buckets with None tpm_limit."""
        venice_bucket = RateLimitBucket(
            bucket_id="test-model",
            rpm_limit=100,
            tpm_limit=None,  # Explicitly None
            models={"test-model"},
        )

        mock_discovery = AsyncMock(spec=RateLimitDiscovery)
        mock_discovery.discover_tiers = AsyncMock(return_value={"test-model": venice_bucket})

        provider = VeniceProvider(rate_limit_discovery=mock_discovery)

        result = await provider.discover_limits()

        assert result["test-model"].tpm_limit is None

    def test_parse_rate_limit_response_with_none_status_code(self):
        """parse_rate_limit_response handles None status_code."""
        provider = VeniceProvider()
        headers = {"x-ratelimit-limit-requests": "100"}

        info = provider.parse_rate_limit_response(headers, status_code=None)

        assert info.rpm_limit == 100
        assert info.is_rate_limited is False

    def test_parse_with_negative_values(self):
        """parse_rate_limit_response handles negative header values."""
        provider = VeniceProvider()
        headers = {
            "x-ratelimit-remaining-requests": "-5",
            "x-ratelimit-limit-requests": "-100",
        }

        info = provider.parse_rate_limit_response(headers, status_code=200)

        # Negative values are parsed as integers, though semantically odd
        assert info.rpm_remaining == -5
        assert info.rpm_limit == -100

    def test_parse_with_whitespace_values(self):
        """parse_rate_limit_response handles values with whitespace."""
        provider = VeniceProvider()
        headers = {
            "x-ratelimit-remaining-requests": "  50  ",
        }

        info = provider.parse_rate_limit_response(headers, status_code=200)

        # float() handles whitespace, so this should work
        assert info.rpm_remaining == 50


class TestProviderInterface:
    """Tests verifying VeniceProvider implements ProviderInterface correctly."""

    def test_implements_name_property(self):
        """VeniceProvider has name property returning string."""
        provider = VeniceProvider()
        assert isinstance(provider.name, str)
        assert provider.name == "venice"

    @pytest.mark.asyncio
    async def test_implements_discover_limits(self):
        """VeniceProvider has discover_limits async method."""
        provider = VeniceProvider()
        result = await provider.discover_limits()
        assert isinstance(result, dict)

    def test_implements_parse_rate_limit_response(self):
        """VeniceProvider has parse_rate_limit_response method."""
        provider = VeniceProvider()
        result = provider.parse_rate_limit_response({})
        assert hasattr(result, "rpm_remaining")
        assert hasattr(result, "rpm_limit")
        assert hasattr(result, "is_rate_limited")

    @pytest.mark.asyncio
    async def test_implements_get_bucket_for_model(self):
        """VeniceProvider has get_bucket_for_model async method."""
        provider = VeniceProvider()
        result = await provider.get_bucket_for_model("test-model")
        assert isinstance(result, str)
