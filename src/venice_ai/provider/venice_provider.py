"""
Venice AI-specific provider implementation.

Features:
- Automatic limit discovery via /api_keys/rate_limits endpoint
- Per-model rate limit buckets
- Venice-specific header parsing
"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # During type-checking, always import the real classes so Pylance can resolve
    # base classes and type annotations correctly.
    from adaptive_rate_limiter import (
        DiscoveredBucket,
        ProviderInterface,
        RateLimitInfo,
    )

    from .._client import VeniceClient

try:
    from adaptive_rate_limiter import (
        DiscoveredBucket as _DiscoveredBucketRuntime,
    )
    from adaptive_rate_limiter import (
        ProviderInterface as _ProviderInterfaceRuntime,
    )
    from adaptive_rate_limiter import (
        RateLimitInfo as _RateLimitInfoRuntime,
    )

    _ADAPTIVE_AVAILABLE = True
except ImportError:
    _ADAPTIVE_AVAILABLE = False
    _ProviderInterfaceRuntime = object  # type: ignore[assignment,misc]
    _DiscoveredBucketRuntime = None  # type: ignore[assignment,misc]
    _RateLimitInfoRuntime = None  # type: ignore[assignment,misc]

if not TYPE_CHECKING:
    ProviderInterface = _ProviderInterfaceRuntime
    DiscoveredBucket = _DiscoveredBucketRuntime
    RateLimitInfo = _RateLimitInfoRuntime

from ..core.rate_limit_discovery import (
    RateLimitDiscovery,
)
from ..utils.parsing import ms_epoch_to_seconds, safe_int

logger = logging.getLogger(__name__)


class VeniceProvider(ProviderInterface):
    """
    Venice AI-specific provider implementation.

    Features:
    - Automatic limit discovery via /api_keys/rate_limits endpoint
    - Per-model rate limit buckets
    - Venice-specific header parsing

    Header Formats (live-verified against the Venice API):
    - x-ratelimit-reset-requests: absolute Unix epoch in MILLISECONDS
    - x-ratelimit-reset-tokens: absolute Unix epoch in MILLISECONDS

    Both reset headers are normalized to absolute Unix *seconds* via
    :func:`venice_ai.utils.parsing.ms_epoch_to_seconds`.
    """

    def __init__(
        self,
        client: "VeniceClient | None" = None,
        rate_limit_discovery: RateLimitDiscovery | None = None,
        account_id: str | None = None,
        cache_duration: int = 300,
    ):
        """
        Initialize the Venice provider.

        Args:
            client: VeniceClient for API calls (used to create discovery if none provided)
            rate_limit_discovery: Optional pre-configured RateLimitDiscovery instance
            account_id: Account identifier (optional)
            cache_duration: Cache duration for rate limits in seconds
        """
        if not _ADAPTIVE_AVAILABLE:
            raise ImportError(
                "adaptive-rate-limiter package is required for VeniceProvider. "
                "Install with: pip install venice-ai[adaptive]"
            )
        self._client = client
        self._discovery = rate_limit_discovery or (
            RateLimitDiscovery(client=client, account_id=account_id, cache_duration=cache_duration)
            if client
            else None
        )

    @property
    def name(self) -> str:
        """Unique provider name."""
        return "venice"

    async def discover_limits(
        self,
        force_refresh: bool = False,
        timeout: float = 30.0,
    ) -> dict[str, DiscoveredBucket]:
        """
        Discover available rate limits from Venice API.

        Wraps RateLimitDiscovery.discover_tiers() and converts to core library format.

        Returns:
            Dictionary mapping bucket_id to RateLimitBucket objects.
        """
        if not self._discovery:
            return {}

        # Use existing discovery mechanism
        tiers = await self._discovery.discover_tiers(force_refresh=force_refresh)

        # Convert Venice RateLimitBucket to core library format
        result: dict[str, DiscoveredBucket] = {}
        for bucket_id, venice_bucket in tiers.items():
            result[bucket_id] = DiscoveredBucket(
                bucket_id=bucket_id,
                rpm_limit=venice_bucket.rpm_limit,
                tpm_limit=venice_bucket.tpm_limit,
            )

        return result

    def parse_rate_limit_response(
        self,
        headers: dict[str, str],
        body: dict[str, Any] | None = None,
        status_code: int | None = None,
    ) -> RateLimitInfo:
        """
        Parse Venice rate limit headers.

        Venice header formats (live-verified):
        - x-ratelimit-reset-requests: absolute Unix epoch in milliseconds
        - x-ratelimit-reset-tokens: absolute Unix epoch in milliseconds

        Both reset values are normalized to absolute Unix seconds.
        """
        # Normalize headers to lowercase
        normalized = {k.lower(): v for k, v in headers.items()}

        return RateLimitInfo(
            rpm_remaining=safe_int(normalized.get("x-ratelimit-remaining-requests")),
            rpm_limit=safe_int(normalized.get("x-ratelimit-limit-requests")),
            rpm_reset=self._parse_timestamp(normalized.get("x-ratelimit-reset-requests")),
            tpm_remaining=safe_int(normalized.get("x-ratelimit-remaining-tokens")),
            tpm_limit=safe_int(normalized.get("x-ratelimit-limit-tokens")),
            tpm_reset=self._parse_timestamp(normalized.get("x-ratelimit-reset-tokens")),
            retry_after=safe_int(normalized.get("retry-after")),
            is_rate_limited=(status_code == 429),
        )

    def _parse_timestamp(self, value: str | None) -> float | None:
        """Parse a Venice reset header to an absolute Unix timestamp in seconds.

        The ``x-ratelimit-reset-requests`` / ``x-ratelimit-reset-tokens`` headers
        arrive as 13-digit absolute Unix epoch milliseconds; they are normalized
        to seconds via :func:`venice_ai.utils.parsing.ms_epoch_to_seconds`.
        """
        if value is None:
            return None
        try:
            parsed = float(value)
        except (ValueError, TypeError):
            return None
        return ms_epoch_to_seconds(parsed)

    async def get_bucket_for_model(self, model_id: str, resource_type: str | None = None) -> str:
        """
        Venice uses per-model rate limit buckets.

        Each model has its own independent rate limit counter.
        """
        if not self._discovery:
            return model_id  # Fallback to model_id

        # Use discovery to get bucket (which maps to tier_id for Venice)
        bucket_id = await self._discovery.get_tier_for_model(model_id, resource_type)

        # If discovery returns None, fallback to model_id
        return bucket_id or model_id
