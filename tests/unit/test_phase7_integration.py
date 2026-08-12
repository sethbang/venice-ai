"""
Integration verification tests.

This module verifies that all Venice Provider components integrate correctly
with the adaptive-rate-limiter.
"""


class TestImportChainVerification:
    """Test that all imports work without circular dependencies."""

    def test_import_venice_provider(self):
        """VeniceProvider imports correctly."""
        from venice_ai.provider import VeniceProvider

        assert VeniceProvider is not None

    def test_import_classifier_adapter(self):
        """VeniceClassifierAdapter imports correctly."""
        from venice_ai.provider import VeniceClassifierAdapter

        assert VeniceClassifierAdapter is not None


class TestVeniceProviderVerification:
    """Test VeniceProvider implements ProviderInterface correctly."""

    def test_provider_name(self):
        """Provider name returns 'venice'."""
        from venice_ai.provider import VeniceProvider

        provider = VeniceProvider()
        assert provider.name == "venice"

    def test_parse_rate_limit_response(self):
        """Parse Venice rate limit headers correctly."""
        from venice_ai.provider import VeniceProvider

        provider = VeniceProvider()

        headers = {
            "x-ratelimit-remaining-requests": "50",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-tokens": "10000",
            "x-ratelimit-limit-tokens": "100000",
        }

        info = provider.parse_rate_limit_response(headers, status_code=200)

        assert info.rpm_remaining == 50
        assert info.rpm_limit == 100
        assert info.tpm_remaining == 10000
        assert info.tpm_limit == 100000
        assert info.is_rate_limited is False

    def test_parse_429_response(self):
        """Detect 429 rate limited responses."""
        from venice_ai.provider import VeniceProvider

        provider = VeniceProvider()

        headers = {"retry-after": "30"}
        info = provider.parse_rate_limit_response(headers, status_code=429)

        assert info.is_rate_limited is True
        assert info.retry_after == 30


# class TestFactorySchedulerSwitching:
#     """Test factory creates correct scheduler based on environment."""

#     def test_internal_scheduler_default(self):
#         """Default to internal scheduler."""
#         from venice_ai.factory import VeniceClientFactory

#         # Clear any env var
#         with patch.dict(os.environ, {"VENICE_SCHEDULER_IMPL": ""}, clear=False):
#             result = VeniceClientFactory._should_use_extracted_scheduler()
#             assert result is False

#     def test_internal_scheduler_explicit(self):
#         """Explicit internal scheduler selection."""
#         from venice_ai.factory import VeniceClientFactory

#         with patch.dict(os.environ, {"VENICE_SCHEDULER_IMPL": "internal"}):
#             result = VeniceClientFactory._should_use_extracted_scheduler()
#             assert result is False

#     def test_extracted_scheduler_selection(self):
#         """Extracted scheduler selection via env var."""
#         from venice_ai.factory import VeniceClientFactory

#         with patch.dict(os.environ, {"VENICE_SCHEDULER_IMPL": "extracted"}):
#             result = VeniceClientFactory._should_use_extracted_scheduler()
#             assert result is True


class TestClientProtocolCompliance:
    """Test VeniceClient implements ClientProtocol."""

    def test_client_has_base_url(self):
        """VeniceClient has base_url property."""
        from venice_ai._client import VeniceClient

        assert hasattr(VeniceClient, "base_url")

    def test_client_has_timeout(self):
        """VeniceClient has timeout property."""
        from venice_ai._client import VeniceClient

        assert hasattr(VeniceClient, "timeout")

    def test_client_has_get_headers(self):
        """VeniceClient has get_headers method."""
        from venice_ai._client import VeniceClient

        assert hasattr(VeniceClient, "get_headers")
        assert callable(getattr(VeniceClient, "get_headers", None))
