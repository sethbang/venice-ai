"""Tests for the SIWE-only auth path on :class:`VeniceClient` (Mode 2).

Covers the new ``auth=X402Auth(...)`` constructor parameter, the validation
that requires either ``api_key`` or ``auth`` to be set, and the per-request
SIWE-header injection / caching behaviour exposed via the private
``_default_siwe_header()`` helper.

Tests skip if the ``[x402]`` extra (``eth_account`` / ``siwe``) is not
installed, since :class:`X402Auth` requires it.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

# Skip the module if the x402 extra isn't available.
pytest.importorskip("eth_account", reason="x402 extra not installed")
pytest.importorskip("siwe", reason="x402 extra not installed")

from venice_ai import VeniceClient
from venice_ai.auth.x402 import X402Auth

# Throwaway key — never funded, never reused.
_THROWAWAY_KEY = "0x" + "a" * 63 + "b"


@pytest.fixture
def auth() -> X402Auth:
    return X402Auth(private_key=_THROWAWAY_KEY)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_init_with_only_auth_succeeds(auth: X402Auth) -> None:
    """Mode 2: auth without api_key is valid; client stores empty api_key."""
    with patch.dict(os.environ, {}, clear=True):
        client = VeniceClient(auth=auth)
    assert client._api_key == ""
    assert client._auth is auth
    assert client._siwe_cache is None


def test_init_with_both_api_key_and_auth_succeeds(auth: X402Auth) -> None:
    """Mode 1+2 combo: api_key wins for default Bearer; auth retained."""
    client = VeniceClient(api_key="vn_test_123456789", auth=auth)
    assert client._api_key == "vn_test_123456789"
    assert client._auth is auth


def test_init_with_neither_raises_with_helpful_message() -> None:
    """No auth at all → ValueError mentioning both api_key and auth options."""
    with (
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(ValueError, match="No authentication provided"),
    ):
        VeniceClient()


def test_init_error_message_mentions_auth_alternative() -> None:
    """The new error message points users at auth=X402Auth(...) explicitly."""
    with patch.dict(os.environ, {}, clear=True):
        try:
            VeniceClient()
        except ValueError as exc:
            assert "auth=" in str(exc) or "auth=X402Auth" in str(exc)
        else:  # pragma: no cover - guarded above
            pytest.fail("expected ValueError")


# ---------------------------------------------------------------------------
# _default_siwe_header — caching + Bearer-precedence semantics
# ---------------------------------------------------------------------------


def test_default_siwe_returns_none_when_only_api_key() -> None:
    """Mode 1: no auth → no SIWE header by default."""
    client = VeniceClient(api_key="vn_test_123456789")
    assert client._default_siwe_header() is None


def test_default_siwe_returns_none_when_both_set(auth: X402Auth) -> None:
    """When both api_key and auth are set, default request auth = Bearer."""
    client = VeniceClient(api_key="vn_test_123456789", auth=auth)
    assert client._default_siwe_header() is None


def test_default_siwe_returns_token_when_only_auth(auth: X402Auth) -> None:
    """Mode 2: SIWE header is generated, base64-ish, and reasonably long."""
    with patch.dict(os.environ, {}, clear=True):
        client = VeniceClient(auth=auth)
    header = client._default_siwe_header()
    assert header is not None
    assert isinstance(header, str)
    # SIWE base64 envelopes are typically 600-800 bytes.
    assert len(header) > 200


def test_default_siwe_caches_within_ttl(auth: X402Auth) -> None:
    """Two consecutive calls within the TTL return the SAME cached token."""
    with patch.dict(os.environ, {}, clear=True):
        client = VeniceClient(auth=auth)
    h1 = client._default_siwe_header()
    h2 = client._default_siwe_header()
    assert h1 == h2
    assert client._siwe_cache is not None


def test_default_siwe_refreshes_when_cache_expired(auth: X402Auth) -> None:
    """When the cached token's expiry is in the past, a new one is signed."""
    with patch.dict(os.environ, {}, clear=True):
        client = VeniceClient(auth=auth)
    h1 = client._default_siwe_header()
    # Force expiry: rewind expires_at to before now.
    assert client._siwe_cache is not None
    cached_header, _expires_at = client._siwe_cache
    client._siwe_cache = (cached_header, 0.0)  # expired in 1970

    h2 = client._default_siwe_header()
    assert h2 != h1  # new nonce + issued_at → different signature


def test_default_siwe_safety_margin_is_30s(auth: X402Auth) -> None:
    """Cached expires_at uses ttl_seconds - 30s for clock-skew tolerance."""
    import time

    with patch.dict(os.environ, {}, clear=True):
        client = VeniceClient(auth=auth)
    before = time.time()
    client._default_siwe_header()
    after = time.time()
    assert client._siwe_cache is not None
    _hdr, expires_at = client._siwe_cache
    # ttl=600, margin=30 → expires within (~570, 570 + tiny epsilon) of now
    expected_lifetime = auth.ttl_seconds - 30
    elapsed_lifetime = expires_at - before
    assert expected_lifetime - 1 <= elapsed_lifetime <= (after - before) + expected_lifetime + 1


def test_default_siwe_returns_none_after_clearing_auth() -> None:
    """If a client is constructed in Mode 2 then auth is unset (sanity), no SIWE."""
    with patch.dict(os.environ, {}, clear=True):
        client = VeniceClient(auth=X402Auth(private_key=_THROWAWAY_KEY))
    client._auth = None  # simulate explicit teardown — defensive coverage
    assert client._default_siwe_header() is None


# ---------------------------------------------------------------------------
# Stored fields
# ---------------------------------------------------------------------------


def test_auth_field_exposed_on_client(auth: X402Auth) -> None:
    """The auth instance is reachable from the client (for per-call use)."""
    with patch.dict(os.environ, {}, clear=True):
        client = VeniceClient(auth=auth)
    assert client._auth is auth


def test_api_key_stripped_when_provided() -> None:
    """Whitespace handling preserved — empty/whitespace api_key + no auth → ValueError."""
    with patch.dict(os.environ, {}, clear=True), pytest.raises(ValueError):
        VeniceClient(api_key="   ")
