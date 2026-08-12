"""Mode-2 (SIWX-only) auth path on :class:`VeniceClient` with a **Solana** wallet.

The Solana parallel of ``test_client_siwe_auth.py``: Venice enabled SIWX
inference authentication for Solana wallets, so the client's default-header
path (used to authenticate inference when no API key is set) must accept a
:class:`~venice_ai.auth.x402_solana.SolanaX402Auth`, not only the EVM
:class:`~venice_ai.auth.x402.X402Auth`.

Skips if the ``[x402-solana]`` extra (``solders``) is not installed.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

# Gated on the Solana extra only — this path never touches eth_account/siwe.
pytest.importorskip("solders", reason="x402-solana extra not installed")

from solders.keypair import Keypair  # noqa: E402

from venice_ai import VeniceClient  # noqa: E402
from venice_ai.auth.x402_solana import SolanaX402Auth  # noqa: E402


@pytest.fixture
def solana_auth() -> SolanaX402Auth:
    """A random throwaway Solana keypair — never funded, never reused."""
    return SolanaX402Auth(private_key=str(Keypair()))


def test_solana_auth_exposes_ttl_seconds(solana_auth: SolanaX402Auth) -> None:
    """SolanaX402Auth exposes ttl_seconds (its SIWX message TTL) so the client
    can size its default-header cache, mirroring X402Auth."""
    assert solana_auth.ttl_seconds == 600


def test_default_siwe_supports_solana_auth(solana_auth: SolanaX402Auth) -> None:
    """Mode 2 with a Solana wallet: the default SIWX header builds without
    crashing (previously raised AttributeError on the missing ttl_seconds)."""
    with patch.dict(os.environ, {}, clear=True):
        client = VeniceClient(auth=solana_auth)
    header = client._default_siwe_header()
    assert header is not None
    assert isinstance(header, str)
    # base64 SIWX envelope — well over 200 bytes.
    assert len(header) > 200


def test_default_siwe_caches_solana_within_ttl(solana_auth: SolanaX402Auth) -> None:
    """Two consecutive calls within the TTL return the SAME cached token,
    exercising the ttl_seconds arithmetic in the cache path."""
    with patch.dict(os.environ, {}, clear=True):
        client = VeniceClient(auth=solana_auth)
    h1 = client._default_siwe_header()
    h2 = client._default_siwe_header()
    assert h1 == h2
    assert client._siwe_cache is not None
