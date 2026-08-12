"""Solana x402 network-id matching (bare + pinned mainnet CAIP-2 only)."""

from __future__ import annotations

import base64
import json

import pytest

pytest.importorskip("solders")

from venice_ai.auth.x402_solana import SOLANA_MAINNET_CAIP2, SolanaX402Auth, is_solana_mainnet


def test_pinned_mainnet_caip2_constant_value() -> None:
    assert SOLANA_MAINNET_CAIP2 == "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp"


@pytest.mark.parametrize("network", ["solana", SOLANA_MAINNET_CAIP2])
def test_accepted_forms(network: str) -> None:
    assert is_solana_mainnet(network) is True


@pytest.mark.parametrize(
    "network",
    [
        None,
        "",
        "eip155:8453",
        "solana:EtWTRABZaYq6iMfeYKouRu166VU2xqa1",  # devnet genesis
        "solana:",
        "solanax",
        "SOLANA",
        "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdX",  # one char off
    ],
)
def test_rejected_forms(network: str | None) -> None:
    assert is_solana_mainnet(network) is False


def _auth() -> SolanaX402Auth:
    """A deterministic throwaway keypair. Not a real wallet, holds no funds."""
    from solders.keypair import Keypair

    kp = Keypair.from_seed(bytes(range(32)))
    # str(Keypair) is the base58 secret; from_base58_string round-trips it.
    return SolanaX402Auth(private_key=str(kp))


def _requirement(network: str) -> dict:
    return {
        "scheme": "exact",
        "network": network,
        "amount": "5000000",
        "asset": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "payTo": "8qUL23aSj7mDWdoLMXGHFvnVCT9wd7jXcysiekroADEL",
        "maxTimeoutSeconds": 300,
        "extra": {
            "name": "USD Coin",
            "version": "2",
            "feePayer": "BFK9TLC3edb13K6v4YyH3DwPb5DSUpkWvb7XnqCL9b4F",
        },
    }


def _build(network: str) -> dict:
    """Build the envelope for ``network`` and return the decoded JSON."""
    hdr = _auth().build_payment_header(
        requirement=_requirement(network),
        recent_blockhash="11111111111111111111111111111111",
        mint_decimals=6,
        token_program="TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
        max_amount_units=5_000_000,
    )
    return json.loads(base64.b64decode(hdr))


def test_caip2_requirement_is_accepted_and_echoed_verbatim() -> None:
    env = _build(SOLANA_MAINNET_CAIP2)
    assert env["accepted"]["network"] == SOLANA_MAINNET_CAIP2


def test_bare_requirement_is_accepted_and_echoed_verbatim() -> None:
    env = _build("solana")
    assert env["accepted"]["network"] == "solana"


def test_wrong_cluster_refuses_to_sign() -> None:
    with pytest.raises(ValueError, match="network mismatch"):
        _build("solana:EtWTRABZaYq6iMfeYKouRu166VU2xqa1")
