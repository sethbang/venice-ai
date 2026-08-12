"""Live E2E: Solana SIWX inference authentication (Mode 2).

Codifies the over-the-wire confirmation that Venice accepts a Solana-signed
``X-Sign-In-With-X`` header on *inference* (not only the ``/x402/*`` reads):
a ``VeniceClient(auth=SolanaX402Auth(...))`` with no API key runs a real chat
completion, debiting the wallet's prepaid ledger.

Gated — skips unless BOTH are present:
- the ``[x402-solana]`` extra (``solders``), and
- ``VENICE_X402_SOLANA_TEST_PRIVATE_KEY``: base58 secret for a wallet that
  carries a Venice prepaid balance (never funded/reused beyond this probe).
"""

from __future__ import annotations

import os

import pytest

solders = pytest.importorskip("solders", reason="x402-solana extra not installed")

from venice_ai import VeniceClient  # noqa: E402
from venice_ai.auth.x402_solana import SolanaX402Auth  # noqa: E402

_SECRET = os.environ.get("VENICE_X402_SOLANA_TEST_PRIVATE_KEY")

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.asyncio,
    pytest.mark.timeout(120),
    pytest.mark.skipif(
        not _SECRET,
        reason="set VENICE_X402_SOLANA_TEST_PRIVATE_KEY (funded Solana wallet) to run",
    ),
]


async def test_solana_siwx_inference_auth_live() -> None:
    """A Solana wallet (Mode 2, no API key) is accepted on a live inference call."""
    auth = SolanaX402Auth(private_key=_SECRET)  # type: ignore[arg-type]

    # No api_key → the SIWX header IS the request auth (the C1 path).
    async with VeniceClient(auth=auth) as client:
        balance = await client.x402.balance(auth=auth)
        if not balance.data.canConsume or balance.data.balanceUsd <= 0:
            pytest.skip(
                f"wallet {auth.wallet_address} has no spendable prepaid balance "
                f"(${balance.data.balanceUsd:.4f}); top up to run this probe"
            )

        model = await client.models.resolve_chat()
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            max_completion_tokens=5,
        )

    # A 200 with a well-formed body is the proof the SIWX header was accepted on
    # inference; an unaccepted header would have raised AuthenticationError.
    assert response.choices, "expected at least one choice from the live call"
    assert response.model
