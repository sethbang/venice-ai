#!/usr/bin/env python3
"""
Venice AI SDK - x402: Solana Settlement
=======================================

Top up your Venice prepaid USDC ledger from a **Solana** wallet, using the
x402 "exact" SVM settlement path. This mirrors the EVM/Base flow
(``client.x402.top_up_with``) but settles in USDC on Solana mainnet.

Two pieces:

1. ``SolanaX402Auth(private_key=...)`` — wraps a base58 Solana secret key and
   builds the base64 ``X-402-Payment`` envelope (a partially-signed
   ``VersionedTransaction``; Venice's facilitator sponsors the network fee).
2. ``client.x402.top_up_with_solana(auth=..., amount_usdc=..., max_amount_usdc=...)``
   — runs the full probe → sign → submit handshake in one call.

**This moves real funds.** USDC on Solana mainnet is real money and on-chain
settlement is irreversible. So this example is a **dry run by default**: it
derives the wallet address and reads its on-chain USDC balance, but does **not**
submit a top-up unless you explicitly opt in by setting::

    export X402_SOLANA_DO_TOPUP=1

The minimum top-up is $5. Install the extra first:

    pip install 'venice-ai[x402-solana]'    # pulls solders

Credentials (never commit): set ``VENICE_X402_SOLANA_TEST_PRIVATE_KEY`` to the
base58 secret key. Optionally set ``VENICE_X402_SOLANA_RPC_URL`` to a private
RPC endpoint (the public default rate-limits).
"""

import asyncio
import os
import sys

import aiohttp

from venice_ai import VeniceClient
from venice_ai.auth.x402_solana import DEFAULT_SOLANA_RPC_URL, USDC_SOLANA_MAINNET
from venice_ai.exceptions import APIError, VeniceError

TOPUP_AMOUNT_USDC = 5.0  # Venice's documented minimum


async def _usdc_balance(rpc_url: str, owner: str) -> float | None:
    """Read the wallet's on-chain USDC balance via Solana JSON-RPC.

    Returns the summed UI amount across the owner's USDC token accounts, or
    None if the read fails (e.g. RPC rate-limit). Read-only — no signing.
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenAccountsByOwner",
        "params": [
            owner,
            {"mint": USDC_SOLANA_MAINNET},
            {"encoding": "jsonParsed"},
        ],
    }
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.post(rpc_url, json=payload) as resp,
        ):
            data = await resp.json()
    except Exception as e:  # network / RPC error — non-fatal for a balance read
        print(f"   ⚠️ Could not read on-chain balance: {e}")
        return None

    accounts = (data.get("result") or {}).get("value") or []
    total = 0.0
    for acct in accounts:
        info = acct["account"]["data"]["parsed"]["info"]["tokenAmount"]
        total += float(info.get("uiAmount") or 0.0)
    return total


async def solana_top_up() -> None:
    """Derive the wallet, read its USDC balance, and (opt-in) top up."""
    print("🪙 x402 Solana settlement")
    print("-" * 40)

    secret = os.environ.get("VENICE_X402_SOLANA_TEST_PRIVATE_KEY")
    if not secret:
        print("   ⏭️  Set VENICE_X402_SOLANA_TEST_PRIVATE_KEY (base58 secret) to run.")
        return

    # Importing SolanaX402Auth requires the [x402-solana] extra (solders).
    try:
        from venice_ai.auth.x402_solana import SolanaX402Auth
    except ImportError:
        print("   ⏭️  Install the extra:  pip install 'venice-ai[x402-solana]'")
        return

    auth = SolanaX402Auth(private_key=secret)
    rpc_url = os.environ.get("VENICE_X402_SOLANA_RPC_URL", DEFAULT_SOLANA_RPC_URL)
    print(f"   Wallet:  {auth.wallet_address}")
    print(f"   RPC:     {rpc_url}")

    balance = await _usdc_balance(rpc_url, auth.wallet_address)
    if balance is not None:
        print(f"   On-chain USDC balance: ${balance:.4f}")

    # Safety gate: only settle real funds when explicitly opted in.
    if os.environ.get("X402_SOLANA_DO_TOPUP") != "1":
        print(f"\n   💤 Dry run — not submitting. This would top up ${TOPUP_AMOUNT_USDC:.2f}")
        print("      (real, irreversible USDC settlement on Solana mainnet).")
        print("      Set X402_SOLANA_DO_TOPUP=1 to execute it.")
        return

    if balance is not None and balance < TOPUP_AMOUNT_USDC:
        print(f"\n   ⏭️  Wallet holds ${balance:.4f} < ${TOPUP_AMOUNT_USDC:.2f} minimum — skipping.")
        return

    print(f"\n   💸 Submitting a ${TOPUP_AMOUNT_USDC:.2f} top-up (real funds)…")
    async with VeniceClient() as client:  # Bearer auth (VENICE_API_KEY) routes the request
        try:
            result = await client.x402.top_up_with_solana(
                auth=auth,
                amount_usdc=TOPUP_AMOUNT_USDC,
                max_amount_usdc=TOPUP_AMOUNT_USDC,  # refuse to sign for more than this
                rpc_url=rpc_url,
            )
        except (VeniceError, APIError) as e:
            print(f"   ❌ Top-up failed: {e}")
            raise

        data = result.data
        print("   ✅ Top-up settled on-chain")
        print(f"      Wallet:      {data.walletAddress}")
        print(f"      Credited:    ${data.amountCredited:.4f}")
        print(f"      New balance: ${data.newBalance:.4f}")
        print(f"      Payment ID:  {data.paymentId}")


async def main() -> None:
    print("🚀 Venice AI x402 — Solana Settlement Example")
    print("=" * 50)

    await solana_top_up()

    print("\n✨ Done.")
    print("\n💡 Key concepts demonstrated:")
    print("   - SolanaX402Auth(private_key=...) → base58 wallet address")
    print("   - Reading on-chain USDC balance before settling")
    print("   - client.x402.top_up_with_solana(amount_usdc=, max_amount_usdc=)")
    print("   - Dry-run-by-default safety gate for irreversible on-chain spend")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
