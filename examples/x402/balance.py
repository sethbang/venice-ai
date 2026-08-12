#!/usr/bin/env python3
"""
Venice AI SDK - x402: Wallet Balance
====================================

Demonstrates ``client.x402.balance(auth=...)`` — Venice's wallet-based
billing endpoint that returns the current prepaid USDC balance for the
wallet authenticated via SIWE (Sign-In-With-Ethereum / EIP-4361).

**Install the optional extra** first::

    pip install 'venice-ai[x402]'

This pulls ``eth-account`` and ``siwe`` for EIP-4361 message signing.

**Private key safety:** Read the key from an environment variable or a
secure vault — never hardcode it. The example below uses
``X402_WALLET_PRIVATE_KEY`` from the environment.
"""

import asyncio
import os
import sys

from venice_ai import VeniceClient
from venice_ai.exceptions import VeniceError


async def show_balance() -> None:
    """Read the wallet balance with SIWE auth."""
    # ---- import x402 auth lazily so the example gives a friendly error
    # if the optional extra isn't installed ------------------------------
    try:
        from venice_ai.auth.x402 import X402Auth
    except ImportError as e:
        print("❌ Missing optional dependency. Install with:")
        print("   pip install 'venice-ai[x402]'")
        print(f"   (original error: {e})")
        return

    private_key = os.environ.get("X402_WALLET_PRIVATE_KEY")
    if not private_key:
        print("❌ X402_WALLET_PRIVATE_KEY is not set.")
        print(
            "   Export your wallet's private key first, e.g.:\n"
            "     export X402_WALLET_PRIVATE_KEY=0xYOUR_PRIVATE_KEY"
        )
        print("   (Use a test wallet, not your main key.)")
        return

    auth = X402Auth(private_key=private_key)
    print(f"🔑 Wallet: {auth.wallet_address}")

    async with VeniceClient() as client:
        try:
            balance = await client.x402.balance(auth=auth)
        except VeniceError as e:
            print(f"❌ Balance lookup failed: {e}")
            return

        data = balance.data
        print("\n💰 x402 Balance")
        print("-" * 30)
        print(f"   Address:        {data.walletAddress}")
        print(f"   Balance (USD):  ${data.balanceUsd:.4f}")
        print(f"   Can consume?    {data.canConsume}")
        if data.minimumTopUpUsd is not None:
            print(f"   Suggested min:  ${data.minimumTopUpUsd:.2f}")
        if data.suggestedTopUpUsd is not None:
            print(f"   Suggested top-up: ${data.suggestedTopUpUsd:.2f}")
        if data.diemBalanceUsd is not None:
            print(f"   Diem balance:   ${data.diemBalanceUsd:.4f}")


async def main() -> None:
    print("🚀 Venice AI x402 — Balance Example")
    print("=" * 50)
    await show_balance()
    print("\n✨ Done.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
