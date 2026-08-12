#!/usr/bin/env python3
"""
Venice AI SDK - x402: Transaction Ledger
========================================

Demonstrates ``client.x402.transactions(auth=...)`` — Venice's wallet
ledger endpoint. Returns the current balance plus a page of past ledger
entries (top-ups and usage debits) for the SIWE-authenticated wallet.

**Install the optional extra** first::

    pip install 'venice-ai[x402]'

**Private key safety:** Read the key from an environment variable or a
secure vault — never hardcode. Use a test wallet when experimenting.
"""

import asyncio
import os
import sys

from venice_ai import VeniceClient
from venice_ai.exceptions import APIError, VeniceError


async def list_transactions() -> bool:
    """Fetch and print the x402 wallet ledger.

    Returns ``True`` on success — including a clean *gated skip* when the
    optional extra or the wallet key is absent (the expected outcome when no
    x402 wallet is configured). Returns ``False`` only when a real ledger
    lookup fails, so an actual API error surfaces as a non-zero process exit
    instead of being masked by the success banner.
    """
    try:
        from venice_ai.auth.x402 import X402Auth
    except ImportError as e:
        print("⏭️ Missing optional dependency — skipping. Install with:")
        print("   pip install 'venice-ai[x402]'")
        print(f"   (original error: {e})")
        return True

    private_key = os.environ.get("X402_WALLET_PRIVATE_KEY")
    if not private_key:
        print(
            "⏭️ Skipping: X402_WALLET_PRIVATE_KEY not set. Use a test wallet; "
            "never commit a production private key."
        )
        return True

    auth = X402Auth(private_key=private_key)
    print(f"🔑 Wallet: {auth.wallet_address}")

    async with VeniceClient() as client:
        try:
            result = await client.x402.transactions(auth=auth)
        except (VeniceError, APIError) as e:
            print(f"❌ Transactions lookup failed: {e}")
            return False

        data = result.data
        print("\n📜 Transaction Ledger")
        print("-" * 30)
        print(f"   Current balance:  ${data.currentBalance:.4f}")
        print(f"   Entries on page:  {len(data.transactions)}")
        print(
            f"   Pagination:       limit={data.pagination.limit}, "
            f"offset={data.pagination.offset}, hasMore={data.pagination.hasMore}"
        )

        if not data.transactions:
            print("\n   (no entries yet — try top_up.py first)")
            return True

        print("\n   Recent entries (newest first):")
        for entry in data.transactions[:10]:
            sign = "+" if entry.amount >= 0 else ""
            print(
                f"   • {entry.createdAt}  {entry.type:<10} "
                f"{sign}${entry.amount:.4f}  (balance after: ${entry.balanceAfter:.4f})"
            )
            if entry.modelId:
                print(f"     model: {entry.modelId}")
            if entry.requestId:
                print(f"     request: {entry.requestId}")

    return True


async def main() -> int:
    """Run the x402 transactions demo.

    Returns ``0`` if the demo succeeded (including a clean gated skip) and
    ``1`` if the ledger lookup failed, so a real API error surfaces as a
    non-zero process exit instead of being masked by the success banner.
    """
    print("🚀 Venice AI x402 — Transactions Example")
    print("=" * 50)

    results: list[tuple[str, bool]] = [
        ("list_transactions", await list_transactions()),
    ]

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("\n✨ Done.")

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
