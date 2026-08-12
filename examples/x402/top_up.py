#!/usr/bin/env python3
"""
Venice AI SDK - x402: Top-Up
============================

Demonstrates ``client.x402.top_up(payment_header=...)`` — Venice's x402
payment-channel top-up endpoint. Unlike ``balance`` and ``transactions``,
top-up uses **Bearer auth** (``VENICE_API_KEY``) plus an optional
``X-402-Payment`` header that carries the signed x402 payment payload.

Two flows:

1. **Discover payment requirements** — an empty POST returns a ``402
   Payment Required`` with the supported chains, assets, and amounts
   the server expects. The SDK surfaces this as a :class:`APIError` whose
   response body contains the requirements; the body shape is the
   standard x402 v2 discovery envelope.

2. **Send a signed payment** — once you've signed the x402 payload
   out-of-band (x402 payment signing is outside this SDK's scope), pass
   the base64 value via ``payment_header=``.
"""

import asyncio
import json
import sys

from venice_ai import VeniceClient
from venice_ai.exceptions import APIError, VeniceError


async def discover_payment_requirements() -> None:
    """Empty POST — server returns 402 with the requirements."""
    print("🔍 Discover x402 Payment Requirements")
    print("-" * 30)

    async with VeniceClient() as client:
        try:
            result = await client.x402.top_up()
            # In the unlikely event the server returns 200 (e.g. in a dry-run
            # environment), we have a credited response instead.
            print(f"✅ Top-up succeeded unexpectedly: {result.data.amountCredited}")
        except (VeniceError, APIError) as e:
            # 402 Payment Required is the documented happy path here. The
            # APIError surfaces both the human-readable message and the
            # structured x402 v2 discovery envelope on `e.body`.
            print("📬 Server responded with payment requirements (expected):")
            print(f"   message: {e}")
            body = getattr(e, "body", None)
            if body is None:
                print("   (no structured body attached)")
            elif isinstance(body, (dict, list)):
                print("   discovery envelope:")
                print(json.dumps(body, indent=2))
            else:
                print(f"   body ({type(body).__name__}): {body}")


async def send_signed_payment(signed_payload_b64: str) -> None:
    """Send a pre-signed x402 payment payload.

    The signing step (SIWE over the payment object) is out-of-scope for
    this SDK; build it with ``@venice-ai/x402-client`` or an equivalent
    library, then pass the resulting base64 string here.
    """
    print("\n💳 Send Signed x402 Payment")
    print("-" * 30)

    async with VeniceClient() as client:
        try:
            result = await client.x402.top_up(payment_header=signed_payload_b64)
        except (VeniceError, APIError) as e:
            print(f"❌ Top-up failed: {e}")
            return

        data = result.data
        print("✅ Top-up completed")
        print(f"   Wallet:         {data.walletAddress}")
        print(f"   Credited:       ${data.amountCredited:.4f}")
        print(f"   New balance:    ${data.newBalance:.4f}")
        print(f"   Payment ID:     {data.paymentId}")


async def main() -> None:
    print("🚀 Venice AI x402 — Top-Up Example")
    print("=" * 50)

    await discover_payment_requirements()

    print("\n💡 To complete a real top-up:")
    print("   1. Build and sign the x402 payment payload off-SDK")
    print("      (e.g. via @venice-ai/x402-client or equivalent).")
    print("   2. Pass the base64-encoded token to:")
    print("        await client.x402.top_up(payment_header=token)")
    print("   3. Confirm the new balance with examples/x402/balance.py.")

    # Uncomment and supply a real signed token to exercise the full flow:
    # await send_signed_payment("eyJ4NDAyVmVyc2lvbiI6Mn0=")

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
