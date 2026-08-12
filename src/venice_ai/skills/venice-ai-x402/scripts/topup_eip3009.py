"""One-shot Venice prepaid ledger top-up via x402 (EIP-3009 USDC on Base).

NOTE: On SDK >= 2.0.0, use ``client.x402.top_up_with(auth=auth, amount_usdc=5.0)`` —
this script exists for users on older SDK versions and as an end-to-end
reference for the full probe → sign → submit flow. The SDK's helper does
the same thing with one async call.

Authorized scope:
- Move $5 USDC from VENICE_X402_TEST_PRIVATE_KEY's wallet to Venice's
  settlement address on Base mainnet (chain 8453).
- Single transaction only. Aborts if requirements deviate from expectations.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import secrets
import time

from dotenv import load_dotenv
from eth_account import Account
from eth_account.messages import encode_typed_data

from venice_ai import VeniceClient
from venice_ai.exceptions import PaymentRequiredError

# Load environment from a `.env` file if present in the working directory or
# its parents. Override `DOTENV_PATH` to point elsewhere; default behavior
# matches python-dotenv's `find_dotenv()` walk.
load_dotenv(os.environ.get("DOTENV_PATH") or None)

# Authorized parameters — abort if the 402 disagrees.
EXPECTED_NETWORK = "eip155:8453"  # Base mainnet
EXPECTED_AMOUNT_MAX_USDC_UNITS = 5_000_000  # $5 USDC (6 decimals)
EXPECTED_ASSET_USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"  # USDC on Base


def _require_env(name: str) -> str:
    """Read a required environment variable, or exit with a usable message.

    Callers resolve credentials through this inside ``main()`` rather than at
    module scope, so merely importing this file — as documentation tooling and
    test collection both do — does not die on a bare ``KeyError`` traceback.
    """
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set. Export it (or add it to your .env) and re-run.")
    return value


def _to_checksum(addr: str) -> str:
    from eth_utils.address import to_checksum_address

    checksummed: str = to_checksum_address(addr)
    return checksummed


def find_exact_requirement(body: dict) -> dict:
    """Pick the first 'exact' scheme requirement on the expected network."""
    accepts: list[dict] = (
        body.get("accepts") or body.get("topUpInstructions", {}).get("accepts") or []
    )
    for req in accepts:
        net = req.get("network", "")
        scheme = req.get("scheme", "exact")
        if net == EXPECTED_NETWORK and scheme in ("exact", "evm-exact", "evm/exact"):
            return req
    raise RuntimeError(f"No matching 'exact' requirement on {EXPECTED_NETWORK} found in: {body}")


def build_eip3009_payment(
    *,
    requirement: dict,
    private_key: str,
) -> dict:
    """Build the X-402-Payment v2 envelope for an EIP-3009 transferWithAuthorization."""
    asset = _to_checksum(requirement["asset"])
    pay_to = _to_checksum(requirement["payTo"])
    value = int(requirement["amount"])
    if value > EXPECTED_AMOUNT_MAX_USDC_UNITS:
        raise RuntimeError(
            f"402 wants {value} units (>{EXPECTED_AMOUNT_MAX_USDC_UNITS}); aborting."
        )
    if asset.lower() != EXPECTED_ASSET_USDC.lower():
        raise RuntimeError(f"Asset {asset} != expected {EXPECTED_ASSET_USDC}; aborting.")

    account = Account.from_key(private_key)
    valid_after = 0
    valid_before = int(time.time()) + 600  # 10-minute window
    nonce = "0x" + secrets.token_hex(32)

    typed_data = {
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
                {"name": "verifyingContract", "type": "address"},
            ],
            "TransferWithAuthorization": [
                {"name": "from", "type": "address"},
                {"name": "to", "type": "address"},
                {"name": "value", "type": "uint256"},
                {"name": "validAfter", "type": "uint256"},
                {"name": "validBefore", "type": "uint256"},
                {"name": "nonce", "type": "bytes32"},
            ],
        },
        "primaryType": "TransferWithAuthorization",
        "domain": {
            "name": "USD Coin",
            "version": "2",
            "chainId": 8453,
            "verifyingContract": asset,
        },
        "message": {
            "from": account.address,
            "to": pay_to,
            "value": value,
            "validAfter": valid_after,
            "validBefore": valid_before,
            "nonce": nonce,
        },
    }

    signable = encode_typed_data(full_message=typed_data)
    signed = account.sign_message(signable)
    sig_hex = signed.signature.hex()
    if not sig_hex.startswith("0x"):
        sig_hex = "0x" + sig_hex

    # x402 V2 envelope: the chosen requirement goes under `accepted` (with
    # `maxTimeoutSeconds`), and there is NO top-level `scheme`/`network`. The
    # flat `{x402Version, scheme, network, payload}` shape is rejected 400 by
    # Venice's V2 facilitator.
    accepted = {k: v for k, v in requirement.items() if k not in ("protocol", "version")}
    accepted.setdefault("scheme", "exact")
    accepted["network"] = EXPECTED_NETWORK
    accepted.setdefault("maxTimeoutSeconds", 300)
    payload_v2 = {
        "x402Version": 2,
        "payload": {
            "signature": sig_hex,
            "authorization": {
                "from": account.address,
                "to": pay_to,
                "value": str(value),
                "validAfter": str(valid_after),
                "validBefore": str(valid_before),
                "nonce": nonce,
            },
        },
        "accepted": accepted,
    }
    return payload_v2


def encode_payment_header(payload: dict) -> str:
    """x402 v2 X-402-Payment header: base64(JSON)."""
    return base64.b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()


async def main() -> None:
    api_key = _require_env("VENICE_API_KEY")
    private_key = _require_env("VENICE_X402_TEST_PRIVATE_KEY")

    # 1. Probe to capture exact requirements.
    print("Step 1: probing /x402/top-up for payment requirements...")
    async with VeniceClient(api_key=api_key) as client:
        try:
            await client.x402.top_up()
            raise RuntimeError("Top-up succeeded without payment header — unexpected; aborting.")
        except PaymentRequiredError as e:
            body = e.body or {}
            print(json.dumps(body, indent=2, default=str))

        requirement = find_exact_requirement(body)
        print(
            f"\nMatched requirement: network={requirement.get('network')} "
            f"scheme={requirement.get('scheme', 'exact')} "
            f"amount={requirement.get('amount')} payTo={requirement.get('payTo')}"
        )

        # 2. Sign EIP-3009 transferWithAuthorization.
        print("\nStep 2: signing EIP-3009 transferWithAuthorization...")
        payload = build_eip3009_payment(requirement=requirement, private_key=private_key)
        print(f"  from   = {payload['payload']['authorization']['from']}")
        print(f"  to     = {payload['payload']['authorization']['to']}")
        print(
            f"  value  = {payload['payload']['authorization']['value']} (= ${int(payload['payload']['authorization']['value']) / 1_000_000} USDC)"
        )
        print(f"  validBefore = {payload['payload']['authorization']['validBefore']}")
        header = encode_payment_header(payload)
        print(f"  X-402-Payment (length={len(header)}): {header[:80]}…")

        # 3. Submit signed payment.
        print("\nStep 3: submitting signed payment to Venice...")
        try:
            result = await client.x402.top_up(payment_header=header)
            print("\n=== TOP-UP RESULT ===")
            print(
                json.dumps(
                    result.model_dump() if hasattr(result, "model_dump") else dict(result),
                    indent=2,
                    default=str,
                )
            )
        except PaymentRequiredError as e:
            print("\nServer rejected the signed payment (still 402):")
            print(json.dumps(e.body, indent=2, default=str))
            raise

        # 4. Verify balance moved.
        print("\nStep 4: verifying new balance...")
        from venice_ai.auth.x402 import X402Auth

        auth = X402Auth(private_key=private_key)
        balance = await client.x402.balance(auth=auth)
        print(f"New balance: ${balance.data.balanceUsd}")


if __name__ == "__main__":
    asyncio.run(main())
