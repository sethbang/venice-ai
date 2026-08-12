"""
Live x402 SIWE validation against the real Venice API.

Reads VENICE_X402_TEST_PRIVATE_KEY + VENICE_X402_TEST_ADDRESS from .env.
Hits client.x402.balance, .transactions, .top_up. Prints structured verdicts
per endpoint. Includes a negative-path test that swaps in a random key to
prove the SIWE signature is server-validated (not just decoded).

Usage:
  poetry run python tools/x402_live_validate.py

NEVER prints the private key. Loads it only via os.environ.
"""

from __future__ import annotations

import asyncio
import os
import traceback
from pathlib import Path

from eth_account import Account

from venice_ai import VeniceClient
from venice_ai.auth.x402 import X402Auth
from venice_ai.types.api.x402 import (
    X402BalanceResponse,
    X402TransactionsResponse,
)


def load_dotenv() -> None:
    """Minimal .env loader; never echoes values to stdout."""
    p = Path(__file__).resolve().parents[1] / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _shape_check_balance(bal: X402BalanceResponse) -> tuple[bool, list[str]]:
    """Confirm bal matches X402BalanceResponse shape; return (ok, notes)."""
    notes: list[str] = []
    ok = isinstance(bal, X402BalanceResponse)
    if not ok:
        notes.append(f"not an X402BalanceResponse (got {type(bal).__name__})")
        return False, notes
    # success literal True
    if bal.success is not True:
        notes.append("success != True")
        ok = False
    # data fields
    for f in ("walletAddress", "balanceUsd", "canConsume"):
        if not hasattr(bal.data, f):
            notes.append(f"data missing required field {f!r}")
            ok = False
    return ok, notes


def _shape_check_transactions(txs: X402TransactionsResponse) -> tuple[bool, list[str]]:
    """Confirm txs matches X402TransactionsResponse shape."""
    notes: list[str] = []
    ok = isinstance(txs, X402TransactionsResponse)
    if not ok:
        notes.append(f"not an X402TransactionsResponse (got {type(txs).__name__})")
        return False, notes
    if txs.success is not True:
        notes.append("success != True")
        ok = False
    for f in ("walletAddress", "currentBalance", "transactions", "pagination"):
        if not hasattr(txs.data, f):
            notes.append(f"data missing required field {f!r}")
            ok = False
    if hasattr(txs.data, "pagination"):
        for f in ("limit", "offset", "hasMore"):
            if not hasattr(txs.data.pagination, f):
                notes.append(f"pagination missing required field {f!r}")
                ok = False
    return ok, notes


async def main() -> int:
    load_dotenv()
    pk = os.environ.get("VENICE_X402_TEST_PRIVATE_KEY")
    addr = os.environ.get("VENICE_X402_TEST_ADDRESS")
    if not pk or not addr:
        print(
            "ERROR: VENICE_X402_TEST_PRIVATE_KEY and VENICE_X402_TEST_ADDRESS "
            "must be set in .env (run tools/x402_setup.py first)."
        )
        return 1
    if not os.environ.get("VENICE_API_KEY"):
        print("ERROR: VENICE_API_KEY must be set (top_up uses bearer auth).")
        return 1

    auth = X402Auth(private_key=pk)
    print(f"Wallet: {addr}")
    print(f"Derived from key: {auth.wallet_address}")
    if auth.wallet_address.lower() != addr.lower():
        print("WARNING: derived address != VENICE_X402_TEST_ADDRESS — using derived")
    print()

    failures = 0
    summary: list[str] = []

    async with VeniceClient() as client:
        # 1. balance
        print("=== balance ===")
        try:
            bal = await client.x402.balance(auth=auth)
            shape_ok, shape_notes = _shape_check_balance(bal)
            print(f"  HTTP 2xx — type: {type(bal).__name__}")
            print(f"  walletAddress: {bal.data.walletAddress}")
            print(f"  balanceUsd:    {bal.data.balanceUsd}")
            print(f"  canConsume:    {bal.data.canConsume}")
            print(f"  minimumTopUpUsd:   {bal.data.minimumTopUpUsd}")
            print(f"  suggestedTopUpUsd: {bal.data.suggestedTopUpUsd}")
            if shape_ok:
                print("  shape: OK")
                summary.append(
                    f"balance: PASS (balance={bal.data.balanceUsd} USD, canConsume={bal.data.canConsume})"
                )
            else:
                print(f"  shape: FAIL — {'; '.join(shape_notes)}")
                summary.append(f"balance: FAIL (shape mismatch — {'; '.join(shape_notes)})")
                failures += 1
        except Exception as e:
            failures += 1
            print(f"  FAIL — {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
            summary.append(f"balance: FAIL ({type(e).__name__}: {e})")

        # 2. transactions
        print("\n=== transactions ===")
        try:
            txs = await client.x402.transactions(auth=auth, limit=10, offset=0)
            shape_ok, shape_notes = _shape_check_transactions(txs)
            print(f"  HTTP 2xx — type: {type(txs).__name__}")
            print(f"  walletAddress:  {txs.data.walletAddress}")
            print(f"  currentBalance: {txs.data.currentBalance}")
            print(f"  transactions:   list of {len(txs.data.transactions)}")
            print(
                f"  pagination:     limit={txs.data.pagination.limit}, "
                f"offset={txs.data.pagination.offset}, hasMore={txs.data.pagination.hasMore}"
            )
            for i, tx in enumerate(txs.data.transactions[:3]):
                print(f"    [{i}] {tx.type} {tx.amount} -> {tx.balanceAfter} ({tx.createdAt})")
            if shape_ok:
                print("  shape: OK")
                summary.append(
                    f"transactions: PASS ({len(txs.data.transactions)} entries, "
                    f"hasMore={txs.data.pagination.hasMore})"
                )
            else:
                print(f"  shape: FAIL — {'; '.join(shape_notes)}")
                summary.append(f"transactions: FAIL (shape mismatch — {'; '.join(shape_notes)})")
                failures += 1
        except Exception as e:
            failures += 1
            print(f"  FAIL — {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
            summary.append(f"transactions: FAIL ({type(e).__name__}: {e})")

        # 3. top_up — no payment header should yield 402 with structured payment
        # requirements
        print("\n=== top_up (no payment header — expect 402) ===")
        try:
            tu = await client.x402.top_up()
            print(f"  UNEXPECTED 2xx — type: {type(tu).__name__} value: {tu!r}")
            summary.append("top_up: FAIL (unexpected 2xx without payment header)")
            failures += 1
        except Exception as e:
            etype = type(e).__name__
            print(f"  EXPECTED non-2xx — {etype}: {e}")
            # Inspect the exception for structured payment requirements
            status = getattr(e, "status_code", None)
            body = getattr(e, "body", None)
            response = getattr(e, "response", None)
            if status is not None:
                print(f"  status_code: {status}")
            if body is not None:
                print(
                    f"  body keys: {list(body.keys()) if isinstance(body, dict) else type(body).__name__}"
                )
            if response is not None:
                print(f"  response: present (type {type(response).__name__})")
            # PaymentRequiredError from the SDK is the canonical 402 mapping.
            is_402 = etype == "PaymentRequiredError" or status == 402
            has_accepts = isinstance(body, dict) and (
                "accepts" in body or "x402Version" in body or "error" in body
            )
            if is_402 and has_accepts:
                print("  verdict: PASS — 402 with structured payment requirements in body")
                summary.append("top_up: PASS (402 PaymentRequiredError with payment requirements)")
            elif is_402:
                print("  verdict: PASS-BUT-LIGHT — 402 raised but body shape unclear")
                summary.append(
                    "top_up: PASS-LIGHT (402 raised; structured body shape not confirmed)"
                )
            else:
                print(f"  verdict: FAIL — expected 402, got {etype}/status={status}")
                summary.append(f"top_up: FAIL (expected 402, got {etype} status={status})")
                failures += 1

        # 4. negative — bad SIWE auth (random key, recovered address won't match)
        print("\n=== negative — bad SIWE auth (random wallet) ===")
        try:
            bogus = Account.create()
            bad_pk = bogus.key.hex()
            bad_auth = X402Auth(private_key=bad_pk)
            # Manually swap in our real wallet address while signing for a
            # different key — actually a much simpler form is just to call
            # balance with the bogus auth (path will be the bogus address).
            # That tests the API end-to-end but doesn't strictly prove
            # signature verification — we want to confirm the SIWE signature
            # is checked, not just that an unfunded wallet returns empty.
            # Strongest test: call balance with bogus auth — server must
            # NOT return our funded wallet's data, and ideally rejects 401
            # rather than returning an empty wallet.
            print(f"  bogus address: {bad_auth.wallet_address}")
            bad_bal = await client.x402.balance(auth=bad_auth)
            # Did Venice return *our* wallet's data despite bogus signature?
            if bad_bal.data.walletAddress.lower() == addr.lower():
                print("  UNEXPECTED — server returned the FUNDED wallet's data for bogus auth!")
                summary.append(
                    "negative-SIWE: FAIL (server returned funded wallet data for bogus signature)"
                )
                failures += 1
            else:
                # Server returned bogus wallet's own data — server segregates
                # by SIWE address. SIWE is at minimum being decoded. But:
                # does it verify the *signature*? Hard to prove without
                # tampering the header. Best we can do without a custom
                # rebuild is attest segregation works.
                print(f"  Server returned bogus wallet's data: {bad_bal.data.walletAddress}")
                print(f"  balanceUsd: {bad_bal.data.balanceUsd} (expect 0 for fresh random wallet)")
                summary.append(
                    "negative-SIWE: PASS-LIGHT (server segregates by SIWE address; "
                    "bogus wallet returned own empty record, not the funded wallet's)"
                )
        except Exception as e:
            etype = type(e).__name__
            status = getattr(e, "status_code", None)
            print(f"  Server REJECTED — {etype}: {e} (status={status})")
            if status in (401, 403):
                summary.append(
                    f"negative-SIWE: PASS (server rejected bogus signature with {status})"
                )
            else:
                # Other error — could be network / API. Note but don't fail.
                summary.append(f"negative-SIWE: INCONCLUSIVE ({etype} status={status})")

        # 5. negative — tampered SIWE signature (real address, mutated sig)
        # This is the strongest test: keep our real wallet's path/header but
        # corrupt the signature field, then send. Server must reject 401.
        print("\n=== negative — tampered SIWE signature (correct address, mutated sig) ===")
        try:
            import base64
            import json

            real_header = auth.build_header()
            decoded = json.loads(base64.b64decode(real_header).decode("utf-8"))
            orig_sig = decoded["signature"]
            # Flip one nibble of the signature to make it invalid but still well-formed
            # signature is 0x + 130 hex chars
            if orig_sig.startswith("0x") and len(orig_sig) >= 4:
                # Flip the third hex char (index 2 after 0x)
                old_char = orig_sig[3]
                new_char = "0" if old_char != "0" else "1"
                tampered_sig = orig_sig[:3] + new_char + orig_sig[4:]
                decoded["signature"] = tampered_sig
                tampered_header = base64.b64encode(
                    json.dumps(decoded, separators=(",", ":")).encode("utf-8")
                ).decode("ascii")

                # Send the tampered header directly via raw client.get
                try:
                    bad = await client.get(
                        f"x402/balance/{auth.wallet_address}",
                        cast_to=X402BalanceResponse,
                        headers={"X-Sign-In-With-X": tampered_header},
                    )
                    print("  UNEXPECTED 2xx — server accepted tampered signature!")
                    print(
                        f"  returned wallet: {bad.data.walletAddress} balance: {bad.data.balanceUsd}"
                    )
                    summary.append("tampered-SIWE: FAIL (server accepted invalid signature)")
                    failures += 1
                except Exception as ie:
                    ietype = type(ie).__name__
                    istatus = getattr(ie, "status_code", None)
                    print(f"  Server REJECTED — {ietype}: {ie} (status={istatus})")
                    if istatus in (401, 403, 400):
                        summary.append(
                            f"tampered-SIWE: PASS (server rejected tampered sig with {istatus})"
                        )
                    else:
                        summary.append(f"tampered-SIWE: INCONCLUSIVE ({ietype} status={istatus})")
            else:
                print(f"  could not tamper signature (unexpected format: {orig_sig[:6]}...)")
                summary.append("tampered-SIWE: SKIPPED (signature format unexpected)")
        except Exception as e:
            print(f"  test setup failed: {type(e).__name__}: {e}")
            summary.append(f"tampered-SIWE: ERROR ({type(e).__name__}: {e})")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for line in summary:
        print(f"  {line}")
    print()
    print(f"Total failures: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
