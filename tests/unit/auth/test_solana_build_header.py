"""TDD: SolanaX402Auth.build_header — the SIGN-IN-WITH-X auth header (audit MED #11).

Client-side correctness only (structure + a verifiable ed25519 signature over the
SIWS message). Server acceptance is confirmed separately by a live /x402/balance
probe with the funded test key.
"""

import base64
import json

import pytest

solders = pytest.importorskip("solders")
from solders.keypair import Keypair  # noqa: E402
from solders.signature import Signature  # noqa: E402

from venice_ai.auth.x402_solana import SolanaX402Auth  # noqa: E402


def _auth():
    kp = Keypair()
    return SolanaX402Auth(private_key=str(kp)), kp


def test_build_header_structure_and_fields():
    auth, kp = _auth()
    obj = json.loads(base64.b64decode(auth.build_header(nonce="0123456789abcdef")))
    assert obj["address"] == auth.wallet_address == str(kp.pubkey())
    assert obj["type"] == "ed25519"
    assert obj["chainId"].startswith("solana:")
    assert isinstance(obj["timestamp"], int)
    assert "wants you to sign in with your Solana account:" in obj["message"]
    assert "Nonce: 0123456789abcdef" in obj["message"]


def test_build_header_signature_verifies():
    auth, kp = _auth()
    obj = json.loads(base64.b64decode(auth.build_header()))
    sig = Signature.from_string(obj["signature"])
    assert sig.verify(kp.pubkey(), obj["message"].encode("utf-8"))


def test_x402_siwe_headers_accept_solana_auth():
    # The SDK's header builder (used by client.x402.balance/transactions) must
    # accept a SolanaX402Auth and emit the X-Sign-In-With-X header.
    from venice_ai.resources.x402 import _siwe_headers

    auth, _ = _auth()
    headers = _siwe_headers(auth)
    assert headers.get("X-Sign-In-With-X")
