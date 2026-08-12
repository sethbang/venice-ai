"""Fixture-corpus smoke proof: the committed captures + collateral + pinned root
verify OFFLINE and deterministically with ``dcap-qvl``.

This is NOT the ``DcapTdxVerifier`` (that lands later); it validates that the
fixtures are real, internally consistent, and usable so every later test can
trust them. Each capture is paired with its OWN collateral snapshot and the
frozen capture epoch as ``now_secs``.
"""

from __future__ import annotations

import hashlib

import pytest

from venice_ai.tee._attestation import _expected_reportdata
from venice_ai.tee._constants import (
    INTEL_SGX_ROOT_CA_DER,
    INTEL_SGX_ROOT_CA_SHA256,
)

from .fixtures import (
    CAPTURE_EPOCH,
    FIXTURE_SLUGS,
    load_attestation,
    load_collateral_json,
)

dcap_qvl = pytest.importorskip("dcap_qvl", reason="requires venice-ai[e2ee-verify]")

#: TCB statuses dcap-qvl reports for a current, unrevoked platform. The Verifier
#: phase pins the exact accept/reject policy; here we only assert the corpus is
#: in an UpToDate-class state so later happy-path tests are not built on a
#: degraded platform.
_UP_TO_DATE_CLASS = {"UpToDate"}


def test_pinned_root_ca_integrity() -> None:
    """The baked Intel SGX Root CA DER hashes to its pinned SHA-256."""
    assert hashlib.sha256(INTEL_SGX_ROOT_CA_DER).hexdigest() == INTEL_SGX_ROOT_CA_SHA256


@pytest.mark.parametrize("slug", FIXTURE_SLUGS)
def test_capture_verifies_offline_against_pinned_root(slug: str) -> None:
    """Each capture verifies offline: signature + PCK-chain-to-pinned-root + TCB
    is UpToDate, AND the parsed MRTD matches the server-reported measurement."""
    payload = load_attestation(slug)
    att = payload["attestation"]
    assert payload["_capture_epoch"] == CAPTURE_EPOCH

    intel_quote = att["intel_quote"]
    assert isinstance(intel_quote, str) and intel_quote, f"{slug}: empty intel_quote"
    raw = bytes.fromhex(intel_quote)

    # Structural parse (no signature check yet).
    quote = dcap_qvl.parse_quote(raw)
    assert quote.is_tdx() is True
    assert quote.fmspc() == "90C06F000000"

    # Trust anchor: offline verify against the PINNED Intel SGX Root CA + frozen
    # now_secs + this capture's own committed collateral.
    collateral = dcap_qvl.QuoteCollateralV3.from_json(load_collateral_json(slug))
    verified = dcap_qvl.verify_with_root_ca(
        raw,
        collateral,
        INTEL_SGX_ROOT_CA_DER,
        CAPTURE_EPOCH,
    )
    assert verified.status in _UP_TO_DATE_CLASS, f"{slug}: TCB status {verified.status!r}"

    # MRTD parsed from the (now cryptographically verified) quote must equal the
    # server-reported measurement — proves the fixture is internally consistent.
    parsed_mrtd = quote.report.mr_td.hex().lower()
    server_mrtd = att["server_verification"]["tdx"]["measurements"]["mrtd"].lower()
    assert parsed_mrtd == server_mrtd, f"{slug}: MRTD mismatch"


@pytest.mark.parametrize("slug", FIXTURE_SLUGS)
def test_intel_quote_reportdata_binds_saved_nonce(slug: str) -> None:
    """``intel_quote`` (NOT ``quote``) is the quote whose REPORTDATA binds the
    signing key to the SAVED client nonce.

    The verifier phase recomputes this exact binding, so the corpus must carry a
    nonce that actually matches the quote's ``report_data`` —
    ``verify_with_root_ca`` does NOT check ``report_data``, so this is asserted
    here separately. Holds even for the two models where ``intel_quote != quote``.
    """
    payload = load_attestation(slug)
    att = payload["attestation"]
    quote = dcap_qvl.parse_quote(bytes.fromhex(att["intel_quote"]))
    actual = quote.report.report_data.hex().lower()
    expected = _expected_reportdata(att["signing_address"], payload["_client_nonce"]).lower()
    assert actual == expected, f"{slug}: intel_quote REPORTDATA does not bind the saved nonce"
