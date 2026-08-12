"""DcapTdxVerifier against the current ``attestation.evidence`` wire shape.

Mirrors the dstack rejection suite: crypto-layer rejections byte-flip the real
capture, and the happy path comes last.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from venice_ai.exceptions import TeeError
from venice_ai.tee._evidence import SCHEMA_EVIDENCE
from venice_ai.tee._verify import DcapTdxVerifier
from venice_ai.tee.types import TeeAttestation

from .fixtures import EVIDENCE_CAPTURE_EPOCH, EVIDENCE_SLUG, load_attestation, load_collateral_json

dcap_qvl = pytest.importorskip("dcap_qvl")


def _attestation() -> TeeAttestation:
    raw = load_attestation(EVIDENCE_SLUG)
    att = TeeAttestation.model_validate(raw["attestation"])
    att.sent_nonce = raw["_client_nonce"]
    return att


def _verifier(**kwargs: Any) -> DcapTdxVerifier:
    kwargs.setdefault(
        "collateral", dcap_qvl.QuoteCollateralV3.from_json(load_collateral_json(EVIDENCE_SLUG))
    )
    kwargs.setdefault("now_secs", EVIDENCE_CAPTURE_EPOCH)
    return DcapTdxVerifier(**kwargs)


def test_tampered_signature_byte_rejects() -> None:
    att = _attestation()
    raw = bytearray(bytes.fromhex(att.intel_quote))
    raw[650] ^= 0xFF
    att.intel_quote = raw.hex()
    with pytest.raises(TeeError):
        _verifier().verify(att)


def test_wrong_root_ca_rejects() -> None:
    att = _attestation()
    with pytest.raises(TeeError):
        _verifier(root_ca_der=b"\x30\x82\x00\x10" + b"\x00" * 12).verify(att)


def test_stale_collateral_rejects() -> None:
    att = _attestation()
    with pytest.raises(TeeError):
        _verifier(now_secs=EVIDENCE_CAPTURE_EPOCH + 10 * 365 * 86400).verify(att)


def test_truncated_quote_rejects() -> None:
    att = _attestation()
    att.intel_quote = att.intel_quote[:200]
    with pytest.raises(TeeError):
        _verifier().verify(att)


def test_wrong_nonce_rejects() -> None:
    att = _attestation()
    att.sent_nonce = "ff" * 32
    with pytest.raises(TeeError, match="REPORTDATA binding mismatch"):
        _verifier().verify(att)


def test_corrupt_event_log_digest_rejects_rtmr_replay() -> None:
    att = _attestation()
    envelope = copy.deepcopy(att.model_extra["attestation"])
    entries = json.loads(envelope["evidence"]["event_log"])
    for entry in entries:
        if entry.get("imr") == 0:
            d = bytearray(bytes.fromhex(entry["digest"]))
            d[0] ^= 0xFF
            entry["digest"] = d.hex()
            break
    envelope["evidence"]["event_log"] = json.dumps(entries)
    att.model_extra["attestation"] = envelope
    with pytest.raises(TeeError, match="event-log replay does not reproduce"):
        _verifier().verify(att)


def test_collateral_none_rejects() -> None:
    att = _attestation()
    with pytest.raises(TeeError):
        DcapTdxVerifier(collateral=None, now_secs=EVIDENCE_CAPTURE_EPOCH).verify(att)


def test_happy_path_verifies_tier_b() -> None:
    att = _attestation()
    v = _verifier()
    assert v.verify(att) is True
    r = v.last_result
    assert r is not None
    assert r["schema"] == SCHEMA_EVIDENCE
    assert r["checks"]["signature_chain"] is True
    assert r["checks"]["reportdata_binding"] is True
    assert r["checks"]["rtmr_replay"] is True
    # Compose identity is not establishable on this wire.
    assert r["checks"]["compose_binding"] == "unavailable"
    assert r["workload_identity_pinned"] is False


def test_unavailable_compose_is_not_reported_as_passing() -> None:
    """The guard against a consumer reading the truthy string as success."""
    att = _attestation()
    v = _verifier()
    v.verify(att)
    assert v.last_result is not None
    assert (v.last_result["checks"]["compose_binding"] is True) is False


def test_expected_compose_hash_rejects_when_wire_omits_it() -> None:
    att = _attestation()
    with pytest.raises(TeeError, match="does not carry a compose hash"):
        _verifier(expected_compose_hash="ab" * 32).verify(att)


def test_mrtd_pinning_still_works() -> None:
    att = _attestation()
    q = dcap_qvl.parse_quote(bytes.fromhex(att.intel_quote))
    mrtd = bytes(q.report.mr_td).hex()
    v = _verifier(expected_measurements={"mrtd": mrtd})
    assert v.verify(att) is True
    assert v.last_result is not None
    assert v.last_result["workload_identity_pinned"] is True


def _load_genuine_unsigned_metadata() -> dict[str, str]:
    """Read the fixture's real (unsigned) ``os_image_hash`` / ``repo_commit``.

    Read at test time from the committed fixture rather than hardcoded, so a
    future re-capture of ``EVIDENCE_SLUG`` can't silently drift this test out
    of sync with the fixture it claims to exercise.
    """
    raw = load_attestation(EVIDENCE_SLUG)
    inner = raw["attestation"]["attestation"]
    vm_config = json.loads(inner["evidence"]["vm_config"])
    os_image_hash = vm_config["os_image_hash"]
    repo_commit = inner["source_provenance"]["repo_commit"]
    assert os_image_hash, "fixture os_image_hash must be non-empty"
    assert repo_commit, "fixture repo_commit must be non-empty"
    return {"os_image_hash": os_image_hash, "repo_commit": repo_commit}


_GENUINE_UNSIGNED_METADATA = _load_genuine_unsigned_metadata()


@pytest.mark.parametrize("key", ["os_image_hash", "repo_commit"])
@pytest.mark.parametrize("value_kind", ["genuine", "garbage"])
def test_unsigned_metadata_is_not_pinnable(key: str, value_kind: str) -> None:
    """Unsigned body fields must be rejected as unknown ``expected_measurements`` keys.

    The design forbids ``os_image_hash`` / ``repo_commit`` from being Tier-A
    dimensions: they live in the attestation JSON body, which the quote
    signature does not cover, so pinning them would let the server assert its
    own compliance. ``_check_expected`` (``_verify.py``) raises on an unknown
    key BEFORE any value comparison, so the rejection is value-independent:
    supplying the fixture's *genuine* value must raise identically to
    supplying pure garbage. Both are parametrized here and matched against the
    unknown-key message specifically, so a regression that instead let
    ``os_image_hash`` become a recognized key (compared against some quote
    field) would show up as a mismatched ``match=`` rather than silently
    passing.
    """
    att = _attestation()
    value = _GENUINE_UNSIGNED_METADATA[key] if value_kind == "genuine" else "deadbeef"
    with pytest.raises(TeeError, match="Unknown expected_measurements key"):
        _verifier(expected_measurements={key: value}).verify(att)
