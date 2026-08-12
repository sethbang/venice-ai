"""Wire-schema normalization for TEE attestations (both dstack and evidence)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from venice_ai.exceptions import TeeError
from venice_ai.tee._evidence import (
    SCHEMA_DSTACK,
    SCHEMA_EVIDENCE,
    detect_schema,
    normalize,
)
from venice_ai.tee.types import TeeAttestation

_QUOTE_HEX = "aabb" * 8
_ADDR = "0x" + "11" * 20
_NONCE = "22" * 32
_DIGEST = "33" * 48


def _dstack_att() -> TeeAttestation:
    att = TeeAttestation.model_validate(
        {
            "signing_address": _ADDR,
            "signing_algo": "ecdsa",
            "nonce": _NONCE,
            "intel_quote": _QUOTE_HEX,
            "verified": True,
            "info": {
                "compose_hash": "44" * 32,
                "tcb_info": {
                    "app_compose": "services: {}",
                    "event_log": [{"imr": 0, "digest": _DIGEST}],
                },
            },
        }
    )
    att.sent_nonce = _NONCE
    return att


def _evidence_att(event_log: Any = None, vm_config: Any = None) -> TeeAttestation:
    if event_log is None:
        event_log = json.dumps([{"imr": 0, "digest": _DIGEST}])
    if vm_config is None:
        vm_config = json.dumps({"os_image_hash": "55" * 32, "image": "dstack-dev-0.5.9"})
    att = TeeAttestation.model_validate(
        {
            "signing_address": _ADDR,
            "signing_algo": "ecdsa",
            "nonce": _NONCE,
            "intel_quote": _QUOTE_HEX,
            "verified": True,
            "attestation": {
                "evidence": {"quote": _QUOTE_HEX, "event_log": event_log, "vm_config": vm_config},
                "source_provenance": {"repo_commit": "58b027d1"},
            },
        }
    )
    att.sent_nonce = _NONCE
    return att


def test_detects_dstack_schema() -> None:
    assert detect_schema(_dstack_att()) == SCHEMA_DSTACK


def test_detects_evidence_schema() -> None:
    assert detect_schema(_evidence_att()) == SCHEMA_EVIDENCE


def test_both_schemas_present_prefers_dstack() -> None:
    """Schema confusion cannot be used to downgrade trust.

    When a payload carries both ``info.tcb_info`` (dstack) and
    ``attestation.evidence`` (current), the stronger, compose-checked dstack
    path must win — a payload can't smuggle in the weaker evidence-only path
    (which reports ``compose_binding`` as ``"unavailable"``) by also including
    a dstack envelope.
    """
    att = _dstack_att()
    envelope = _evidence_att().model_extra["attestation"]
    att.model_extra["attestation"] = envelope
    assert detect_schema(att) == SCHEMA_DSTACK
    assert normalize(att).compose_binding_available is True


def test_normalizes_dstack() -> None:
    n = normalize(_dstack_att())
    assert n.schema == SCHEMA_DSTACK
    assert n.raw_quote == bytes.fromhex(_QUOTE_HEX)
    assert n.event_log == [{"imr": 0, "digest": _DIGEST}]
    assert n.signing_address == _ADDR
    assert n.sent_nonce == _NONCE
    assert n.app_compose == "services: {}"
    assert n.compose_binding_available is True


def test_normalizes_evidence_and_parses_json_event_log() -> None:
    n = normalize(_evidence_att())
    assert n.schema == SCHEMA_EVIDENCE
    assert n.event_log == [{"imr": 0, "digest": _DIGEST}]
    assert n.os_image_hash == "55" * 32
    assert n.repo_commit == "58b027d1"
    # Compose identity is NOT verifiable on this wire.
    assert n.compose_hash is None
    assert n.app_compose is None
    assert n.compose_binding_available is False


def test_unknown_schema_fails_closed() -> None:
    att = TeeAttestation.model_validate(
        {
            "signing_address": _ADDR,
            "signing_algo": "ecdsa",
            "nonce": _NONCE,
            "intel_quote": _QUOTE_HEX,
            "verified": True,
        }
    )
    with pytest.raises(TeeError, match="matches no known evidence schema"):
        detect_schema(att)


def test_non_json_event_log_fails_closed() -> None:
    with pytest.raises(TeeError, match="event_log is not valid JSON"):
        normalize(_evidence_att(event_log="not json"))


def test_non_list_event_log_fails_closed() -> None:
    with pytest.raises(TeeError, match="event_log is missing or not a list"):
        normalize(_evidence_att(event_log=json.dumps({"imr": 0})))


def test_missing_quote_fails_closed() -> None:
    att = _evidence_att()
    att.intel_quote = None
    # The evidence wire also carries the quote nested under evidence.quote;
    # clear both sources so this exercises the true "no quote anywhere" case.
    att.model_extra["attestation"]["evidence"].pop("quote", None)
    with pytest.raises(TeeError, match="intel_quote is missing or empty"):
        normalize(att)


def test_odd_length_quote_fails_closed() -> None:
    att = _evidence_att()
    att.intel_quote = _QUOTE_HEX + "a"
    with pytest.raises(TeeError, match="intel_quote has odd hex length"):
        normalize(att)


def test_malformed_vm_config_is_tolerated_as_absent() -> None:
    # vm_config is unsigned informational metadata; a bad value must not break
    # verification, it simply yields no os_image_hash.
    n = normalize(_evidence_att(vm_config="not json"))
    assert n.os_image_hash is None
