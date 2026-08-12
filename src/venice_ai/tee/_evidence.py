"""Wire-schema adaptation for Venice TEE attestations.

Venice has shipped two attestation shapes. The original dstack layout nests the
measurement evidence under ``info``::

    info.tcb_info.event_log   # list of {imr, digest}
    info.tcb_info.app_compose # string
    info.compose_hash         # string

The current layout nests it under a top-level ``attestation`` envelope, with the
event log carried as a JSON *string* rather than a decoded list::

    attestation.evidence.quote
    attestation.evidence.event_log   # JSON string -> list of {imr, digest, ...}
    attestation.evidence.vm_config   # JSON string -> {os_image_hash, image, ...}
    attestation.source_provenance    # {repo_url, repo_commit}

:func:`normalize` maps either shape onto :class:`NormalizedEvidence` so
:class:`~venice_ai.tee._verify.DcapTdxVerifier` holds one code path. Detection is
positive-only: a payload matching neither shape raises rather than guessing.

**Signed vs unsigned.** ``raw_quote`` and the RTMRs it carries are covered by the
quote signature. Everything else here is server-supplied and unsigned.
``event_log`` earns trust only because the verifier replays it and requires it to
reproduce the quote-signed RTMRs. ``os_image_hash``, ``repo_commit`` and
``vm_config`` have no such reconciliation, so they are exposed as informational
metadata only and must never be treated as verified measurements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..exceptions import TeeError

#: The original dstack layout (evidence under ``info``).
SCHEMA_DSTACK = "dstack-info"

#: The current layout (evidence under ``attestation.evidence``).
SCHEMA_EVIDENCE = "evidence"


@dataclass(frozen=True)
class NormalizedEvidence:
    """Schema-independent verification inputs read off an attestation.

    Attributes:
        raw_quote: Decoded Intel TDX quote bytes. Quote-signed.
        event_log: Measurement entries as ``{"imr": int, "digest": hex}`` dicts.
            Unsigned; trustworthy only after RTMR replay reconciles it.
        signing_address: The enclave signing address bound into REPORTDATA.
        sent_nonce: The nonce the client sent, needed for the REPORTDATA binding.
        schema: Which wire shape produced this — :data:`SCHEMA_DSTACK` or
            :data:`SCHEMA_EVIDENCE`.
        compose_hash: dstack only; ``None`` on the current wire.
        app_compose: dstack only; ``None`` on the current wire.
        os_image_hash: Unsigned metadata. Never a verified measurement.
        repo_commit: Unsigned metadata. Never a verified measurement.
        vm_config: Unsigned metadata mapping, when parseable.
    """

    raw_quote: bytes
    event_log: list[dict[str, Any]]
    signing_address: str
    sent_nonce: str | None
    schema: str
    compose_hash: str | None = None
    app_compose: str | None = None
    os_image_hash: str | None = None
    repo_commit: str | None = None
    vm_config: dict[str, Any] | None = None

    @property
    def compose_binding_available(self) -> bool:
        """Whether ``SHA256(app_compose) == compose_hash`` can be checked."""
        return self.compose_hash is not None and self.app_compose is not None


def _mapping(value: Any) -> dict[str, Any] | None:
    """Return ``value`` as a mapping, parsing a JSON string, else ``None``."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    return value if isinstance(value, dict) else None


def _envelope(attestation: Any) -> dict[str, Any] | None:
    """Return the top-level ``attestation`` envelope of the current wire shape."""
    extra = getattr(attestation, "model_extra", None) or {}
    return _mapping(extra.get("attestation"))


def _info(attestation: Any) -> dict[str, Any] | None:
    """Return the dstack ``info`` mapping, from the model or its extras."""
    info = getattr(attestation, "info", None)
    if info is None:
        extra = getattr(attestation, "model_extra", None) or {}
        info = extra.get("info")
    return _mapping(info)


def detect_schema(attestation: Any) -> str:
    """Return which wire shape ``attestation`` uses.

    Raises:
        TeeError: If the payload matches neither known shape (fail-closed — the
            caller must never guess at a layout it cannot identify).
    """
    info = _info(attestation)
    if info is not None and _mapping(info.get("tcb_info")) is not None:
        return SCHEMA_DSTACK

    envelope = _envelope(attestation)
    if envelope is not None and _mapping(envelope.get("evidence")) is not None:
        return SCHEMA_EVIDENCE

    raise TeeError(
        "TEE attestation matches no known evidence schema: expected either "
        "info.tcb_info (dstack) or attestation.evidence (fail-closed)."
    )


def _decode_quote(attestation: Any, evidence: dict[str, Any] | None) -> bytes:
    """Decode the key-binding Intel quote, preferring ``intel_quote``."""
    raw = getattr(attestation, "intel_quote", None)
    if not isinstance(raw, str) or not raw:
        raw = (evidence or {}).get("quote")
    if not isinstance(raw, str) or not raw:
        raise TeeError("TEE attestation intel_quote is missing or empty (fail-closed).")
    if len(raw) % 2 != 0:
        raise TeeError("TEE attestation intel_quote has odd hex length (fail-closed).")
    try:
        return bytes.fromhex(raw)
    except ValueError as exc:
        raise TeeError("TEE attestation intel_quote is not valid hex (fail-closed).") from exc


def _event_log(value: Any) -> list[dict[str, Any]]:
    """Coerce an event log to a list, parsing a JSON string when needed."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError) as exc:
            raise TeeError("TEE attestation event_log is not valid JSON (fail-closed).") from exc
    if not isinstance(value, list):
        raise TeeError("TEE attestation event_log is missing or not a list (fail-closed).")
    return value


def normalize(attestation: Any) -> NormalizedEvidence:
    """Read schema-independent verification inputs off ``attestation``.

    Raises:
        TeeError: On an unrecognized schema, a missing/!hex quote, or an event
            log that is absent, non-JSON, or not a list (all fail-closed).
    """
    schema = detect_schema(attestation)
    signing_address = getattr(attestation, "signing_address", None) or ""
    sent_nonce = getattr(attestation, "sent_nonce", None)

    if schema == SCHEMA_DSTACK:
        info = _info(attestation) or {}
        tcb_info = _mapping(info.get("tcb_info")) or {}
        compose_hash = info.get("compose_hash")
        app_compose = tcb_info.get("app_compose")
        return NormalizedEvidence(
            raw_quote=_decode_quote(attestation, None),
            event_log=_event_log(tcb_info.get("event_log")),
            signing_address=signing_address,
            sent_nonce=sent_nonce,
            schema=schema,
            compose_hash=compose_hash if isinstance(compose_hash, str) else None,
            app_compose=app_compose if isinstance(app_compose, str) else None,
        )

    envelope = _envelope(attestation) or {}
    evidence = _mapping(envelope.get("evidence")) or {}
    vm_config = _mapping(evidence.get("vm_config"))
    provenance = _mapping(envelope.get("source_provenance")) or {}
    os_image_hash = (vm_config or {}).get("os_image_hash")
    repo_commit = provenance.get("repo_commit")
    return NormalizedEvidence(
        raw_quote=_decode_quote(attestation, evidence),
        event_log=_event_log(evidence.get("event_log")),
        signing_address=signing_address,
        sent_nonce=sent_nonce,
        schema=schema,
        os_image_hash=os_image_hash if isinstance(os_image_hash, str) else None,
        repo_commit=repo_commit if isinstance(repo_commit, str) else None,
        vm_config=vm_config,
    )
