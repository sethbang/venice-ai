"""Baseline TEE attestation verification (fail-closed).

This module verifies a :class:`~venice_ai.tee.types.TeeAttestation` returned by
``GET /tee/attestation`` against the client-generated nonce. The baseline checks
(live-verified against the Venice API on 2026-06-04; see
``tests/unit/tee/fixtures/attestation_gemma.json`` for the captured response)
are:

1. ``verified == true`` — the server's own verification claim.
2. **Nonce echo** — the attestation's echoed ``nonce`` equals the
   client-generated ``sent_nonce``. (Freshness; non-vacuous only because
   ``sent_nonce`` is supplied by the caller, not read back from the response.)
3. **REPORTDATA binding** — ``server_verification.tdx.reportData`` (64 bytes,
   128 hex) equals ``signing_address``(20B, lowercased, no ``0x``) ``|| 12 zero
   bytes || sent_nonce``(32B). Binds the model encryption key to the enclave and
   to this fresh request.
4. **TDX debug-mode flag rejected** — bit 0 (``TUD.DEBUG``) of the little-endian
   ``measurements.tdAttributes`` field must be clear.

.. warning::
    **SECURITY LIMITATION — the baseline trusts Venice's server-side**
    ``verified`` **claim.** It does **NOT** independently parse and validate the
    raw Intel TDX quote signature / X.509 certificate chain to Intel roots, and
    it does **NOT** verify the NVIDIA GPU attestation via NVIDIA's NRAS. A
    compromised or malicious Venice endpoint that forges ``verified:true``
    together with a self-consistent ``reportData`` (it controls both fields in
    the response it returns) would **NOT** be detected by baseline verification.
    The nonce/REPORTDATA binding proves freshness and key-to-enclave binding *as
    asserted by the server*, but not cryptographically to Intel/NVIDIA roots
    independently of the server.

    Closing this gap for Intel TDX is shipped: :class:`venice_ai.tee.DcapTdxVerifier`
    (the optional ``[e2ee-verify]`` extra) is a concrete :class:`FullQuoteVerifier`
    that verifies the Intel TDX quote signature + PCK cert-chain-to-Intel-root +
    TCB status offline. Pass it via ``verifier=`` (or
    ``TeeOptions(verifier=...)``); it runs after these baseline checks. NVIDIA GPU
    attestation via NRAS is **not** shipped. The raw ``intel_quote`` /
    ``nvidia_payload`` are retained on the attestation for the verifier. Callers
    whose threat model includes a malicious Venice operator MUST NOT rely on the
    baseline alone — supply a :class:`DcapTdxVerifier`.

The baseline is fail-closed: when ``fail_closed=True`` (the default) any failed
check — including a *missing or malformed* field it was supposed to check —
raises :class:`~venice_ai.exceptions.TeeAttestationError`. Absence of evidence
fails closed; it never silently passes.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..exceptions import TeeAttestationError
from .types import TeeVerificationResult

if TYPE_CHECKING:
    from .types import TeeAttestation

#: Required nonce length in raw bytes (32) and hex characters (64). The API
#: rejects shorter nonces (e.g. 16-byte) outright.
NONCE_LEN_BYTES = 32
NONCE_LEN_HEX = NONCE_LEN_BYTES * 2

#: REPORTDATA layout: signing_address(20B) || zero-pad(12B) || nonce(32B) = 64B.
_ADDR_LEN_BYTES = 20
_ZERO_PAD_LEN_BYTES = 12
_REPORTDATA_LEN_HEX = (_ADDR_LEN_BYTES + _ZERO_PAD_LEN_BYTES + NONCE_LEN_BYTES) * 2

#: Bit 0 of the little-endian TD_ATTRIBUTES field is ``TUD.DEBUG``.
_TDX_DEBUG_BIT = 0x1


@runtime_checkable
class FullQuoteVerifier(Protocol):
    """Extension point for full client-side quote verification.

    The baseline verifier does **not** parse the raw Intel TDX quote signature
    or the NVIDIA GPU attestation payload. A caller who needs cryptographic
    proof to Intel / NVIDIA roots can supply an object implementing this
    protocol; :func:`verify_attestation` invokes it after the baseline checks
    pass and treats a ``False`` return (or an exception) as a verification
    failure. The raw ``intel_quote`` / ``nvidia_payload`` remain on the
    :class:`~venice_ai.tee.types.TeeAttestation` for the verifier to consume.

    Implementations should return ``True`` on success and ``False`` (or raise)
    on failure. The default behaviour ships no such verifier (baseline only).
    """

    def verify(self, attestation: TeeAttestation) -> bool:
        """Return ``True`` iff the full quote chain validates."""
        ...


def preflight_nonce(nonce: str) -> str:
    """Validate a client nonce and return it lowercased.

    A valid nonce is exactly 32 bytes encoded as 64 lowercase hex characters.
    The Venice API rejects shorter nonces, so we reject them client-side too.

    Args:
        nonce: The candidate nonce (hex string).

    Returns:
        The nonce, lowercased.

    Raises:
        TeeAttestationError: If ``nonce`` is not exactly 64 hex characters.
    """
    if not isinstance(nonce, str):  # pragma: no cover - defensive
        raise TeeAttestationError(
            f"TEE attestation nonce must be a string, got {type(nonce).__name__}."
        )
    candidate = nonce.lower()
    if len(candidate) != NONCE_LEN_HEX:
        raise TeeAttestationError(
            f"TEE attestation nonce must be exactly {NONCE_LEN_BYTES} bytes "
            f"({NONCE_LEN_HEX} hex chars); got {len(candidate)} chars."
        )
    try:
        bytes.fromhex(candidate)
    except ValueError as exc:
        raise TeeAttestationError("TEE attestation nonce must be valid hexadecimal.") from exc
    return candidate


def _normalize_address(signing_address: str) -> str:
    """Return ``signing_address`` lowercased with any ``0x`` prefix removed."""
    return signing_address.lower().removeprefix("0x")


def _expected_reportdata(signing_address: str, sent_nonce: str) -> str:
    """Build the expected 128-hex REPORTDATA for ``signing_address`` + ``sent_nonce``."""
    addr = _normalize_address(signing_address)
    return addr + ("00" * _ZERO_PAD_LEN_BYTES) + sent_nonce.lower()


def _get_tdx(attestation: TeeAttestation) -> dict[str, Any]:
    """Locate the parsed ``server_verification.tdx`` mapping, fail-closed.

    Raises:
        TeeAttestationError: If ``server_verification`` or its ``tdx`` member is
            absent or not a mapping. Absence of the evidence we must check is a
            verification failure, not a pass.
    """
    sv = attestation.server_verification
    if not isinstance(sv, dict):
        raise TeeAttestationError(
            "TEE attestation is missing server_verification; cannot verify the "
            "report-data binding (fail-closed)."
        )
    tdx = sv.get("tdx")
    if not isinstance(tdx, dict):
        raise TeeAttestationError(
            "TEE attestation server_verification has no tdx report; cannot "
            "verify the report-data binding (fail-closed)."
        )
    return tdx


def _decode_model_public_key(signing_public_key: str | None) -> bytes | None:
    """Best-effort decode of the 130-hex signing key to 65 raw bytes."""
    if not isinstance(signing_public_key, str):
        return None
    try:
        return bytes.fromhex(signing_public_key.lower())
    except ValueError:
        return None


def verify_attestation(
    attestation: TeeAttestation,
    sent_nonce: str,
    *,
    fail_closed: bool = True,
    verifier: FullQuoteVerifier | None = None,
) -> TeeVerificationResult:
    """Run baseline TEE attestation verification, fail-closed by default.

    Performs the four baseline checks (server ``verified``; nonce echo;
    REPORTDATA binding; TDX debug-flag rejection) against the
    **client-generated** ``sent_nonce``. If ``verifier`` is supplied, it is
    invoked after the baseline checks pass as a :class:`FullQuoteVerifier`
    extension point.

    Args:
        attestation: The parsed attestation response.
        sent_nonce: The nonce the client actually sent (NOT read from the
            response echo). Validated to be 64 hex chars.
        fail_closed: When ``True`` (default), the first failed check raises
            :class:`~venice_ai.exceptions.TeeAttestationError`. When ``False``,
            all failures are collected into the returned
            :class:`~venice_ai.tee.types.TeeVerificationResult` (``ok=False``)
            and a single :class:`UserWarning` is emitted instead of raising.
        verifier: Optional full-quote verifier. The baseline ships no verifier.

    Returns:
        A :class:`~venice_ai.tee.types.TeeVerificationResult`.

    Raises:
        TeeAttestationError: When ``fail_closed`` and any check fails, including
            a missing/malformed field that a check depends on.
    """
    sent_nonce = preflight_nonce(sent_nonce)

    checks: dict[str, bool] = {}
    failures: list[str] = []

    def fail(name: str, message: str) -> None:
        checks[name] = False
        failures.append(message)
        if fail_closed:
            raise TeeAttestationError(message)

    # (1) Server verification claim.
    if attestation.verified is True:
        checks["verified"] = True
    else:
        fail(
            "verified",
            "TEE attestation reported verified=false; refusing to proceed (fail-closed).",
        )

    # (2) Nonce echo — the server's echoed nonce must equal the client nonce.
    echoed = (attestation.nonce or "").lower()
    if echoed == sent_nonce:
        checks["nonce_echo"] = True
    else:
        fail(
            "nonce_echo",
            "TEE attestation nonce echo does not match the client-sent nonce "
            f"(expected {sent_nonce!r}, got {echoed!r}).",
        )

    # (3) REPORTDATA binding. _get_tdx is fail-closed on missing evidence.
    if fail_closed:
        tdx = _get_tdx(attestation)
    else:
        try:
            tdx = _get_tdx(attestation)
        except TeeAttestationError as exc:
            tdx = {}
            fail("reportdata_binding", str(exc))

    if checks.get("reportdata_binding") is not False:
        report_data = tdx.get("reportData")
        if not isinstance(report_data, str):
            fail(
                "reportdata_binding",
                "TEE attestation tdx.reportData is missing or not a string; "
                "cannot verify the report-data binding (fail-closed).",
            )
        else:
            expected = _expected_reportdata(attestation.signing_address, sent_nonce)
            actual = report_data.lower()
            if len(actual) != _REPORTDATA_LEN_HEX:
                fail(
                    "reportdata_binding",
                    "TEE attestation tdx.reportData has unexpected length "
                    f"({len(actual)} hex chars, expected {_REPORTDATA_LEN_HEX}).",
                )
            elif actual == expected:
                checks["reportdata_binding"] = True
            else:
                fail(
                    "reportdata_binding",
                    "TEE attestation report-data binding mismatch: reportData "
                    "does not equal signing_address || zero-pad || sent_nonce.",
                )

    # (4) TDX debug-mode flag must be clear.
    if checks.get("reportdata_binding") is not False or not fail_closed:
        measurements = tdx.get("measurements") if isinstance(tdx, dict) else None
        td_attributes = measurements.get("tdAttributes") if isinstance(measurements, dict) else None
        if not isinstance(td_attributes, str):
            fail(
                "debug_flag",
                "TEE attestation tdx.measurements.tdAttributes is absent; "
                "cannot confirm TDX debug mode is disabled (fail-closed).",
            )
        else:
            debug_set = _check_debug_flag(td_attributes)
            if debug_set is None:
                fail(
                    "debug_flag",
                    "TEE attestation tdAttributes is malformed; cannot confirm "
                    "TDX debug mode is disabled (fail-closed).",
                )
            elif debug_set:
                fail(
                    "debug_flag",
                    "TEE attestation enclave is in TDX DEBUG mode; refusing to "
                    "proceed (fail-closed).",
                )
            else:
                checks["debug_flag"] = True

    # (5) Optional full-quote verifier (extension point). Only run if the
    # baseline passed; a False return or exception is a failure.
    if verifier is not None and not failures:
        try:
            ok = verifier.verify(attestation)
        except Exception as exc:  # noqa: BLE001 - any verifier failure is fatal
            checks["full_quote"] = False
            failures.append(f"Full-quote verifier raised: {exc!r}")
            if fail_closed:
                raise TeeAttestationError(f"TEE full-quote verifier raised: {exc!r}") from exc
        else:
            if ok:
                checks["full_quote"] = True
            else:
                fail(
                    "full_quote",
                    "TEE full-quote verifier returned False (quote did not validate).",
                )

    ok = not failures
    if not ok and not fail_closed:
        warnings.warn(
            "TEE attestation verification failed (fail_closed=False): " + "; ".join(failures),
            UserWarning,
            stacklevel=2,
        )

    return TeeVerificationResult(
        ok=ok,
        checks=checks,
        failures=failures,
        signing_address=_normalize_address(attestation.signing_address),
        model_public_key=_decode_model_public_key(attestation.signing_public_key),
    )


def _check_debug_flag(td_attributes: str) -> bool | None:
    """Return whether the TDX DEBUG bit is set, or ``None`` if unparseable.

    ``tdAttributes`` is the 8-byte TD_ATTRIBUTES field as natural-byte-order
    hex; read it little-endian and test bit 0 (``TUD.DEBUG``). (The live sample
    ``0000001000000000`` decodes little-endian to ``SEPT_VE_DISABLE`` (bit 28),
    a real production flag — confirming little-endian byte order.)
    """
    try:
        raw = bytes.fromhex(td_attributes.lower())
    except ValueError:
        return None
    value = int.from_bytes(raw, "little")
    return bool(value & _TDX_DEBUG_BIT)
