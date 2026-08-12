"""Venice TEE (Trusted Execution Environment) end-to-end encryption.

This package implements client-side support for Venice's confidential-compute
E2EE chat path (``e2ee-*`` models): attestation verification and per-message
encryption/decryption over secp256k1 ECDH + HKDF-SHA256 + AES-256-GCM.

``cryptography`` is an **optional** dependency (the ``[e2ee]`` extra); this
package imports cleanly on a bare install and only requires it lazily when an
encryption path is actually used (see :func:`_require_crypto`).

.. warning::
    The baseline attestation verifier trusts Venice's server-side ``verified``
    claim. Full client-side TDX / NVIDIA quote verification is NOT performed by
    the baseline. For full **offline** Intel TDX verification, supply a
    :class:`DcapTdxVerifier` (the ``[e2ee-verify]`` extra) via the
    :class:`FullQuoteVerifier` extension point; NVIDIA GPU attestation is not yet
    shipped.

Public symbols are re-exported here: the typed TEE exceptions
(:class:`TeeError`, :class:`TeeAttestationError`, :class:`TeeEncryptionError`),
the attestation and quote verifiers (:class:`FullQuoteVerifier`,
:class:`DcapTdxVerifier`), the encrypted session (:class:`TeeSession`),
:class:`TeeOptions`, and the :func:`_require_crypto` helper.
"""

from __future__ import annotations

from ..exceptions import TeeAttestationError, TeeEncryptionError, TeeError
from ._attestation import FullQuoteVerifier
from ._crypto import _require_crypto
from ._session import TeeSession
from ._verify import DcapTdxVerifier
from .types import TeeOptions

__all__ = [
    "TeeError",
    "TeeAttestationError",
    "TeeEncryptionError",
    "FullQuoteVerifier",
    "DcapTdxVerifier",
    "TeeOptions",
    "TeeSession",
    "_require_crypto",
]
