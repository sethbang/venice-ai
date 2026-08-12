"""Live-verified protocol constants for Venice TEE end-to-end encryption.

Every value here was confirmed against the live Venice API on 2026-06-04. Do
not change them without a fresh live probe — the wire format depends on them
exactly.
"""

from __future__ import annotations

# --- Key agreement / KDF / cipher --------------------------------------------

#: Elliptic curve used for ECDH key agreement (and the model signing key).
CURVE_NAME = "secp256k1"

#: HKDF ``info`` parameter. The shared secret is the raw 32-byte big-endian X
#: coordinate (NOT hashed); HKDF-SHA256(salt=None, info=HKDF_INFO, length=32)
#: derives the AES-256 key.
HKDF_INFO = b"ecdsa_encryption"

#: HKDF output length in bytes → AES-256 key.
HKDF_LENGTH = 32

#: AES-256-GCM nonce length in bytes (random per message; AAD is None).
GCM_NONCE_LEN = 12

#: Uncompressed SEC1 public-key length in bytes (0x04 || X || Y).
UNCOMPRESSED_PUBKEY_LEN = 65

#: Minimum length, in hex characters, for an encrypted field to be treated as
#: ciphertext (the ``looks_encrypted`` heuristic). 93 bytes = 65 (ephemeral
#: pub) + 12 (gcm nonce) + 16 (tag) → 186 hex chars.
MIN_ENCRYPTED_HEX_LEN = 186

# --- Confirmed request header names ------------------------------------------

#: Session public key (130-hex, uncompressed). Its private half decrypts the
#: streamed response.
HEADER_CLIENT_PUB_KEY = "X-Venice-TEE-Client-Pub-Key"

#: The model's ``signing_public_key`` (130-hex) — also its ECDH/encryption key.
HEADER_MODEL_PUB_KEY = "X-Venice-TEE-Model-Pub-Key"

#: Signing algorithm; always ``"ecdsa"``.
HEADER_SIGNING_ALGO = "X-Venice-TEE-Signing-Algo"

# --- Full client-side quote verification ([e2ee-verify] extra) ----------------

#: No-auth collateral source URLs for ``dcap-qvl`` (Intel-signed TCB info / QE
#: identity / CRLs). Both serve the same Intel-signed collateral; the Phala PCCS
#: mirror is the default because it requires no Intel subscription key.
PHALA_PCCS_URL = "https://pccs.phala.network"
INTEL_PCS_URL = "https://api.trustedservices.intel.com"

#: Pinned **Intel SGX Root CA** (self-signed, valid 2018-05-21 → 2049-12-31),
#: extracted from the PCK cert chain embedded in a live Venice TDX quote on
#: 2026-06-04 and confirmed against the well-known Intel root. The PCK chain
#: (leaf → PCK Platform CA → this root) is embedded in every quote, so local
#: verification needs only this pinned root as the trust anchor.
#:
#: .. warning::
#:     A silently stale pin would reject **all** quotes after a (rare) Intel
#:     root rotation. The :class:`~venice_ai.tee._verify.DcapTdxVerifier`
#:     constructor accepts a ``root_ca_der=`` override for the rotation path.
INTEL_SGX_ROOT_CA_DER = bytes.fromhex(
    "3082028f30820234a003020102021422650cd65a9d3489f383b49552bf501b392706ac"
    "300a06082a8648ce3d0403023068311a301806035504030c11496e74656c2053475820"
    "526f6f74204341311a3018060355040a0c11496e74656c20436f72706f726174696f6e"
    "3114301206035504070c0b53616e746120436c617261310b300906035504080c024341"
    "310b300906035504061302555330"
    "1e170d3138303532313130343531305a170d343931"
    "323331323335393539"
    "5a3068311a301806035504030c11496e74656c2053475820526f"
    "6f74204341311a3018060355040a0c11496e74656c20436f72706f726174696f6e3114"
    "301206035504070c0b53616e746120436c617261310b300906035504080c024341310b"
    "30090603550406130255533059301306072a8648ce3d020106082a8648ce3d03010703"
    "4200040ba9c4c0c0c86193a3fe23d6b02cda10a8bbd4e88e48b4458561a36e705525f56"
    "7918e2edc88e40d860bd0cc4ee26aacc988e505a953558c453f6b0904ae7394a381bb30"
    "81b8301f0603551d2304183016801422650cd65a9d3489f383b49552bf501b392706ac"
    "30520603551d1f044b30493047a045a043864168747470733a2f2f6365727469666963"
    "617465732e7472757374656473657276696365732e696e74656c2e636f6d2f496e7465"
    "6c534758526f6f7443412e646572301d0603551d0e0416041422650cd65a9d3489f383"
    "b49552bf501b392706ac300e0603551d0f0101ff04040302010630120603551d130101"
    "ff040830060101ff020101300a06082a8648ce3d0403020349003046022100e5bfe509"
    "11f92f428920dc368a302ee3d12ec5867ff622ec6497f78060c13c20022100e09d25ac"
    "7a0cb3e5e8e68fec5fa3bd416c47440bd950639d450edcbea4576aa2"
)

#: SHA-256 of :data:`INTEL_SGX_ROOT_CA_DER`, for pin-integrity self-checks.
INTEL_SGX_ROOT_CA_SHA256 = "44a0196b2b99f889b8e149e95b807a350e7424964399e885a7cbb8ccfab674d3"
