#!/usr/bin/env python3
"""
Venice AI SDK - TEE Client-Side End-to-End Encryption (E2EE)
============================================================

Demonstrates Venice confidential-compute (TEE) chat with client-side E2EE.
Messages are encrypted in-process to the attested model key and the streamed
response deltas are decrypted locally — Venice's infrastructure never sees the
plaintext of your ``user`` / ``system`` content.

Requires the ``[e2ee]`` extra (pulls in ``cryptography``)::

    pip install 'venice-ai[e2ee]'

For the optional full client-side TDX quote verification step, also install the
``[e2ee-verify]`` extra (pulls in ``dcap-qvl``; arm64-macOS wheel available)::

    pip install 'venice-ai[e2ee-verify]'

Features Demonstrated:
    - Dynamic discovery of an E2EE-capable model (no hardcoded id)
    - client.tee.get_attestation(model=...) → fail-closed baseline verification
    - Full client-side Intel TDX quote verification via DcapTdxVerifier
    - Streaming chat with e2ee=True, consuming decrypted text deltas
    - A stream=False call returning a reassembled (decrypted) response

Security note:
    The *baseline* attestation verifier TRUSTS Venice's server-side ``verified``
    claim. ``DcapTdxVerifier`` (shown below) removes that trust by verifying the
    Intel TDX quote client-side, but by default proves only GENUINE non-debug
    TDX hardware + a self-consistent dstack workload (Tier B) — not that it is
    the legitimate Venice image, which requires an independently-pinned
    reference (``expected_compose_hash`` / ``expected_measurements``, Tier A).
"""

import asyncio
import sys

from venice_ai import VeniceClient
from venice_ai.exceptions import TeeAttestationError, TeeError
from venice_ai.types.api import SystemMessage, UserMessage

# Prefix that marks a Venice confidential-compute (TEE) chat model. Mirrors
# ``_E2EE_MODEL_PREFIX`` in venice_ai.resources.chat.completions — we match on
# the prefix rather than hardcoding any specific model id.
E2EE_MODEL_PREFIX = "e2ee-"


async def discover_e2ee_model(client: VeniceClient) -> str | None:
    """Return the id of an E2EE-capable text model, or None if none are listed.

    Reads capabilities directly off each list entry (no per-model round-trip).
    Primary signal is the ``e2ee-`` id prefix; we also honor any text model
    whose ``capabilities.supportsE2EE`` flag is set, in case a non-prefixed
    E2EE model ever appears.
    """
    models = await client.models.list(type="text")
    for m in models.data:
        if m.id.startswith(E2EE_MODEL_PREFIX):
            return m.id
        caps = getattr(getattr(m, "model_spec", None), "capabilities", None)
        if getattr(caps, "supportsE2EE", False):
            return m.id
    return None


async def tee_e2ee_demo() -> bool:
    """Discover an E2EE model, attest it, then run encrypted chat.

    Returns ``True`` on success (including the graceful skips below where no
    E2EE model is entitled or the ``[e2ee]`` extra is absent), ``False`` only if
    a load-bearing step (baseline attestation or the core encrypted chat)
    fails. The optional Tier-B full-quote verification degrading does NOT count
    as a failure.
    """
    print("🔐 Venice TEE End-to-End Encryption")
    print("-" * 40)

    async with VeniceClient() as client:
        model = await discover_e2ee_model(client)
        if model is None:
            # No E2EE-capable model is available to this account. That's an
            # entitlement / availability condition, not a code error, so we
            # report it and skip gracefully (ok=True) rather than fail.
            print("   ⏭️  No E2EE-capable (e2ee-*) model is available on this account.")
            print("      Confidential-compute models are gated; nothing to demo.")
            return True
        print(f"📍 Using E2EE model: {model}")

        # --- (1) Attestation ------------------------------------------------
        # Fetch and BASELINE-verify the enclave's attestation. fail_closed=True
        # (the default) means any failed check raises TeeAttestationError, so if
        # this returns, the attestation passed every baseline check.
        # NOTE: this is a standalone demonstration of the attestation surface;
        # the encrypted create() calls below re-attest internally when they open
        # their session, so seeing two attestation fetches is expected.
        print("\n🪪  Fetching + baseline-verifying attestation...")
        attestation = await client.tee.get_attestation(model=model)
        print(f"   verified (server claim): {attestation.verified}")
        print("   ✅ Attestation is BASELINE-verified (nonce echo + report-data")
        print("      binding + TDX debug-flag checks all passed, fail-closed).")
        print("   ⚠️  SECURITY LIMITATION: baseline verification TRUSTS Venice's")
        print("      server-side 'verified' claim and does NOT perform full")
        print("      client-side Intel TDX quote verification.")

        # --- (1b) Full client-side TDX verification (optional) --------------
        # The [e2ee-verify] extra adds DcapTdxVerifier, which verifies the raw
        # Intel TDX quote itself (signature → pinned Intel SGX Root CA + TCB
        # status + QE identity + debug-flag), binds the E2EE key to the enclave
        # (REPORTDATA), and replays the event log to the quoted RTMRs — entirely
        # client-side, instead of trusting Venice's 'verified' flag. It runs on
        # Apple Silicon (dcap-qvl ships an arm64 wheel).
        try:
            from venice_ai.tee import DcapTdxVerifier

            print("\n🔎 Full client-side TDX verification ([e2ee-verify])...")
            # with_fetched_collateral makes the ONE no-auth PCCS call; verify()
            # itself is fully offline. We verify the SAME attestation object.
            verifier = await DcapTdxVerifier.with_fetched_collateral(attestation.intel_quote)
            ok = verifier.verify(attestation)
            result = verifier.last_result or {}
            print(f"   full quote verified: {ok}")
            print(f"   TCB status: {result.get('tcb_status')!r}  (fail-closed; reject-by-default)")
            print(f"   workload_identity_pinned: {result.get('workload_identity_pinned')}")
            print("   ℹ️  Tier B: this proves GENUINE non-debug Intel TDX hardware +")
            print("      a self-consistent dstack workload — but NOT that it is the")
            print("      legitimate Venice image. For per-dimension Tier A, pin an")
            print("      INDEPENDENTLY-obtained reference, e.g.:")
            print("        DcapTdxVerifier(collateral=..., expected_compose_hash='<ref>')")
            print("      and attach it to chat via e2ee=TeeOptions(verifier=verifier).")
        except ImportError as e:
            print("\n   ⏭️  Full TDX verification needs the [e2ee-verify] extra.")
            print(f"      ({e})")
            print("      Install it with: pip install 'venice-ai[e2ee-verify]'")
        except TeeError as e:
            # Full quote verification is fail-closed: if it cannot complete we
            # report it and continue with the baseline-verified attestation
            # rather than silently claiming a stronger guarantee.
            print("\n   ⏭️  Full client-side TDX verification did not complete")
            print(f"      ({e})")

        messages = [
            SystemMessage(content="You are a concise, helpful assistant."),
            UserMessage(content="In one sentence, what does end-to-end encryption protect?"),
        ]

        # The streaming + non-streaming encrypted calls below require the
        # [e2ee] extra (cryptography) to generate the session keypair. On a bare
        # install open_session raises ImportError with a pip-install hint; we
        # catch it so the example degrades gracefully instead of crashing.
        try:
            # --- (2) Streaming encrypted chat -------------------------------
            # create(e2ee=True, stream=True) returns a ChatStream. The SDK emits
            # a one-time UserWarning about the baseline-attestation limitation on
            # this call (we don't re-raise or suppress it). text_deltas() yields
            # already-DECRYPTED text — decryption happens locally per chunk.
            print("\n🤖 Assistant (E2EE streaming): ", end="", flush=True)
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                e2ee=True,
                stream=True,
                max_completion_tokens=120,
                temperature=0.3,
            )
            async with stream:
                async for text in stream.text_deltas():
                    print(text, end="", flush=True)
            print()

            # --- (3) Non-streaming encrypted chat ---------------------------
            # stream=False still runs the encrypted wire flow under the hood,
            # but the SDK reassembles the decrypted deltas into a normal
            # ChatCompletionResponse for you.
            print("\n📦 Non-streaming E2EE response:")
            response = await client.chat.completions.create(
                model=model,
                messages=[UserMessage(content="Reply with exactly: ENCRYPTED OK")],
                e2ee=True,
                stream=False,
                max_completion_tokens=20,
                temperature=0.0,
            )
            print(f"   {response.text}")
        except ImportError as e:
            # The [e2ee] extra (cryptography) is not installed — the encrypting
            # session cannot be opened. The attestation step above still ran, so
            # this is a graceful skip (ok), not a failure.
            print("\n   ⏭️  Encrypted chat needs the [e2ee] extra.")
            print(f"      ({e})")
            print("      Install it with: pip install 'venice-ai[e2ee]'")

    return True


async def main() -> int:
    """Run the TEE E2EE example.

    Returns ``0`` only if the demo succeeded (or gracefully skipped), ``1`` if a
    load-bearing step failed, so a real failure surfaces as a non-zero process
    exit instead of being masked by the success banner.
    """
    print("🚀 Venice AI TEE E2EE Example")
    print("=" * 50)

    ok = await tee_e2ee_demo()

    if ok:
        print("\n✨ TEE E2EE example completed!")
    else:
        print("\n⚠️ TEE E2EE example failed.")

    print("\n💡 Key concepts demonstrated:")
    print("   - Discovering an e2ee-* model dynamically (no hardcoded id)")
    print("   - client.tee.get_attestation(model=...) → fail-closed baseline verify")
    print("   - DcapTdxVerifier full client-side TDX quote verification ([e2ee-verify])")
    print("   - chat.completions.create(e2ee=True, stream=True) decrypted deltas")
    print("   - chat.completions.create(e2ee=True, stream=False) reassembled response")
    print("   - Tier B vs caller-pinned Tier A (TeeOptions(verifier=...))")

    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except TeeAttestationError as e:
        # Baseline attestation is fail-closed: a failed baseline check raises and
        # MUST surface loudly (non-zero exit), never silently degrade.
        print(f"\n❌ Attestation verification failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
