"""Live regression: the verifier must handle the wire Venice serves today.

The schema-drift defect this guards against was invisible to the offline suite,
because the committed captures were taken before the wire changed. This test
fetches a real attestation and asserts the normalizer recognizes it and the full
verifier accepts it. Skipped without credentials.
"""

from __future__ import annotations

import os

import pytest

from venice_ai import VeniceClient
from venice_ai.tee._evidence import detect_schema, normalize

pytest.importorskip("dcap_qvl")

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not os.environ.get("VENICE_API_KEY"),
        reason="live TEE schema check requires VENICE_API_KEY",
    ),
]


async def _first_e2ee_model(client: VeniceClient) -> str | None:
    models = await client.models.list(type="text")
    return next((m.id for m in models.data if m.id.startswith("e2ee-")), None)


async def test_live_attestation_schema_is_recognized() -> None:
    async with VeniceClient() as client:
        model = await _first_e2ee_model(client)
        if model is None:
            pytest.skip("no e2ee-* model entitled on this account")
        attestation = await client.tee.get_attestation(model=model)

        # Fails loudly if Venice changes the wire again.
        schema = detect_schema(attestation)
        evidence = normalize(attestation)
        assert evidence.raw_quote, "live attestation carried no usable quote"
        assert evidence.event_log, f"live attestation ({schema}) carried an empty event log"
        assert all("imr" in e and "digest" in e for e in evidence.event_log)


async def test_live_attestation_fully_verifies() -> None:
    from venice_ai.tee import DcapTdxVerifier

    async with VeniceClient() as client:
        model = await _first_e2ee_model(client)
        if model is None:
            pytest.skip("no e2ee-* model entitled on this account")
        attestation = await client.tee.get_attestation(model=model)

        verifier = await DcapTdxVerifier.with_fetched_collateral(attestation.intel_quote)
        assert verifier.verify(attestation) is True
        result = verifier.last_result
        assert result is not None
        assert result["checks"]["signature_chain"] is True
        assert result["checks"]["reportdata_binding"] is True
        assert result["checks"]["rtmr_replay"] is True
