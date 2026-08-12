"""TDD: Seedance face-media consent attestations on video requests.

Swagger ``QueueVideoRequest.consents.seedance`` defines three required boolean
attestations (each ``enum: [true]``); the API returns a 409 ``needs_consent``
when face-bearing media is submitted without them. These must be expressible
through the SDK's video request models and serialized into the queue body.
"""

import pytest
from pydantic import ValidationError

from venice_ai.types.api.requests.video import VideoTextToVideoRequest

CONSENTS = {
    "seedance": {
        "confirmed_terms_and_privacy": True,
        "confirmed_legal_right": True,
        "confirmed_screening_acknowledged": True,
    }
}


def _req(**extra):
    return VideoTextToVideoRequest(
        model="seedance-2-0-text-to-video",
        prompt="a person waving at the camera",
        duration="5s",
        **extra,
    )  # type: ignore


class TestVideoSeedanceConsents:
    def test_consents_serialized_into_request_body(self):
        body = _req(consents=CONSENTS).model_dump(exclude_none=True)
        assert body.get("consents", {}).get("seedance") == CONSENTS["seedance"]

    def test_consents_optional_absent_by_default(self):
        assert "consents" not in _req().model_dump(exclude_none=True)

    def test_seedance_consent_flags_must_be_true(self):
        # Each flag is enum [true]; a False attestation is invalid.
        with pytest.raises(ValidationError):
            _req(
                consents={
                    "seedance": {
                        "confirmed_terms_and_privacy": False,
                        "confirmed_legal_right": True,
                        "confirmed_screening_acknowledged": True,
                    }
                }
            )

    def test_all_three_seedance_flags_required(self):
        # Swagger marks all three required within the seedance object.
        with pytest.raises(ValidationError):
            _req(consents={"seedance": {"confirmed_terms_and_privacy": True}})
