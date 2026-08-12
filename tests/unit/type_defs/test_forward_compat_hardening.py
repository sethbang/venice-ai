"""TDD: forward-compat hardening for /response models (audit MED #2,#7,#10,#13,#18,#20).

(1) Response models whose swagger schema has no ``additionalProperties: false``
    must use ``extra="allow"`` so a future server field is preserved, not rejected
    with a hard ValidationError.
(2) Closed ``Literal`` enums on /models response constraints are relaxed to open
    ``str`` (mirroring the existing ``quantization`` policy) so a new server-side
    value can't crash the entire /models parse.
"""

import pytest

from venice_ai.core.models.common import Balances, VeniceParametersResponse
from venice_ai.types.api.characters import (
    CharacterResponse,
    CharacterReviewsResponse,
    CharactersListResponse,
)
from venice_ai.types.api.models import ImageModelConstraints, VideoModelConstraints
from venice_ai.types.api.music import (
    MusicCompletedStatus,
    MusicCompleteResponse,
    MusicFailedStatus,
    MusicProcessingStatus,
    MusicQuoteResponse,
)

EXTRA_ALLOW_MODELS = [
    Balances,
    VeniceParametersResponse,
    MusicQuoteResponse,
    MusicProcessingStatus,
    MusicFailedStatus,
    MusicCompletedStatus,
    MusicCompleteResponse,
    CharactersListResponse,
    CharacterResponse,
    CharacterReviewsResponse,
]


@pytest.mark.parametrize("model", EXTRA_ALLOW_MODELS, ids=lambda m: m.__name__)
def test_response_model_tolerates_extra_fields(model):
    assert model.model_config.get("extra") == "allow", (
        f"{model.__name__} must preserve forward-compatible server fields, not reject them"
    )


def test_image_constraints_accept_novel_quality():
    c = ImageModelConstraints.model_validate(
        {
            "promptCharacterLimit": 1500,
            "steps": {"default": 8, "max": 50},
            "widthHeightDivisor": 8,
            "defaultQuality": "ultra",  # not in the old Literal
            "qualities": ["ultra", "max"],
        }
    )
    assert c.defaultQuality == "ultra"
    assert c.qualities == ["ultra", "max"]


def test_video_constraints_accept_novel_model_type():
    c = VideoModelConstraints.model_validate(
        {"model_type": "text-to-3d", "resolutions": [], "durations": []}  # novel value
    )
    assert c.model_type == "text-to-3d"
