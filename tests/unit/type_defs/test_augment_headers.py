"""TDD: augment scrape/search must surface response headers (audit MED #19).

These were plain BaseModel — no header accessor — so the documented
``X-Balance-Remaining`` header was silently discarded. Switching to
VeniceBaseModel gives them the ``.headers`` accessor (the resource already
posts with cast_to, which attaches the raw response).
"""

from venice_ai.core.models.common import VeniceBaseModel
from venice_ai.types.api.augment import AugmentScrapeResponse, AugmentSearchResponse


def test_augment_responses_use_venice_base_for_header_access():
    assert issubclass(AugmentScrapeResponse, VeniceBaseModel)
    assert issubclass(AugmentSearchResponse, VeniceBaseModel)


def test_augment_responses_tolerate_extra_fields():
    assert AugmentScrapeResponse.model_config.get("extra") == "allow"
    assert AugmentSearchResponse.model_config.get("extra") == "allow"


def test_scrape_response_exposes_headers():
    r = AugmentScrapeResponse.model_validate(
        {"url": "https://e.com", "content": "hi", "format": "markdown"}
    )

    class _Resp:
        headers = {"x-balance-remaining": "42"}

    object.__setattr__(r, "_response", _Resp())
    assert r.headers is not None and r.headers.get("x-balance-remaining") == "42"
