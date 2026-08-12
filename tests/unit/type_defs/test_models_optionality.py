"""TDD: /models response optionality (audit MED #8, #9).

#9 swagger ImageModelPricing marks only `upscale` required; an upscale-only
   model omits `generation`, which the SDK wrongly required.
#8 swagger model_spec does not require `name`; the SDK declared it required.
"""

from venice_ai.types.api.models import ImageModelPricing, ModelSpec


def test_image_pricing_generation_optional():
    p = ImageModelPricing.model_validate(
        {"upscale": {"2x": {"usd": 0.01, "diem": 0.0}, "4x": {"usd": 0.02, "diem": 0.0}}}
    )
    assert p.generation is None
    assert p.upscale.x2.usd == 0.01


def test_model_spec_name_optional():
    assert ModelSpec.model_fields["name"].is_required() is False
