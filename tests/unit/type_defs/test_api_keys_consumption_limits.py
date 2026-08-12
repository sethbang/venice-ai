"""
Unit tests for ``ConsumptionLimits`` VCU field.

The Venice API still accepts the ``vcu`` key inside ``consumptionLimit``
(legacy Diem, deprecated). See
``api-reference/endpoint/api_keys/create.md`` — the docs note VCU is being
phased out but the API continues to honor it. The SDK must round-trip the
field so inbound responses don't drop it and outbound requests can still set it.
"""

from venice_ai.types.api.api_keys import ConsumptionLimits


def test_consumption_limits_roundtrip_with_vcu() -> None:
    model = ConsumptionLimits.model_validate({"usd": 1.0, "diem": 2.0, "vcu": 3.0})
    assert model.usd == 1.0
    assert model.diem == 2.0
    assert model.vcu == 3.0

    dumped = model.model_dump(exclude_none=True)
    assert dumped == {"usd": 1.0, "diem": 2.0, "vcu": 3.0}


def test_consumption_limits_vcu_defaults_to_none() -> None:
    model = ConsumptionLimits.model_validate({"usd": 1.0, "diem": 2.0})
    assert model.vcu is None
    # Legacy callers that only care about usd/diem see unchanged dumped shape.
    assert model.model_dump(exclude_none=True) == {"usd": 1.0, "diem": 2.0}


def test_consumption_limits_accepts_vcu_only() -> None:
    model = ConsumptionLimits.model_validate({"vcu": 5.0})
    assert model.vcu == 5.0
    assert model.usd is None
    assert model.diem is None
