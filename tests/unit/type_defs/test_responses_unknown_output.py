"""TDD: Responses API must not crash on an unknown output-item type (audit MED #3).

A future/unmodeled output block ``type`` makes every union variant fail, raising
a ValidationError that fails the whole /responses parse. A permissive catch-all
variant (appended last) preserves forward-compatibility while keeping known
types deserialising into their specific models.
"""

from venice_ai.types.api.responses import (
    ResponsesResponse,
    ResponsesWebSearchCallOutput,
)

BASE = {"id": "r", "object": "response", "created_at": 1, "model": "m", "status": "completed"}


def test_unknown_output_type_does_not_crash():
    r = ResponsesResponse.model_validate(
        {**BASE, "output": [{"type": "some_future_block", "id": "o1", "foo": "bar"}]}
    )
    assert len(r.output) == 1
    assert r.output[0].type == "some_future_block"


def test_known_output_type_still_parses_to_specific_model():
    r = ResponsesResponse.model_validate(
        {**BASE, "output": [{"type": "web_search_call", "id": "o1", "status": "completed"}]}
    )
    assert isinstance(r.output[0], ResponsesWebSearchCallOutput)
