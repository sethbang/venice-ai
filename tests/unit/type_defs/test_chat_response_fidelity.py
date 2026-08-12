"""TDD: chat response fidelity (audit MED #0, #1).

#0 swagger marks `choices` NOT required ("certain models may not return this
   field"); the SDK declared it required and would crash on a spec-legal omission.
#1 `web_search_citations` was a phantom top-level field (always []); the real,
   populated data lives in `venice_parameters.web_search_citations`.
"""

from venice_ai.types.api.chat import ChatCompletionResponse

BASE = {"id": "x", "object": "chat.completion", "created": 1, "model": "m"}
VP = {
    "enable_web_search": "auto",
    "enable_web_scraping": False,
    "enable_web_citations": True,
    "include_venice_system_prompt": True,
    "include_search_results_in_stream": False,
    "return_search_results_as_documents": False,
    "strip_thinking_response": False,
    "disable_thinking": False,
    "web_search_citations": [
        {"title": "T", "url": "https://e.com", "content": "c", "date": "2026-01-01"}
    ],
}


def test_choices_optional_and_accessors_guard_empty():
    resp = ChatCompletionResponse.model_validate(BASE)  # choices omitted (spec-legal)
    assert resp.choices == []
    assert resp.text is None
    assert resp.parsed is None


def test_web_search_citations_is_not_a_phantom_field():
    assert "web_search_citations" not in ChatCompletionResponse.model_fields


def test_web_search_citations_delegates_to_venice_parameters():
    resp = ChatCompletionResponse.model_validate({**BASE, "venice_parameters": VP})
    assert len(resp.web_search_citations) == 1
    assert resp.web_search_citations == resp.venice_parameters.web_search_citations


def test_web_search_citations_empty_without_venice_parameters():
    assert ChatCompletionResponse.model_validate(BASE).web_search_citations == []
