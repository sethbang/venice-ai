"""Regression guard for Audio resource docstrings.

The class- and method-level docstrings in ``venice_ai.resources.audio`` ship a
number of TTS examples. Those examples are the first thing AI assistants and
human readers copy-paste, so any ``Voice.*`` enum member referenced in them
must actually exist on :class:`venice_ai.types.enums.Voice`, and any literal
``voice="..."`` string must be a valid Voice value.

Outside-agent feedback (see ``audio.py`` history) flagged ``Voice.KOKORO_DEFAULT``
and the literal ``"kokoro-default"`` as a trap — neither resolves at the SDK or
API layer. This test exists to make sure the docstrings can never silently
regress to that state again.
"""

from __future__ import annotations

import inspect
import re

from venice_ai.resources import audio as audio_module
from venice_ai.types.enums import Voice


def _all_audio_docstrings() -> str:
    """Concatenate every docstring in ``venice_ai.resources.audio`` that we
    expect to teach users how to call the API.

    We pull from the module, the ``Audio`` class, and every public method on it
    so future docstring drift on any of them is caught in one place.
    """
    chunks: list[str] = []
    if audio_module.__doc__:
        chunks.append(audio_module.__doc__)
    if audio_module.Audio.__doc__:
        chunks.append(audio_module.Audio.__doc__)
    for name, member in inspect.getmembers(audio_module.Audio):
        if name.startswith("_"):
            continue
        doc = getattr(member, "__doc__", None)
        if doc:
            chunks.append(doc)
    return "\n\n".join(chunks)


def test_voice_enum_references_in_docstrings_resolve() -> None:
    """Every ``Voice.SOMETHING`` token in an audio docstring must be a real enum member."""
    text = _all_audio_docstrings()
    referenced = set(re.findall(r"\bVoice\.([A-Z][A-Z0-9_]*)\b", text))
    assert referenced, "expected at least one Voice.* reference in audio docstrings"
    valid = {member.name for member in Voice}
    missing = referenced - valid
    assert not missing, (
        f"Audio docstrings reference Voice members that do not exist: {sorted(missing)}. "
        f"Valid kokoro-family voices include AF_HEART, AF_ALLOY, AF_JADZIA. "
        f"Use ``client.audio.get_voices(model_id=...)`` for the live catalog."
    )


def test_voice_string_literals_in_docstrings_are_valid() -> None:
    """Every ``voice="..."`` literal in an audio docstring must be a real Voice value.

    ``voice="kokoro-default"`` is the canonical trap — it looks plausible but is
    not in the Voice enum and the API rejects it with a 400.
    """
    text = _all_audio_docstrings()
    referenced = set(re.findall(r'voice\s*=\s*"([^"]+)"', text))
    assert referenced, "expected at least one voice='...' literal in audio docstrings"
    valid_values = {member.value for member in Voice}
    invalid = referenced - valid_values
    assert not invalid, (
        f"Audio docstrings use voice='...' literals that aren't in the Voice enum: "
        f"{sorted(invalid)}. Pick a real value (e.g. 'af_heart', 'af_alloy') or "
        f"point readers at ``client.audio.get_voices(model_id=...)``."
    )
