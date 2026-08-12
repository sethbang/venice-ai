"""End-to-end tests for venice_ai.audio_helpers against the live Venice API.

These tests are designed to be **audibly validated**. Each test that runs
writes an mp3 to ``tests/e2e/audio_helper_outputs/`` and prints the path so
the operator can play it back (e.g. ``afplay <path>``) and confirm the audio
is complete, in order, and free of obvious artifacts.

Tests require ``VENICE_API_KEY`` and ``RUN_AUDIBLE_E2E=1`` to skip when not
explicitly requested — they cost credits and take 20-60 s each.

Run with::

    RUN_AUDIBLE_E2E=1 poetry run pytest tests/e2e/test_audio_helpers_e2e.py -s

The ``-s`` flag is important — it lets the printed file paths reach your
terminal so you can copy them into ``afplay``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import pytest_asyncio

from venice_ai import VeniceClient
from venice_ai.audio_helpers import split_text_for_tts

pytestmark = [pytest.mark.e2e]

# Gate these tests behind an explicit env flag — they cost credits and
# take meaningful wall time. They're meant to be run intentionally and the
# output listened to, not as part of the default CI loop.
RUN_AUDIBLE = os.environ.get("RUN_AUDIBLE_E2E") == "1"
audible_only = pytest.mark.skipif(
    not RUN_AUDIBLE,
    reason="Set RUN_AUDIBLE_E2E=1 to enable audible-validation e2e tests",
)

OUTPUT_DIR = Path(__file__).parent / "audio_helper_outputs"


# The 12-line poem used throughout the investigation. Chosen to exceed the
# qwen3 15.9 s output cap (would render as 38 s of audio on kokoro) so the
# helper's split-and-stitch behavior is the only path that produces complete
# output for qwen3.
POEM = (
    "Beneath the silver moon's soft glow,\n"
    "Where rivers hum and willows sway,\n"
    "The night air holds a tender flow,\n"
    "As dreamers drift in twilight's play.\n"
    "\n"
    "A whispered breeze through pines aloft,\n"
    "Carries tales of stars long-burned,\n"
    "Their light, though distant, lingers soft,\n"
    "On every wave the ocean turned.\n"
    "\n"
    "And in the hush before the dawn,\n"
    "When shadows learn the language of light,\n"
    "A quiet hope is gently drawn,\n"
    "To carry day past every night."
)


def _ffprobe_duration(path: Path) -> float | None:
    """Return audio duration in seconds via ffprobe, or None if not installed."""
    if shutil.which("ffprobe") is None:
        return None
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if r.returncode != 0:
        return None
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except (KeyError, ValueError):
        return None


@pytest_asyncio.fixture
async def live_client():
    """Live VeniceClient — requires VENICE_API_KEY in the environment."""
    api_key = os.environ.get("VENICE_API_KEY")
    if not api_key:
        pytest.skip("VENICE_API_KEY required for live audio helpers e2e tests")
    client = VeniceClient(api_key=api_key)
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture(autouse=True)
def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Sanity / regression
# ---------------------------------------------------------------------------


def test_qwen3_word_budget_matches_observed_cap():
    """Regression: confirms the qwen3 budget hasn't drifted from the
    empirically-measured 15.9 s output cap.

    If this fails because the budget was changed, ALSO update the docstring
    in audio_helpers.py to reflect the new evidence. The budget is the SDK's
    last line of defense against the server-side cap.
    """
    from venice_ai.audio_helpers import MODEL_WORD_BUDGETS

    assert MODEL_WORD_BUDGETS["tts-qwen3-0-6b"] == 25
    assert MODEL_WORD_BUDGETS["tts-qwen3-1-7b"] == 25


def test_poem_splits_into_expected_segments_on_qwen3():
    """The 12-line poem (74 words) at 25 words/segment splits into 4 segments:

    * stanza 1: 25 words exactly → 1 segment
    * stanza 2: 23 words → 1 segment
    * stanza 3: 26 words → splits after the 3rd line (20 words) because the
      4th line (6 words) would push to 26, over budget.

    If this changes audibly, investigate before adjusting the splitter or
    the qwen3 budget — both have empirical justification.
    """
    segments = split_text_for_tts(POEM, model="tts-qwen3-1-7b")
    assert len(segments) == 4, f"Expected 4 qwen3 segments, got {len(segments)}:\n" + "\n--\n".join(
        segments
    )
    assert "Beneath" in segments[0] and "twilight's play" in segments[0]
    assert "whispered breeze" in segments[1]
    assert "hush before the dawn" in segments[2]
    assert "carry day past every night" in segments[3]
    # Word-count sanity
    word_counts = [len(s.split()) for s in segments]
    assert word_counts == [25, 23, 20, 6]


# ---------------------------------------------------------------------------
# Live audible tests
# ---------------------------------------------------------------------------


@audible_only
@pytest.mark.asyncio
async def test_audible_qwen3_long_text_produces_full_audio(live_client):
    """The reproduction of the original bug: full 12-line poem on qwen3-1-7b.

    Without the helper, qwen3 truncates at 15.9 s of output.
    With the helper, expected duration is ~35-40 s (3 stanzas, each ~12 s).

    AUDIBLE CHECK: play the resulting file and confirm you hear ALL THREE
    STANZAS to completion. The third stanza ends with "To carry day past
    every night."
    """
    output_path = OUTPUT_DIR / "qwen3_long_via_helper.mp3"

    segment_timings: list[tuple[int, float, int]] = []
    t0 = time.monotonic()

    chunks: list[bytes] = []
    stream = await live_client.audio.stream_long_text(
        input=POEM,
        model="tts-qwen3-1-7b",
        voice="Serena",
        response_format="mp3",
        max_concurrency=4,
        on_segment_complete=lambda i, t, b: segment_timings.append((i, t, b)),
        timeout=180.0,
    )
    async for chunk in stream:
        chunks.append(chunk)

    elapsed = time.monotonic() - t0
    body = b"".join(chunks)
    output_path.write_bytes(body)

    duration = _ffprobe_duration(output_path)

    print(
        f"\n[AUDIBLE TEST] qwen3 long-text helper output:\n"
        f"  path:         {output_path}\n"
        f"  bytes:        {len(body)}\n"
        f"  wall time:    {elapsed:.2f}s\n"
        f"  ffprobe dur:  {duration}s\n"
        f"  segment times: {segment_timings}\n"
        f"  to play:      afplay {output_path}\n"
    )

    assert len(body) > 0, "Helper produced empty output"

    # Hard assertion: the original bug capped output at exactly 15.896875s.
    # If our helper does not exceed that, we did not fix anything.
    if duration is not None:
        assert duration > 16.0, (
            f"qwen3 output is {duration}s — at or below the known 15.9s cap, "
            f"meaning the helper did NOT actually split the input. Check the "
            f"splitter and the MODEL_WORD_BUDGETS entry."
        )
        # Soft check: with 3 stanzas each ~10-13s we expect ~30-40s of audio
        assert duration > 25.0, (
            f"qwen3 output is only {duration}s; expected ~30-40s for the "
            f"full 3-stanza poem. Possible mid-segment truncation."
        )


@audible_only
@pytest.mark.asyncio
async def test_audible_qwen3_short_text_uses_fast_path(live_client):
    """A full stanza (25 words — exactly at the qwen3 budget) should still
    hit the single-call fast path and render as one uninterrupted clip.

    AUDIBLE CHECK: should sound like a clean read of the full first stanza,
    no concatenation artifacts, since the helper short-circuits to a single
    ``create_speech`` call when input fits the budget.
    """
    # First stanza of the test poem — 25 words, exactly at the qwen3 budget
    short_text = (
        "Beneath the silver moon's soft glow, "
        "Where rivers hum and willows sway, "
        "The night air holds a tender flow, "
        "As dreamers drift in twilight's play."
    )
    output_path = OUTPUT_DIR / "qwen3_short_fast_path.mp3"

    segment_timings: list[tuple[int, float, int]] = []

    chunks: list[bytes] = []
    stream = await live_client.audio.stream_long_text(
        input=short_text,
        model="tts-qwen3-1-7b",
        voice="Serena",
        response_format="mp3",
        on_segment_complete=lambda i, t, b: segment_timings.append((i, t, b)),
        timeout=120.0,
    )
    async for chunk in stream:
        chunks.append(chunk)

    body = b"".join(chunks)
    output_path.write_bytes(body)
    duration = _ffprobe_duration(output_path)

    # Fast path: no segment callback invoked because there's no split
    assert segment_timings == [], (
        f"Short text should hit single-segment fast path; got "
        f"{len(segment_timings)} segment callbacks"
    )

    print(
        f"\n[AUDIBLE TEST] qwen3 short-text fast-path output:\n"
        f"  path:         {output_path}\n"
        f"  bytes:        {len(body)}\n"
        f"  ffprobe dur:  {duration}s\n"
        f"  to play:      afplay {output_path}\n"
    )

    assert len(body) > 0


@audible_only
@pytest.mark.asyncio
async def test_audible_kokoro_under_budget_passes_through_to_single_call(live_client):
    """Kokoro on the 74-word poem fits under the 100-word default budget, so
    the helper should pass the input through to one ``create_speech`` call
    unchanged. Output should be identical to calling ``create_speech``
    directly — same byte count, same duration (~37 s).

    AUDIBLE CHECK: play the file. Should sound exactly like a normal
    kokoro rendering of the full poem — no concatenation, no pacing breaks.
    """
    output_path = OUTPUT_DIR / "kokoro_under_budget_passthrough.mp3"

    segment_timings: list[tuple[int, float, int]] = []
    t0 = time.monotonic()

    chunks: list[bytes] = []
    stream = await live_client.audio.stream_long_text(
        input=POEM,
        model="tts-kokoro",
        voice="af_alloy",
        response_format="mp3",
        max_concurrency=4,
        on_segment_complete=lambda i, t, b: segment_timings.append((i, t, b)),
        timeout=180.0,
    )
    async for chunk in stream:
        chunks.append(chunk)

    elapsed = time.monotonic() - t0
    body = b"".join(chunks)
    output_path.write_bytes(body)
    duration = _ffprobe_duration(output_path)

    print(
        f"\n[AUDIBLE TEST] kokoro under-budget passthrough output:\n"
        f"  path:         {output_path}\n"
        f"  bytes:        {len(body)}\n"
        f"  wall time:    {elapsed:.2f}s\n"
        f"  ffprobe dur:  {duration}s\n"
        f"  segment times: {segment_timings}\n"
        f"  to play:      afplay {output_path}\n"
    )

    assert len(body) > 0
    # Under budget → fast path → no segment callbacks
    assert segment_timings == [], (
        f"Under-budget input should hit single-segment fast path; "
        f"got {len(segment_timings)} segment callbacks. Splitter regression?"
    )
    if duration is not None:
        # Match the single-call baseline (~37 s) within tolerance
        assert duration > 30.0, (
            f"Kokoro full-poem duration {duration}s well below expected ~37s — "
            f"investigate whether create_speech itself regressed."
        )


@audible_only
@pytest.mark.asyncio
async def test_audible_kokoro_forced_split_via_low_budget(live_client):
    """When the caller deliberately lowers the budget below the input size,
    the helper splits even on a model that wouldn't need it. This exercises
    the parallel queue + stitcher code paths against a known-good model.

    AUDIBLE CHECK: the audio plays the entire poem in order, but you may
    hear slight pacing differences between segments (kokoro renders shorter
    chunks ~30 % faster than longer ones — documented in audio_helpers.py).
    """
    output_path = OUTPUT_DIR / "kokoro_forced_split.mp3"

    segment_timings: list[tuple[int, float, int]] = []

    chunks: list[bytes] = []
    stream = await live_client.audio.stream_long_text(
        input=POEM,
        model="tts-kokoro",
        voice="af_alloy",
        response_format="mp3",
        max_words_per_segment=25,  # force a split
        max_concurrency=4,
        on_segment_complete=lambda i, t, b: segment_timings.append((i, t, b)),
        timeout=180.0,
    )
    async for chunk in stream:
        chunks.append(chunk)

    body = b"".join(chunks)
    output_path.write_bytes(body)
    duration = _ffprobe_duration(output_path)

    print(
        f"\n[AUDIBLE TEST] kokoro forced-split output:\n"
        f"  path:         {output_path}\n"
        f"  bytes:        {len(body)}\n"
        f"  ffprobe dur:  {duration}s\n"
        f"  segment times: {segment_timings}\n"
        f"  to play:      afplay {output_path}\n"
    )

    assert len(body) > 0
    # With 25-word budget, the 74-word poem must produce >= 3 segments
    assert len(segment_timings) >= 3, (
        f"Forced-split kokoro should produce >=3 segment callbacks; got {len(segment_timings)}."
    )
