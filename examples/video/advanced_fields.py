#!/usr/bin/env python3
"""
Venice AI SDK - Video: Advanced Body Fields
===========================================

Demonstrates the advanced fields on ``client.video.submit()`` that go
beyond a basic text-to-video or image-to-video request:

- ``end_image_url`` — the final-frame reference for models that support
  transitions.
- ``audio_url`` — a reference audio track (WAV/MP3, max 30s / 15 MB) that
  the model tries to match timing to.
- ``reference_image_urls`` — up to 9 references for character / style
  consistency.
- ``reference_video_urls`` — up to 3 reference *video* donors (R2V) for
  Seedance 2.0 R2V models; the clip's subject motion / camera movement /
  style is inherited by the generation.
- ``elements`` — up to 4 structured character / object elements for
  element-aware models (e.g. Kling O3 R2V). Reference them in the prompt
  as ``@Element1``, ``@Element2``, … Each element may also carry a
  per-element ``video_url`` motion donor.
- ``scene_image_urls`` — up to 4 scene references; address as ``@Image1``,
  ``@Image2``, … in the prompt.

Dry-run by default
------------------
Queuing a video job costs real money, so this example is **dry-run by
default**: each section resolves a compatible model and builds the exact
payload it *would* submit, then prints what it would queue **without
calling** ``client.video.run`` — no cost, no orphaned server-side jobs.

To actually submit (and incur video-generation cost), set::

    VENICE_RUN_PAID_VIDEO=1

A live submission also needs **real reference media**: the inline 1×1 PNG
is a valid *image* and the synthesized silent WAV is valid *audio*, but the
``reference_video_urls`` / per-element ``video_url`` fields require a real
``.mp4``/``.mov`` clip — this example points those at a small public sample
clip (``SAMPLE_VIDEO_URL``) so the paid payloads are correct.

SDK caveat: ``VideoJob.__aexit__`` currently calls ``/video/complete`` on a
freshly-queued job, which the server rejects with "Request ID is invalid"
(a known SDK limitation, not addressed here). It surfaces only on the paid
path; the dry-run path never opens a job.

``/video/quote`` prices by model + duration + resolution only and does
not accept prompt or reference fields; use ``submit()`` for the advanced
payload and call ``quote()`` separately to sanity-check cost.

Note on model selection: the public ``VideoModelConstraints`` schema
exposes ``model_type`` / ``resolutions`` / ``durations`` / ``aspect_ratios``
/ ``audio*`` / ``video_input``, but does *not* advertise per-feature flags
for ``reference_image_urls`` / ``end_image_url`` / ``elements``. We
therefore inspect ``models.list(type="video")`` and pick a model whose
``id`` matches the documented family that supports each feature
(``*-reference-to-video``, ``*-transition``, Kling O3 R2V). If no such
model is available we skip the sub-function.
"""

import asyncio
import base64
import io
import os
import struct
import sys
import wave

from venice_ai import VeniceClient
from venice_ai.exceptions import APIError, VeniceError
from venice_ai.types.api.models import ModelResponse

# Submitting a video job costs money and (currently) leaves an orphaned
# server-side job on cleanup. Default to a dry run; opt in explicitly to pay.
RUN_PAID = os.environ.get("VENICE_RUN_PAID_VIDEO") == "1"

# Inline data URL for tiny sample images (a 1×1 red PNG). This is a *valid*
# image; for real usage, pass an HTTPS URL or a larger inline data URL.
_TINY_RED_PNG_DATA_URL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAA"
    "AADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
)

# A small, publicly-hosted sample ``.mp4`` for the *video* reference fields
# (``reference_video_urls`` and per-element ``video_url``). Unlike images,
# there is no cheap inline placeholder for a valid video, so we point at a
# real clip — the payloads below are therefore correct for a paid submission.
SAMPLE_VIDEO_URL = (
    "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"
)


def _silent_wav_data_url(seconds: int = 2, sample_rate: int = 8000) -> str:
    """Build a small valid mono WAV data URL (silence) for R2V reference audio.

    ``reference_audio_urls`` requires real ``.wav``/``.mp3`` clips of 2–15s. We
    synthesize a short silent WAV inline so the example is self-contained — for
    a real workflow, point at a publicly-hosted clip with actual vocals/SFX.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack("<h", 0) * sample_rate * seconds)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


async def _list_video_models(client: VeniceClient) -> list[ModelResponse]:
    """Fetch available video models, filtering offline/beta where possible."""
    listing = await client.models.list(type="video")
    out: list[ModelResponse] = []
    for m in listing.data:
        spec = getattr(m, "model_spec", None)
        if spec is None:
            continue
        if getattr(spec, "offline", False):
            continue
        out.append(m)
    return out


def _pick_by_id_substring(
    models: list[ModelResponse],
    *,
    contains: tuple[str, ...],
    require_model_type: str | None = None,
) -> str | None:
    """Pick the first non-beta model whose id contains any of ``contains``.

    Falls back to a beta model only if no non-beta candidate is found.
    """
    beta_fallback: str | None = None
    for m in models:
        spec = m.model_spec
        if spec is None:
            continue
        constraints = getattr(spec, "constraints", None)
        if require_model_type is not None:
            mt = getattr(constraints, "model_type", None) if constraints else None
            if mt != require_model_type:
                continue
        mid = m.id.lower()
        if not any(token in mid for token in contains):
            continue
        if getattr(spec, "beta", False):
            beta_fallback = beta_fallback or m.id
            continue
        return m.id
    return beta_fallback


def _advanced_fields(payload: dict) -> str:
    """Comma-listed advanced fields in ``payload`` (excludes the basics).

    Keeps the dry-run summary line honest: it is derived from the *same* dict
    that the paid path submits, so the two can never drift apart.
    """
    basics = {"model", "prompt", "duration_seconds"}
    extras = [k for k in payload if k not in basics]
    return ", ".join(extras) if extras else "(none)"


async def _run_or_dry_run(payload: dict) -> bool:
    """Either dry-run (default) or actually submit ``payload``.

    Dry-run (``VENICE_RUN_PAID_VIDEO`` unset): print what would be queued and
    return True without calling ``client.video.run`` — no cost, no orphaned
    jobs. Paid (``VENICE_RUN_PAID_VIDEO=1``): queue the job and print the
    ``queue_id``; on a genuine API/Venice error, fail loud (return False).
    """
    model = payload["model"]
    if not RUN_PAID:
        print(
            f"🧪 dry run — would queue on {model} with {_advanced_fields(payload)}; "
            "set VENICE_RUN_PAID_VIDEO=1 to actually submit (incurs video cost)"
        )
        return True

    async with VeniceClient() as client:
        try:
            # NOTE: the SDK's VideoJob.__aexit__ calls /video/complete on this
            # freshly-queued job, which the server currently rejects with
            # "Request ID is invalid" — a known SDK limitation, not fixed here.
            async with await client.video.run(**payload) as job:
                print(f"📼 Queue ID: {job.queue_id}")
            return True
        except (VeniceError, APIError) as e:
            print(f"❌ Queue failed: {e}")
            return False


async def queue_with_reference_images() -> bool:
    """Queue a job that uses ``reference_image_urls`` for style consistency.

    Picks a model from the ``*-reference-to-video`` family, which is the
    documented family that accepts multi-reference style/character inputs.
    Returns True on success (or legitimate skip), False on genuine failure.
    """
    print("🖼️  queue() with reference_image_urls")
    print("-" * 30)

    async with VeniceClient() as client:
        try:
            models = await _list_video_models(client)
        except (VeniceError, APIError) as e:
            print(f"❌ Failed to list video models: {e}")
            return False

    # Prefer pixverse-c1-reference-to-video (cheapest fast path), then any
    # other ``*-reference-to-video`` model.
    video_model = _pick_by_id_substring(
        models,
        contains=("pixverse-c1-reference-to-video",),
        require_model_type="image-to-video",
    ) or _pick_by_id_substring(
        models,
        contains=("-reference-to-video",),
        require_model_type="image-to-video",
    )
    if video_model is None:
        print("⏭️  skipped (no *-reference-to-video model in catalog)")
        return True  # legitimate skip, not a failure

    print(f"📍 Model: {video_model}")
    payload = {
        "model": video_model,
        "prompt": "A character with consistent features walking through a meadow",
        "duration_seconds": "5s",
        "image_url": _TINY_RED_PNG_DATA_URL,
        "reference_image_urls": [_TINY_RED_PNG_DATA_URL] * 2,
        # ``*-reference-to-video`` models require ``aspect_ratio`` explicitly
        # because they don't infer it from the input image.
        "aspect_ratio": "16:9",
    }
    return await _run_or_dry_run(payload)


async def queue_with_end_image_url() -> bool:
    """Queue a job with a start + end frame for a transition.

    Uses a ``*-transition`` model, which is the documented family for
    start→end-frame image-to-video transitions.
    """
    print("\n🎬 queue() with end_image_url (transition)")
    print("-" * 30)

    async with VeniceClient() as client:
        try:
            models = await _list_video_models(client)
        except (VeniceError, APIError) as e:
            print(f"❌ Failed to list video models: {e}")
            return False

    # Prefer pixverse-c1-transition then pixverse-v5.6-transition.
    video_model = _pick_by_id_substring(
        models,
        contains=("pixverse-c1-transition",),
        require_model_type="image-to-video",
    ) or _pick_by_id_substring(
        models,
        contains=("-transition",),
        require_model_type="image-to-video",
    )
    if video_model is None:
        print("⏭️  skipped (no *-transition model in catalog)")
        return True

    print(f"📍 Model: {video_model}")
    payload = {
        "model": video_model,
        "prompt": "Smoothly transition from the first frame to the second",
        "duration_seconds": "5s",
        "image_url": _TINY_RED_PNG_DATA_URL,
        "end_image_url": _TINY_RED_PNG_DATA_URL,
        # ``*-transition`` models require ``aspect_ratio`` explicitly.
        "aspect_ratio": "16:9",
    }
    return await _run_or_dry_run(payload)


async def queue_with_reference_audio() -> bool:
    """Queue a Seedance 2.0 R2V job that uses ``reference_audio_urls``.

    Reference-to-video (R2V) models such as Seedance 2.0 accept up to 3
    ``reference_audio_urls`` — donor clips for vocal timbre / narration / SFX —
    which **must be paired with at least one reference image** (or reference
    video). Each clip is 2–15s, ``.wav``/``.mp3``, aggregate ≤15s, supplied as a
    public URL or a data URL.

    Picks a ``seedance-*-reference-to-video`` model; skips if no R2V model is in
    the catalog.
    """
    print("\n🔊 queue() with reference_audio_urls (Seedance R2V)")
    print("-" * 30)

    async with VeniceClient() as client:
        try:
            models = await _list_video_models(client)
        except (VeniceError, APIError) as e:
            print(f"❌ Failed to list video models: {e}")
            return False

    # Prefer the cheaper fast R2V model, then the standard one, then any
    # seedance reference-to-video model.
    video_model = (
        _pick_by_id_substring(
            models,
            contains=("seedance-2-0-fast-reference-to-video",),
            require_model_type="image-to-video",
        )
        or _pick_by_id_substring(
            models,
            contains=("seedance-2-0-reference-to-video",),
            require_model_type="image-to-video",
        )
        or _pick_by_id_substring(
            models,
            contains=("seedance", "-reference-to-video"),
            require_model_type="image-to-video",
        )
    )
    if video_model is None:
        print("⏭️  skipped (no Seedance R2V model in catalog)")
        return True

    print(f"📍 Model: {video_model}")
    payload = {
        "model": video_model,
        "prompt": "A narrator's voice carries over a slow pan across a misty valley",
        "duration_seconds": "5s",
        "image_url": _TINY_RED_PNG_DATA_URL,
        # reference_audio_urls MUST be paired with a reference image/video.
        "reference_image_urls": [_TINY_RED_PNG_DATA_URL],
        "reference_audio_urls": [_silent_wav_data_url(seconds=2)],
        "aspect_ratio": "16:9",
    }
    return await _run_or_dry_run(payload)


async def queue_with_reference_video_urls() -> bool:
    """Queue a Seedance 2.0 R2V job that uses ``reference_video_urls`` (R2V).

    Reference-to-video (R2V) models such as Seedance 2.0 accept up to 3
    ``reference_video_urls`` — donor clips the model uses to **inherit subject
    motion, camera movement, and overall style**. Each clip is 2–15s,
    ``.mp4``/``.mov``, ≤50 MB, aggregate ≤15s, supplied as a public URL or a
    data URL. This example uses ``SAMPLE_VIDEO_URL`` (a small public ``.mp4``)
    so the payload is a real, submittable R2V request.

    Picks a ``seedance-*-reference-to-video`` model; skips if no R2V model is in
    the catalog.
    """
    print("\n🎞️  queue() with reference_video_urls (Seedance R2V)")
    print("-" * 30)

    async with VeniceClient() as client:
        try:
            models = await _list_video_models(client)
        except (VeniceError, APIError) as e:
            print(f"❌ Failed to list video models: {e}")
            return False

    # Prefer the cheaper fast R2V model, then the standard one, then any
    # seedance reference-to-video model.
    video_model = (
        _pick_by_id_substring(
            models,
            contains=("seedance-2-0-fast-reference-to-video",),
            require_model_type="image-to-video",
        )
        or _pick_by_id_substring(
            models,
            contains=("seedance-2-0-reference-to-video",),
            require_model_type="image-to-video",
        )
        or _pick_by_id_substring(
            models,
            contains=("seedance", "-reference-to-video"),
            require_model_type="image-to-video",
        )
    )
    if video_model is None:
        print("⏭️  skipped (no Seedance R2V model in catalog)")
        return True

    print(f"📍 Model: {video_model}")
    payload = {
        "model": video_model,
        "prompt": "Inherit the donor clip's camera motion across a neon city street",
        "duration_seconds": "5s",
        "image_url": _TINY_RED_PNG_DATA_URL,
        # reference_video_urls are real motion/style donors (.mp4/.mov).
        "reference_video_urls": [SAMPLE_VIDEO_URL],
        "aspect_ratio": "16:9",
    }
    return await _run_or_dry_run(payload)


async def queue_with_elements_and_scenes() -> bool:
    """Queue a Kling-O3-style request with structured elements + scene refs.

    Element-aware models (Kling O3 R2V and similar) accept the structured
    ``elements`` payload plus ``scene_image_urls`` referenced as
    ``@Element1`` / ``@Image1`` etc. in the prompt. Each element may also carry
    a per-element ``video_url`` motion donor (``VideoElement.video_url``) used
    by models that accept per-element motion references; this example uses
    ``SAMPLE_VIDEO_URL`` (a real ``.mp4``) for that field.

    Picks a Kling O3 R2V model; skips if no element-aware model is in the
    catalog.
    """
    print("\n🎭 queue() with elements + scene_image_urls")
    print("-" * 30)

    async with VeniceClient() as client:
        try:
            models = await _list_video_models(client)
        except (VeniceError, APIError) as e:
            print(f"❌ Failed to list video models: {e}")
            return False

    # Prefer Kling O3 reference-to-video, then any kling-o3 model as a fallback
    # (the API may accept ``elements`` on related families even though the
    # contract is documented for Kling O3).
    video_model = _pick_by_id_substring(
        models,
        contains=("kling-o3-pro-reference-to-video", "kling-o3-standard-reference-to-video"),
        require_model_type="image-to-video",
    ) or _pick_by_id_substring(
        models,
        contains=("kling-o3",),
        require_model_type="image-to-video",
    )
    if video_model is None:
        print("⏭️  skipped (no element-aware Kling O3 R2V model in catalog)")
        return True

    print(f"📍 Model: {video_model}")
    payload = {
        "model": video_model,
        "prompt": "Show @Element1 exploring @Image1 in a cinematic tracking shot",
        "duration_seconds": "5s",
        "elements": [
            {
                "frontal_image_url": _TINY_RED_PNG_DATA_URL,
                "reference_image_urls": [_TINY_RED_PNG_DATA_URL],
                # Per-element motion donor (VideoElement.video_url): a real .mp4.
                "video_url": SAMPLE_VIDEO_URL,
            },
        ],
        "scene_image_urls": [_TINY_RED_PNG_DATA_URL],
        # Kling O3 R2V requires an explicit aspect_ratio.
        "aspect_ratio": "16:9",
    }
    return await _run_or_dry_run(payload)


async def main() -> int:
    print("🚀 Venice AI Video — Advanced Fields Example")
    print("=" * 50)
    if RUN_PAID:
        print("💸 VENICE_RUN_PAID_VIDEO=1 — will submit real (paid) video jobs.")
    else:
        print("🧪 Dry run (default). Set VENICE_RUN_PAID_VIDEO=1 to actually submit.")

    results: dict[str, bool] = {}
    results["reference_image_urls"] = await queue_with_reference_images()
    results["end_image_url"] = await queue_with_end_image_url()
    results["reference_audio_urls"] = await queue_with_reference_audio()
    results["reference_video_urls"] = await queue_with_reference_video_urls()
    results["elements_and_scenes"] = await queue_with_elements_and_scenes()

    print("\n✨ Done.")
    print("\n💡 Tip: call quote() beforehand with just model + duration +")
    print("   resolution to get a cost estimate; pricing does not depend")
    print("   on prompt text or reference images.")

    print("\n📊 Summary:")
    for name, ok in results.items():
        print(f"   {'✅' if ok else '❌'} {name}")

    failed = [name for name, ok in results.items() if not ok]
    if failed:
        print(f"\n❌ Failed sub-functions: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        rc = asyncio.run(main())
        sys.exit(rc)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
