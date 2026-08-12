#!/usr/bin/env python3
"""
Venice AI SDK - Text-to-Video Generation
=========================================

This example demonstrates how to generate videos from text prompts using the
Venice AI SDK. Video generation is **queue-based** and asynchronous; the SDK
exposes the lifecycle through :class:`venice_ai.VideoJob`:

    1. ``client.video.quote(...)`` — get an estimated cost in USD.
    2. ``client.video.run(...)`` — submit and receive a ``VideoJob``.
    3. ``async with job:`` — use as a context manager to guarantee server-side
       cleanup on exit.
    4. ``await job.wait(on_progress=...)`` — poll until ``VideoCompletedStatus``
       (raises :class:`VideoGenerationError` on failure or ``TimeoutError`` on
       poll exhaustion).
    5. ``await job.download(path, status)`` — write the video to disk using
       the SDK's managed HTTP session (no separate ``aiohttp`` session needed).

Each section below is self-contained — you can run them independently.
"""

import asyncio
import sys
from pathlib import Path

from venice_ai import VeniceClient, VideoGenerationError, VideoJob
from venice_ai.types.api.video import VideoProcessingStatus

# Resolve results dir relative to this file's location.
# All example scripts live one level below examples/ (e.g., examples/video/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_progress(status: VideoProcessingStatus) -> None:
    """Render a one-line progress update suitable for ``VideoJob.wait``."""
    remaining_s = status.estimated_remaining_ms / 1000 if status.estimated_remaining_ms else 0
    print(f"⏳ Processing… {status.progress_percent:.0f}% (~{remaining_s:.0f}s remaining)")


async def _wait_and_save(job: VideoJob, output_path: Path) -> bool:
    """Wait for completion and save to disk. Returns True on success."""
    print(f"⏳ Polling for completion (queue_id={job.queue_id}) ...")
    try:
        status = await job.wait(on_progress=_print_progress)
    except VideoGenerationError as e:
        print(f"❌ Generation failed: {e} (code={e.error_code})")
        return False
    except TimeoutError as e:
        print(f"⚠️ {e}")
        return False

    saved = await job.download(output_path, status)
    size = saved.stat().st_size
    print(f"✅ Video ready — saved to {saved} ({size} bytes)")
    return True


# ---------------------------------------------------------------------------
# Section 1 — Basic text-to-video generation
# ---------------------------------------------------------------------------


async def basic_text_to_video() -> bool:
    """Generate a video from a simple text prompt.

    Returns ``True`` on success, ``False`` if submission, generation, or the
    save step failed.
    """
    print("🎬 Basic Text-to-Video Generation")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            # Dynamically select a text-to-video model
            model = await client.models.resolve_video(video_type="text-to-video")
            prompt = "A serene mountain landscape at sunrise with golden light spilling over snow-capped peaks"

            print(f"📍 Using model: {model}")
            print(f"📝 Prompt: {prompt}")

            # run() returns a VideoJob; using it as an async context
            # manager guarantees server-side cleanup (job.cancel()) runs
            # on exit, even if wait()/download() raise.
            async with await client.video.run(
                model=model,
                prompt=prompt,
                duration_seconds=5,
                aspect_ratio="16:9",
            ) as job:
                print(f"📨 Queued — queue_id: {job.queue_id}")
                ok = await _wait_and_save(job, RESULTS_DIR / "basic_video.mp4")

        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Note: Video generation requires appropriate API access")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# Section 2 — Cost estimation with quote()
# ---------------------------------------------------------------------------


async def cost_estimation() -> bool:
    """Get a price quote before committing to generation.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print("\n💰 Cost Estimation with quote()")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            model = await client.models.resolve_video(video_type="text-to-video")

            # Compare costs for different durations
            durations = [5, 10]
            for duration in durations:
                quote_resp = await client.video.quote(
                    model=model,
                    duration_seconds=duration,
                    resolution="720p",
                    aspect_ratio="16:9",
                )
                print(f"💵 {duration}s video at 720p: ${float(quote_resp.quote):.4f}")

            # Compare resolutions at fixed duration. A resolution that the model
            # does not support is an expected, informational outcome — not a
            # failure of the demo — so it does not flip ``ok``.
            print()
            resolutions = ["720p", "1080p"]
            for resolution in resolutions:
                try:
                    quote_resp = await client.video.quote(
                        model=model,
                        duration_seconds=5,
                        resolution=resolution,
                        aspect_ratio="16:9",
                    )
                    print(f"💵 5s video at {resolution}: ${float(quote_resp.quote):.4f}")
                except Exception as e:
                    print(f"ℹ️  {resolution} not available for this model: {e}")

        except Exception as e:
            print(f"❌ Error: {e}")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# Section 3 — Advanced options
# ---------------------------------------------------------------------------


async def advanced_options() -> bool:
    """Demonstrate advanced generation parameters.

    Returns ``True`` on success, ``False`` if submission, generation, or the
    save step failed.
    """
    print("\n⚙️  Advanced Options")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            model = await client.models.resolve_video(video_type="text-to-video")
            prompt = (
                "A slow-motion closeup of ocean waves crashing on a rocky shore, "
                "cinematic lighting, 4K detail"
            )
            negative_prompt = "blurry, low quality, text, watermark"

            print(f"📍 Model: {model}")
            print(f"📝 Prompt: {prompt}")
            print(f"🚫 Negative: {negative_prompt}")

            async with await client.video.run(
                model=model,
                prompt=prompt,
                duration_seconds=5,
                resolution="720p",
                aspect_ratio="16:9",
                audio=False,
                negative_prompt=negative_prompt,
            ) as job:
                print(f"📨 Queued — queue_id: {job.queue_id}")
                ok = await _wait_and_save(job, RESULTS_DIR / "advanced_video.mp4")

        except Exception as e:
            print(f"❌ Error: {e}")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# Section 4 — Model discovery
# ---------------------------------------------------------------------------


async def model_discovery() -> bool:
    """Use the model selector to find available text-to-video models.

    Returns ``True`` on success, ``False`` if submission, generation, or the
    save step failed.
    """
    print("\n🔍 Model Discovery")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            # Intelligently select the best T2V model
            selected = await client.models.resolve_video(video_type="text-to-video")
            print(f"\n🏆 Selected T2V model: {selected}")

            # Get a price quote with the selected model
            quote_resp = await client.video.quote(
                model=selected,
                duration_seconds=5,
            )
            print(f"💵 Estimated cost: ${float(quote_resp.quote):.4f}")

            async with await client.video.run(
                model=selected,
                prompt="A golden retriever running through a meadow",
                duration_seconds=5,
                aspect_ratio="16:9",
            ) as job:
                print(f"📨 Queued — queue_id: {job.queue_id}")
                ok = await _wait_and_save(job, RESULTS_DIR / "discovered_model_video.mp4")

        except Exception as e:
            print(f"❌ Error: {e}")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    """Run all text-to-video examples.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real
    submit/generation failure surfaces as a non-zero process exit instead of
    being masked by the success banner.
    """
    print("🚀 Venice AI Text-to-Video Examples")
    print("=" * 60)

    results: list[tuple[str, bool]] = [
        ("basic_text_to_video", await basic_text_to_video()),
        ("cost_estimation", await cost_estimation()),
        ("advanced_options", await advanced_options()),
        ("model_discovery", await model_discovery()),
    ]

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("\n✨ Text-to-video examples completed!")

    print("\n💡 Key concepts demonstrated:")
    print("   - VideoJob context manager (auto cleanup on exit)")
    print("   - job.wait(on_progress=...) replaces hand-rolled polling")
    print("   - job.download(path, status) reuses the SDK's HTTP session")
    print("   - Cost estimation with quote() before committing")
    print("   - Advanced parameters (negative_prompt, resolution, audio)")
    print("   - Dynamic model discovery and selection")

    # Only advertise generated files when every generation demo succeeded —
    # otherwise we'd imply outputs exist that a failed run never wrote.
    if not failed:
        print(f"\n📁 Generated files in {RESULTS_DIR}/:")
        print("   - basic_video.mp4")
        print("   - advanced_video.mp4")
        print("   - discovered_model_video.mp4")

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
