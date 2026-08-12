#!/usr/bin/env python3
"""
Venice AI SDK - Image-to-Video Generation
==========================================

This example demonstrates how to animate still images into videos using the
Venice AI SDK's **image-to-video (I2V)** capabilities.  The workflow is
identical to text-to-video except that you supply an ``image_url`` parameter
pointing to a publicly accessible reference image:

    1. ``client.video.quote(...)`` — estimate the cost.
    2. ``client.video.run(..., image_url=...)`` — submit and receive a
       :class:`venice_ai.VideoJob`.
    3. ``async with job:`` — guarantees server-side cleanup on exit.
    4. ``await job.wait(on_progress=...)`` — polls until completion (raises
       :class:`VideoGenerationError` on failure or ``TimeoutError`` on
       poll exhaustion).
    5. ``await job.download(path, status)`` — saves to disk via the SDK's
       managed HTTP session.

The ``prompt`` describes the **desired motion or animation** — not the image
content itself.  For example, given a photo of a mountain lake you might
prompt *"gentle ripples spread across the water as a breeze picks up"*.

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

# A publicly-accessible sample image (Picsum Photos CDN — forest scenery).
# Uses the direct fastly CDN URL to avoid 302 redirects that the Venice
# API may not follow.
SAMPLE_IMAGE_URL = (
    "https://fastly.picsum.photos/id/10/800/600.jpg"
    "?hmac=9u_ZYBasFb_VEVrBgjTZor_IfBxtpq9zl_CjKJr7-cs"
)


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
# Section 1 — Basic image-to-video generation
# ---------------------------------------------------------------------------


async def basic_image_to_video() -> bool:
    """Animate a reference image into a short video clip.

    Returns ``True`` on success, ``False`` if generation/save failed.
    """
    print("🖼️  Basic Image-to-Video Generation")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            # Dynamically select an image-to-video model
            model = await client.models.resolve_video(video_type="image-to-video")
            prompt = (
                "The scene slowly comes to life — colours gently shift and a "
                "soft breeze seems to move through the frame"
            )

            print(f"📍 Using model: {model}")
            print(f"🖼️  Image URL: {SAMPLE_IMAGE_URL[:80]}…")
            print(f"📝 Prompt (motion): {prompt}")
            print()
            print(
                "ℹ️  Note: image_url must be publicly accessible (http://, https://, or data: URI)"
            )

            # run() with image_url switches to I2V mode and returns
            # a VideoJob; the context manager handles server-side cleanup.
            async with await client.video.run(
                model=model,
                prompt=prompt,
                image_url=SAMPLE_IMAGE_URL,
                duration_seconds=5,
            ) as job:
                print(f"📨 Queued — queue_id: {job.queue_id}")
                return await _wait_and_save(job, RESULTS_DIR / "i2v_basic.mp4")

        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Note: Image-to-video requires appropriate API access")
            return False


# ---------------------------------------------------------------------------
# Section 2 — Cost estimation for I2V
# ---------------------------------------------------------------------------


async def cost_estimation() -> bool:
    """Get a price quote for image-to-video generation.

    Per-duration/resolution failures are tolerated (some combos may be
    unavailable); returns ``False`` only if the demo itself errored out.
    """
    print("\n💰 I2V Cost Estimation with quote()")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            model = await client.models.resolve_video(video_type="image-to-video")

            # Compare durations
            print("📊 Cost comparison by duration:")
            durations = [5, 10]
            for duration in durations:
                try:
                    quote_resp = await client.video.quote(
                        model=model,
                        duration_seconds=duration,
                    )
                    print(f"   💵 {duration}s video: ${float(quote_resp.quote):.4f}")
                except Exception as e:
                    print(f"   ℹ️  {duration}s not available: {e}")

            # Compare resolutions
            print("\n📊 Cost comparison by resolution:")
            resolutions = ["720p", "1080p"]
            for resolution in resolutions:
                try:
                    quote_resp = await client.video.quote(
                        model=model,
                        duration_seconds=5,
                        resolution=resolution,
                    )
                    print(f"   💵 5s video at {resolution}: ${float(quote_resp.quote):.4f}")
                except Exception as e:
                    print(f"   ℹ️  {resolution} not available: {e}")

            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False


# ---------------------------------------------------------------------------
# Section 3 — Creative animation prompts
# ---------------------------------------------------------------------------


async def creative_prompts() -> bool:
    """Show how different prompts animate the same image differently.

    The image stays the same, but the *motion* described in the prompt
    controls the animation style.  Compare "gentle breeze" with
    "dramatic storm" to see how prompt wording steers the output.

    Returns ``True`` only if every variation generated and saved.
    """
    print("\n🎨 Creative Animation Prompts")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            model = await client.models.resolve_video(video_type="image-to-video")

            # Two contrasting motion prompts for the same image
            variations = [
                {
                    "label": "Gentle",
                    "prompt": (
                        "A gentle breeze blows through the scene, creating soft, calming motion"
                    ),
                    "filename": "i2v_gentle.mp4",
                },
                {
                    "label": "Dramatic",
                    "prompt": (
                        "A dramatic storm builds — wind gusts intensify, "
                        "objects sway violently, and lighting shifts rapidly"
                    ),
                    "filename": "i2v_dramatic.mp4",
                },
            ]

            print(f"🖼️  Same image: {SAMPLE_IMAGE_URL[:60]}…")
            print("🎯 Different prompts → different animations\n")

            ok = True
            for var in variations:
                print(f"🎬 Variation: {var['label']}")
                print(f"   📝 Motion prompt: {var['prompt']}")

                async with await client.video.run(
                    model=model,
                    prompt=var["prompt"],
                    image_url=SAMPLE_IMAGE_URL,
                    duration_seconds=5,
                ) as job:
                    print(f"   📨 Queued — queue_id: {job.queue_id}")
                    if not await _wait_and_save(job, RESULTS_DIR / var["filename"]):
                        ok = False
                print()

            return ok

        except Exception as e:
            print(f"❌ Error: {e}")
            return False


# ---------------------------------------------------------------------------
# Section 4 — Model discovery for I2V
# ---------------------------------------------------------------------------


async def model_discovery() -> bool:
    """Use the model selector to find available image-to-video models.

    Returns ``True`` on success, ``False`` if generation/save failed.
    """
    print("\n🔍 I2V Model Discovery")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            # Select the best I2V model specifically
            selected = await client.models.resolve_video(video_type="image-to-video")
            print(f"\n🏆 Selected I2V model: {selected}")

            # Get a cost estimate with the selected model
            quote_resp = await client.video.quote(
                model=selected,
                duration_seconds=5,
            )
            print(f"💵 Estimated cost: ${float(quote_resp.quote):.4f}")

            async with await client.video.run(
                model=selected,
                prompt="Gentle wind blows through the scene",
                image_url=SAMPLE_IMAGE_URL,
                duration_seconds=5,
            ) as job:
                print(f"📨 Queued — queue_id: {job.queue_id}")
                return await _wait_and_save(job, RESULTS_DIR / "i2v_discovered_model.mp4")

        except Exception as e:
            print(f"❌ Error: {e}")
            return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    """Run all image-to-video examples.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    print("🚀 Venice AI Image-to-Video Examples")
    print("=" * 60)

    results: list[tuple[str, bool]] = [
        ("basic_image_to_video", await basic_image_to_video()),
        ("cost_estimation", await cost_estimation()),
        ("creative_prompts", await creative_prompts()),
        ("model_discovery", await model_discovery()),
    ]

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("\n✨ Image-to-video examples completed!")

    print("\n💡 Key concepts demonstrated:")
    print("   - image_url switches run() to I2V mode")
    print("   - VideoJob context manager handles cleanup automatically")
    print("   - Prompt describes desired motion, not image content")
    print("   - Cost estimation with quote()")
    print("   - Same image + different prompts → different animations")
    print("   - Dynamic I2V model discovery via client.models.resolve_video()")

    print(f"\n📁 Generated files in {RESULTS_DIR}/:")
    print("   - i2v_basic.mp4")
    print("   - i2v_gentle.mp4")
    print("   - i2v_dramatic.mp4")
    print("   - i2v_discovered_model.mp4")

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
