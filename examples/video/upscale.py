#!/usr/bin/env python3
"""
Venice AI SDK - Video: Upscale (topaz-video-upscale)
====================================================

Demonstrates the video-upscale flow on ``client.video.submit()`` using the
``topaz-video-upscale`` model. Upscaling is distinct from generation:

- Provide ``video_url`` (the source video) instead of a generation prompt.
- Use ``upscale_factor`` (1 / 2 / 4) instead of ``resolution``.
  ``1`` = quality-only enhancement, ``2`` = double dimensions (default),
  ``4`` = quadruple.
- Duration / FPS are detected automatically from the source file.

The flow is otherwise identical to a normal ``queue → retrieve → complete``
lifecycle, and ``client.video.run()`` wraps the whole thing in a
``VideoJob`` context manager if you want the lifecycle handled for you.

Model resolution: ``client.models.list(type="upscale")`` returns the *image*
upscaler, not the video upscaler. Use the dedicated shortcut
``client.models.resolve_video_upscale()`` instead — it filters
``type="video"`` for ``model_type="video"`` + ``video_input=True`` and
returns ``topaz-video-upscale``.
"""

import asyncio
import sys

from venice_ai import VeniceClient
from venice_ai.exceptions import APIError, VeniceError

# A small public-domain sample clip so this example produces a real quote
# out of the box. Replace with your own source video URL (HTTPS or a
# ``data:`` URI). Data URIs work but get large fast — prefer a signed URL
# for real content.
SOURCE_VIDEO_URL = (
    "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"
)


async def quote_upscale() -> bool:
    """Get a cost estimate for upscaling a video 2× without running it.

    Returns True on success, False on failure.
    """
    print("💰 Quote an Upscale Job")
    print("-" * 30)

    async with VeniceClient() as client:
        # Pick the upscale model dynamically — never hardcode.
        try:
            upscale_model = await client.models.resolve_video_upscale()
        except (ValueError, VeniceError, APIError) as e:
            print(f"❌ Could not resolve a video-upscale model: {e}")
            return False
        print(f"📍 Model: {upscale_model}")

        try:
            quote = await client.video.quote(
                model=upscale_model,
                duration_seconds="Auto",  # detected from the source video
                video_url=SOURCE_VIDEO_URL,
                upscale_factor=2,
            )
            print(f"💰 Estimated cost: ${quote.quote:.4f}")
            return True
        except (VeniceError, APIError) as e:
            # A genuine failure (e.g. the server can't fetch SOURCE_VIDEO_URL,
            # or it isn't a supported video) — report it honestly and fail.
            # Don't mask it as success: a shipped example should print a real
            # quote, so a 4xx here means the URL/params need fixing.
            print(f"❌ Quote failed: {e}")
            print("   Ensure SOURCE_VIDEO_URL points at a fetchable video the")
            print("   server can inspect (a public HTTPS URL or a data: URI).")
            return False


async def queue_upscale_job() -> bool:
    """Queue an upscale job (cost) — uncomment to actually run.

    Returns True on success (model resolved), False on failure.
    """
    print("\n🚀 Queue an Upscale Job")
    print("-" * 30)
    print("⚠️  This incurs real cost — uncomment the call below to run.")

    async with VeniceClient() as client:
        try:
            upscale_model = await client.models.resolve_video_upscale()
        except (ValueError, VeniceError, APIError) as e:
            print(f"❌ Could not resolve a video-upscale model: {e}")
            return False

        # Uncomment to actually queue the upscale:
        # try:
        #     result = await client.video.submit(
        #         model=upscale_model,
        #         prompt="upscale",
        #         duration_seconds="Auto",
        #         video_url=SOURCE_VIDEO_URL,
        #         upscale_factor=2,  # 1, 2, or 4
        #     )
        #     print(f"✅ Queued: {result.queue_id} on {result.model}")
        #     # ...then poll client.video.retrieve(model=..., queue_id=...)
        #     # until complete, download the file, and call
        #     # client.video.cancel(...) to release server-side storage.
        #     #
        #     # Or simpler: use client.video.run(...) for the whole
        #     # lifecycle wrapped in a VideoJob context manager.
        # except (VeniceError, APIError) as e:
        #     print(f"❌ Queue failed: {e}")
        print(f"   (would have queued on {upscale_model})")
        return True


async def main() -> int:
    print("🚀 Venice AI Video — Upscale Example")
    print("=" * 50)

    results: dict[str, bool] = {}
    results["quote"] = await quote_upscale()
    results["queue"] = await queue_upscale_job()

    print("\n✨ Done.")
    print("\n💡 Tips:")
    print("   - upscale_factor=1 means quality-only enhancement (same size).")
    print("   - upscale_factor=4 quadruples width × height; expensive.")
    print("   - Pass 'duration_seconds=Auto' so the server detects length from the")
    print("     source file; don't hardcode a duration for upscale.")

    failed = [name for name, ok in results.items() if not ok]
    if failed:
        print(f"\n❌ Failed steps: {failed}")
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
