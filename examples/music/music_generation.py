#!/usr/bin/env python3
"""
Venice AI SDK — Music Generation
================================

End-to-end example of the async music generation flow. The SDK exposes
music generation on its own resource — :attr:`VeniceClient.music` — even
though it shares the underlying ``/audio/*`` queue endpoints with TTS /
ASR. The lifecycle mirrors video: ``submit`` → ``retrieve`` → ``cancel``,
or use ``run()`` for the high-level :class:`MusicJob` context manager.

Key features covered:
- Dynamic model selection via ``client.models.resolve_music()`` (never
  hardcode model IDs)
- Price quoting before submission
- The high-level :class:`MusicJob` lifecycle — context-managed, guarantees
  server-side cleanup
- Low-level ``client.music.submit`` / ``retrieve`` / ``cancel``
- Saving the audio to disk
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

from venice_ai import MusicGenerationError, VeniceClient
from venice_ai.types.api.music import (
    MusicCompletedStatus,
    MusicFailedStatus,
)

# Resolve results dir relative to this file's location.
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _ts() -> str:
    """Compact timestamp suffix for unique output filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


async def quote_then_generate() -> bool:
    """Demonstrate the end-to-end flow with dynamic model selection.

    Returns ``True`` on success, ``False`` if the quote, generation, or
    download failed — so a real failure is tallied rather than masked.
    """
    print("🎵 Music Generation — Quote → Generate → Download")
    print("-" * 60)

    try:
        async with VeniceClient() as client:
            # Prefer a flexible-duration model; different models enforce different
            # duration tiers (e.g. ACE-Step only accepts 60 / 90 / 120 / ...),
            # so inspect ``/models?type=music`` for per-model constraints.
            model = await client.models.resolve_music(preferred_models=["elevenlabs-music"])
            print(f"   🤖 Using music model: {model}")

            prompt = (
                "Upbeat cinematic orchestral opener with bright strings, "
                "light percussion, and a warm brass swell at the end."
            )
            duration_seconds = 60  # Valid for elevenlabs-music and ACE-Step tier 1

            # Price estimate before committing to the generation.
            quote = await client.music.quote(
                model=model,
                duration_seconds=duration_seconds,
                character_count=len(prompt),
            )
            print(f"   💰 Quoted cost: ${quote.quote:.4f}")

            # Use MusicJob as an async context manager — guarantees server-side
            # cleanup even if download / wait fails.
            output_path = RESULTS_DIR / f"music_out_{_ts()}.mp3"
            async with await client.music.run(
                model=model,
                prompt=prompt,
                duration_seconds=duration_seconds,
                force_instrumental=True,
            ) as job:
                print(f"   ⏳ Queued with queue_id={job.queue_id}")

                def _on_progress(status) -> None:
                    pct = status.progress_percent
                    print(f"   ⏱  Progress: {pct:5.1f}%", end="\r")

                status = await job.wait(poll_interval=3.0, on_progress=_on_progress)
                print()  # newline after progress indicator

                saved = await job.download(output_path, status)
                print(f"   💾 Saved to: {saved.resolve()}")
    except MusicGenerationError as e:
        # A server-side generation failure is a genuine failure, not a demo —
        # report it and flag the demo so the run exits non-zero.
        print(f"   ❌ Generation failed: {e} (code={e.error_code})")
        return False
    except Exception as e:
        print(f"   ❌ Error in music generation: {e}")
        return False

    return True


async def low_level_flow() -> bool:
    """Same task using the raw queue/retrieve/complete methods.

    Useful if you want to interleave other work between polls, or integrate
    the job state into a custom task system.

    Returns ``True`` only if the job completed; ``False`` if the job failed,
    the poll loop timed out, or any error was raised.
    """
    print("\n🎚️  Music Generation — Low-level submit/retrieve/cancel")
    print("-" * 60)

    try:
        async with VeniceClient() as client:
            model = await client.models.resolve_music(preferred_models=["elevenlabs-music"])

            queue_response = await client.music.submit(
                model=model,
                prompt="Gentle lo-fi hip-hop beat with warm vinyl crackle.",
                duration_seconds=60,
                force_instrumental=True,
            )
            print(f"   ⏳ queue_id={queue_response.queue_id}")

            # Poll until completion. MusicJob's wait() hides this pattern — shown
            # here verbatim for illustration.
            completed = False
            for _ in range(120):
                status = await client.music.retrieve(model=model, queue_id=queue_response.queue_id)
                print(f"   → {type(status).__name__}")
                if isinstance(status, MusicCompletedStatus):
                    # When returned inline the audio bytes are on ``status.data``;
                    # otherwise ``status.url`` carries a download link.
                    if status.data:
                        out = RESULTS_DIR / f"music_out_lowlevel_{_ts()}.mp3"
                        out.write_bytes(status.data)
                        print(f"   💾 Wrote inline bytes to {out}")
                    elif status.url:
                        print(f"   🔗 Download URL: {status.url}")
                    completed = True
                    break
                if isinstance(status, MusicFailedStatus):
                    print(f"   ❌ Failed: {status.error}")
                    break
                await asyncio.sleep(3.0)

            if not completed:
                print("   ❌ Job did not complete (failed or timed out)")

            await client.music.cancel(model=model, queue_id=queue_response.queue_id)
            print("   🧹 Cleanup complete")
    except Exception as e:
        print(f"   ❌ Error in low-level flow: {e}")
        return False

    return completed


async def main() -> int:
    """Run all music generation demos.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    or generation failure surfaces as a non-zero process exit instead of being
    masked by the success banner.
    """
    print("🚀 Venice AI Music Generation Examples")
    print("=" * 60)

    results: list[tuple[str, bool]] = [
        ("quote_then_generate", await quote_then_generate()),
        ("low_level_flow", await low_level_flow()),
    ]

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("\n✨ Done!")

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
