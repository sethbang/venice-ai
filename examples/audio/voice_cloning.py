#!/usr/bin/env python3
"""
Venice AI SDK - Voice Cloning
=============================

Clone a voice from a short audio sample, then synthesize new speech in that
voice. Two steps:

1. ``client.audio.create_voice(file=...)`` uploads the sample to
   ``POST /v1/audio/voices`` and returns a :class:`ClonedVoice` whose ``id`` is
   a ``vv_<id>`` handle, paired with the model on ``.model``.
2. ``client.audio.create_speech(input=..., model=voice.model, voice=voice.id)``
   generates speech using that handle.

**Pair the handle with the same model it was created for** — that's why we read
``voice.model`` back rather than resolving a TTS model independently for the
synthesis call.

The sample at ``examples/voice-to-clone.wav`` is used as input. A clean 5–10s
speech recording works best. Output is written to ``examples/results/``.

Handles expire after the per-model retention window (currently ~7 days).
"""

import asyncio
import sys
from pathlib import Path

from venice_ai import VeniceClient
from venice_ai.exceptions import APIError
from venice_ai.types.audio import ResponseFormat

# Resolve sample + results dir relative to this file's location.
SAMPLE_PATH = Path(__file__).resolve().parent.parent / "voice-to-clone.wav"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


async def clone_and_speak() -> None:
    """Clone the sample voice, then synthesize a new line in it."""
    print("🎙️  Voice cloning")
    print("-" * 40)

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Voice sample not found at {SAMPLE_PATH}. Drop a short WAV/MP3 there."
        )

    async with VeniceClient() as client:
        # Step 1 — clone. Omitting model lets the API pick its default
        # (currently tts-chatterbox-hd) and report it back on voice.model.
        print(f"   Uploading sample: {SAMPLE_PATH.name}")
        try:
            voice = await client.audio.create_voice(file=SAMPLE_PATH)
        except APIError as e:
            # Voice cloning is a gated capability — some accounts/models don't
            # have access. That's an entitlement condition, not a code error,
            # so we report it and skip rather than fail. Any *other* API error
            # still propagates.
            if e.status_code == 403:
                print("   ⏭️  Voice cloning is not enabled on this account.")
                print(f"      ({e})")
                print("      Contact support@venice.ai to request access.")
                return
            raise
        print(f"   ✅ Cloned handle: {voice.id}")
        print(f"      paired model: {voice.model}")

        # Step 2 — synthesize using the handle + its paired model.
        line = "Hello! This sentence was generated in a cloned voice using the Venice AI SDK."
        response = await client.audio.create_speech(
            input=line,
            model=voice.model,  # MUST match the model the handle was made for
            voice=voice.id,  # the vv_<id> handle as a plain string
            response_format=ResponseFormat.MP3,
        )

        out_path = RESULTS_DIR / "cloned_voice.mp3"
        saved = response.save(out_path, overwrite=True)
        print(f"   🔊 Saved synthesized speech → {saved}")


async def main() -> None:
    """Run the voice-cloning example."""
    print("🚀 Venice AI Voice Cloning")
    print("=" * 50)

    await clone_and_speak()

    print("\n✨ Voice cloning example completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - client.audio.create_voice(file=...) → ClonedVoice (vv_<id>)")
    print("   - Pairing voice.id with voice.model on create_speech")
    print("   - Saving the synthesized audio to results/")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
