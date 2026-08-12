#!/usr/bin/env python3
"""
Venice AI SDK - Speech-to-Text Transcription
=============================================

This example demonstrates how to transcribe audio to text using the Venice AI SDK.
Learn how to convert speech audio into text with word-level timestamps, different
input methods, and model discovery.

The demo is self-contained: it first generates a short TTS audio clip, then
transcribes it back to text — a complete round-trip without external audio files.
"""

import asyncio
import sys
from pathlib import Path

from venice_ai import VeniceClient
from venice_ai.types.audio import ResponseFormat, Voice

# Resolve results dir relative to this file's location.
# All example scripts live one level below examples/ (e.g., examples/audio/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


async def basic_transcription() -> bool:
    """Transcribe an audio file from a file path and print the text.

    Returns ``True`` on success, ``False`` if the round-trip failed.
    """
    print("🎤 Basic Speech-to-Text Transcription")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            # --- Step 1: Generate a short TTS audio clip for the demo ---
            tts_model = await client.models.resolve_tts()

            sample_text = "Hello! This is a speech-to-text demonstration using the Venice AI SDK."
            print(f"🔊 Generating TTS audio with model: {tts_model}")
            print(f"📝 Original text: {sample_text}")

            response = await client.audio.create_speech(
                model=tts_model,
                input=sample_text,
                voice=Voice.AF_ALLOY,
                response_format=ResponseFormat.WAV,
                speed=1.0,
            )

            audio_path = RESULTS_DIR / "stt_demo.wav"
            response.save(audio_path, overwrite=True)
            print(f"💾 Saved TTS audio: {audio_path} ({len(response.content)} bytes)")

            # --- Step 2: Transcribe the generated audio ---
            asr_model = await client.models.resolve_asr()
            print(f"🎤 Using ASR model: {asr_model}")
            print("\n📝 Transcribing audio...")

            result = await client.audio.transcribe(
                file=str(audio_path),
                model=asr_model,
            )

            print(f"✅ Transcription: {result.text}")

        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Note: Speech-to-text requires appropriate API access")
            ok = False

    return ok


async def transcription_with_timestamps() -> bool:
    """Use timestamps=True to get word-level timing data.

    Returns ``True`` on success, ``False`` if the round-trip failed.
    """
    print("\n⏱️ Transcription with Word-Level Timestamps")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            # Generate a TTS clip to transcribe
            tts_model = await client.models.resolve_tts()

            sample_text = "Word-level timestamps let you know exactly when each word was spoken."
            print("🔊 Generating TTS audio for timestamp demo...")

            response = await client.audio.create_speech(
                model=tts_model,
                input=sample_text,
                voice=Voice.AM_ADAM,
                response_format=ResponseFormat.WAV,
                speed=1.0,
            )

            audio_path = RESULTS_DIR / "stt_timestamps_demo.wav"
            response.save(audio_path, overwrite=True)
            print(f"💾 Saved TTS audio: {audio_path}")

            # Transcribe with timestamps enabled — prefer a model that returns
            # word-level timings. Not every ASR model populates `result.words`.
            asr_model = await client.models.resolve_asr(
                preferred_models=["elevenlabs/scribe-v2", "stt-xai-v1"],
            )
            print(f"🎤 Using ASR model: {asr_model}")
            print("📝 Transcribing with timestamps=True...")

            result = await client.audio.transcribe(
                file=str(audio_path),
                model=asr_model,
                timestamps=True,
            )

            print(f"✅ Full text: {result.text}")

            if result.words:
                print(f"\n⏱️ Word-level timestamps ({len(result.words)} words):")
                for word_info in result.words:
                    start = f"{word_info.start:.2f}s" if word_info.start is not None else "N/A"
                    end = f"{word_info.end:.2f}s" if word_info.end is not None else "N/A"
                    print(f"   📌 {word_info.word:<20} {start} → {end}")
            else:
                print("ℹ️ No word-level timestamps returned by the model")

        except Exception as e:
            print(f"❌ Error: {e}")
            ok = False

    return ok


async def different_input_methods() -> bool:
    """Show file path (str), bytes, and BinaryIO (open file) inputs.

    Returns ``True`` on success, ``False`` if any input method failed.
    """
    print("\n📂 Different Input Methods")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            # Generate a TTS clip to work with
            tts_model = await client.models.resolve_tts()

            sample_text = "Testing different input methods for the transcription API."

            response = await client.audio.create_speech(
                model=tts_model,
                input=sample_text,
                voice=Voice.BF_ALICE,
                response_format=ResponseFormat.WAV,
                speed=1.0,
            )

            audio_path = RESULTS_DIR / "stt_input_methods.wav"
            response.save(audio_path, overwrite=True)
            print(f"🔊 Generated test audio: {audio_path}")

            asr_model = await client.models.resolve_asr()
            print(f"🎤 Using ASR model: {asr_model}")

            # --- Method 1: File path as string ---
            print("\n1️⃣ Input method: file path (str)")
            result = await client.audio.transcribe(
                file=str(audio_path),
                model=asr_model,
            )
            print(f"   ✅ Result: {result.text}")

            # --- Method 2: Raw bytes ---
            print("\n2️⃣ Input method: bytes")
            audio_bytes = audio_path.read_bytes()
            result = await client.audio.transcribe(
                file=audio_bytes,
                model=asr_model,
            )
            print(f"   ✅ Result: {result.text}")

            # --- Method 3: File-like object (BinaryIO) ---
            print("\n3️⃣ Input method: BinaryIO (open file)")
            with open(audio_path, "rb") as f:
                result = await client.audio.transcribe(
                    file=f,
                    model=asr_model,
                )
            print(f"   ✅ Result: {result.text}")

            print("\n📊 All three input methods produced transcriptions successfully!")

        except Exception as e:
            print(f"❌ Error: {e}")
            ok = False

    return ok


async def model_discovery() -> bool:
    """Use resolve_asr() to find available ASR models.

    Returns ``True`` on success, ``False`` if discovery failed. The inner
    ``resolve_*`` handlers are graceful fallbacks for optional model
    availability and do not, on their own, flip the demo to failed.
    """
    print("\n🔍 ASR Model Discovery")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            # Use resolve_asr() for automatic best-model selection
            print("🎯 Automatic ASR model selection...")
            try:
                selected = await client.models.resolve_asr()
                print(f"✅ Selected ASR model: {selected}")
            except Exception as e:
                print(f"ℹ️ resolve_asr() unavailable or no models: {e}")
                print("💡 Fallback: nvidia/parakeet-tdt-0.6b-v3")

            # Also show TTS models for comparison
            print("\n🔊 Available TTS models (for reference):")
            try:
                tts_model = await client.models.resolve_tts()
                print(f"   🎵 {tts_model}")
            except Exception as e:
                print(f"   ℹ️ Could not resolve TTS model: {e}")

        except Exception as e:
            print(f"❌ Error: {e}")
            ok = False

    return ok


async def response_format_options() -> bool:
    """Show different response_format values for transcription.

    Returns ``True`` on success, ``False`` if the round-trip failed.
    """
    print("\n📋 Response Format Options")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            # Generate a TTS clip
            tts_model = await client.models.resolve_tts()

            sample_text = "Exploring different response formats for the transcription endpoint."

            response = await client.audio.create_speech(
                model=tts_model,
                input=sample_text,
                voice=Voice.AF_NOVA,
                response_format=ResponseFormat.WAV,
                speed=1.0,
            )

            audio_path = RESULTS_DIR / "stt_format_demo.wav"
            response.save(audio_path, overwrite=True)
            print(f"🔊 Generated test audio: {audio_path}")

            asr_model = await client.models.resolve_asr()
            print(f"🎤 Using ASR model: {asr_model}")

            # Default format (no response_format specified)
            print("\n1️⃣ Default response format:")
            result = await client.audio.transcribe(
                file=str(audio_path),
                model=asr_model,
            )
            print(f"   ✅ text: {result.text}")
            print(f"   📊 words present: {result.words is not None}")

            # JSON response format
            print("\n2️⃣ JSON response format:")
            result = await client.audio.transcribe(
                file=str(audio_path),
                model=asr_model,
                response_format="json",
            )
            print(f"   ✅ text: {result.text}")
            print(f"   📊 words present: {result.words is not None}")

            # With language hint
            print("\n3️⃣ With language hint (en):")
            result = await client.audio.transcribe(
                file=str(audio_path),
                model=asr_model,
                language="en",
            )
            print(f"   ✅ text: {result.text}")

        except Exception as e:
            print(f"❌ Error: {e}")
            ok = False

    return ok


async def main() -> int:
    """Run all speech-to-text examples.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    print("🚀 Venice AI Speech-to-Text Examples")
    print("=" * 60)

    results: list[tuple[str, bool]] = [
        ("basic_transcription", await basic_transcription()),
        ("transcription_with_timestamps", await transcription_with_timestamps()),
        ("different_input_methods", await different_input_methods()),
        ("model_discovery", await model_discovery()),
        ("response_format_options", await response_format_options()),
    ]

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("\n✨ Speech-to-text examples completed!")

    print("\n💡 Key concepts demonstrated:")
    print("   - Basic audio-to-text transcription")
    print("   - Word-level timestamps with timing data")
    print("   - Multiple input methods (file path, bytes, BinaryIO)")
    print("   - ASR model discovery and selection")
    print("   - Response format options and language hints")
    print("   - Round-trip TTS → STT demo (self-contained)")

    if not failed:
        print("\n📁 Generated files in examples/results/:")
        print("   - stt_demo.wav")
        print("   - stt_timestamps_demo.wav")
        print("   - stt_input_methods.wav")
        print("   - stt_format_demo.wav")

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print(
            "Check that your API key is valid and you have audio transcription access.",
            file=sys.stderr,
        )
        sys.exit(1)
