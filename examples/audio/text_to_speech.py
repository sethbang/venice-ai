#!/usr/bin/env python3
"""
Venice AI SDK - Text-to-Speech Generation
==========================================

This example demonstrates how to generate speech from text using the Venice AI SDK.
Learn how to create high-quality audio from text with various voice options and formats.
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


async def basic_text_to_speech() -> bool:
    """Generate basic speech from text.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print("🎤 Basic Text-to-Speech Generation")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            # Get available audio model dynamically
            audio_model = await client.models.resolve_tts()

            print(f"📍 Using audio model: {audio_model}")

            # Generate speech with basic settings
            text = "Hello! Welcome to Venice AI's text-to-speech service. This is a demonstration of basic speech generation."

            response = await client.audio.create_speech(
                model=audio_model,
                input=text,
                voice=Voice.AF_ALLOY,  # American female voice
                response_format=ResponseFormat.MP3,
                speed=1.0,
            )

            # Save the audio file
            filename = RESULTS_DIR / "basic_speech.mp3"
            response.save(filename, overwrite=True)

            print("✅ Generated speech audio")
            print(f"💾 Saved as: {filename}")
            print(f"📏 File size: {len(response.content)} bytes")
            print(f"📝 Text: {text}")

            # Show response headers if available
            if response.headers:
                print(f"📊 Response headers available: {len(response.headers)} headers")

        except Exception as e:
            print(f"❌ Error generating speech: {e}")
            print("💡 Note: Text-to-speech requires appropriate API access")
            ok = False

    return ok


async def different_formats() -> bool:
    """Generate speech in different audio formats.

    Returns ``True`` on success, ``False`` if any format failed.
    """
    print("\n🎧 Different Audio Formats")
    print("-" * 40)

    text = "This is a test of different audio formats."

    # Test different formats
    formats = [
        (ResponseFormat.MP3, "mp3"),
        (ResponseFormat.WAV, "wav"),
        (ResponseFormat.FLAC, "flac"),
        (ResponseFormat.AAC, "aac"),
    ]

    ok = True
    async with VeniceClient() as client:
        try:
            # Get available audio model dynamically
            audio_model = await client.models.resolve_tts()

            print(f"📍 Using audio model: {audio_model}")

            for format_enum, extension in formats:
                print(f"\n🎵 Generating {extension.upper()} format...")

                try:
                    response = await client.audio.create_speech(
                        model=audio_model,
                        input=text,
                        voice=Voice.AM_ADAM,  # American male voice
                        response_format=format_enum,
                        speed=1.0,
                    )

                    filename = RESULTS_DIR / f"format_test.{extension}"
                    response.save(filename, overwrite=True)

                    print(f"✅ Generated {extension.upper()}: {filename}")
                    print(f"📏 Size: {len(response.content)} bytes")

                except Exception as e:
                    print(f"❌ Failed to generate {extension.upper()}: {e}")
                    ok = False

        except Exception as e:
            print(f"❌ Error in format testing: {e}")
            ok = False

    return ok


async def speed_variations() -> bool:
    """Generate speech at different speeds.

    Returns ``True`` on success, ``False`` if any speed failed.
    """
    print("\n⚡ Speed Variations")
    print("-" * 40)

    text = (
        "This sentence will be spoken at different speeds to demonstrate the speed control feature."
    )

    # Different speed settings
    speeds = [0.5, 1.0, 1.5, 2.0]

    ok = True
    async with VeniceClient() as client:
        try:
            # Get available audio model dynamically
            audio_model = await client.models.resolve_tts()

            print(f"📍 Using audio model: {audio_model}")

            for speed in speeds:
                print(f"\n🏃 Generating speech at {speed}x speed...")

                try:
                    response = await client.audio.create_speech(
                        model=audio_model,
                        input=text,
                        voice=Voice.AF_NOVA,  # American female voice
                        response_format=ResponseFormat.MP3,
                        speed=speed,
                    )

                    filename = RESULTS_DIR / f"speed_{speed}x.mp3"
                    response.save(filename, overwrite=True)

                    print(f"✅ Generated {speed}x speed: {filename}")
                    print(f"📏 Size: {len(response.content)} bytes")

                except Exception as e:
                    print(f"❌ Failed to generate {speed}x speed: {e}")
                    ok = False

        except Exception as e:
            print(f"❌ Error in speed variation testing: {e}")
            ok = False

    return ok


async def streaming_speech() -> bool:
    """Generate speech using streaming for real-time processing.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print("\n🌊 Streaming Speech Generation")
    print("-" * 40)

    text = "This is a demonstration of streaming text-to-speech generation. The audio is generated and delivered in chunks for real-time processing applications."

    ok = True
    async with VeniceClient() as client:
        try:
            # Get available audio model dynamically
            audio_model = await client.models.resolve_tts()

            print(f"📍 Using audio model: {audio_model}")
            print(f"📝 Text: {text}")
            print("🔄 Starting streaming generation...")

            # Create streaming speech
            stream = await client.audio.create_speech(
                model=audio_model,
                input=text,
                voice=Voice.AM_ECHO,  # American male voice
                response_format=ResponseFormat.MP3,
                speed=1.0,
                stream=True,  # Enable streaming
            )

            # Process chunks as they arrive
            filename = RESULTS_DIR / "streamed_speech.mp3"
            chunk_count = 0
            total_bytes = 0

            with open(filename, "wb") as f:
                async for chunk in stream:
                    chunk_count += 1
                    chunk_size = len(chunk)
                    total_bytes += chunk_size
                    f.write(chunk)
                    print(f"📦 Received chunk {chunk_count}: {chunk_size} bytes")

            print("✅ Streaming complete!")
            print(f"💾 Saved as: {filename}")
            print(f"📊 Total chunks: {chunk_count}")
            print(f"📏 Total size: {total_bytes} bytes")

        except Exception as e:
            print(f"❌ Error in streaming generation: {e}")
            print("💡 Note: Streaming requires compatible model support")
            ok = False

    return ok


async def batch_text_processing() -> bool:
    """Generate speech for multiple texts efficiently.

    Returns ``True`` on success, ``False`` if any text failed.
    """
    print("\n📚 Batch Text Processing")
    print("-" * 40)

    # Multiple texts to convert
    texts = [
        "Welcome to our service.",
        "Thank you for your order.",
        "Your appointment has been confirmed.",
        "Have a great day!",
    ]

    ok = True
    async with VeniceClient() as client:
        try:
            # Get available audio model dynamically
            audio_model = await client.models.resolve_tts()

            print(f"📍 Using audio model: {audio_model}")
            print(f"📝 Processing {len(texts)} texts...")

            for i, text in enumerate(texts):
                print(f"\n🎤 Generating audio {i + 1}/{len(texts)}: {text}")

                try:
                    response = await client.audio.create_speech(
                        model=audio_model,
                        input=text,
                        voice=Voice.BF_ALICE,  # British female voice
                        response_format=ResponseFormat.MP3,
                        speed=1.0,
                    )

                    filename = RESULTS_DIR / f"batch_{i + 1:02d}.mp3"
                    response.save(filename, overwrite=True)

                    print(f"✅ Generated: {filename} ({len(response.content)} bytes)")

                except Exception as e:
                    print(f"❌ Failed to generate audio {i + 1}: {e}")
                    ok = False

        except Exception as e:
            print(f"❌ Error in batch processing: {e}")
            ok = False

    return ok


async def multilingual_with_language_hint() -> bool:
    """Pass an optional ``language`` hint for multi-lingual TTS models.

    Accepted values are model-specific:
      * Qwen 3 accepts full names (``"English"``, ``"Chinese"``, …).
      * xAI and ElevenLabs accept ISO 639-1 codes (``"en"``, ``"ja"``, …).
      * MiniMax accepts full names.

    Unsupported values are silently ignored by the server, so it is safe to
    pass a hint even when targeting models that don't yet support it.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print("\n🌍 Multi-lingual TTS with language hint")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            tts_model = await client.models.resolve_tts()
            print(f"📍 Using TTS model: {tts_model}")

            response = await client.audio.create_speech(
                model=tts_model,
                input="Hello, how are you today?",
                voice=Voice.AF_SKY,
                response_format=ResponseFormat.MP3,
                language="en",  # ISO 639-1 for xAI/ElevenLabs models
            )

            filename = RESULTS_DIR / "speech_language_en.mp3"
            response.save(filename, overwrite=True)
            print(f"✅ Saved: {filename} ({len(response.content)} bytes)")

        except Exception as e:
            print(f"❌ Error generating multi-lingual speech: {e}")
            ok = False

    return ok


async def main() -> int:
    """Run all text-to-speech examples.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    print("🚀 Venice AI Text-to-Speech Examples")
    print("=" * 60)

    results: list[tuple[str, bool]] = [
        ("basic_text_to_speech", await basic_text_to_speech()),
        ("different_formats", await different_formats()),
        ("speed_variations", await speed_variations()),
        ("streaming_speech", await streaming_speech()),
        ("batch_text_processing", await batch_text_processing()),
        ("multilingual_with_language_hint", await multilingual_with_language_hint()),
    ]

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("\n✨ Text-to-speech examples completed!")

    print("\n💡 Key concepts demonstrated:")
    print("   - Basic text-to-speech generation")
    print("   - Multiple audio formats (MP3, WAV, FLAC, AAC)")
    print("   - Speed control (0.5x to 2.0x)")
    print("   - Streaming audio generation")
    print("   - Batch text processing")
    print("   - Dynamic model selection")
    print("   - Multi-lingual language hint (language='en')")
    print("   - Error handling and file management")

    print("\n📁 Generated files in examples/results/:")
    print("   - basic_speech.mp3")
    print("   - format_test.* (multiple formats)")
    print("   - speed_*.mp3 (different speeds)")
    print("   - streamed_speech.mp3")
    print("   - batch_*.mp3 (multiple files)")
    print("   - speech_language_en.mp3")

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
            "Check that your API key is valid and you have audio generation access.",
            file=sys.stderr,
        )
        sys.exit(1)
