#!/usr/bin/env python3
"""
Venice AI SDK - Voice Options Demonstration
===========================================

This example demonstrates the various voice options available in the Venice AI SDK.
Learn how to use different voices, discover available voices, and customize speech characteristics.
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


async def demonstrate_voice_variety():
    """Demonstrate different voice options with the same text."""
    print("🎭 Voice Variety Demonstration")
    print("-" * 40)

    text = "Hello! I'm demonstrating different voice characteristics available in Venice AI."

    # Different voice samples from various categories
    voice_samples = [
        (Voice.AF_ALLOY, "American Female - Alloy"),
        (Voice.AM_ADAM, "American Male - Adam"),
        (Voice.BF_ALICE, "British Female - Alice"),
        (Voice.BM_GEORGE, "British Male - George"),
        (Voice.ZF_XIAONI, "Chinese Female - Xiaoni"),
        (Voice.ZM_YUNXI, "Chinese Male - Yunxi"),
        (Voice.JF_ALPHA, "Japanese Female - Alpha"),
        (Voice.JM_KUMO, "Japanese Male - Kumo"),
    ]

    async with VeniceClient() as client:
        try:
            # Get available audio model dynamically
            audio_model = await client.models.resolve_tts()

            print(f"📍 Using audio model: {audio_model}")
            print(f"📝 Text: {text}")

            for voice, description in voice_samples:
                print(f"\n🎤 Generating with {description}...")

                try:
                    response = await client.audio.create_speech(
                        model=audio_model,
                        input=text,
                        voice=voice,
                        response_format=ResponseFormat.MP3,
                        speed=1.0,
                    )

                    # Create descriptive filename
                    voice_name = voice.value.replace("_", "-")
                    filename = RESULTS_DIR / f"voice_{voice_name}.mp3"
                    response.save(filename, overwrite=True)

                    print(f"✅ Generated: {filename} ({len(response.content)} bytes)")

                except Exception as e:
                    print(f"❌ Failed to generate {description}: {e}")

        except Exception as e:
            print(f"❌ Error in voice demonstration: {e}")


async def list_available_voices():
    """List and categorize available voices."""
    print("\n📋 Available Voice Discovery")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            # First try to get TTS models directly
            try:
                tts_models = await client.models.list(type="tts")
                print(f"📊 Found {len(tts_models.data)} TTS models:")
                for model in tts_models.data:
                    print(f"   🎤 {model.id}")
            except Exception as e:
                print(f"⚠️ Could not fetch TTS models directly: {e}")

            # Try the audio.get_voices() method
            try:
                voice_list = await client.audio.get_voices()
                print(f"📊 Found {len(voice_list.data)} available voices via get_voices()")

                if voice_list.data:
                    # Categorize voices by language/region
                    categories = {}
                    for voice in voice_list.data:
                        language = voice.language or "Unknown"
                        if language not in categories:
                            categories[language] = []
                        categories[language].append(voice)

                    # Display voices by category
                    for language, voices in categories.items():
                        print(f"\n🌍 {language} Voices:")
                        for voice in voices[:5]:  # Show first 5 per category
                            gender_icon = (
                                "♀️"
                                if voice.gender == "female"
                                else "♂️"
                                if voice.gender == "male"
                                else "❓"
                            )
                            accent_info = f" ({voice.accent})" if voice.accent else ""
                            print(f"   {gender_icon} {voice.id}{accent_info}")

                        if len(voices) > 5:
                            print(f"   ... and {len(voices) - 5} more")

                    # Show filter examples
                    print("\n🔍 Filter Examples:")

                    # Filter by gender
                    female_voices = await client.audio.get_voices(gender="female")
                    male_voices = await client.audio.get_voices(gender="male")

                    print(f"   Female voices: {len(female_voices.data)}")
                    print(f"   Male voices: {len(male_voices.data)}")
                else:
                    print("   ℹ️ No voices returned by get_voices() method")
                    print("   💡 This might mean the TTS models use a different voice system")

            except Exception as e:
                print(f"⚠️ Error with get_voices() method: {e}")

            # Show available voice enum options as fallback
            print("\n🎭 Available Voice Enum Options (from SDK):")
            from venice_ai.types.audio import Voice

            voice_options = list(Voice)
            print(f"   📊 Found {len(voice_options)} predefined voice options:")
            for voice in voice_options[:10]:  # Show first 10
                print(f"   🎤 {voice.value}")
            if len(voice_options) > 10:
                print(f"   ... and {len(voice_options) - 10} more")

        except Exception as e:
            print(f"❌ Error in voice discovery: {e}")
            print("💡 Note: Voice listing requires appropriate API access")


async def regional_accent_showcase():
    """Showcase voices from different regions with the same text."""
    print("\n🌏 Regional Accent Showcase")
    print("-" * 40)

    text = "This text demonstrates regional accent variations."

    # Regional voice variations
    regional_voices = [
        (Voice.AF_ALLOY, "American English"),
        (Voice.BF_ALICE, "British English"),
        (Voice.ZF_XIAONI, "Mandarin Chinese"),
        (Voice.JF_ALPHA, "Japanese"),
        (Voice.FF_SIWIS, "French"),
        (Voice.IF_SARA, "Italian"),
        (Voice.PF_DORA, "Portuguese"),
        (Voice.EF_DORA, "Spanish"),
    ]

    async with VeniceClient() as client:
        try:
            # Get available audio model dynamically
            audio_model = await client.models.resolve_tts()

            print(f"📍 Using audio model: {audio_model}")
            print(f"📝 Text: {text}")

            for voice, region in regional_voices:
                print(f"\n🗣️ Generating {region} accent...")

                try:
                    response = await client.audio.create_speech(
                        model=audio_model,
                        input=text,
                        voice=voice,
                        response_format=ResponseFormat.MP3,
                        speed=1.0,
                    )

                    # Create descriptive filename
                    region_name = region.lower().replace(" ", "_")
                    filename = RESULTS_DIR / f"accent_{region_name}.mp3"
                    response.save(filename, overwrite=True)

                    print(f"✅ Generated {region}: {filename} ({len(response.content)} bytes)")

                except Exception as e:
                    print(f"❌ Failed to generate {region}: {e}")

        except Exception as e:
            print(f"❌ Error in regional showcase: {e}")


async def gender_voice_comparison():
    """Compare male and female voices for the same content."""
    print("\n⚧️ Gender Voice Comparison")
    print("-" * 40)

    text = "This demonstrates the difference between male and female voice characteristics."

    # Male and female voice pairs
    voice_pairs = [
        ("American", Voice.AF_ALLOY, Voice.AM_ADAM),
        ("British", Voice.BF_ALICE, Voice.BM_GEORGE),
        ("Chinese", Voice.ZF_XIAONI, Voice.ZM_YUNXI),
        ("Japanese", Voice.JF_ALPHA, Voice.JM_KUMO),
    ]

    async with VeniceClient() as client:
        try:
            # Get available audio model dynamically
            audio_model = await client.models.resolve_tts()

            print(f"📍 Using audio model: {audio_model}")
            print(f"📝 Text: {text}")

            for region, female_voice, male_voice in voice_pairs:
                print(f"\n🌍 {region} Voice Comparison:")

                # Generate female voice
                try:
                    response = await client.audio.create_speech(
                        model=audio_model,
                        input=text,
                        voice=female_voice,
                        response_format=ResponseFormat.MP3,
                        speed=1.0,
                    )

                    filename = RESULTS_DIR / f"gender_{region.lower()}_female.mp3"
                    response.save(filename, overwrite=True)

                    print(f"  ♀️ Female: {filename} ({len(response.content)} bytes)")

                except Exception as e:
                    print(f"  ❌ Failed female voice: {e}")

                # Generate male voice
                try:
                    response = await client.audio.create_speech(
                        model=audio_model,
                        input=text,
                        voice=male_voice,
                        response_format=ResponseFormat.MP3,
                        speed=1.0,
                    )

                    filename = RESULTS_DIR / f"gender_{region.lower()}_male.mp3"
                    response.save(filename, overwrite=True)

                    print(f"  ♂️ Male: {filename} ({len(response.content)} bytes)")

                except Exception as e:
                    print(f"  ❌ Failed male voice: {e}")

        except Exception as e:
            print(f"❌ Error in gender comparison: {e}")


async def voice_personality_showcase():
    """Showcase different voice personalities and characteristics."""
    print("\n✨ Voice Personality Showcase")
    print("-" * 40)

    # Different texts that highlight voice characteristics
    personality_tests = [
        (
            "Warm & Friendly",
            "Hello! I'm so excited to help you today. How can I make your day better?",
            Voice.AF_NOVA,
        ),
        (
            "Professional",
            "Good morning. I'll be assisting you with your inquiries in a professional manner.",
            Voice.AM_ADAM,
        ),
        (
            "Energetic",
            "Hey there! Ready for an amazing adventure? Let's dive right in!",
            Voice.AF_SKY,
        ),
        (
            "Calm & Soothing",
            "Take a deep breath and relax. Everything will be just fine.",
            Voice.BF_ALICE,
        ),
        (
            "Authoritative",
            "Please follow these instructions carefully and precisely.",
            Voice.AM_ONYX,
        ),
    ]

    async with VeniceClient() as client:
        try:
            # Get available audio model dynamically
            audio_model = await client.models.resolve_tts()

            print(f"📍 Using audio model: {audio_model}")

            for personality, text, voice in personality_tests:
                print(f"\n🎭 {personality} Personality:")
                print(f"   Text: {text}")

                try:
                    response = await client.audio.create_speech(
                        model=audio_model,
                        input=text,
                        voice=voice,
                        response_format=ResponseFormat.MP3,
                        speed=1.0,
                    )

                    personality_filename = personality.lower().replace(" & ", "_").replace(" ", "_")
                    filename = RESULTS_DIR / f"personality_{personality_filename}.mp3"
                    response.save(filename, overwrite=True)

                    print(f"   ✅ Generated: {filename} ({len(response.content)} bytes)")

                except Exception as e:
                    print(f"   ❌ Failed to generate {personality}: {e}")

        except Exception as e:
            print(f"❌ Error in personality showcase: {e}")


async def main():
    """Run all voice option examples."""
    print("🚀 Venice AI Voice Options Examples")
    print("=" * 60)

    await demonstrate_voice_variety()
    await list_available_voices()
    await regional_accent_showcase()
    await gender_voice_comparison()
    await voice_personality_showcase()

    print("\n✨ Voice options examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Voice variety and characteristics")
    print("   - Available voice discovery and filtering")
    print("   - Regional accent variations")
    print("   - Male vs female voice comparisons")
    print("   - Voice personality and tone")
    print("   - Dynamic model selection")
    print("   - Comprehensive voice catalog usage")

    print("\n📁 Generated files in examples/results/:")
    print("   - voice_*.mp3 (various voice samples)")
    print("   - accent_*.mp3 (regional accents)")
    print("   - gender_*.mp3 (male/female comparisons)")
    print("   - personality_*.mp3 (personality variations)")


if __name__ == "__main__":
    try:
        asyncio.run(main())
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
