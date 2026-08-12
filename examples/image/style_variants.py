#!/usr/bin/env python3
"""
Venice AI SDK - Image Style Variations
=======================================

This example demonstrates how to generate images in different artistic styles.
Learn how to control the aesthetic and artistic direction of AI-generated images.
"""

import asyncio
import sys
from pathlib import Path

from venice_ai import RetryOptions, VeniceClient
from venice_ai.types.api import ImageGenerationResponse

# Resolve results dir relative to this file's location.
# All example scripts live one level below examples/ (e.g., examples/image/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# z-image-turbo is sequential here; ~3-4s pacing keeps 32 calls under the rate limit.
INTER_REQUEST_SLEEP = 3.5

# Per-call retry policy used by every sub-example below. The SDK retries on
# 429 / 5xx by default; we just bump max_attempts and the base delay so the
# 32-call sweep can ride out a transient rate-limit hiccup without bailing.
_RATE_LIMIT_RETRIES = RetryOptions(max_attempts=4, base_delay=2.0)


def _report_section(section: str, results: list[tuple[str, bool, str | None]]) -> bool:
    succeeded = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    print(f"\n📊 {section}: {len(succeeded)}/{len(results)} generated")
    if failed:
        print("   Failures:")
        for name, _ok, err in failed:
            print(f"   - {name}: {err}")
    return len(failed) == 0


async def artistic_styles() -> list[tuple[str, bool, str | None]]:
    """Generate the same subject in different artistic styles."""
    print("🎨 Artistic Style Variations")
    print("-" * 40)

    results: list[tuple[str, bool, str | None]] = []

    async with VeniceClient() as client, client.with_retries(_RATE_LIMIT_RETRIES):
        image_model = await client.models.resolve_image()
        print(f"📍 Using image model: {image_model}")

        base_subject = "a majestic lion"

        styles = [
            ("photorealistic", "professional wildlife photography, 8k, detailed"),
            ("oil_painting", "classical oil painting style, Renaissance art"),
            ("watercolor", "soft watercolor painting, delicate brush strokes"),
            ("anime", "anime art style, Studio Ghibli inspired"),
            ("pixel_art", "retro pixel art, 16-bit game style"),
            ("comic_book", "comic book art, bold lines, vibrant colors"),
            ("abstract", "abstract art, geometric shapes, modern art"),
            ("impressionist", "impressionist painting, Monet style, soft brush strokes"),
        ]

        print(f"\n🖼️ Generating '{base_subject}' in {len(styles)} different styles...")

        for style_name, style_description in styles:
            print(f"\n🎨 Style: {style_name}")

            full_prompt = f"{base_subject}, {style_description}"

            try:
                response: ImageGenerationResponse = await client.image.create(
                    model=image_model,
                    prompt=full_prompt,
                    width=512,
                    height=512,
                    num_images=1,
                    return_binary=False,
                )

                filename = response.save(RESULTS_DIR / f"lion_{style_name}", overwrite=True)
                print(f"   ✅ Generated: {filename}")
                print(f"   📏 Size: {filename.stat().st_size} bytes")

                if response.timing:
                    print(f"   ⏱️ Time: {response.timing.inferenceDuration}ms")

                results.append((style_name, True, None))

            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results.append((style_name, False, str(e)))

            await asyncio.sleep(INTER_REQUEST_SLEEP)

    _report_section("Artistic styles", results)
    return results


async def medium_variations() -> list[tuple[str, bool, str | None]]:
    """Demonstrate different artistic mediums."""
    print("\n🖌️ Artistic Medium Variations")
    print("-" * 40)

    results: list[tuple[str, bool, str | None]] = []

    async with VeniceClient() as client, client.with_retries(_RATE_LIMIT_RETRIES):
        image_model = await client.models.resolve_image()
        print(f"📍 Using image model: {image_model}")

        subject = "a serene Japanese garden with cherry blossoms"

        mediums = [
            ("pencil_sketch", "detailed pencil sketch, graphite drawing"),
            ("charcoal", "charcoal drawing, dramatic shadows"),
            ("ink_drawing", "traditional ink drawing, Japanese sumi-e style"),
            ("pastel", "soft pastel art, gentle colors"),
            ("acrylic", "vibrant acrylic painting, bold colors"),
            ("digital_art", "modern digital art, clean lines"),
        ]

        print(f"\n🖼️ Generating in {len(mediums)} different mediums...")

        for medium_name, medium_description in mediums:
            print(f"\n🎨 Medium: {medium_name}")

            full_prompt = f"{subject}, {medium_description}"

            try:
                response: ImageGenerationResponse = await client.image.create(
                    model=image_model,
                    prompt=full_prompt,
                    width=512,
                    height=512,
                    num_images=1,
                    return_binary=False,
                )

                filename = response.save(RESULTS_DIR / f"garden_{medium_name}", overwrite=True)
                print(f"   ✅ Saved: {filename}")
                results.append((medium_name, True, None))

            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results.append((medium_name, False, str(e)))

            await asyncio.sleep(INTER_REQUEST_SLEEP)

    _report_section("Mediums", results)
    return results


async def era_and_movement_styles() -> list[tuple[str, bool, str | None]]:
    """Generate images in styles from different art eras and movements."""
    print("\n🏛️ Art Era and Movement Styles")
    print("-" * 40)

    results: list[tuple[str, bool, str | None]] = []

    async with VeniceClient() as client, client.with_retries(_RATE_LIMIT_RETRIES):
        image_model = await client.models.resolve_image()
        print(f"📍 Using image model: {image_model}")

        subject = "a woman reading a book by candlelight"

        movements = [
            ("baroque", "Baroque style, Rembrandt lighting, dramatic chiaroscuro"),
            ("art_nouveau", "Art Nouveau style, flowing organic lines, decorative"),
            ("art_deco", "Art Deco style, geometric patterns, luxurious"),
            ("cubist", "Cubist style, Picasso inspired, fragmented forms"),
            ("surrealist", "Surrealist style, Dali inspired, dreamlike"),
            ("pop_art", "Pop Art style, Warhol inspired, bold colors"),
        ]

        print(f"\n🖼️ Generating in {len(movements)} art movements...")

        for movement_name, movement_description in movements:
            print(f"\n🎨 Movement: {movement_name}")

            full_prompt = f"{subject}, {movement_description}"

            try:
                response: ImageGenerationResponse = await client.image.create(
                    model=image_model,
                    prompt=full_prompt,
                    width=512,
                    height=512,
                    num_images=1,
                    return_binary=False,
                )

                filename = response.save(RESULTS_DIR / f"reading_{movement_name}", overwrite=True)
                print(f"   ✅ Created: {filename}")
                results.append((movement_name, True, None))

            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results.append((movement_name, False, str(e)))

            await asyncio.sleep(INTER_REQUEST_SLEEP)

    _report_section("Art movements", results)
    return results


async def mood_and_atmosphere() -> list[tuple[str, bool, str | None]]:
    """Generate images with different moods and atmospheres."""
    print("\n🌈 Mood and Atmosphere Variations")
    print("-" * 40)

    results: list[tuple[str, bool, str | None]] = []

    async with VeniceClient() as client, client.with_retries(_RATE_LIMIT_RETRIES):
        image_model = await client.models.resolve_image()
        print(f"📍 Using image model: {image_model}")

        base_scene = "a forest path"

        moods = [
            ("cheerful", "bright sunny day, vibrant colors, joyful atmosphere"),
            ("mysterious", "foggy evening, mysterious atmosphere, ethereal"),
            ("dramatic", "stormy weather, dramatic clouds, intense"),
            ("peaceful", "soft morning light, tranquil, serene"),
            ("melancholic", "autumn colors, gentle rain, nostalgic mood"),
            ("magical", "glowing lights, fantasy atmosphere, enchanted"),
        ]

        print(f"\n🖼️ Generating scene in {len(moods)} different moods...")

        for mood_name, mood_description in moods:
            print(f"\n✨ Mood: {mood_name}")

            full_prompt = f"{base_scene}, {mood_description}"

            try:
                response: ImageGenerationResponse = await client.image.create(
                    model=image_model,
                    prompt=full_prompt,
                    width=512,
                    height=512,
                    num_images=1,
                    return_binary=False,
                )

                filename = response.save(RESULTS_DIR / f"forest_{mood_name}", overwrite=True)
                print(f"   ✅ Saved: {filename}")
                results.append((mood_name, True, None))

            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results.append((mood_name, False, str(e)))

            await asyncio.sleep(INTER_REQUEST_SLEEP)

    _report_section("Moods", results)
    return results


async def color_palette_styles() -> list[tuple[str, bool, str | None]]:
    """Generate images with different color palettes."""
    print("\n🎨 Color Palette Variations")
    print("-" * 40)

    results: list[tuple[str, bool, str | None]] = []

    async with VeniceClient() as client, client.with_retries(_RATE_LIMIT_RETRIES):
        image_model = await client.models.resolve_image()
        print(f"📍 Using image model: {image_model}")

        subject = "a cityscape at dusk"

        palettes = [
            ("warm", "warm color palette, oranges and reds, sunset tones"),
            ("cool", "cool color palette, blues and purples, twilight"),
            ("monochrome", "black and white, high contrast, noir style"),
            ("pastel", "soft pastel colors, gentle tones"),
            ("neon", "bright neon colors, cyberpunk palette, vivid"),
            ("earth_tones", "natural earth tones, browns and greens"),
        ]

        print(f"\n🖼️ Generating scene in {len(palettes)} color palettes...")

        for palette_name, palette_description in palettes:
            print(f"\n🎨 Palette: {palette_name}")

            full_prompt = f"{subject}, {palette_description}"

            try:
                response: ImageGenerationResponse = await client.image.create(
                    model=image_model,
                    prompt=full_prompt,
                    width=512,
                    height=512,
                    num_images=1,
                    return_binary=False,
                )

                filename = response.save(RESULTS_DIR / f"city_{palette_name}", overwrite=True)
                print(f"   ✅ Generated: {filename}")
                results.append((palette_name, True, None))

            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results.append((palette_name, False, str(e)))

            await asyncio.sleep(INTER_REQUEST_SLEEP)

    _report_section("Color palettes", results)
    return results


async def main() -> int:
    """Run all style variation examples."""
    print("🚀 Venice AI Image Style Variations Examples")
    print("=" * 50)

    sections: list[tuple[str, str, list[tuple[str, bool, str | None]]]] = []
    sections.append(("Artistic styles", "lion_", await artistic_styles()))
    sections.append(("Mediums", "garden_", await medium_variations()))
    sections.append(("Art movements", "reading_", await era_and_movement_styles()))
    sections.append(("Moods", "forest_", await mood_and_atmosphere()))
    sections.append(("Color palettes", "city_", await color_palette_styles()))

    total = sum(len(r) for _, _, r in sections)
    succeeded = sum(1 for _, _, r in sections for entry in r if entry[1])
    failed = total - succeeded

    print("\n" + "=" * 50)
    print(f"📊 Overall: {succeeded}/{total} generations succeeded")

    if failed:
        print(f"\n❌ {failed} generation(s) failed:")
        for section_name, _prefix, results in sections:
            for name, ok, err in results:
                if not ok:
                    print(f"   [{section_name}] {name}: {err}")
    else:
        print("\n✨ Style variation examples completed!")

    print("\n💡 Key concepts demonstrated:")
    print("   - Artistic style variations (photorealistic, anime, etc.)")
    print("   - Different artistic mediums (pencil, oil, watercolor)")
    print("   - Art movements and eras (Baroque, Cubism, Pop Art)")
    print("   - Mood and atmosphere control")
    print("   - Color palette manipulation")
    print("   - Consistent subject across styles")

    print("\n📁 Generated files in examples/results/:")
    prefixes = [prefix for _, prefix, _ in sections]
    on_disk: list[Path] = []
    for prefix in prefixes:
        for ext in ("png", "webp", "jpg"):
            on_disk.extend(sorted(RESULTS_DIR.glob(f"{prefix}*.{ext}")))
    if on_disk:
        for path in on_disk:
            print(f"   - {path.name}")
    else:
        print("   (none)")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
