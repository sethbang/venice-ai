#!/usr/bin/env python3
"""
Venice AI SDK - Text-to-Image Generation
========================================

This example demonstrates how to generate images from text prompts using the Venice AI SDK.
Learn how to create compelling visual content with AI image generation models.
"""

import asyncio
import sys
from pathlib import Path

from venice_ai import VeniceClient
from venice_ai.types.api import ImageGenerationResponse

# Resolve results dir relative to this file's location.
# All example scripts live one level below examples/ (e.g., examples/image/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


async def basic_image_generation() -> bool:
    """Generate a simple image from text.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print("🎨 Basic Image Generation")
    print("-" * 30)

    ok = True
    async with VeniceClient() as client:
        try:
            # Get available image model dynamically
            image_model = await client.models.resolve_image()
            print(f"📍 Using image model: {image_model}")

            # Generate an image from text (get structured response)
            response: ImageGenerationResponse = await client.image.create(
                model=image_model,
                prompt="A serene mountain landscape at sunset with a crystal clear lake",
                width=512,
                height=512,
                num_images=1,
                return_binary=False,  # Get structured response
            )

            print(f"✅ Generated {len(response.images)} image(s)")

            # save_all() auto-detects the format from magic bytes when ext=None,
            # so WebP-from-z-image-turbo lands as .webp instead of mislabeled .png.
            saved = response.save_all(RESULTS_DIR, prefix="generated_landscape", overwrite=True)
            for path in saved:
                size = path.stat().st_size
                print(f"💾 Saved image as: {path}")
                print(f"📏 Image size: {size} bytes")

            # Show timing information
            if response.timing:
                print(f"⏱️ Generation time: {response.timing.inferenceDuration}ms")

        except Exception as e:
            print(f"❌ Error generating image: {e}")
            print("💡 Note: Image generation requires appropriate API access")
            ok = False

    return ok


async def batch_image_generation() -> bool:
    """Generate multiple images with different settings.

    Returns ``True`` only if every image generated, ``False`` otherwise.
    """
    print("\n🖼️ Batch Image Generation")
    print("-" * 30)

    # Different prompts to generate
    prompts = [
        "A futuristic city with flying cars",
        "A peaceful forest with magical creatures",
        "An abstract art piece with vibrant colors",
    ]

    ok = True
    async with VeniceClient() as client:
        try:
            # Get available image model dynamically
            image_model = await client.models.resolve_image()
            print(f"📍 Using image model: {image_model}")

            for i, prompt in enumerate(prompts):
                print(f"\n🎨 Generating image {i + 1}: {prompt[:50]}...")

                try:
                    response: ImageGenerationResponse = await client.image.create(
                        model=image_model,
                        prompt=prompt,
                        width=512,
                        height=512,
                        num_images=1,
                        return_binary=False,  # Get structured response
                    )

                    saved = response.save(RESULTS_DIR / f"batch_image_{i + 1}", overwrite=True)
                    print(f"✅ Generated and saved: {saved}")

                except Exception as e:
                    print(f"❌ Failed to generate image {i + 1}: {e}")
                    ok = False

        except Exception as e:
            print(f"❌ Error in batch generation: {e}")
            ok = False

    return ok


async def style_variations() -> bool:
    """Generate images with different artistic styles.

    Returns ``True`` only if every style generated, ``False`` otherwise.
    """
    print("\n🎭 Style Variations")
    print("-" * 30)

    base_prompt = "A majestic lion in its natural habitat"

    # Different style modifiers
    style_prompts = [
        f"{base_prompt}, photorealistic style",
        f"{base_prompt}, oil painting style",
        f"{base_prompt}, cartoon animation style",
        f"{base_prompt}, abstract digital art style",
    ]

    ok = True
    async with VeniceClient() as client:
        try:
            # Get available image model dynamically
            image_model = await client.models.resolve_image()
            print(f"📍 Using image model: {image_model}")

            for styled_prompt in style_prompts:
                style_name = styled_prompt.split(", ")[-1].replace(" style", "")
                print(f"\n🖌️ Generating {style_name} version...")

                try:
                    response: ImageGenerationResponse = await client.image.create(
                        model=image_model,
                        prompt=styled_prompt,
                        width=512,
                        height=512,
                        num_images=1,
                        return_binary=False,  # Get structured response
                    )

                    saved = response.save(
                        RESULTS_DIR / f"lion_{style_name.replace(' ', '_')}",
                        overwrite=True,
                    )
                    print(f"✅ Generated {style_name} version: {saved}")

                except Exception as e:
                    print(f"❌ Failed to generate {style_name} version: {e}")
                    ok = False

        except Exception as e:
            print(f"❌ Error in style variations: {e}")
            ok = False

    return ok


async def parameter_exploration() -> bool:
    """Explore different generation parameters.

    Returns ``True`` only if every parameter combination generated, ``False``
    otherwise.
    """
    print("\n⚙️ Parameter Exploration")
    print("-" * 30)

    base_prompt = "A beautiful garden with colorful flowers"

    # Different parameter combinations
    param_combinations = [
        {"width": 512, "height": 512, "description": "Square format"},
        {"width": 768, "height": 512, "description": "Landscape format"},
        {"width": 512, "height": 768, "description": "Portrait format"},
    ]

    ok = True
    async with VeniceClient() as client:
        try:
            # Get available image model dynamically
            image_model = await client.models.resolve_image()
            print(f"📍 Using image model: {image_model}")

            for params in param_combinations:
                description = params.pop("description")
                print(f"\n📐 Generating {description} ({params['width']}x{params['height']})...")

                try:
                    response: ImageGenerationResponse = await client.image.create(
                        model=image_model,
                        prompt=base_prompt,
                        num_images=1,
                        return_binary=False,  # Get structured response
                        **params,
                    )

                    saved = response.save(
                        RESULTS_DIR / f"garden_{params['width']}x{params['height']}",
                        overwrite=True,
                    )
                    print(f"✅ Generated {description}: {saved}")
                    print(f"📏 File size: {saved.stat().st_size} bytes")

                except Exception as e:
                    print(f"❌ Failed to generate {description}: {e}")
                    ok = False

        except Exception as e:
            print(f"❌ Error in parameter exploration: {e}")
            ok = False

    return ok


async def main() -> int:
    """Run all image generation examples.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    print("🚀 Venice AI Image Generation Examples")
    print("=" * 50)

    results: list[tuple[str, bool]] = [
        ("basic_image_generation", await basic_image_generation()),
        ("batch_image_generation", await batch_image_generation()),
        ("style_variations", await style_variations()),
        ("parameter_exploration", await parameter_exploration()),
    ]

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("\n✨ Image generation examples completed!")

    print("\n💡 Key concepts demonstrated:")
    print("   - Basic text-to-image generation")
    print("   - Batch processing multiple prompts")
    print("   - Style variations and artistic control")
    print("   - Parameter tuning (dimensions, formats)")
    print("   - Dynamic model selection")
    print("   - Image data handling and file saving")

    # Images save in the model's native format (typically .webp), auto-detected
    # from magic bytes — actual paths are printed by each demo above.
    print("\n📁 Generated files in examples/results/:")
    print("   - generated_landscape_*")
    print("   - batch_image_*")
    print("   - lion_*")
    print("   - garden_*")

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
