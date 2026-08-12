#!/usr/bin/env python3
"""
Venice AI SDK - Background Removal
====================================

This example demonstrates how to remove backgrounds from images using the Venice AI SDK.
Learn how to isolate subjects, create transparent PNGs, and build compositing workflows
with AI-powered background removal.
"""

import asyncio
import base64
import sys
from io import BytesIO
from pathlib import Path

from venice_ai import VeniceClient, detect_image_format
from venice_ai.types.api import ImageGenerationResponse

# Resolve results dir relative to this file's location.
# All example scripts live one level below examples/ (e.g., examples/image/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


async def basic_background_removal() -> bool:
    """Generate an image of an object and remove its background.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print("✂️ Basic Background Removal")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            # Step 1: Generate an image with a clear subject
            image_model = await client.models.resolve_image()
            print(f"📍 Using image model: {image_model}")

            print("🎨 Generating image of a red sports car on a city street...")
            response: ImageGenerationResponse = await client.image.create(
                model=image_model,
                prompt="A red sports car parked on a city street, clear subject, studio lighting",
                width=512,
                height=512,
                num_images=1,
                return_binary=False,
            )

            # Decode and save the original image
            image_data = response.images[0]
            original_bytes = base64.b64decode(image_data)
            original_path = (
                RESULTS_DIR / f"bg_removal_original.{detect_image_format(original_bytes)[0]}"
            )
            with open(original_path, "wb") as f:
                f.write(original_bytes)
            print(f"💾 Original image saved: {original_path}")
            print(f"📏 Original size: {len(original_bytes)} bytes")

            # Step 2: Remove the background
            print("✂️ Removing background...")
            result = await client.image.background_remove(
                image=original_bytes,
            )

            # Save the transparent PNG
            output_path = RESULTS_DIR / f"bg_removal_transparent.{detect_image_format(result)[0]}"
            with open(output_path, "wb") as f:
                f.write(result)

            print("✅ Background removed successfully!")
            print(f"💾 Transparent PNG saved: {output_path}")
            print(f"📏 Result size: {len(result)} bytes")

        except Exception as e:
            print(f"❌ Error during background removal: {e}")
            ok = False

    return ok


async def background_removal_from_url() -> bool:
    """Remove background from an image using a URL.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print("\n🌐 Background Removal from URL")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            # Use a publicly accessible image URL (must be a direct image link)
            image_url = "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400"
            print(f"🔗 Image URL: {image_url[:60]}...")

            print("✂️ Removing background from URL image...")
            result = await client.image.background_remove(
                image_url=image_url,
            )

            # Save the result
            output_path = RESULTS_DIR / f"bg_removal_from_url.{detect_image_format(result)[0]}"
            with open(output_path, "wb") as f:
                f.write(result)

            print("✅ Background removed from URL image!")
            print(f"💾 Saved to: {output_path}")
            print(f"📏 Result size: {len(result)} bytes")

        except Exception as e:
            print(f"❌ Error removing background from URL: {e}")
            print("💡 Note: The URL must be publicly accessible")
            ok = False

    return ok


async def different_input_methods() -> bool:
    """Demonstrate different ways to provide images for background removal.

    Returns ``True`` on success, ``False`` if any input method failed.
    """
    print("\n📂 Different Input Methods")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            # First, generate a sample image to work with
            image_model = await client.models.resolve_image()

            print("🎨 Generating sample image...")
            response: ImageGenerationResponse = await client.image.create(
                model=image_model,
                prompt="A golden retriever sitting on grass, clear subject",
                width=512,
                height=512,
                num_images=1,
                return_binary=False,
            )

            image_data = response.images[0]
            sample_bytes = base64.b64decode(image_data)
            sample_path = (
                RESULTS_DIR / f"bg_removal_sample_dog.{detect_image_format(sample_bytes)[0]}"
            )
            with open(sample_path, "wb") as f:
                f.write(sample_bytes)
            print(f"✅ Sample image saved: {sample_path}")

            # Method 1: File path (string)
            print("\n📁 Method 1: File path (string)")
            result = await client.image.background_remove(
                image=str(sample_path),
            )
            output_path = RESULTS_DIR / f"bg_removal_from_path.{detect_image_format(result)[0]}"
            with open(output_path, "wb") as f:
                f.write(result)
            print(f"   ✅ Saved: {output_path} ({len(result)} bytes)")

            # Method 2: Raw bytes
            print("🔢 Method 2: Raw bytes")
            result = await client.image.background_remove(
                image=sample_bytes,
            )
            output_path = RESULTS_DIR / f"bg_removal_from_bytes.{detect_image_format(result)[0]}"
            with open(output_path, "wb") as f:
                f.write(result)
            print(f"   ✅ Saved: {output_path} ({len(result)} bytes)")

            # Method 3: File-like object (BinaryIO)
            print("📖 Method 3: File-like object (BinaryIO)")
            bio = BytesIO(sample_bytes)
            result = await client.image.background_remove(
                image=bio,
            )
            output_path = RESULTS_DIR / f"bg_removal_from_fileobj.{detect_image_format(result)[0]}"
            with open(output_path, "wb") as f:
                f.write(result)
            print(f"   ✅ Saved: {output_path} ({len(result)} bytes)")

            # Method 4: Path object
            print("🗂️ Method 4: Path object")
            result = await client.image.background_remove(
                image=sample_path,
            )
            output_path = RESULTS_DIR / f"bg_removal_from_pathobj.{detect_image_format(result)[0]}"
            with open(output_path, "wb") as f:
                f.write(result)
            print(f"   ✅ Saved: {output_path} ({len(result)} bytes)")

            print("\n✅ All input methods work correctly!")

        except Exception as e:
            print(f"❌ Error testing input methods: {e}")
            ok = False

    return ok


async def practical_use_cases() -> bool:
    """Show practical workflows using background removal.

    Returns ``True`` on success, ``False`` if any use case failed.
    """
    print("\n🎯 Practical Use Cases")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            image_model = await client.models.resolve_image()

            # Use case 1: Product photography — isolate a product
            print("🛍️ Use case 1: Product photography")
            print("   Generating product image...")
            response: ImageGenerationResponse = await client.image.create(
                model=image_model,
                prompt="A sleek wireless headphone on a wooden desk, product photography",
                width=512,
                height=512,
                num_images=1,
                return_binary=False,
            )

            product_bytes = response.bytes(0)
            product_path = (
                RESULTS_DIR / f"bg_removal_product_original.{detect_image_format(product_bytes)[0]}"
            )
            with open(product_path, "wb") as f:
                f.write(product_bytes)

            print("   ✂️ Isolating product from background...")
            isolated = await client.image.background_remove(image=product_bytes)
            isolated_path = (
                RESULTS_DIR / f"bg_removal_product_isolated.{detect_image_format(isolated)[0]}"
            )
            with open(isolated_path, "wb") as f:
                f.write(isolated)
            print(f"   ✅ Product isolated: {isolated_path}")
            print(f"   📏 Transparent PNG: {len(isolated)} bytes")
            print("   💡 Ready for e-commerce listing or catalog compositing!")

            # Use case 2: Profile picture — remove distracting background
            print("\n👤 Use case 2: Profile picture")
            print("   Generating portrait...")
            response = await client.image.create(
                model=image_model,
                prompt="Professional headshot portrait of a person in an office, shallow depth of field",
                width=512,
                height=512,
                num_images=1,
                return_binary=False,
            )

            portrait_bytes = response.bytes(0)
            portrait_path = (
                RESULTS_DIR
                / f"bg_removal_portrait_original.{detect_image_format(portrait_bytes)[0]}"
            )
            with open(portrait_path, "wb") as f:
                f.write(portrait_bytes)

            print("   ✂️ Removing background from portrait...")
            cutout = await client.image.background_remove(image=portrait_bytes)
            cutout_path = (
                RESULTS_DIR / f"bg_removal_portrait_cutout.{detect_image_format(cutout)[0]}"
            )
            with open(cutout_path, "wb") as f:
                f.write(cutout)
            print(f"   ✅ Portrait cutout: {cutout_path}")
            print(f"   📏 Transparent PNG: {len(cutout)} bytes")
            print("   💡 Ready for profile avatars or compositing onto custom backgrounds!")

            # Use case 3: Batch processing — multiple items
            print("\n📦 Use case 3: Batch processing pipeline")
            items = [
                "A ceramic coffee mug on a kitchen counter",
                "Running shoes on a track field",
                "A potted succulent plant on a shelf",
            ]

            batch_failed = 0
            for i, prompt in enumerate(items):
                item_name = prompt.split(" on ")[0].removeprefix("A ").lower().replace(" ", "_")
                print(f"   🎨 Generating item {i + 1}: {item_name}...")

                try:
                    response = await client.image.create(
                        model=image_model,
                        prompt=prompt,
                        width=512,
                        height=512,
                        num_images=1,
                        return_binary=False,
                    )

                    item_bytes = response.bytes(0)
                    result = await client.image.background_remove(image=item_bytes)

                    item_path = (
                        RESULTS_DIR
                        / f"bg_removal_batch_{item_name}.{detect_image_format(result)[0]}"
                    )
                    with open(item_path, "wb") as f:
                        f.write(result)
                    print(f"   ✅ {item_name}: {item_path} ({len(result)} bytes)")

                except Exception as e:
                    print(f"   ❌ Failed for {item_name}: {e}")
                    ok = False
                    batch_failed += 1

            if batch_failed:
                print(
                    f"\n⚠️ Batch processing finished with {batch_failed} of {len(items)} items failed"
                )
            else:
                print("\n✅ Batch processing complete!")
                print("   💡 All items are now transparent PNGs ready for compositing")

        except Exception as e:
            print(f"❌ Error in practical use cases: {e}")
            ok = False

    return ok


async def main() -> int:
    """Run all background removal examples.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    print("🚀 Venice AI Background Removal Examples")
    print("=" * 50)

    results: list[tuple[str, bool]] = [
        ("basic_background_removal", await basic_background_removal()),
        ("background_removal_from_url", await background_removal_from_url()),
        ("different_input_methods", await different_input_methods()),
        ("practical_use_cases", await practical_use_cases()),
    ]

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("\n✨ Background removal examples completed!")

    print("\n💡 Key concepts demonstrated:")
    print("   - Basic background removal (generate → remove)")
    print("   - Background removal from URL (image_url parameter)")
    print("   - Multiple input methods (path, bytes, BinaryIO, Path)")
    print("   - Product photography isolation")
    print("   - Profile picture cutouts")
    print("   - Batch processing pipeline")
    print("\n📁 Generated files in examples/results/:")
    print("   - bg_removal_original.png (source image)")
    print("   - bg_removal_transparent.png (background removed)")
    print("   - bg_removal_from_url.png (URL input)")
    print("   - bg_removal_from_*.png (different input methods)")
    print("   - bg_removal_product_*.png (product photography)")
    print("   - bg_removal_portrait_*.png (profile picture)")
    print("   - bg_removal_batch_*.png (batch processing)")

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
