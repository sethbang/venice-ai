#!/usr/bin/env python3
"""
Venice AI SDK - Batch Image Generation
=======================================

This example demonstrates efficient batch generation of multiple images.
Learn how to generate multiple images concurrently and manage batch workflows.
"""

import asyncio
import base64
import sys
import time
from pathlib import Path

from venice_ai import RateLimitError, VeniceClient, detect_image_format
from venice_ai.types.api import ImageGenerationResponse

# Resolve results dir relative to this file's location.
# All example scripts live one level below examples/ (e.g., examples/image/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


async def simple_batch_generation() -> bool:
    """Generate multiple images from different prompts.

    Returns ``True`` only if every image generated successfully, ``False`` if
    any image failed (so the caller can surface a non-zero exit).
    """
    print("📦 Simple Batch Generation")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            # Get image model
            image_model = await client.models.resolve_image()
            print(f"📍 Using image model: {image_model}")

            # Different prompts to generate
            prompts = [
                "A futuristic city with flying cars",
                "A peaceful mountain landscape",
                "An abstract art piece with vibrant colors",
                "A cozy coffee shop interior",
                "A magical forest with glowing mushrooms",
            ]

            print(f"\n🎨 Generating {len(prompts)} images sequentially...")
            start_time = time.time()

            for i, prompt in enumerate(prompts, 1):
                print(f"\n📝 Image {i}/{len(prompts)}: {prompt[:40]}...")

                try:
                    response: ImageGenerationResponse = await client.image.create(
                        model=image_model,
                        prompt=prompt,
                        width=512,
                        height=512,
                        num_images=1,
                        return_binary=False,
                    )

                    # Save image
                    image_data = response.images[0]
                    image_bytes = base64.b64decode(image_data)

                    filename = RESULTS_DIR / f"batch_seq_{i}.{detect_image_format(image_bytes)[0]}"
                    with open(filename, "wb") as f:
                        f.write(image_bytes)

                    print(f"   ✅ Generated: {filename}")
                    print(f"   📏 Size: {len(image_bytes)} bytes")

                    if response.timing:
                        print(f"   ⏱️ Time: {response.timing.inferenceDuration}ms")

                except RateLimitError as e:
                    print(f"   ❌ Rate limited (retry after {e.retry_after_seconds}s): {e}")
                    ok = False
                except Exception as e:
                    print(f"   ❌ Failed: {e}")
                    ok = False

                # Rate limit delay between sequential requests
                await asyncio.sleep(1.5)

            elapsed = time.time() - start_time
            print(f"\n⏱️ Total time: {elapsed:.2f}s ({elapsed / len(prompts):.2f}s per image)")

        except Exception as e:
            print(f"❌ Error in batch generation: {e}")
            ok = False

    return ok


async def concurrent_batch_generation() -> bool:
    """Generate multiple images concurrently for better performance.

    Uses ``client.gather`` with a concurrency cap so the batch does not
    self-inflict 429s. Returns ``True`` only if every image succeeded.
    """
    print("\n⚡ Concurrent Batch Generation")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            # Get image model
            image_model = await client.models.resolve_image()
            print(f"📍 Using image model: {image_model}")

            # Different prompts
            prompts = [
                "A cyberpunk street scene at night",
                "A serene Japanese zen garden",
                "A steampunk airship in the clouds",
                "A tropical beach at sunset",
            ]

            async def generate_and_save(prompt: str, index: int) -> dict:
                """Generate a single image and save it."""
                try:
                    response: ImageGenerationResponse = await client.image.create(
                        model=image_model,
                        prompt=prompt,
                        width=512,
                        height=512,
                        num_images=1,
                        return_binary=False,
                    )

                    image_data = response.images[0]
                    image_bytes = base64.b64decode(image_data)

                    filename = (
                        RESULTS_DIR
                        / f"batch_concurrent_{index}.{detect_image_format(image_bytes)[0]}"
                    )
                    with open(filename, "wb") as f:
                        f.write(image_bytes)

                    return {
                        "index": index,
                        "prompt": prompt,
                        "filename": str(filename),
                        "size": len(image_bytes),
                        "success": True,
                        "timing": response.timing.inferenceDuration if response.timing else None,
                    }

                except RateLimitError as e:
                    return {
                        "index": index,
                        "prompt": prompt,
                        "success": False,
                        "error": f"rate limited (retry after {e.retry_after_seconds}s): {e}",
                    }
                except Exception as e:
                    return {"index": index, "prompt": prompt, "success": False, "error": str(e)}

            print(f"\n🚀 Generating {len(prompts)} images concurrently...")
            start_time = time.time()

            # Create awaitables for concurrent generation
            tasks = [generate_and_save(prompt, i + 1) for i, prompt in enumerate(prompts)]

            # Execute with a bounded concurrency cap to avoid self-inflicted 429s.
            # The inner helper already returns dicts (never raises), so results
            # are dicts in input order.
            results = await client.gather(tasks, max_concurrency=2)

            elapsed = time.time() - start_time

            # Display results
            print("\n📊 Results:")
            successful = 0
            failed = 0

            for result in results:
                if result["success"]:
                    successful += 1
                    print(f"\n✅ Image {result['index']}")
                    print(f"   Prompt: {result['prompt'][:40]}...")
                    print(f"   File: {result['filename']}")
                    print(f"   Size: {result['size']} bytes")
                    if result["timing"]:
                        print(f"   Time: {result['timing']}ms")
                else:
                    failed += 1
                    print(f"\n❌ Image {result['index']}")
                    print(f"   Prompt: {result['prompt'][:40]}...")
                    print(f"   Error: {result['error']}")

            print(f"\n⏱️ Total time: {elapsed:.2f}s")
            print(f"📈 Average: {elapsed / len(prompts):.2f}s per image (with concurrency)")
            print(f"✅ Successful: {successful}/{len(prompts)}")
            if failed > 0:
                print(f"❌ Failed: {failed}/{len(prompts)}")
                ok = False

        except Exception as e:
            print(f"❌ Error in concurrent generation: {e}")
            ok = False

    return ok


async def batch_with_variations() -> bool:
    """Generate multiple variations of the same prompt.

    Returns ``True`` only if every variation succeeded.
    """
    print("\n🎲 Batch Variations of Same Prompt")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            # Get image model
            image_model = await client.models.resolve_image()
            print(f"📍 Using image model: {image_model}")

            base_prompt = "A majestic dragon perched on a mountain peak"
            num_variations = 4

            print(f"\n🎨 Generating {num_variations} variations of:")
            print(f"   '{base_prompt}'")

            async def generate_variation(seed: int, index: int) -> dict:
                """Generate a variation with a specific seed."""
                try:
                    response: ImageGenerationResponse = await client.image.create(
                        model=image_model,
                        prompt=base_prompt,
                        width=512,
                        height=512,
                        num_images=1,
                        seed=seed,  # Different seed for variation
                        return_binary=False,
                    )

                    image_data = response.images[0]
                    image_bytes = base64.b64decode(image_data)

                    filename = (
                        RESULTS_DIR
                        / f"variation_{index}_seed{seed}.{detect_image_format(image_bytes)[0]}"
                    )
                    with open(filename, "wb") as f:
                        f.write(image_bytes)

                    return {
                        "index": index,
                        "seed": seed,
                        "filename": str(filename),
                        "success": True,
                    }

                except RateLimitError as e:
                    return {
                        "index": index,
                        "seed": seed,
                        "success": False,
                        "error": f"rate limited (retry after {e.retry_after_seconds}s): {e}",
                    }
                except Exception as e:
                    return {"index": index, "seed": seed, "success": False, "error": str(e)}

            # Generate with different seeds concurrently, capped to avoid 429s
            seeds = [42, 123, 456, 789]
            tasks = [generate_variation(seed, i + 1) for i, seed in enumerate(seeds)]

            results = await client.gather(tasks, max_concurrency=2)

            # Display results
            print("\n📊 Variations generated:")
            for result in results:
                if result["success"]:
                    print(
                        f"   ✅ Variation {result['index']} (seed: {result['seed']}): {result['filename']}"
                    )
                else:
                    print(f"   ❌ Variation {result['index']} failed: {result['error']}")
                    ok = False

        except Exception as e:
            print(f"❌ Error in variation generation: {e}")
            ok = False

    return ok


async def progressive_batch_generation() -> bool:
    """Generate images in progressive batches with feedback.

    Returns ``True`` only if every image across all batches succeeded.
    """
    print("\n📈 Progressive Batch Generation")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            # Get image model
            image_model = await client.models.resolve_image()
            print(f"📍 Using image model: {image_model}")

            # All prompts to generate
            all_prompts = [
                "A red sports car",
                "A blue ocean wave",
                "A green forest path",
                "A yellow sunflower field",
                "A purple sunset sky",
                "A white snowy mountain",
            ]

            batch_size = 2
            total_batches = (len(all_prompts) + batch_size - 1) // batch_size

            print(f"\n📦 Processing {len(all_prompts)} prompts in batches of {batch_size}")
            print(f"   Total batches: {total_batches}")

            all_results = []

            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(all_prompts))
                batch_prompts = all_prompts[start_idx:end_idx]

                print(f"\n🔄 Batch {batch_num + 1}/{total_batches} ({len(batch_prompts)} images)")

                async def generate_in_batch(prompt: str, global_idx: int) -> dict:
                    """Generate image for this batch."""
                    try:
                        response: ImageGenerationResponse = await client.image.create(
                            model=image_model,
                            prompt=prompt,
                            width=512,
                            height=512,
                            num_images=1,
                            return_binary=False,
                        )

                        image_data = response.images[0]
                        image_bytes = base64.b64decode(image_data)

                        filename = (
                            RESULTS_DIR
                            / f"progressive_{global_idx}.{detect_image_format(image_bytes)[0]}"
                        )
                        with open(filename, "wb") as f:
                            f.write(image_bytes)

                        return {"success": True, "prompt": prompt, "filename": str(filename)}

                    except RateLimitError as e:
                        return {
                            "success": False,
                            "prompt": prompt,
                            "error": f"rate limited (retry after {e.retry_after_seconds}s): {e}",
                        }
                    except Exception as e:
                        return {"success": False, "prompt": prompt, "error": str(e)}

                # Generate this batch concurrently, capped to avoid 429s
                tasks = [
                    generate_in_batch(prompt, start_idx + i)
                    for i, prompt in enumerate(batch_prompts)
                ]

                batch_results = await client.gather(tasks, max_concurrency=2)
                all_results.extend(batch_results)

                # Show progress
                successful = sum(1 for r in batch_results if r["success"])
                print(f"   ✅ Completed: {successful}/{len(batch_results)}")

            # Final summary
            total_success = sum(1 for r in all_results if r["success"])
            total_failed = len(all_results) - total_success

            print("\n📊 Final Summary:")
            print(f"   Total images: {len(all_results)}")
            print(f"   ✅ Successful: {total_success}")
            if total_failed > 0:
                print(f"   ❌ Failed: {total_failed}")
                ok = False

        except Exception as e:
            print(f"❌ Error in progressive generation: {e}")
            ok = False

    return ok


async def batch_with_mixed_parameters() -> bool:
    """Generate batch with different parameters per image.

    Returns ``True`` only if every configured image succeeded.
    """
    print("\n⚙️ Batch with Mixed Parameters")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            # Get image model
            image_model = await client.models.resolve_image()
            print(f"📍 Using image model: {image_model}")

            # Different configurations
            configs = [
                {"prompt": "A landscape painting", "width": 512, "height": 512, "name": "square"},
                {
                    "prompt": "A portrait photograph",
                    "width": 512,
                    "height": 768,
                    "name": "portrait",
                },
                {"prompt": "A panoramic view", "width": 768, "height": 512, "name": "landscape"},
            ]

            print(f"\n🎨 Generating {len(configs)} images with different parameters...")

            async def generate_with_config(config: dict, index: int) -> dict:
                """Generate image with specific configuration."""
                try:
                    response: ImageGenerationResponse = await client.image.create(
                        model=image_model,
                        prompt=config["prompt"],
                        width=config["width"],
                        height=config["height"],
                        num_images=1,
                        return_binary=False,
                    )

                    image_data = response.images[0]
                    image_bytes = base64.b64decode(image_data)

                    filename = (
                        RESULTS_DIR
                        / f"mixed_{config['name']}.{detect_image_format(image_bytes)[0]}"
                    )
                    with open(filename, "wb") as f:
                        f.write(image_bytes)

                    return {
                        "success": True,
                        "config": config,
                        "filename": str(filename),
                        "size": len(image_bytes),
                    }

                except RateLimitError as e:
                    return {
                        "success": False,
                        "config": config,
                        "error": f"rate limited (retry after {e.retry_after_seconds}s): {e}",
                    }
                except Exception as e:
                    return {"success": False, "config": config, "error": str(e)}

            results = []
            for i, config in enumerate(configs):
                result = await generate_with_config(config, i)
                results.append(result)
                # Rate limit delay between sequential requests
                await asyncio.sleep(1.5)

            # Display results
            print("\n📊 Results:")
            for result in results:
                if result["success"]:
                    config = result["config"]
                    print(f"\n✅ {config['name'].capitalize()} format")
                    print(f"   Dimensions: {config['width']}x{config['height']}")
                    print(f"   File: {result['filename']}")
                    print(f"   Size: {result['size']} bytes")
                else:
                    print(f"\n❌ {result['config']['name']} failed: {result['error']}")
                    ok = False

        except Exception as e:
            print(f"❌ Error in mixed parameter generation: {e}")
            ok = False

    return ok


async def main() -> int:
    """Run all batch generation examples.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure (including a partial batch where some images failed) surfaces as a
    non-zero process exit instead of being masked by the success banner.
    """
    print("🚀 Venice AI Batch Image Generation Examples")
    print("=" * 50)

    # Aggregate per-demo results. The inter-demo sleeps are anti-429 pacing.
    results: list[tuple[str, bool]] = []
    results.append(("simple_batch_generation", await simple_batch_generation()))
    await asyncio.sleep(5)
    results.append(("concurrent_batch_generation", await concurrent_batch_generation()))
    await asyncio.sleep(5)
    results.append(("batch_with_variations", await batch_with_variations()))
    await asyncio.sleep(5)
    results.append(("progressive_batch_generation", await progressive_batch_generation()))
    await asyncio.sleep(5)
    results.append(("batch_with_mixed_parameters", await batch_with_mixed_parameters()))

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("\n✨ Batch generation examples completed!")

    print("\n💡 Key concepts demonstrated:")
    print("   - Sequential batch generation")
    print("   - Concurrent generation for performance")
    print("   - Generating variations with different seeds")
    print("   - Progressive batch processing")
    print("   - Mixed parameters in batches")
    print("   - Bounded concurrency via client.gather(max_concurrency=N)")
    print("   - Error handling in batch operations")

    print("\n📁 Generated files in examples/results/:")
    print("   - batch_seq_*.png (sequential)")
    print("   - batch_concurrent_*.png (concurrent)")
    print("   - variation_*.png (same prompt variations)")
    print("   - progressive_*.png (progressive batches)")
    print("   - mixed_*.png (different parameters)")

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
