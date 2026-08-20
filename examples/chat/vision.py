#!/usr/bin/env python3
"""
Venice AI SDK - Vision / Multimodal Chat Completions
=====================================================

This example demonstrates how to send images to vision-capable models using
the Venice AI SDK. Every image is fetched and encoded as a base64 ``data:``
URI before sending (see "Why we pre-fetch and resize the source images"
below) — learn how to analyze a single image, generate-then-analyze a
self-contained image, compare multiple images, and use different analysis
prompt strategies.

Why we pre-fetch and resize the source images
---------------------------------------------
Some Venice vision models accept smaller maximum input dimensions than others
(e.g. ``venice-uncensored-1-2`` and ``venice-uncensored-role-play`` will return
HTTP 500 ``"Inference processing failed"`` when handed a multi-megapixel
image, while ``qwen3-vl-235b-a22b`` accepts the same payload). To make this
example work across every vision-capable model — including the
venice-uncensored family — we download the source images once and run them
through ``fit_image_bytes(max_dim=1024)`` before encoding as base64. That
sidesteps the image-size limit on the picky models without any model
exclusion list.
"""

import asyncio
import base64
import sys
from pathlib import Path

import aiohttp

from venice_ai import VeniceClient, detect_image_format, fit_image_bytes
from venice_ai.types.api import (
    SystemMessage,
    UserMessage,
)

# Resolve results dir relative to this file's location.
# All example scripts live one level below examples/ (e.g., examples/chat/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Public image URLs (Wikimedia Commons — stable, direct, no signed redirects).
# ---------------------------------------------------------------------------
# Wikimedia originals are full-resolution photographs (multi-megabyte / multi-
# megapixel). We download once and run them through fit_image_bytes() before
# sending so every Venice vision model — including the venice-uncensored
# family with its tighter image-size limit — accepts them.
#
# IMAGE_1: tabby cat on a stone wall  → expect "cat", "tabby", "feline"
# IMAGE_2: tri-color beagle on a leash → expect "dog", "beagle", "canine"
SAMPLE_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg"
SAMPLE_IMAGE_URL_2 = "https://upload.wikimedia.org/wikipedia/commons/5/55/Beagle_600.jpg"


async def _fetch_and_fit_data_uri(
    session: aiohttp.ClientSession, url: str, *, max_dim: int = 1024
) -> str:
    """Fetch *url*, fit to ``max_dim``, return a base64 ``data:`` URI.

    Sets a friendly User-Agent — Wikimedia and many other CDNs reject the
    default aiohttp UA.
    """
    headers = {"User-Agent": "venice-py-sdk-examples/1.0 (vision example)"}
    async with session.get(url, headers=headers) as resp:
        resp.raise_for_status()
        raw = await resp.read()
    fitted = fit_image_bytes(raw, max_dim=max_dim)
    _ext, mime = detect_image_format(fitted)
    return f"data:{mime};base64,{base64.b64encode(fitted).decode('ascii')}"


def _check_keywords(content: str, keywords: list[str], label: str) -> None:
    """Print a caveat warning if response doesn't mention any expected keyword."""
    lowered = content.lower()
    if not any(kw.lower() in lowered for kw in keywords):
        print(f"⚠️  {label}: response didn't mention expected keyword(s) {keywords}")


# ============================================================================
# 1. Basic Image Analysis
# ============================================================================


async def basic_image_analysis(cat_uri: str):
    print("👁️  Basic Image Analysis")
    print("-" * 40)

    async with VeniceClient() as client:
        vision_model = await client.models.resolve_chat(require_vision=True)
        print(f"📍 Using vision model: {vision_model}")

        # Pattern 1: fluent builder — most readable for incremental construction.
        # cat_uri is a fitted base64 data URI prepared at startup.
        message = (
            UserMessage.builder()
            .text("What do you see in this image? Describe it in detail.")
            .image(cat_uri)
            .build()
        )

        response = await client.chat.completions.create(
            model=vision_model,
            messages=[message],
            max_completion_tokens=300,
            temperature=0.5,
        )

        content = response.text or ""
        print(f"🤖 Assistant: {content}")

        if response.usage:
            usage = response.usage
            print("\n📊 Token Usage:")
            print(f"   Input tokens:  {usage.prompt_tokens}")
            print(f"   Output tokens: {usage.completion_tokens}")
            print(f"   Total tokens:  {usage.total_tokens}")

        # SAMPLE_IMAGE_URL is a tabby cat on a stone wall
        _check_keywords(content, ["cat", "tabby", "feline", "kitten"], "basic_image_analysis")


# ============================================================================
# 2. Base64 Image Input (self-contained)
# ============================================================================


async def base64_image_input():
    print("\n🖼️  Base64 Image Input (generated → analysed)")
    print("-" * 40)

    async with VeniceClient() as client:
        # --- Step 1: Generate an image so the demo is self-contained ---
        image_model = await client.models.resolve(type="image")
        print(f"🎨 Generating image with: {image_model}")

        gen_response = await client.image.create(
            model=image_model,
            prompt="A small red wooden boat on a calm blue lake at sunrise",
            width=512,
            height=512,
            num_images=1,
            return_binary=False,
        )
        b64_data = gen_response.images[0]
        print(f"✅ Generated image ({len(b64_data)} base64 chars)")

        # Save the generated image — gen_response.save() picks the right
        # extension from magic bytes (z-image-turbo returns WebP, not PNG).
        decoded_bytes = base64.b64decode(b64_data)
        result_path = gen_response.save(RESULTS_DIR / "vision_generated_sample", overwrite=True)
        print(f"💾 Saved generated image to {result_path}")

        # --- Step 2: Send the base64 image to a vision model ---
        vision_model = await client.models.resolve_chat(require_vision=True)
        print(f"📍 Analysing with vision model: {vision_model}")

        # Use UserMessage.builder() — same fluent style as basic_image_analysis,
        # but with a base64 data URI built from the bytes we just generated.
        _ext, mime_type = detect_image_format(decoded_bytes)
        data_uri = f"data:{mime_type};base64,{b64_data}"
        message = (
            UserMessage.builder()
            .text("Describe this image in one paragraph.")
            .image(data_uri)
            .build()
        )

        response = await client.chat.completions.create(
            model=vision_model,
            messages=[message],
            max_completion_tokens=250,
            temperature=0.5,
        )

        content = response.text or ""
        print(f"🤖 Assistant: {content}")

        # Image was generated from prompt: red wooden boat on a blue lake at sunrise
        _check_keywords(
            content,
            ["boat", "lake", "water", "sunrise", "ship", "vessel"],
            "base64_image_input",
        )


# ============================================================================
# 3. Image Comparison
# ============================================================================


async def image_comparison(cat_uri: str, dog_uri: str):
    print("\n🔍 Image Comparison (two images)")
    print("-" * 40)

    async with VeniceClient() as client:
        vision_model = await client.models.resolve_chat(require_vision=True)
        print(f"📍 Using vision model: {vision_model}")

        # Builder makes multi-image messages especially compact
        message = (
            UserMessage.builder()
            .text(
                "I'm showing you two images. "
                "Please compare them: what are the main differences and similarities?"
            )
            .image(cat_uri)
            .image(dog_uri)
            .build()
        )

        response = await client.chat.completions.create(
            model=vision_model,
            messages=[
                SystemMessage(
                    content="You are an observant image analyst. Be concise but thorough.",
                ),
                message,
            ],
            max_completion_tokens=400,
            temperature=0.4,
        )

        content = response.text or ""
        print(f"🤖 Comparison: {content}")

        # Comparison should mention something from at least one of the two
        # images (cat/feline from image 1 or dog/beagle from image 2).
        _check_keywords(
            content,
            ["cat", "feline", "tabby", "dog", "beagle", "canine"],
            "image_comparison",
        )


# ============================================================================
# 4. Detailed Analysis Prompts
# ============================================================================


async def detailed_analysis_prompts(cat_uri: str):
    print("\n📝 Detailed Analysis Prompts")
    print("-" * 40)

    # Various prompt strategies for the same image
    strategies = [
        ("Description", "Describe exactly what you see in this image."),
        (
            "OCR / Text extraction",
            "Extract any visible text from this image. If there is no text, say so.",
        ),
        ("Object counting", "List and count all distinct objects visible in this image."),
        (
            "Spatial reasoning",
            "Describe the spatial layout of the image: foreground, middle-ground, and background.",
        ),
    ]

    async with VeniceClient() as client:
        vision_model = await client.models.resolve_chat(require_vision=True)
        print(f"📍 Using vision model: {vision_model}")

        # Aggregate descriptions for a single keyword check across strategies.
        all_text_parts: list[str] = []

        for label, prompt_text in strategies:
            print(f"\n🔎 Strategy: {label}")
            message = UserMessage.builder().text(prompt_text).image(cat_uri).build()

            try:
                response = await client.chat.completions.create(
                    model=vision_model,
                    messages=[message],
                    max_completion_tokens=200,
                    temperature=0.3,
                )

                content = response.text or ""
                print(f"   🤖 {content}")
                all_text_parts.append(content)
            except Exception as e:
                print(f"   ❌ Error: {e}")
                # Re-raise so the outer try/except marks this sub-example failed.
                raise

        # Verify at least one strategy mentioned something cat-like.
        _check_keywords(
            "\n".join(all_text_parts),
            ["cat", "tabby", "feline", "kitten"],
            "detailed_analysis_prompts",
        )


# ============================================================================
# 5. Model Discovery — list vision-capable models
# ============================================================================


async def model_discovery():
    print("\n🗺️  Vision Model Discovery")
    print("-" * 40)

    from venice_ai.types.api import TextModelSpec

    async with VeniceClient() as client:
        # Vision is a chat/text capability — restrict the listing to text
        # models so ``capabilities`` is typed.
        all_models = await client.models.list(type="text")

        vision_models = []
        for model in all_models.data:
            spec = model.model_spec
            if not isinstance(spec, TextModelSpec):
                continue
            caps = spec.capabilities
            if caps is not None and caps.supportsVision:
                vision_models.append(model.id)

        print(f"✅ Found {len(vision_models)} vision-capable model(s):")
        for m in sorted(vision_models):
            print(f"   • {m}")

        # Also show the default vision model selected by resolve
        default_vision = await client.models.resolve_chat(require_vision=True)
        print(f"\n⭐ Default vision model (via resolve_chat): {default_vision}")


# ============================================================================
# Main
# ============================================================================


async def main() -> int:
    print("🚀 Venice AI Vision / Multimodal Examples")
    print("=" * 50)

    # Pre-fetch and fit the source images once. fit_image_bytes(max_dim=1024)
    # ensures the resulting data URIs are accepted by every Venice vision
    # model — including the venice-uncensored family which has tighter input
    # size limits than other vision models.
    print("\n⬇️  Pre-fetching source images and fitting to 1024 px...")
    async with aiohttp.ClientSession() as session:
        cat_uri, dog_uri = await asyncio.gather(
            _fetch_and_fit_data_uri(session, SAMPLE_IMAGE_URL),
            _fetch_and_fit_data_uri(session, SAMPLE_IMAGE_URL_2),
        )
    print(f"   ✅ cat ({len(cat_uri)} chars), dog ({len(dog_uri)} chars)")

    sub_examples = [
        ("basic_image_analysis", basic_image_analysis(cat_uri)),
        ("base64_image_input", base64_image_input()),
        ("image_comparison", image_comparison(cat_uri, dog_uri)),
        ("detailed_analysis_prompts", detailed_analysis_prompts(cat_uri)),
        ("model_discovery", model_discovery()),
    ]

    results: list[tuple[str, bool]] = []
    for name, coro in sub_examples:
        try:
            await coro
            results.append((name, True))
        except Exception as e:
            print(f"\n❌ Sub-example '{name}' failed: {e!r}", file=sys.stderr)
            results.append((name, False))

    # Summary
    succeeded = sum(1 for _, ok in results if ok)
    total = len(results)
    print("\n" + "=" * 50)
    print(f"✨ {succeeded}/{total} vision examples completed")
    for name, ok in results:
        marker = "✅" if ok else "❌"
        print(f"   {marker} {name}")

    print("\n💡 Key concepts demonstrated:")
    print("   - fit_image_bytes() to keep images within all models' size limits")
    print("   - Sending images as base64 data URIs")
    print("   - Comparing multiple images in one message")
    print("   - Different analysis prompt strategies")
    print("   - Discovering vision-capable models")

    return 0 if succeeded == total else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(exit_code)
