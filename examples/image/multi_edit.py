#!/usr/bin/env python3
"""
Venice AI SDK - Multi-Layer Image Editing
=========================================

This example demonstrates how to use the Venice AI SDK's most advanced image
editing endpoint: ``client.image.multi_edit()``.  Unlike the single-image
``edit()`` method, ``multi_edit()`` accepts up to **three layered inputs**
(base image + up to two overlay layers) and composites them according to a
text prompt.

The ``multi_edit()`` method returns raw image **bytes** — save the result with
``open("output.png", "wb")``.
"""

import asyncio
import sys
from pathlib import Path

from venice_ai import VeniceClient, detect_image_format

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import generate_base_image as _generate_base_image

# Resolve results dir relative to this file's location.
# All example scripts live one level below examples/ (e.g., examples/image/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Accumulator of files that were actually written to disk, in write order.
# Each demo appends to this immediately after a successful ``f.write()`` so the
# end-of-run summary reflects exactly what hit disk — with real extensions —
# even when a demo fails partway through.
WRITTEN: list[str] = []


# ---------------------------------------------------------------------------
# 1. Basic multi-edit with a single image
# ---------------------------------------------------------------------------


async def basic_multi_edit() -> bool:
    """Use multi_edit() with a single image and prompt (simplest case).

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print("🔀 Basic Multi-Edit (Single Image)")
    print("-" * 30)

    ok = True
    async with VeniceClient() as client:
        try:
            # Step 1 — generate a base image
            print("🎨 Generating base image …")
            base_bytes = await _generate_base_image(
                client,
                "A sunlit park with a stone fountain and tall oak trees",
            )

            base_path = str(
                RESULTS_DIR / f"multi_edit_base_park.{detect_image_format(base_bytes)[0]}"
            )
            with open(base_path, "wb") as f:
                f.write(base_bytes)
            WRITTEN.append(base_path)
            print(f"💾 Base image saved: {base_path} ({len(base_bytes)} bytes)")

            # Step 2 — select an inpaint / edit model
            inpaint_model = await client.models.resolve_inpaint()
            print(f"📍 Using edit model: {inpaint_model}")

            # Step 3 — multi-edit with just one image + prompt
            # This is similar to edit() but uses the multi-edit endpoint.
            print("🔀 Applying multi-edit: adding autumn foliage …")
            edited_bytes = await client.image.multi_edit(
                prompt="Transform the trees to show vibrant autumn foliage with orange and red leaves",
                model=inpaint_model,
                image=base_bytes,
            )

            out_path = str(
                RESULTS_DIR / f"multi_edit_autumn.{detect_image_format(edited_bytes)[0]}"
            )
            with open(out_path, "wb") as f:
                f.write(edited_bytes)
            WRITTEN.append(out_path)
            print(f"✅ Edited image saved: {out_path} ({len(edited_bytes)} bytes)")

        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Note: Multi-edit requires appropriate API access")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# 2. Two-layer editing
# ---------------------------------------------------------------------------


async def two_layer_editing() -> bool:
    """Combine two generated images using multi_edit().

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print("\n🖼️  Two-Layer Editing")
    print("-" * 30)

    ok = True
    async with VeniceClient() as client:
        try:
            # Generate two base images
            print("🎨 Generating layer 1: landscape …")
            layer1_bytes = await _generate_base_image(
                client,
                "A dramatic desert landscape with sand dunes at golden hour",
            )
            layer1_path = str(
                RESULTS_DIR / f"multi_edit_layer1_desert.{detect_image_format(layer1_bytes)[0]}"
            )
            with open(layer1_path, "wb") as f:
                f.write(layer1_bytes)
            WRITTEN.append(layer1_path)
            print(f"💾 Layer 1 saved: {layer1_path}")

            print("🎨 Generating layer 2: sky overlay …")
            layer2_bytes = await _generate_base_image(
                client,
                "A starry night sky with the Milky Way and shooting stars",
            )
            layer2_path = str(
                RESULTS_DIR / f"multi_edit_layer2_sky.{detect_image_format(layer2_bytes)[0]}"
            )
            with open(layer2_path, "wb") as f:
                f.write(layer2_bytes)
            WRITTEN.append(layer2_path)
            print(f"💾 Layer 2 saved: {layer2_path}")

            # Select an inpaint model
            inpaint_model = await client.models.resolve_inpaint()
            print(f"📍 Using edit model: {inpaint_model}")

            # Combine the two images with multi_edit()
            print("🔀 Compositing two layers …")
            composited = await client.image.multi_edit(
                prompt="Blend the desert landscape with the starry night sky, creating a magical twilight scene",
                model=inpaint_model,
                image=layer1_bytes,
                image_2=layer2_bytes,
            )

            out_path = str(
                RESULTS_DIR / f"multi_edit_two_layer.{detect_image_format(composited)[0]}"
            )
            with open(out_path, "wb") as f:
                f.write(composited)
            WRITTEN.append(out_path)
            print(f"✅ Composited image saved: {out_path} ({len(composited)} bytes)")

        except Exception as e:
            print(f"❌ Error: {e}")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# 3. Three-layer composition
# ---------------------------------------------------------------------------


async def three_layer_composition() -> bool:
    """Demonstrate full three-layer compositing with multi_edit().

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print("\n🎭 Three-Layer Composition")
    print("-" * 30)

    ok = True
    async with VeniceClient() as client:
        try:
            # Generate three base images
            print("🎨 Generating layer 1: background environment …")
            bg_bytes = await _generate_base_image(
                client,
                "A misty enchanted forest with ancient trees and soft green light",
            )
            bg_path = str(RESULTS_DIR / f"multi_edit_layer_bg.{detect_image_format(bg_bytes)[0]}")
            with open(bg_path, "wb") as f:
                f.write(bg_bytes)
            WRITTEN.append(bg_path)
            print(f"💾 Background saved: {bg_path}")

            print("🎨 Generating layer 2: mid-ground element …")
            mid_bytes = await _generate_base_image(
                client,
                "A crystal-clear woodland stream with mossy rocks and ferns",
            )
            mid_path = str(
                RESULTS_DIR / f"multi_edit_layer_mid.{detect_image_format(mid_bytes)[0]}"
            )
            with open(mid_path, "wb") as f:
                f.write(mid_bytes)
            WRITTEN.append(mid_path)
            print(f"💾 Mid-ground saved: {mid_path}")

            print("🎨 Generating layer 3: foreground accent …")
            fg_bytes = await _generate_base_image(
                client,
                "Glowing fireflies and magical floating lanterns in a dark scene",
            )
            fg_path = str(RESULTS_DIR / f"multi_edit_layer_fg.{detect_image_format(fg_bytes)[0]}")
            with open(fg_path, "wb") as f:
                f.write(fg_bytes)
            WRITTEN.append(fg_path)
            print(f"💾 Foreground saved: {fg_path}")

            # Select inpaint model
            inpaint_model = await client.models.resolve_inpaint()
            print(f"📍 Using edit model: {inpaint_model}")

            # Three-layer composition
            print("🔀 Compositing three layers into a single scene …")
            final = await client.image.multi_edit(
                prompt=(
                    "Merge these three layers into a cohesive enchanted forest scene: "
                    "the misty forest as the background, the stream as the mid-ground, "
                    "and the glowing fireflies scattered throughout the foreground"
                ),
                model=inpaint_model,
                image=bg_bytes,
                image_2=mid_bytes,
                image_3=fg_bytes,
            )

            out_path = str(RESULTS_DIR / f"multi_edit_three_layer.{detect_image_format(final)[0]}")
            with open(out_path, "wb") as f:
                f.write(final)
            WRITTEN.append(out_path)
            print(f"✅ Final composition saved: {out_path} ({len(final)} bytes)")

            # Show the full parameter recap
            print("\n📋 Parameters used:")
            print("   • image   → enchanted forest (background)")
            print("   • image_2 → woodland stream (mid-ground)")
            print("   • image_3 → fireflies / lanterns (foreground)")
            print("   • prompt  → merge instruction")
            print("   • model   → inpaint model")

        except Exception as e:
            print(f"❌ Error: {e}")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# 4. Model selection for multi-edit
# ---------------------------------------------------------------------------


async def model_selection() -> bool:
    """Discover and choose models compatible with multi-edit.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print("\n🔍 Model Selection for Multi-Edit")
    print("-" * 30)

    ok = True
    async with VeniceClient() as client:
        try:
            # List available inpaint / edit models (multi-edit uses the same pool)
            inpaint_models_response = await client.models.list(type="inpaint")
            inpaint_model_ids = [m.id for m in inpaint_models_response.data]
            print(f"🎨 Available inpaint/edit models ({len(inpaint_model_ids)}):")
            for model_id in inpaint_model_ids:
                print(f"   • {model_id}")

            # Auto-select the best available model
            default_model = await client.models.resolve_inpaint()
            print(f"\n📍 Auto-selected model: {default_model}")

            # Try preferred models in order. A missing preferred model is an
            # expected, handled condition — it must not fail this demo.
            preferred = ["flux-2-max-edit", "gpt-image-1-5-edit"]
            try:
                preferred_model = await client.models.resolve_inpaint(
                    preferred_models=preferred,
                )
                print(f"⭐ Preferred model selected: {preferred_model}")
            except ValueError:
                print("⚠️  None of the preferred models are available")

            # Quick demo with the selected model
            if inpaint_model_ids:
                print(f"\n🔀 Quick multi-edit demo with {default_model} …")
                base_bytes = await _generate_base_image(
                    client,
                    "A simple wooden table with a white coffee mug",
                    width=512,
                    height=512,
                )

                edited = await client.image.multi_edit(
                    prompt="Add steam rising from the mug and a plate of cookies beside it",
                    model=default_model,
                    image=base_bytes,
                )

                out_path = str(
                    RESULTS_DIR / f"multi_edit_model_demo.{detect_image_format(edited)[0]}"
                )
                with open(out_path, "wb") as f:
                    f.write(edited)
                WRITTEN.append(out_path)
                print(f"✅ Saved: {out_path} ({len(edited)} bytes)")

        except Exception as e:
            print(f"❌ Error: {e}")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    """Run all multi-edit examples.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    print("🚀 Venice AI Multi-Layer Image Editing Examples")
    print("=" * 50)

    results: list[tuple[str, bool]] = [
        ("basic_multi_edit", await basic_multi_edit()),
        ("two_layer_editing", await two_layer_editing()),
        ("three_layer_composition", await three_layer_composition()),
        ("model_selection", await model_selection()),
    ]

    failed = [name for name, ok in results if not ok]

    print("\n" + "=" * 50)
    if failed:
        print(f"⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("✨ Multi-edit examples completed!")

    print("\n💡 Key concepts demonstrated:")
    print("   - Single-image multi-edit (simplest case)")
    print("   - Two-layer compositing (image + image_2)")
    print("   - Three-layer composition (image + image_2 + image_3)")
    print("   - Dynamic inpaint model selection")
    print("   - Self-contained generate → multi-edit workflow")

    # Honest summary: list only the files that were actually written, with the
    # extensions they were saved under (.webp, .png, …) — never a fixed list.
    print("\n📁 Generated files in examples/results/:")
    if WRITTEN:
        for path in WRITTEN:
            print(f"   - {path}")
    else:
        print("   (no files written)")

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
