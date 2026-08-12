#!/usr/bin/env python3
"""
Venice AI SDK - Image Editing
=============================

This example demonstrates how to edit and modify existing images using
the Venice AI SDK's image editing capabilities.  Learn how to use
``client.image.edit()`` to apply AI-powered edits such as adding objects,
changing scenes, removing elements, and prompt-based targeted editing.

The ``edit()`` method returns raw image **bytes** — save the result with
``open("output.png", "wb")``.

The ``model`` kwarg is forwarded to the API, so a call like
``client.image.edit(model=await client.models.resolve_image(), ...)``
edits with the model you select. Use ``safe_mode=False`` to disable the
default adult-content blur on models that support it.
"""

import asyncio
import sys
from pathlib import Path

from venice_ai import VeniceClient, detect_image_format
from venice_ai.exceptions import InvalidRequestError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import generate_base_image as _generate_base_image

# Resolve results dir relative to this file's location.
# All example scripts live one level below examples/ (e.g., examples/image/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Basic image editing
# ---------------------------------------------------------------------------


async def basic_image_editing() -> bool:
    """Generate a base image, then edit it with a text prompt.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print("✏️  Basic Image Editing")
    print("-" * 30)

    ok = True
    async with VeniceClient() as client:
        try:
            # Step 1 — generate a base landscape image
            print("🎨 Generating base landscape image …")
            base_bytes = await _generate_base_image(
                client,
                "A serene mountain landscape with a clear blue sky and green meadows",
            )

            base_path = str(
                RESULTS_DIR / f"edit_base_landscape.{detect_image_format(base_bytes)[0]}"
            )
            with open(base_path, "wb") as f:
                f.write(base_bytes)
            print(f"💾 Base image saved: {base_path} ({len(base_bytes)} bytes)")

            # Step 2 — select an inpaint / edit model
            inpaint_model = await client.models.resolve_inpaint()
            print(f"📍 Using edit model: {inpaint_model}")

            # Step 3 — edit the image: add a rainbow to the sky
            print("✏️  Editing image: adding a vibrant rainbow …")
            edited_bytes = await client.image.edit(
                prompt="Add a vivid rainbow arching across the sky",
                model=inpaint_model,
                image=base_bytes,  # pass raw bytes directly
            )

            edited_path = str(
                RESULTS_DIR / f"edit_with_rainbow.{detect_image_format(edited_bytes)[0]}"
            )
            with open(edited_path, "wb") as f:
                f.write(edited_bytes)
            print(f"✅ Edited image saved: {edited_path} ({len(edited_bytes)} bytes)")

        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Note: Image editing requires appropriate API access")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# 2. Different input methods
# ---------------------------------------------------------------------------


async def different_input_methods() -> bool:
    """Demonstrate the various ways to pass an image to ``edit()``.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print("\n🖼️  Different Input Methods")
    print("-" * 30)

    ok = True
    async with VeniceClient() as client:
        try:
            # Generate a base image to work with
            print("🎨 Generating base image …")
            base_bytes = await _generate_base_image(
                client,
                "A cozy cottage in a flower garden",
            )

            base_path = str(RESULTS_DIR / f"edit_base_cottage.{detect_image_format(base_bytes)[0]}")
            with open(base_path, "wb") as f:
                f.write(base_bytes)
            print(f"💾 Base image saved: {base_path}")

            inpaint_model = await client.models.resolve_inpaint()
            print(f"📍 Using edit model: {inpaint_model}")

            # --- Method A: raw bytes ---
            print("\n📦 Method A — raw bytes …")
            edited_a = await client.image.edit(
                prompt="Add a cat sitting on the doorstep",
                model=inpaint_model,
                image=base_bytes,
            )
            path_a = str(RESULTS_DIR / f"edit_input_bytes.{detect_image_format(edited_a)[0]}")
            with open(path_a, "wb") as f:
                f.write(edited_a)
            print(f"✅ Saved: {path_a} ({len(edited_a)} bytes)")

            # --- Method B: file path string ---
            print("\n📂 Method B — file path string …")
            edited_b = await client.image.edit(
                prompt="Add a cat sitting on the doorstep",
                model=inpaint_model,
                image=base_path,  # pass the path as a string
            )
            path_b = str(RESULTS_DIR / f"edit_input_filepath.{detect_image_format(edited_b)[0]}")
            with open(path_b, "wb") as f:
                f.write(edited_b)
            print(f"✅ Saved: {path_b} ({len(edited_b)} bytes)")

            # --- Method C: file-like object ---
            print("\n📄 Method C — file-like object …")
            with open(base_path, "rb") as img_file:
                edited_c = await client.image.edit(
                    prompt="Add a cat sitting on the doorstep",
                    model=inpaint_model,
                    image=img_file,
                )
            path_c = str(RESULTS_DIR / f"edit_input_fileobj.{detect_image_format(edited_c)[0]}")
            with open(path_c, "wb") as f:
                f.write(edited_c)
            print(f"✅ Saved: {path_c} ({len(edited_c)} bytes)")

            # --- Method D: URL (JSON mode) ---
            print("\n🌐 Method D — URL string (JSON mode) …")
            print("   ℹ️  URLs are sent via JSON rather than multipart upload.")
            print('   ℹ️  Example: image="https://example.com/photo.jpg"')
            print("   (Skipping live call — replace with a real URL to test)")

        except Exception as e:
            print(f"❌ Error: {e}")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# 3. Prompt-based targeted editing
# ---------------------------------------------------------------------------


async def editing_with_prompt() -> bool:
    """Demonstrate targeted edits driven entirely by the text prompt.

    The edit endpoint is prompt-driven: the model decides what to change based
    on your instruction (there is no separate mask parameter).

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print("\n🎭 Prompt-Based Targeted Editing")
    print("-" * 30)

    ok = True
    async with VeniceClient() as client:
        try:
            print("ℹ️  How targeted edits work:")
            print("   • The model interprets the prompt to decide what to change.")
            print("   • Be specific about the region/subject in your instruction")
            print("     (e.g. 'replace the sunflowers with red roses').\n")

            # Generate a base image
            print("🎨 Generating base image …")
            base_bytes = await _generate_base_image(
                client,
                "A still life painting of a vase of sunflowers on a wooden table",
            )

            base_path = str(
                RESULTS_DIR / f"edit_base_stilllife.{detect_image_format(base_bytes)[0]}"
            )
            with open(base_path, "wb") as f:
                f.write(base_bytes)
            print(f"💾 Base image saved: {base_path}")

            inpaint_model = await client.models.resolve_inpaint()
            print(f"📍 Using edit model: {inpaint_model}")

            # The model freely interprets the prompt to perform the edit.
            print("\n✏️  Editing (model interprets the prompt) …")
            edited = await client.image.edit(
                prompt="Replace the sunflowers with red roses",
                model=inpaint_model,
                image=base_bytes,
            )
            edited_path = str(RESULTS_DIR / f"edit_roses.{detect_image_format(edited)[0]}")
            with open(edited_path, "wb") as f:
                f.write(edited)
            print(f"✅ Saved: {edited_path} ({len(edited)} bytes)")

        except Exception as e:
            print(f"❌ Error: {e}")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# 4. Model selection for image editing
# ---------------------------------------------------------------------------


async def model_selection() -> bool:
    """Show how to discover and choose image editing models.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print("\n🔍 Model Selection for Image Editing")
    print("-" * 30)

    ok = True
    async with VeniceClient() as client:
        try:
            # List all available inpaint / edit models
            inpaint_models_response = await client.models.list(type="inpaint")
            inpaint_model_ids = [m.id for m in inpaint_models_response.data]
            print(f"🎨 Available inpaint/edit models ({len(inpaint_model_ids)}):")
            for model_id in inpaint_model_ids:
                print(f"   • {model_id}")

            # Auto-select the best available inpaint model
            default_model = await client.models.resolve_inpaint()
            print(f"\n📍 Auto-selected model: {default_model}")

            # Select with preferred model list (tries each in order)
            preferred = ["flux-2-max-edit", "gpt-image-1-5-edit", "qwen-edit"]
            try:
                preferred_model = await client.models.resolve_inpaint(
                    preferred_models=preferred,
                )
                print(f"⭐ Preferred model selected: {preferred_model}")
            except ValueError:
                print("⚠️  None of the preferred models are available")

            # Quick demo: edit with the selected model
            if inpaint_model_ids:
                print(f"\n✏️  Quick edit demo with {default_model} …")
                base_bytes = await _generate_base_image(
                    client,
                    "A simple red car parked on a street",
                    width=512,
                    height=512,
                )

                edited = await client.image.edit(
                    prompt="Change the car color to bright blue",
                    model=default_model,
                    image=base_bytes,
                )

                out_path = str(
                    RESULTS_DIR / f"edit_model_selection.{detect_image_format(edited)[0]}"
                )
                with open(out_path, "wb") as f:
                    f.write(edited)
                print(f"✅ Saved: {out_path} ({len(edited)} bytes)")

        except Exception as e:
            print(f"❌ Error: {e}")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# 5. Disabling safe_mode
# ---------------------------------------------------------------------------


async def disable_safe_mode() -> bool:
    """Disable the default adult-content blur on a capable edit model.

    Returns ``True`` on success, ``False`` if the API call failed.
    """
    print("\n🔓 Disabling safe_mode")
    print("-" * 30)

    ok = True
    async with VeniceClient() as client:
        try:
            inpaint_model = await client.models.resolve_inpaint()
            base_bytes = await _generate_base_image(
                client,
                "A studio portrait of a person in casual clothing",
                width=512,
                height=512,
            )

            # Pass safe_mode=False to opt out of the server-side adult-content
            # blur. Only adult-capable edit models honour this.
            edited = await client.image.edit(
                prompt="Add a dramatic cinematic lighting effect",
                model=inpaint_model,
                image=base_bytes,
                safe_mode=False,
            )

            out_path = str(RESULTS_DIR / f"edit_safe_mode_off.{detect_image_format(edited)[0]}")
            with open(out_path, "wb") as f:
                f.write(edited)
            print(f"✅ Saved: {out_path} ({len(edited)} bytes)")

        except Exception as e:
            print(f"❌ Error: {e}")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# 6. Resolution tier + request timeout
# ---------------------------------------------------------------------------


async def edit_with_resolution_and_timeout() -> bool:
    """Demonstrate the ``resolution`` and ``timeout`` parameters on ``edit()``.

    - ``resolution`` (``"1K"`` / ``"2K"`` / ``"4K"``) requests an output tier on
      models with resolution-based pricing. Models that don't price by
      resolution ignore it, so we pass it and fall back gracefully.
    - ``timeout`` caps the request — high-resolution edits take longer, so a
      generous per-call timeout (a float in seconds, or an
      ``aiohttp.ClientTimeout``) avoids a premature client-side abort. The same
      ``timeout`` parameter exists on ``upscale()`` (see image_upscaling.py).

    ``multi_edit()`` accepts ``resolution`` too, with the same semantics.

    Returns ``True`` on success, ``False`` if the API call failed. The targeted
    ``except InvalidRequestError`` resolution fallback below is an intentional
    demo of handling a resolution rejection and still counts as success.
    """
    print("\n📐 Resolution tier + timeout")
    print("-" * 30)

    ok = True
    async with VeniceClient() as client:
        try:
            inpaint_model = await client.models.resolve_inpaint()
            print(f"📍 Using edit model: {inpaint_model}")

            base_bytes = await _generate_base_image(
                client,
                "A lighthouse on a rocky coast at golden hour",
                width=512,
                height=512,
            )

            prompt = "Add dramatic storm clouds gathering on the horizon"
            # Not every edit model prices by resolution, and the catalog doesn't
            # expose a per-model flag for it. So we try with resolution='2K' and,
            # if the model rejects it, transparently retry without — the edit
            # still succeeds either way. `timeout` is always honored.
            try:
                print("✏️  Editing at resolution='2K' with a 180s timeout …")
                edited = await client.image.edit(
                    prompt=prompt,
                    model=inpaint_model,
                    image=base_bytes,
                    resolution="2K",  # honored by resolution-priced models
                    timeout=180.0,  # seconds; generous cap for higher-res edits
                )
                used_resolution = True
            except InvalidRequestError as e:
                if "resolution" not in str(e).lower():
                    raise
                print(f"   ℹ️ {inpaint_model} doesn't price by resolution — retrying without it.")
                edited = await client.image.edit(
                    prompt=prompt,
                    model=inpaint_model,
                    image=base_bytes,
                    timeout=180.0,
                )
                used_resolution = False

            tag = "2k" if used_resolution else "default_res"
            out_path = str(RESULTS_DIR / f"edit_resolution_{tag}.{detect_image_format(edited)[0]}")
            with open(out_path, "wb") as f:
                f.write(edited)
            print(f"✅ Saved: {out_path} ({len(edited)} bytes)")

        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Note: `resolution` (1K/2K/4K) only affects resolution-priced models.")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    """Run all image editing examples.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    print("🚀 Venice AI Image Editing Examples")
    print("=" * 50)

    results: list[tuple[str, bool]] = [
        ("basic_image_editing", await basic_image_editing()),
        ("different_input_methods", await different_input_methods()),
        ("editing_with_prompt", await editing_with_prompt()),
        ("model_selection", await model_selection()),
        ("disable_safe_mode", await disable_safe_mode()),
        ("edit_with_resolution_and_timeout", await edit_with_resolution_and_timeout()),
    ]

    failed = [name for name, ok in results if not ok]

    print("\n" + "=" * 50)
    if failed:
        print(f"⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("✨ Image editing examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Generate → edit workflow (self-contained)")
    print("   - Multiple input methods (bytes, file path, file object, URL)")
    print("   - Prompt-based targeted editing")
    print("   - Dynamic inpaint model selection")
    print("   - safe_mode=False for adult-capable edit models")
    print("   - Saving raw bytes output to files")

    # Extensions are auto-detected from the returned format (e.g. .webp), so
    # the listing below uses an <ext> placeholder rather than a hardcoded .png.
    print("\n📁 Generated files in examples/results/:")
    print("   - edit_base_landscape.<ext>   (base image)")
    print("   - edit_with_rainbow.<ext>     (edited image)")
    print("   - edit_base_cottage.<ext>     (base for input methods)")
    print("   - edit_input_bytes.<ext>      (edited via bytes)")
    print("   - edit_input_filepath.<ext>   (edited via file path)")
    print("   - edit_input_fileobj.<ext>    (edited via file object)")
    print("   - edit_base_stilllife.<ext>   (base for edit demo)")
    print("   - edit_roses.<ext>            (prompt-based edit result)")
    print("   - edit_model_selection.<ext>  (model selection demo)")

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
