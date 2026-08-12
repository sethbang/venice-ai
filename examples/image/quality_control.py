#!/usr/bin/env python3
"""
Venice AI SDK - Image Quality Control
=====================================

Quality-aware image models (e.g. GPT Image 2) accept a native ``quality``
parameter — ``"low"`` / ``"medium"`` / ``"high"`` — that trades render time and
cost against fidelity. This example:

1. **Discovers** which image model is quality-aware by reading
   ``ImageModelConstraints.qualities`` / ``defaultQuality`` off the model spec —
   no hardcoded model ID.
2. **Generates** one image per supported tier with
   ``client.image.create(..., quality=...)`` and saves them to
   ``examples/results/``.

If no quality-aware model is in the catalog, the example explains that the
``quality`` parameter is still accepted (it's a no-op for non-quality models)
and exits cleanly — the capability is model-gated, not a code error.

See ``models/model_lifecycle.py`` for the read-only metadata view.
"""

import asyncio
import sys
from pathlib import Path

import aiohttp

from venice_ai import VeniceClient, detect_image_format
from venice_ai.exceptions import VeniceError
from venice_ai.types.api import ImageModelSpec

# Make the shared image helpers importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import generate_base_image  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = "A single ripe pomegranate on a marble countertop, soft window light"

# High-quality renders are genuinely slow and can exceed the SDK default
# per-request timeout (DEFAULT_TIMEOUT total=120s), raising APITimeoutError.
# ``Image.create`` has no per-call timeout kwarg, so we widen the timeout on
# the client itself for the quality-tier loop.
QUALITY_RENDER_TIMEOUT = aiohttp.ClientTimeout(total=300.0)


async def _find_quality_aware_model(client: VeniceClient) -> tuple[str, list[str]] | None:
    """Return (model_id, supported_quality_tiers) for the first quality-aware model."""
    models = await client.models.list(type="image")
    for entry in models.data:
        spec = entry.model_spec
        if isinstance(spec, ImageModelSpec) and spec.constraints:
            tiers = spec.constraints.qualities
            if tiers:
                return entry.id, list(tiers)
    return None


async def quality_tiers() -> bool:
    """Generate one image per supported quality tier on a quality-aware model.

    Returns ``True`` on success (or a clean model-gated skip), ``False`` if a
    render failed.
    """
    print("🎨 Image quality tiers")
    print("-" * 40)

    # Widen the per-request timeout for this loop: the ``high`` tier render is
    # genuinely slow and can exceed the 120s default, raising APITimeoutError.
    async with VeniceClient(timeout=QUALITY_RENDER_TIMEOUT) as client:
        try:
            found = await _find_quality_aware_model(client)
            if found is None:
                print("   ⏭️  No quality-aware image model in the catalog right now.")
                print("      The `quality` parameter is still accepted by create() —")
                print("      it simply has no effect on models without quality tiers.")
                return True

            model, tiers = found
            print(f"   Quality-aware model: {model}")
            print(f"   Supported tiers: {', '.join(tiers)}\n")

            for tier in tiers:
                print(f"   ⏳ Rendering quality='{tier}' …")
                response = await client.image.create(
                    model=model,
                    prompt=PROMPT,
                    quality=tier,  # native low/medium/high
                )
                saved = response.save_all(RESULTS_DIR, prefix=f"quality_{tier}", overwrite=True)
                print(f"      ✅ saved → {', '.join(str(p) for p in saved)}")
        except Exception as e:
            print(f"   ❌ Error rendering quality tiers: {e}")
            return False

    return True


async def edit_output_controls() -> bool:
    """Demonstrate output controls on the edit endpoints.

    ``client.image.edit()`` accepts ``output_format`` (``jpeg``/``png``/``webp``);
    ``client.image.multi_edit()`` additionally accepts ``quality`` and
    ``aspect_ratio``. The edit model is discovered dynamically via
    ``resolve_inpaint()`` — no hardcoded model ID. If no edit-capable model is
    entitled, the section explains and exits cleanly.

    Returns ``True`` on success (or a clean model-gated skip), ``False`` if an
    edit call failed.
    """
    print("\n✂️  Edit output controls (output_format / quality / aspect_ratio)")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            edit_model = await client.models.resolve_inpaint()
        except (VeniceError, ValueError) as e:
            print(f"   ⏭️  No edit-capable model available — skipping ({e}).")
            print("      output_format/quality/aspect_ratio are accepted by edit()/")
            print("      multi_edit() whenever an edit model is entitled.")
            return True

        print(f"   Edit model: {edit_model}")

        try:
            # A self-contained source image to edit.
            base_bytes = await generate_base_image(client, "A plain ceramic bowl on a wooden table")

            # 1. edit(): force a specific output_format (PNG here).
            print("   ⏳ edit() with output_format='png' …")
            edited = await client.image.edit(
                prompt="Fill the bowl with fresh strawberries",
                image=base_bytes,
                model=edit_model,
                output_format="png",
            )
            ext = detect_image_format(edited)[0]
            out = RESULTS_DIR / f"quality_edit_output.{ext}"
            out.write_bytes(edited)
            print(f"      ✅ saved → {out}")

            # 2. multi_edit(): output_format + quality + aspect_ratio together.
            print(
                "   ⏳ multi_edit() with output_format='png', quality='low', aspect_ratio='1:1' …"
            )
            multi = await client.image.multi_edit(
                prompt="Add soft morning light from the left",
                image=base_bytes,
                model=edit_model,
                output_format="png",
                quality="low",  # keep the demo cheap; low/medium/high on quality-aware models
                aspect_ratio="1:1",
            )
            m_ext = detect_image_format(multi)[0]
            m_out = RESULTS_DIR / f"quality_multi_edit_output.{m_ext}"
            m_out.write_bytes(multi)
            print(f"      ✅ saved → {m_out}")
        except Exception as e:
            print(f"   ❌ Error in edit output controls: {e}")
            return False

    return True


async def main() -> int:
    """Run the image quality-control example.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    print("🚀 Venice AI Image Quality Control")
    print("=" * 50)

    results: list[tuple[str, bool]] = [
        ("quality_tiers", await quality_tiers()),
        ("edit_output_controls", await edit_output_controls()),
    ]

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("\n✨ Image quality control example completed!")

    print("\n💡 Key concepts demonstrated:")
    print("   - Discovering defaultQuality / qualities from ImageModelConstraints")
    print("   - client.image.create(..., quality='low'|'medium'|'high')")
    print("   - Widening the client timeout for slow high-quality renders")
    print("   - client.image.edit(..., output_format=...)")
    print("   - client.image.multi_edit(..., output_format=..., quality=..., aspect_ratio=...)")
    print("   - Letting save_all() auto-detect the output extension")

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
