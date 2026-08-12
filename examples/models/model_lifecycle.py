#!/usr/bin/env python3
"""
Venice AI SDK - Model Lifecycle & Capability Metadata
=====================================================

Newer fields on the model catalog let you inspect a model's *lifecycle* and
*capability envelope* without a trial-and-error API call. This example reads
them straight off ``client.models.list(...)`` and the per-model lookups
``client.models.get(...)`` / ``client.models.get_capabilities(...)``:

- ``ModelResponse.context_length`` — max context window in tokens, surfaced as
  a typed top-level field (mirrors ``model_spec.availableContextTokens``).
- ``model_spec.deprecation`` (:class:`ModelDeprecation`) — ``startsAt`` /
  ``removesAt`` lifecycle instants, ``replacementModelId`` (where to migrate),
  and ``autoRemap`` (whether Venice silently re-routes the retired ID).
- Text capabilities — ``reasoningEffortOptions`` / ``defaultReasoningEffort``
  tell you which ``reasoning_effort`` values a model accepts *before* you send
  one, instead of guessing.
- Image constraints — ``defaultQuality`` / ``qualities`` tell you which
  ``quality`` tiers a quality-aware image model supports.
- ``client.models.get(model_id)`` — fetch one model's full
  :class:`ModelResponse` by id (the SDK abstracts the list-and-filter pattern
  Venice's catalog otherwise requires; there is no per-model GET endpoint).
- ``client.models.get_capabilities(model_id)`` — a typed, snake_case
  :class:`Capabilities` view (polymorphic by model type, e.g.
  :class:`ChatCapabilities`) so you can introspect feature flags directly
  instead of probing ``resolve_chat(require_function_calling=True)`` etc.

All of this is a read-only ``GET /models`` — running this example costs nothing.

See also ``models/model_selection.py`` for resolver-based selection, and
``image/quality_control.py`` for *using* the discovered ``quality`` tiers.
"""

import asyncio
import sys

from venice_ai import VeniceClient
from venice_ai.types.api import ImageModelSpec, TextModelSpec


async def text_model_capabilities() -> bool:
    """Surface context length and reasoning-effort options for text models.

    Returns ``True`` on success, ``False`` if the catalog read failed.
    """
    print("🧠 Text model capabilities")
    print("-" * 40)

    try:
        async with VeniceClient() as client:
            models = await client.models.list(type="text")

            for entry in models.data:
                spec = entry.model_spec
                # context_length is a typed top-level field (None for non-text).
                ctx = entry.context_length
                ctx_str = f"{ctx:,} tokens" if ctx else "—"
                print(f"\n📦 {entry.id}")
                print(f"   context_length: {ctx_str}")

                if isinstance(spec, TextModelSpec) and spec.capabilities:
                    caps = spec.capabilities
                    if caps.supportsReasoningEffort:
                        opts = caps.reasoningEffortOptions or []
                        default = caps.defaultReasoningEffort or "—"
                        print(f"   reasoning_effort options: {', '.join(opts) or '—'}")
                        print(f"   default reasoning_effort: {default}")
                    else:
                        print("   reasoning_effort: not supported")
    except Exception as e:
        print(f"❌ Error reading text model capabilities: {e}")
        return False

    return True


async def image_quality_tiers() -> bool:
    """Surface the quality tiers each quality-aware image model supports.

    Returns ``True`` on success, ``False`` if the catalog read failed.
    """
    print("\n\n🎨 Image model quality tiers")
    print("-" * 40)

    try:
        async with VeniceClient() as client:
            models = await client.models.list(type="image")

            any_quality_aware = False
            for entry in models.data:
                spec = entry.model_spec
                if not isinstance(spec, ImageModelSpec) or spec.constraints is None:
                    continue
                constraints = spec.constraints
                # defaultQuality / qualities are present only on quality-aware models.
                if constraints.qualities or constraints.defaultQuality:
                    any_quality_aware = True
                    tiers = constraints.qualities or []
                    print(f"\n📦 {entry.id}")
                    print(f"   qualities: {', '.join(tiers) or '—'}")
                    print(f"   defaultQuality: {constraints.defaultQuality or '—'}")

            if not any_quality_aware:
                print("\n   ℹ️ No quality-aware image models currently in the catalog.")
                print("      (The fields are still typed — they just read as None here.)")
    except Exception as e:
        print(f"❌ Error reading image quality tiers: {e}")
        return False

    return True


async def deprecation_report() -> bool:
    """Flag any model carrying deprecation metadata and where to migrate.

    Returns ``True`` on success, ``False`` if the catalog read failed. A model
    *type* being unavailable on this tier is tolerated (skip + continue), not a
    failure — that's expected fan-out across the whole catalog.
    """
    print("\n\n⏳ Deprecation lifecycle report")
    print("-" * 40)

    try:
        async with VeniceClient() as client:
            flagged = 0
            # Walk every model type so we catch deprecations anywhere in the catalog.
            for model_type in ("text", "image", "embedding", "tts", "asr", "video", "music"):
                try:
                    models = await client.models.list(type=model_type)
                except Exception:
                    continue  # a type may be unavailable on some tiers
                for entry in models.data:
                    dep = entry.model_spec.deprecation
                    if dep is None:
                        continue
                    flagged += 1
                    print(f"\n⚠️  {entry.id} ({model_type})")
                    if dep.startsAt:
                        print(f"   warnings active from: {dep.startsAt}")
                    if dep.removesAt:
                        print(f"   removed from catalog:  {dep.removesAt}")
                    if dep.date:
                        print(f"   legacy sunset date:    {dep.date}")
                    if dep.replacementModelId:
                        print(f"   ➡️  migrate to: {dep.replacementModelId}")
                    print(f"   auto-remap retired ID: {dep.autoRemap}")

            if flagged == 0:
                print("\n   ✅ No models currently carry deprecation metadata.")
                print("      Venice publishes it here (and via response headers) when a")
                print("      model is scheduled for retirement — check before pinning.")
    except Exception as e:
        print(f"❌ Error building deprecation report: {e}")
        return False

    return True


async def model_detail_and_capabilities() -> bool:
    """Look up one model's detail + typed capabilities by id.

    Demonstrates the per-model lookups ``client.models.get(model_id)`` and
    ``client.models.get_capabilities(model_id)``. The id comes from a resolver
    (never hardcoded) so this keeps working as the catalog evolves.

    Returns ``True`` on success, ``False`` if either lookup failed.
    """
    print("\n\n🔎 Per-model detail & typed capabilities")
    print("-" * 40)

    try:
        async with VeniceClient() as client:
            # Resolve a real chat model id dynamically — no hardcoded IDs.
            model_id = await client.models.resolve_chat()
            print(f"🤖 Resolved chat model: {model_id}")

            # ── client.models.get(model_id) ────────────────────────────────
            # Returns the full ModelResponse for a single id (the SDK abstracts
            # the list-and-filter pattern; Venice has no per-model GET endpoint).
            detail = await client.models.get(model_id)
            print("\n📦 models.get() detail")
            print(f"   id:             {detail.id}")
            print(f"   type:           {detail.type}")
            print(f"   owned_by:       {detail.owned_by}")
            ctx = detail.context_length
            print(f"   context_length: {f'{ctx:,} tokens' if ctx else '—'}")
            print(f"   name:           {detail.model_spec.name or '—'}")
            print(f"   beta:           {detail.model_spec.beta}")
            if detail.model_spec.deprecation is not None:
                dep = detail.model_spec.deprecation
                print(f"   ⚠️  deprecated — replacement: {dep.replacementModelId or '—'}")

            # ── client.models.get_capabilities(model_id) ───────────────────
            # Polymorphic, snake_case Capabilities view. For a chat model this
            # is a ChatCapabilities; print every flag generically via
            # model_dump() so the demo survives the discriminated union cleanly.
            caps = await client.models.get_capabilities(model_id)
            print("\n🧩 models.get_capabilities() flags")
            print(f"   (capabilities type: {caps.type})")
            for field, value in caps.model_dump().items():
                if field == "type":
                    continue
                print(f"   {field}: {value}")
    except Exception as e:
        print(f"❌ Error looking up model detail/capabilities: {e}")
        return False

    return True


async def main() -> int:
    """Run the model-lifecycle metadata report (read-only, no cost).

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    print("🚀 Venice AI Model Lifecycle & Capability Metadata")
    print("=" * 60)

    results: list[tuple[str, bool]] = [
        ("text_model_capabilities", await text_model_capabilities()),
        ("image_quality_tiers", await image_quality_tiers()),
        ("deprecation_report", await deprecation_report()),
        ("model_detail_and_capabilities", await model_detail_and_capabilities()),
    ]

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("\n\n✨ Done.")

    print("\n💡 Key concepts demonstrated:")
    print("   - context_length as a typed top-level field")
    print("   - reasoningEffortOptions / defaultReasoningEffort discovery")
    print("   - defaultQuality / qualities for quality-aware image models")
    print("   - ModelDeprecation: startsAt / removesAt / replacementModelId")
    print("   - models.get(model_id) for a single model's full detail")
    print("   - models.get_capabilities(model_id) for typed feature flags")

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("Check that VENICE_API_KEY is set and valid.", file=sys.stderr)
        sys.exit(1)
