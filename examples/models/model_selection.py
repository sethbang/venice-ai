#!/usr/bin/env python3
"""
Venice AI SDK - Model Selection by Traits and Compatibility
===========================================================

This example demonstrates how to use Venice AI's intelligent model selection features:
- Selecting models by semantic traits (fastest, default)
- Cross-platform compatibility mappings for seamless migration
- Smart model discovery for different use cases
"""

import asyncio
import sys

from venice_ai import VeniceClient
from venice_ai.types.api import TextModelSpec


async def discover_models_by_traits() -> bool:
    """Discover models using semantic traits for easy selection."""
    print("🏷️ Model Discovery by Traits")
    print("-" * 40)

    # Only text and image types have traits
    model_types = ["text", "image"]

    ok = True
    async with VeniceClient() as client:
        for model_type in model_types:
            print(f"\n🎯 {model_type.upper()} Model Traits:")

            try:
                traits_response = await client.models.list_traits(type=model_type)

                if traits_response.data:
                    print(f"   📊 Found {len(traits_response.data)} traits for {model_type} models")

                    # Display all actual traits returned by the API
                    for trait_name, model_id in traits_response.data.items():
                        # Format trait name for display
                        formatted_trait = trait_name.replace("_", " ").title()
                        print(f"     🔹 {formatted_trait}")
                        print(f"       → Model: {model_id}")

                        # Get additional info about this model if possible
                        try:
                            models_list = await client.models.list(type=model_type)
                            model_info = next(
                                (m for m in models_list.data if m.id == model_id), None
                            )
                            if model_info and hasattr(model_info, "model_spec"):
                                spec = model_info.model_spec
                                if hasattr(spec, "name"):
                                    print(f"       → Name: {spec.name}")
                                # ``availableContextTokens`` only exists on
                                # ``TextModelSpec``; narrow before accessing.
                                if isinstance(spec, TextModelSpec) and spec.availableContextTokens:
                                    print(
                                        f"       → Context: {spec.availableContextTokens:,.0f} tokens"
                                    )
                        except Exception:
                            pass  # Skip if we can't get additional info

                        print()  # Empty line for readability
                else:
                    print("   ℹ️ No traits found for this model type")

            except Exception as e:
                print(f"   ❌ Error getting {model_type} traits: {e}")
                ok = False

    return ok


async def demonstrate_trait_based_selection() -> bool:
    """Show practical examples of using traits for model selection."""
    print("\n🎯 Practical Trait-Based Selection")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        # Example scenarios
        scenarios = [
            {
                "name": "💬 Chat Application",
                "type": "text",
                "recommended_traits": ["default", "most_uncensored", "fastest"],
                "description": "For a chat application, you want balanced performance",
            },
            {
                "name": "🎨 Image Generation",
                "type": "image",
                "recommended_traits": ["highest_quality", "default"],
                "description": "For creative work, quality is often preferred over speed",
            },
        ]

        for scenario in scenarios:
            print(f"\n{scenario['name']}")
            print(f"   📝 {scenario['description']}")

            try:
                traits_response = await client.models.list_traits(type=scenario["type"])

                if traits_response.data:
                    print("   🎯 Recommended selection order:")

                    for i, trait in enumerate(scenario["recommended_traits"], 1):
                        model_id = traits_response.data.get(trait)
                        if model_id:
                            print(f"      {i}. {trait.title()}: {model_id}")
                        else:
                            print(f"      {i}. {trait.title()}: ❌ Not available")

                    # Show what we'd actually use
                    selected_model = None
                    selected_trait = None
                    for trait in scenario["recommended_traits"]:
                        if trait in traits_response.data:
                            selected_model = traits_response.data[trait]
                            selected_trait = trait
                            break

                    if selected_model:
                        print(f"   ✅ Would select: {selected_model} ({selected_trait})")
                    else:
                        print("   ❌ No suitable model found")
                else:
                    print(f"   ❌ No traits available for {scenario['type']} models")

            except Exception as e:
                print(f"   ❌ Error: {e}")
                ok = False

    return ok


async def explore_compatibility_mappings() -> bool:
    """Explore cross-platform compatibility for easy migration."""
    print("\n🔄 Cross-Platform Compatibility")
    print("-" * 40)

    # Well-known external model names to check by type
    test_models_by_type = {
        "text": ["gpt-4.1", "o1-mini", "gpt-3.5-turbo", "claude-3-5-haiku-20241022"],
        "image": ["flux-dev-uncensored-11"],
        "embedding": ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"],
    }

    # Per-type compat lookups legitimately skip types with no mappings, so we
    # key success off whether *any* list_compatibility call returned, not off
    # the result being non-empty. Start False; flip True on the first call that
    # completes. If every call raises (e.g. bad key) this stays False.
    ok = False
    async with VeniceClient() as client:
        all_mappings = {}
        total_found = 0

        # Query compatibility mappings for each type
        for model_type in ["text", "image", "embedding", "tts"]:
            try:
                compatibility_response = await client.models.list_compatibility(type=model_type)
                ok = True
                if compatibility_response.data:
                    all_mappings.update(compatibility_response.data)
            except Exception:
                pass  # Skip if this type doesn't have compatibility mappings

        if all_mappings:
            print(f"📊 Found {len(all_mappings)} total compatibility mappings")

            # Check our test models by type
            for model_type, test_models in test_models_by_type.items():
                if test_models:
                    print(f"\n🔍 {model_type.upper()} models:")
                    type_found = 0

                    for external_model in test_models:
                        venice_equivalent = all_mappings.get(external_model)
                        if venice_equivalent:
                            print(f"   ✅ {external_model} → {venice_equivalent}")
                            type_found += 1
                            total_found += 1
                        else:
                            print(f"   ❌ {external_model} → No mapping found")

                    if type_found > 0:
                        print(
                            f"   📈 Found mappings for {type_found}/{len(test_models)} {model_type} models"
                        )

            print(f"\n📊 Total: Found mappings for {total_found} models across all types")

            # Show some additional mappings not in our test set
            all_test_models = [m for models in test_models_by_type.values() for m in models]
            other_mappings = {k: v for k, v in all_mappings.items() if k not in all_test_models}

            if other_mappings:
                print("\n🔍 Additional mappings (showing first 10):")
                for _i, (external, venice) in enumerate(list(other_mappings.items())[:10]):
                    print(f"   📄 {external} → {venice}")

                if len(other_mappings) > 10:
                    print(f"   ... and {len(other_mappings) - 10} more mappings")
        else:
            print("❌ No compatibility mappings found")

    return ok


async def demonstrate_migration_workflow() -> bool:
    """Show a complete migration workflow using compatibility mappings."""
    print("\n🚀 Migration Workflow Example")
    print("-" * 40)

    # Example migration scenario with diverse model types
    outside_models_by_type = {
        "text": ["gpt-4.1", "gpt-3.5-turbo", "claude-3-5-sonnet-20241022"],
        "embedding": ["text-embedding-ada-002", "text-embedding-3-small"],
        "image": ["flux-dev-uncensored-11", "dall-e-3"],
        "tts": ["tts-1"],
    }

    # Flatten for display
    all_outside_models = []
    for models in outside_models_by_type.values():
        all_outside_models.extend(models)

    print("📋 Migrating application from multiple platforms to Venice AI")
    print("   Legacy models in use:", ", ".join(all_outside_models))

    # As in explore_compatibility_mappings: per-type compat lookups legitimately
    # skip types with no mappings, so key success off any call completing rather
    # than off a non-empty result. Start False; flip True on the first call that
    # returns. The outer except below also forces False on an unexpected error.
    ok = False
    async with VeniceClient() as client:
        migration_plan = {}
        all_compatibility_mappings = {}

        try:
            # Get compatibility mappings for each type
            for model_type in ["text", "image", "embedding", "tts"]:
                try:
                    compatibility_response = await client.models.list_compatibility(type=model_type)
                    ok = True
                    if compatibility_response.data:
                        all_compatibility_mappings.update(compatibility_response.data)
                except Exception:
                    pass  # Skip if this type doesn't have compatibility mappings

            print("\n🔍 Migration Analysis:")

            for model_type, outside_models in outside_models_by_type.items():
                if outside_models:
                    print(f"\n📦 {model_type.upper()} Models:")

                    for out_model in outside_models:
                        venice_equivalent = all_compatibility_mappings.get(out_model)

                        if venice_equivalent:
                            print(f"   ✅ {out_model} → {venice_equivalent}")
                            migration_plan[out_model] = venice_equivalent

                            # Get additional info about the Venice model
                            try:
                                models_list = await client.models.list(type=model_type)
                                venice_model_info = next(
                                    (m for m in models_list.data if m.id == venice_equivalent), None
                                )

                                if venice_model_info and hasattr(venice_model_info, "model_spec"):
                                    spec = venice_model_info.model_spec
                                    print(f"     📋 Venice model: {spec.name or venice_equivalent}")
                                    # ``availableContextTokens`` and ``capabilities``
                                    # are on ``TextModelSpec`` only.
                                    if (
                                        isinstance(spec, TextModelSpec)
                                        and spec.availableContextTokens
                                    ):
                                        print(
                                            f"     📏 Context length: {spec.availableContextTokens:,.0f} tokens"
                                        )

                                    # Show capabilities for text models
                                    if (
                                        model_type == "text"
                                        and isinstance(spec, TextModelSpec)
                                        and spec.capabilities
                                    ):
                                        caps = spec.capabilities
                                        features = []
                                        if caps.supportsFunctionCalling:
                                            features.append("Function Calling")
                                        if caps.supportsVision:
                                            features.append("Vision")
                                        if caps.supportsWebSearch:
                                            features.append("Web Search")
                                        if features:
                                            print(f"     🔧 Features: {', '.join(features)}")

                            except Exception:
                                pass  # Skip detailed info if error

                        else:
                            print(f"   ❌ {out_model} → No direct equivalent found")

                            # Suggest alternatives based on traits for text and image types
                            if model_type in ["text", "image"]:
                                try:
                                    traits_response = await client.models.list_traits(
                                        type=model_type
                                    )
                                    if traits_response.data:
                                        recommended = traits_response.data.get("default")
                                        if recommended:
                                            print(
                                                f"     💡 Suggested alternative: {recommended} (default {model_type} model)"
                                            )
                                            migration_plan[out_model] = recommended
                                except Exception:
                                    pass
                            # For other types, suggest the first available model
                            elif model_type in ["embedding", "tts"]:
                                try:
                                    models_list = await client.models.list(type=model_type)
                                    if models_list.data:
                                        recommended = models_list.data[0].id
                                        print(
                                            f"     💡 Suggested alternative: {recommended} (available {model_type} model)"
                                        )
                                        migration_plan[out_model] = recommended
                                except Exception:
                                    pass

            # Summary
            print("\n📊 Migration Summary:")
            direct_mappings = len([k for k in migration_plan if all_compatibility_mappings.get(k)])
            alternatives = len(migration_plan) - direct_mappings
            no_solution = len(all_outside_models) - len(migration_plan)

            print(f"   ✅ Direct mappings: {direct_mappings}")
            print(f"   💡 Alternative suggestions: {alternatives}")
            print(f"   ❌ No solution found: {no_solution}")

            if migration_plan:
                print("\n📝 Final Migration Plan:")
                for legacy, venice in migration_plan.items():
                    mapping_type = (
                        "Direct" if all_compatibility_mappings.get(legacy) else "Alternative"
                    )
                    print(f"   {legacy} → {venice} ({mapping_type})")

        except Exception as e:
            print(f"❌ Error creating migration plan: {e}")
            ok = False

    return ok


async def main() -> int:
    """Run all model selection and compatibility examples.

    Returns ``0`` only if every sub-section succeeded, ``1`` otherwise, so a
    real API failure surfaces as a non-zero process exit.
    """
    print("🚀 Venice AI Model Selection & Compatibility Examples")
    print("=" * 70)

    results: list[tuple[str, bool]] = [
        ("discover_models_by_traits", await discover_models_by_traits()),
        ("demonstrate_trait_based_selection", await demonstrate_trait_based_selection()),
        ("explore_compatibility_mappings", await explore_compatibility_mappings()),
        ("demonstrate_migration_workflow", await demonstrate_migration_workflow()),
    ]

    failed = [name for name, ok in results if not ok]

    print("\n✨ Model selection examples completed!")
    if failed:
        print(f"\n❌ {len(failed)} section(s) failed: {', '.join(failed)}")
    print("\n💡 Key concepts demonstrated:")
    print("   - Semantic trait-based model discovery (default, fastest)")
    print("   - Practical model selection for different use cases")
    print("   - Cross-platform compatibility mappings")
    print("   - Complete migration workflow from other AI platforms")
    print("   - Model capabilities and specifications analysis")
    print("   - Fallback strategies when direct mappings aren't available")

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("Check that your API key is valid and you have appropriate access.", file=sys.stderr)
        sys.exit(1)
