#!/usr/bin/env python3
"""
Venice AI SDK - Model Discovery and Listing
============================================

This example demonstrates how to discover and explore available models in the Venice AI SDK.
Learn how to browse models by type, understand their capabilities, and access detailed metadata.
"""

import asyncio
import sys

from venice_ai import VeniceClient
from venice_ai.types.api import TextModelSpec


async def list_all_models() -> bool:
    """List all available models with basic information."""
    print("📋 All Available Models")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            # `models.list(type=None)` returns the full catalog across every model
            # type in a single call — no per-type loop needed.
            all_models = await client.models.list(type=None)
            total = len(all_models.data)

            # Group by capability type for display. ``capabilities`` is declared only
            # on ``TextModelSpec``, so an isinstance check is the cleanest type
            # discriminator.
            models_by_type: dict[str, list] = {}
            for m in all_models.data:
                spec = m.model_spec
                type_key = (
                    "text"
                    if isinstance(spec, TextModelSpec) and spec.capabilities is not None
                    else "other"
                )
                models_by_type.setdefault(type_key, []).append(m)

            print(f"📊 {total} models total across {len(models_by_type)} types")

            # Display models grouped by type
            for model_type, models in models_by_type.items():
                print(f"\n🏷️ {model_type.upper()} Models ({len(models)}):")
                for model in models[:5]:  # Show first 5 per type
                    model_name = model.id
                    model_id = model.id
                    owner = model.owned_by or "venice"
                    print(f"   📄 {model_name} (ID: {model_id}) - by {owner}")

                if len(models) > 5:
                    print(f"   ... and {len(models) - 5} more {model_type} models")

            return True

        except Exception as e:
            print(f"❌ Error listing models: {e}")
            print("💡 Note: Model listing requires appropriate API access")
            return False


async def list_models_by_type() -> bool:
    """List models filtered by specific types."""
    print("\n🎯 Models by Type")
    print("-" * 40)

    # Different model types to explore
    model_types = ["text", "image", "embedding", "tts"]

    ok = True
    async with VeniceClient() as client:
        for model_type in model_types:
            print(f"\n🔍 {model_type.upper()} Models:")

            try:
                models_response = await client.models.list(type=model_type)

                if models_response.data:
                    print(f"   📊 Found {len(models_response.data)} {model_type} models")

                    for model in models_response.data[:3]:  # Show first 3
                        model_name = model.id

                        # Show model capabilities if available — only text-typed
                        # specs carry ``capabilities``.
                        spec = model.model_spec
                        capabilities = (
                            spec.capabilities if isinstance(spec, TextModelSpec) else None
                        )
                        if capabilities is not None:
                            caps = []
                            if capabilities.supportsFunctionCalling:
                                caps.append("🔧 Function Calling")
                            if capabilities.supportsVision:
                                caps.append("👁️ Vision")
                            if capabilities.supportsWebSearch:
                                caps.append("🌐 Web Search")
                            if capabilities.optimizedForCode:
                                caps.append("💻 Code Optimized")
                            if capabilities.supportsReasoning:
                                caps.append("🧠 Reasoning")

                            caps_str = ", ".join(caps) if caps else "Basic"
                            print(f"     📄 {model_name} - {caps_str}")
                        else:
                            print(f"     📄 {model_name}")

                    if len(models_response.data) > 3:
                        print(f"     ... and {len(models_response.data) - 3} more")
                else:
                    print("   ℹ️ No models found for this type")

            except Exception as e:
                print(f"   ❌ Error getting {model_type} models: {e}")
                ok = False

    return ok


async def explore_model_details() -> bool:
    """Explore detailed model information and capabilities."""
    print("\n🔬 Model Details Exploration")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            # Get text models for detailed analysis
            text_models = await client.models.list(type="text")

            if text_models.data:
                # Analyze first few text models
                print(f"📊 Analyzing {min(3, len(text_models.data))} text models:")

                for i, model in enumerate(text_models.data[:3]):
                    model_name = model.id
                    model_id = model.id

                    print(f"\n🤖 Model {i + 1}: {model_name}")
                    print(f"   🏷️ ID: {model_id}")
                    print(f"   🏢 Owner: {model.owned_by or 'venice'}")
                    print(f"   📅 Created: {model.created or 'unknown'}")

                    # Show model specifications — text models dispatch to
                    # ``TextModelSpec`` which carries ``capabilities``.
                    spec = model.model_spec
                    print("   📋 Specifications:")

                    # Show capabilities (Optional — older models may not have them)
                    caps = spec.capabilities if isinstance(spec, TextModelSpec) else None
                    if caps is not None:
                        print(f"      🔧 Function Calling: {caps.supportsFunctionCalling}")
                        print(f"      👁️ Vision Support: {caps.supportsVision}")
                        print(f"      🌐 Web Search: {caps.supportsWebSearch}")
                        print(f"      💻 Code Optimized: {caps.optimizedForCode}")
                        print(f"      🧠 Reasoning: {caps.supportsReasoning}")

                    # traits is `list[str]` (defaults to empty); truthy iff non-empty
                    if spec.traits:
                        print(f"      🏷️ Traits: {', '.join(spec.traits)}")

                    # Show pricing if available
                    if model.model_spec and model.model_spec.pricing:
                        pricing = model.model_spec.pricing
                        print("   💰 Pricing:")
                        try:
                            from venice_ai.types.api.models import (
                                ImageModelPricing,
                                LLMModelPricing,
                            )

                            if isinstance(pricing, LLMModelPricing):
                                # LLM pricing - these values are per million tokens
                                input_price = pricing.input.usd
                                output_price = pricing.output.usd
                                print(f"      📥 Input: ${input_price} USD per million tokens")
                                print(f"      📤 Output: ${output_price} USD per million tokens")
                                if isinstance(input_price, (int, float)):
                                    print(f"         (${input_price / 1_000_000:.8f} per token)")
                                if isinstance(output_price, (int, float)):
                                    print(f"         (${output_price / 1_000_000:.8f} per token)")
                            elif isinstance(pricing, ImageModelPricing):
                                # Image pricing
                                gen_price = pricing.generation.usd
                                print(f"      🎨 Generation: ${gen_price} USD per image")
                                print("      🔍 Upscale available")
                            elif hasattr(pricing, "input"):
                                # Audio/other pricing
                                input_price = pricing.input.usd  # type: ignore[union-attr]
                                print(f"      🎤 Input: ${input_price} USD per unit")
                            else:
                                print("      💵 Pricing available (format not recognized)")
                        except Exception as e:
                            print(f"      ❌ Error displaying pricing: {e}")

            return True

        except Exception as e:
            print(f"❌ Error exploring model details: {e}")
            return False


async def compare_model_resolve_results() -> bool:
    """Compare model resolve results with direct API calls."""
    print("\n⚖️ Model Resolve vs Direct API Comparison")
    print("-" * 40)

    # Each resolve/list call below has its own inner try, so a single outer
    # try would not catch a failure when (say) every resolve fails. Track
    # success explicitly so any failed call yields a non-zero process exit.
    ok = True
    async with VeniceClient() as client:
        try:
            print("🤖 Model Resolve Recommendations:")

            # Get chat model
            try:
                chat_model = await client.models.resolve_chat()
                print(f"   💬 Chat Model: {chat_model}")
            except Exception as e:
                print(f"   ❌ Chat Model Error: {e}")
                ok = False

            # Get embedding model
            try:
                embedding_model = await client.models.resolve_embedding()
                print(f"   🔢 Embedding Model: {embedding_model}")
            except Exception as e:
                print(f"   ❌ Embedding Model Error: {e}")
                ok = False

            # Get image model
            try:
                image_model = await client.models.resolve_image()
                print(f"   🎨 Image Model: {image_model}")
            except Exception as e:
                print(f"   ❌ Image Model Error: {e}")
                ok = False

            # Compare with direct API results
            print("\n📋 Direct API Results:")

            for model_type in ["text", "embedding", "image"]:
                try:
                    models = await client.models.list(type=model_type)
                    if models.data:
                        first_model = models.data[0]
                        print(f"   🏷️ First {model_type} model: {first_model.id}")
                    else:
                        print(f"   ℹ️ No {model_type} models found")
                except Exception as e:
                    print(f"   ❌ Error getting {model_type} models: {e}")
                    ok = False

        except Exception as e:
            print(f"❌ Error in model comparison: {e}")
            ok = False

    return ok


async def main() -> int:
    """Run all model listing examples.

    Returns ``0`` only if every sub-section succeeded, ``1`` otherwise, so a
    real API failure surfaces as a non-zero process exit.
    """
    print("🚀 Venice AI Model Discovery Examples")
    print("=" * 60)

    results: list[tuple[str, bool]] = [
        ("list_all_models", await list_all_models()),
        ("list_models_by_type", await list_models_by_type()),
        ("explore_model_details", await explore_model_details()),
        ("compare_model_resolve_results", await compare_model_resolve_results()),
    ]

    failed = [name for name, ok in results if not ok]

    print("\n✨ Model discovery examples completed!")
    if failed:
        print(f"\n❌ {len(failed)} section(s) failed: {', '.join(failed)}")
    print("\n💡 Key concepts demonstrated:")
    print("   - Listing all available models")
    print("   - Filtering models by type")
    print("   - Exploring model capabilities and specifications")
    print("   - Understanding model pricing information")
    print("   - Comparing model resolve vs direct API")
    print("   - Model metadata and owner information")

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
