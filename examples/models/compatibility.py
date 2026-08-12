#!/usr/bin/env python3
"""
Venice AI SDK - API Compatibility and Migration
================================================

This example demonstrates how to migrate from other AI APIs to Venice AI.
Learn how to use compatibility mappings for seamless transitions from OpenAI,
Anthropic, and other platforms.
"""

import asyncio
import sys

from venice_ai import VeniceClient
from venice_ai.types import ModelCompatibilityResponse


async def openai_to_venice_chat(compat: ModelCompatibilityResponse) -> bool:
    """Demonstrate migrating OpenAI chat completion code to Venice AI.

    Receives the shared ``type="text"`` compatibility mapping so the whole
    guide fetches it from the network only once.
    """
    print("🔄 OpenAI → Venice AI: Chat Completions")
    print("-" * 50)

    # 'gpt-4' has no direct Venice mapping (see the results below), so the
    # AFTER snippet uses a model that *does* map. Keeping the snippet honest
    # avoids implying 'gpt-4' resolves on its own.
    print("📝 Original OpenAI Code Pattern:")
    print("```python")
    print("from openai import OpenAI")
    print("client = OpenAI(api_key='...')")
    print("response = client.chat.completions.create(")
    print("    model='gpt-4o',")
    print("    messages=[{'role': 'user', 'content': 'Hello!'}]")
    print(")")
    print("```")

    print("\n✨ Venice AI Equivalent:")
    print("```python")
    print("from venice_ai import VeniceClient")
    print("client = VeniceClient()")
    print("response = await client.chat.completions.create(")
    print("    model='gpt-4o',  # Has a direct mapping (see results below)")
    print("    messages=[UserMessage(content='Hello!')]")
    print(")")
    print("```")

    try:
        # Common OpenAI model names
        openai_models = ["gpt-4", "gpt-4.1", "gpt-3.5-turbo", "gpt-4o"]

        print("\n🔍 Model Mapping Results:")
        for openai_model in openai_models:
            venice_model = compat.data.get(openai_model)
            if venice_model:
                print(f"   ✅ {openai_model:20s} → {venice_model}")
            else:
                print(f"   ❌ {openai_model:20s} → No mapping (use a trait, e.g. resolve_chat())")

        print("\n💡 Key Differences:")
        print("   • Venice AI is async-first (use await)")
        print("   • Use Pydantic models for messages (UserMessage, etc.)")
        print("   • Model names are mapped when a direct equivalent exists;")
        print("     otherwise resolve by trait (e.g. client.models.resolve_chat())")
        print("   • Same response structure for easy migration")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def anthropic_to_venice_chat(compat: ModelCompatibilityResponse) -> bool:
    """Demonstrate migrating Anthropic code to Venice AI.

    Reuses the shared ``type="text"`` compatibility mapping passed from
    :func:`main` rather than re-fetching it.
    """
    print("\n🔄 Anthropic → Venice AI: Chat Completions")
    print("-" * 50)

    print("📝 Original Anthropic Code Pattern:")
    print("```python")
    print("from anthropic import Anthropic")
    print("client = Anthropic(api_key='...')")
    print("message = client.messages.create(")
    print("    model='claude-3-5-sonnet-20241022',")
    print("    messages=[{'role': 'user', 'content': 'Hello!'}]")
    print(")")
    print("```")

    print("\n✨ Venice AI Equivalent:")
    print("```python")
    print("from venice_ai import VeniceClient")
    print("client = VeniceClient()")
    print("response = await client.chat.completions.create(")
    print("    model='claude-3-5-sonnet-20241022',  # Auto-mapped")
    print("    messages=[UserMessage(content='Hello!')]")
    print(")")
    print("```")

    try:
        # Common Anthropic model names
        anthropic_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ]

        print("\n🔍 Model Mapping Results:")
        for anthropic_model in anthropic_models:
            venice_model = compat.data.get(anthropic_model)
            if venice_model:
                print(f"   ✅ {anthropic_model:35s} → {venice_model}")
            else:
                print(f"   ❌ {anthropic_model:35s} → No mapping (resolve by trait)")

        print("\n💡 Key Differences:")
        print("   • Venice AI uses .chat.completions.create() instead of .messages.create()")
        print("   • Async-first design (use await)")
        print("   • Unified interface across all providers")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def embedding_migration() -> bool:
    """Demonstrate migrating embedding code to Venice AI."""
    print("\n🔄 Embeddings Migration")
    print("-" * 50)

    print("📝 OpenAI Embeddings Pattern:")
    print("```python")
    print("from openai import OpenAI")
    print("client = OpenAI(api_key='...')")
    print("response = client.embeddings.create(")
    print("    model='text-embedding-ada-002',")
    print("    input=['text to embed']")
    print(")")
    print("```")

    print("\n✨ Venice AI Equivalent:")
    print("```python")
    print("from venice_ai import VeniceClient")
    print("client = VeniceClient()")
    print("response = await client.embeddings.create(")
    print("    model='text-embedding-ada-002',  # Auto-mapped")
    print("    input=['text to embed']")
    print(")")
    print("```")

    async with VeniceClient() as client:
        try:
            # Get compatibility mapping for embeddings
            compat = await client.models.list_compatibility(type="embedding")

            # Common embedding models
            embedding_models = [
                "text-embedding-ada-002",
                "text-embedding-3-small",
                "text-embedding-3-large",
            ]

            print("\n🔍 Model Mapping Results:")
            for embed_model in embedding_models:
                venice_model = compat.data.get(embed_model)
                if venice_model:
                    print(f"   ✅ {embed_model:30s} → {venice_model}")
                else:
                    print(f"   ❌ {embed_model:30s} → No mapping")

            print("\n💡 Migration Notes:")
            print("   • Same API structure as OpenAI")
            print("   • Compatible response format")
            print("   • Async-first (use await)")

            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False


async def parameter_mapping_guide() -> bool:
    """Show how common parameters map between APIs."""
    print("\n📊 Parameter Mapping Guide")
    print("-" * 50)

    print("\n🔹 Chat Completion Parameters:")
    print("\nOpenAI/Anthropic → Venice AI")
    print("─" * 40)

    param_mappings = [
        ("model", "model", "Model identifier (auto-mapped)"),
        ("messages", "messages", "Message array (use Pydantic models)"),
        ("temperature", "temperature", "Same (0.0 - 2.0)"),
        ("max_tokens", "max_completion_tokens", "Maximum tokens to generate"),
        ("top_p", "top_p", "Same (0.0 - 1.0)"),
        ("frequency_penalty", "frequency_penalty", "Same (0.0 - 2.0)"),
        ("presence_penalty", "presence_penalty", "Same (0.0 - 2.0)"),
        ("stream", "stream", "Same (boolean)"),
        ("n", "n", "Number of completions"),
        ("seed", "seed", "Deterministic sampling"),
    ]

    for old_param, new_param, description in param_mappings:
        indicator = "→" if old_param == new_param else "⚠️"
        print(f"   {indicator} {old_param:20s} → {new_param:25s} | {description}")

    print("\n💡 Notable Differences:")
    print("   • max_tokens → max_completion_tokens (more explicit)")
    print("   • Venice AI uses Pydantic models for type safety")
    print("   • All operations are async (use await)")
    return True


async def response_structure_comparison() -> bool:
    """Compare response structures between APIs."""
    print("\n📦 Response Structure Comparison")
    print("-" * 50)

    print("\n✅ Venice AI maintains OpenAI-compatible response structure:")
    print("\nBoth APIs return:")
    print("```python")
    print("response = {")
    print("    'id': '...',")
    print("    'model': '...',")
    print("    'created': 1234567890,")
    print("    'choices': [")
    print("        {")
    print("            'index': 0,")
    print("            'message': {")
    print("                'role': 'assistant',")
    print("                'content': '...'")
    print("            },")
    print("            'finish_reason': 'stop'")
    print("        }")
    print("    ],")
    print("    'usage': {")
    print("        'prompt_tokens': 10,")
    print("        'completion_tokens': 20,")
    print("        'total_tokens': 30")
    print("    }")
    print("}")
    print("```")

    print("\n💡 This means:")
    print("   • Minimal code changes needed for migration")
    print("   • Existing response parsing code works as-is")
    print("   • Type-safe Pydantic models available")
    return True


async def migration_checklist() -> bool:
    """Provide a migration checklist for developers."""
    print("\n✅ Migration Checklist")
    print("-" * 50)

    checklist_items = [
        ("Install Venice AI SDK", "pip install venice-ai"),
        ("Update imports", "from venice_ai import VeniceClient"),
        ("Set up API key", "export VENICE_API_KEY='your-key'"),
        ("Convert to async", "Use async/await for all API calls"),
        ("Update message format", "Use Pydantic models (UserMessage, etc.)"),
        ("Check model names", "Use compatibility mappings or trait-based selection"),
        ("Update parameters", "max_tokens → max_completion_tokens"),
        ("Test responses", "Verify response structure matches expectations"),
        ("Update error handling", "Handle Venice-specific exceptions"),
        ("Performance testing", "Verify latency and throughput"),
    ]

    print("\n📋 Step-by-Step Migration:")
    for i, (step, details) in enumerate(checklist_items, 1):
        print(f"\n{i}. {step}")
        print(f"   → {details}")

    print("\n🔍 Common Issues & Solutions:")
    print("\n   ❓ Model not found")
    print("   → Use list_compatibility() to find Venice equivalent")
    print("   → Or use trait-based selection (default, fastest)")

    print("\n   ❓ Different response format")
    print("   → Venice AI maintains OpenAI compatibility")
    print("   → Use Pydantic models for type safety")

    print("\n   ❓ Sync vs Async")
    print("   → Venice AI is async-first")
    print("   → Wrap calls in async functions and use await")
    return True


async def practical_migration_example(compat: ModelCompatibilityResponse) -> bool:
    """Show a complete before/after migration example.

    Reuses the shared ``type="text"`` compatibility mapping rather than
    re-fetching it from the network.
    """
    print("\n🚀 Complete Migration Example")
    print("-" * 50)

    print("\n❌ BEFORE (OpenAI):")
    print("```python")
    print("from openai import OpenAI")
    print("")
    print("def get_response(prompt: str) -> str:")
    print("    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))")
    print("    ")
    print("    response = client.chat.completions.create(")
    print("        model='gpt-4',")
    print("        messages=[")
    print("            {'role': 'system', 'content': 'You are helpful.'},")
    print("            {'role': 'user', 'content': prompt}")
    print("        ],")
    print("        max_tokens=100")
    print("    )")
    print("    ")
    print("    return response.choices[0].message.content")
    print("")
    print("result = get_response('Hello!')")
    print("```")

    # 'gpt-4' has no direct Venice mapping, so the AFTER snippet resolves the
    # chat model by trait instead of pretending 'gpt-4' maps to something.
    print("\n✅ AFTER (Venice AI):")
    print("```python")
    print("from venice_ai import VeniceClient")
    print("from venice_ai.types.api import SystemMessage, UserMessage")
    print("")
    print("async def get_response(prompt: str) -> str:")
    print("    async with VeniceClient() as client:")
    print("        # 'gpt-4' has no direct mapping — resolve by trait instead")
    print("        model = await client.models.resolve_chat()")
    print("        response = await client.chat.completions.create(")
    print("            model=model,")
    print("            messages=[")
    print("                SystemMessage(content='You are helpful.'),")
    print("                UserMessage(content=prompt)")
    print("            ],")
    print("            max_completion_tokens=100")
    print("        )")
    print("        ")
    print("        return response.choices[0].message.content or ''")
    print("")
    print("result = await get_response('Hello!')")
    print("```")

    print("\n🔑 Key Changes:")
    print("   1. Import VeniceClient instead of OpenAI")
    print("   2. Make function async and use await")
    print("   3. Use Pydantic message models (UserMessage, SystemMessage)")
    print("   4. max_tokens → max_completion_tokens")
    print("   5. Handle potential None values (type safety)")

    try:
        # Demonstrate the honest mapping outcome for 'gpt-4'.
        gpt4_equivalent = compat.data.get("gpt-4")

        print("\n✨ Model Mapping:")
        if gpt4_equivalent:
            print(f"   OpenAI 'gpt-4' → Venice '{gpt4_equivalent}'")
            print("   Migration ready!")
        else:
            print("   OpenAI 'gpt-4' → no direct mapping")
            print("   Resolve by trait instead, e.g. await client.models.resolve_chat()")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


async def main() -> int:
    """Run all compatibility and migration examples.

    Returns ``0`` only if every sub-section succeeded, ``1`` otherwise, so a
    real API failure surfaces as a non-zero process exit.
    """
    print("🚀 Venice AI API Compatibility & Migration Guide")
    print("=" * 60)

    # Fetch the text-model compatibility mapping once and share it across every
    # section that needs it, instead of hitting the network three times.
    async with VeniceClient() as client:
        text_compat = await client.models.list_compatibility(type="text")

    results: list[tuple[str, bool]] = [
        ("openai_to_venice_chat", await openai_to_venice_chat(text_compat)),
        ("anthropic_to_venice_chat", await anthropic_to_venice_chat(text_compat)),
        ("embedding_migration", await embedding_migration()),
        ("parameter_mapping_guide", await parameter_mapping_guide()),
        ("response_structure_comparison", await response_structure_comparison()),
        ("migration_checklist", await migration_checklist()),
        ("practical_migration_example", await practical_migration_example(text_compat)),
    ]

    failed = [name for name, ok in results if not ok]

    print("\n✨ Migration guide completed!")
    if failed:
        print(f"\n❌ {len(failed)} section(s) failed: {', '.join(failed)}")
    print("\n💡 Key Takeaways:")
    print("   - Venice AI maintains API compatibility with major providers")
    print("   - Automatic model name mapping for seamless transitions")
    print("   - Async-first design for better performance")
    print("   - Type-safe Pydantic models for reliability")
    print("   - Minimal code changes required for migration")
    print("\n📚 Next Steps:")
    print("   - Check compatibility mappings for your specific models")
    print("   - Test with your existing code patterns")
    print("   - Leverage Venice-specific features (uncensored models, etc.)")

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
