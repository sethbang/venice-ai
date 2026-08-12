#!/usr/bin/env python3
"""
Venice AI SDK - Character Discovery and Integration
===================================================

This example demonstrates how to discover and work with AI characters in the Venice AI SDK:
- Listing available AI characters and personalities
- Understanding character metadata and capabilities
- Categorizing characters by specialization and use case
- Analyzing character statistics and popularity
- Learning how to integrate characters into chat applications
"""

import asyncio
import sys
from collections import defaultdict

from venice_ai import VeniceClient


async def discover_characters():
    """Discover and explore available AI characters."""
    print("🎭 Character Discovery")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            # Get all available characters
            characters_response = await client.characters.list()

            characters = characters_response.data

            if characters:
                print(f"🎪 Found {len(characters)} available characters")

                # Show basic character information
                print("\n📋 Character Catalog:")
                for i, character in enumerate(characters[:10], 1):  # Show first 10
                    name = character.name
                    slug = character.slug
                    description = character.description or "No description available"

                    print(f"\n   {i}. {name}")
                    print(f"      🔗 Slug: {slug}")
                    print(
                        f"      📝 Description: {description[:100]}{'...' if len(description) > 100 else ''}"
                    )

                    # Show additional metadata if available
                    if character.tags:
                        tags = character.tags[:5]  # Show first 5 tags
                        print(
                            f"      🏷️ Tags: {', '.join(tags)}{'...' if len(character.tags) > 5 else ''}"
                        )

                    # Show web availability
                    web_status = "✅ Web Enabled" if character.webEnabled else "🌐 Web Restricted"
                    print(f"      {web_status}")

                if len(characters) > 10:
                    print(f"\n   ... and {len(characters) - 10} more characters")

                return characters

            else:
                print("ℹ️ No characters found")
                print(
                    "💡 This might be due to API access limitations or the service being unavailable"
                )
                return []

        except Exception as e:
            print(f"❌ Error discovering characters: {e}")
            print("💡 Note: Characters API is in Preview and requires appropriate access")
            return []


async def analyze_character_categories(characters):
    """Analyze and categorize characters by their specializations."""
    if not characters:
        return

    print("\n📊 Character Analysis & Categorization")
    print("-" * 40)

    # Categorize by tags
    tag_counts = defaultdict(int)
    category_characters = defaultdict(list)

    for character in characters:
        name = character.name
        description = (character.description or "").lower()
        tags = character.tags or []

        # Count tags
        for tag in tags:
            tag_counts[tag] += 1

        # Categorize by common themes
        if any(keyword in description for keyword in ["assistant", "help", "support"]):
            category_characters["Assistants"].append(name)
        elif any(keyword in description for keyword in ["creative", "story", "writing", "art"]):
            category_characters["Creative"].append(name)
        elif any(keyword in description for keyword in ["teacher", "tutor", "education", "learn"]):
            category_characters["Educational"].append(name)
        elif any(
            keyword in description for keyword in ["game", "roleplay", "adventure", "fantasy"]
        ):
            category_characters["Gaming & Roleplay"].append(name)
        elif any(keyword in description for keyword in ["professional", "business", "work"]):
            category_characters["Professional"].append(name)
        else:
            category_characters["General"].append(name)

    # Show most popular tags
    if tag_counts:
        print("🏷️ Most Popular Character Tags:")
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        for tag, count in sorted_tags[:10]:
            print(f"   📌 {tag}: {count} characters")

    # Show character categories
    print("\n📂 Character Categories:")
    for category, char_names in category_characters.items():
        if char_names:
            print(f"\n   🎯 {category} ({len(char_names)} characters):")
            for name in char_names[:5]:  # Show first 5 per category
                print(f"      • {name}")
            if len(char_names) > 5:
                print(f"      ... and {len(char_names) - 5} more")


async def analyze_character_statistics(characters):
    """Analyze character statistics and popularity metrics."""
    if not characters:
        return

    print("\n📈 Character Statistics Analysis")
    print("-" * 40)

    characters_with_stats = []

    for character in characters:
        name = character.name
        stats = character.stats

        if stats:
            characters_with_stats.append((name, stats))

    if characters_with_stats:
        print(f"📊 Found statistics for {len(characters_with_stats)} characters")

        def _stats_to_dict(stats):
            """Convert stats Pydantic model to dict."""
            return stats.model_dump()

        # Analyze available statistics
        stat_keys = set()
        for _name, stats in characters_with_stats:
            stats_dict = _stats_to_dict(stats)
            stat_keys.update(stats_dict.keys())

        if stat_keys:
            print("\n📋 Available Statistics Types:")
            for stat_type in sorted(stat_keys):
                print(f"   📊 {stat_type}")

            # Show top characters by different metrics
            for stat_type in sorted(stat_keys)[:3]:  # Show first 3 stat types
                print(f"\n🏆 Top Characters by {stat_type}:")

                # Extract values for this stat type
                char_values = []
                for name, stats in characters_with_stats:
                    stats_dict = _stats_to_dict(stats)
                    if stat_type in stats_dict:
                        value = stats_dict[stat_type]
                        if isinstance(value, (int, float)):
                            char_values.append((name, value))

                # Sort and show top characters
                char_values.sort(key=lambda x: x[1], reverse=True)
                for i, (name, value) in enumerate(char_values[:5], 1):
                    print(f"   {i}. {name}: {value}")

        else:
            print("📊 Statistics data format not recognized")

    else:
        print("ℹ️ No character statistics available")
        print("💡 Statistics may be available in the future or require special access")


async def demonstrate_character_integration():
    """Demonstrate how to integrate characters into chat applications."""
    print("\n🔗 Character Integration Guide")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            # Get characters for demonstration
            characters_response = await client.characters.list()
            characters = characters_response.data

            if characters:
                # Pick a few example characters
                example_chars = characters[:3]

                print("💡 Character Integration Examples:")
                print(
                    "\nTo use characters in chat completions, reference their slug in your chat request:"
                )

                for i, character in enumerate(example_chars, 1):
                    name = character.name
                    slug = character.slug
                    description = character.description or "No description"

                    print(f"\n{i}. {name}")
                    print(f"   Slug: '{slug}'")
                    print(
                        f"   Use case: {description[:80]}{'...' if len(description) > 80 else ''}"
                    )
                    print("   Integration example:")
                    print("   ```python")
                    print("   model = await client.models.resolve_chat()")
                    print("   response = await client.chat.completions.create(")
                    print("       model=model,")
                    print(f"       venice_parameters={{'character_slug': '{slug}'}},")
                    print("       messages=[")
                    print("           {'role': 'user', 'content': 'Hello! Can you help me?'}")
                    print("       ]")
                    print("   )")
                    print("   ```")

                print("\n🔧 Integration Best Practices:")
                print("   ✅ Always use the character's slug (not name) for API calls")
                print("   ✅ Test character responses to understand their personality")
                print("   ✅ Choose characters that match your application's purpose")
                print("   ✅ Consider character tags when filtering for specific needs")
                print("   ✅ Handle cases where characters might not be available")
                print("   ⚠️ Remember that Characters API is in Preview")

            else:
                print("ℹ️ No characters available for integration examples")

        except Exception as e:
            print(f"❌ Error demonstrating integration: {e}")


async def character_selection_guide(characters):
    """Provide guidance on selecting appropriate characters for different use cases."""
    if not characters:
        return

    print("\n🎯 Character Selection Guide")
    print("-" * 40)

    use_cases = {
        "Customer Support": ["helpful", "assistant", "support", "professional"],
        "Creative Writing": ["creative", "storytelling", "writing", "narrative"],
        "Education & Tutoring": ["teacher", "tutor", "educational", "learning"],
        "Entertainment": ["fun", "entertaining", "humor", "comedy"],
        "Technical Help": ["technical", "coding", "programming", "developer"],
        "Personal Assistant": ["assistant", "productivity", "organization"],
        "Gaming": ["game", "roleplay", "adventure", "fantasy", "rpg"],
    }

    print("📋 Recommended Characters by Use Case:")

    for use_case, keywords in use_cases.items():
        print(f"\n🎯 {use_case}:")

        # Find characters matching keywords
        matching_chars = []
        for character in characters:
            name = character.name
            description = (character.description or "").lower()
            tags = [tag.lower() for tag in (character.tags or [])]

            # Check if any keywords match description or tags
            if any(keyword in description for keyword in keywords) or any(
                keyword in tag for keyword in keywords for tag in tags
            ):
                matching_chars.append(name)

        if matching_chars:
            for char_name in matching_chars[:3]:  # Show top 3 matches
                print(f"   • {char_name}")
            if len(matching_chars) > 3:
                print(f"   ... and {len(matching_chars) - 3} more")
        else:
            print("   ℹ️ No specific matches found (use general characters)")

    print("\n💡 Selection Tips:")
    print("   🔍 Review character descriptions carefully")
    print("   🏷️ Use tags to filter characters by capability")
    print("   📊 Consider character statistics for popularity indicators")
    print("   🧪 Test interactions before production deployment")
    print("   📱 Check web enablement for browser-based applications")


async def main():
    """Run all character discovery and analysis examples."""
    print("🚀 Venice AI Character Discovery & Integration Examples")
    print("=" * 70)

    # Discover characters
    characters = await discover_characters()

    # Analyze the discovered characters
    if characters:
        await analyze_character_categories(characters)
        await analyze_character_statistics(characters)
        await character_selection_guide(characters)

    # Show integration examples
    await demonstrate_character_integration()

    print("\n✨ Character discovery examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Discovering available AI characters and personalities")
    print("   - Understanding character metadata and capabilities")
    print("   - Categorizing characters by specialization and use case")
    print("   - Analyzing character statistics and popularity metrics")
    print("   - Integrating characters into chat applications")
    print("   - Selecting appropriate characters for different scenarios")
    print("\n⚠️ Important Notes:")
    print("   - Characters API is currently in Preview")
    print("   - Features and availability may change in future releases")
    print("   - Always test character interactions before production use")
    print("   - Use character slugs (not names) for API integration")
    print("\n🎭 Character Benefits:")
    print("   - Pre-configured AI personalities without custom prompts")
    print("   - Consistent behavior patterns and knowledge domains")
    print("   - Enhanced user engagement through distinct personas")
    print("   - Specialized assistants for various use cases")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("Check that your API key is valid and you have character access.", file=sys.stderr)
        sys.exit(1)
