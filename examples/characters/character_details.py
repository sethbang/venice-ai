#!/usr/bin/env python3
"""
Venice AI SDK - Character Details and Individual Character Access
================================================================

This example demonstrates how to access individual character details using the new
GET /characters/{slug} endpoint introduced in the Venice AI API. It showcases:

- Fetching detailed information for specific characters
- Accessing new character fields (photoUrl, shareUrl, modelId)
- Using character details for enhanced integration
- Best practices for character-specific workflows

Key Features Demonstrated:
- Individual character retrieval by slug
- Complete character metadata access
- Enhanced character integration patterns
- Character-specific model usage
- Character URL and photo management
"""

import asyncio
import random
import sys

from venice_ai import VeniceClient


async def get_character_details() -> bool:
    """Demonstrate getting detailed information for specific characters.

    Returns ``True`` on success, ``False`` if the list call failed or every
    sampled character lookup failed.
    """
    print("🔍 Character Details Access")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            # First, get a list of characters to choose from
            print("📋 Getting character list...")
            characters_list = await client.characters.list()

            if not characters_list.data:
                print("ℹ️ No characters available")
                return True

            # Pick up to 10 random characters to get details for
            num_to_sample = min(10, len(characters_list.data))
            example_characters = random.sample(characters_list.data, num_to_sample)

            print(f"✅ Found {len(characters_list.data)} characters")
            print(f"🔍 Getting detailed information for {len(example_characters)} characters...\n")

            detail_failures = 0
            for i, char_summary in enumerate(example_characters, 1):
                print(f"═══ Character {i}: {char_summary.name} ═══")

                try:
                    # Get detailed character information
                    character_details = await client.characters.get(char_summary.slug)
                    char_data = character_details.data

                    print("📊 Basic Information:")
                    print(f"   Name: {char_data.name}")
                    print(f"   Slug: {char_data.slug}")
                    print(f"   Model ID: {char_data.modelId}")
                    print(f"   Created: {char_data.createdAt}")
                    print(f"   Updated: {char_data.updatedAt}")

                    print("\n📝 Description:")
                    if char_data.description:
                        desc = char_data.description
                        print(f"   {desc[:200]}{'...' if len(desc) > 200 else ''}")
                    else:
                        print("   (No description available)")

                    print("\n🔗 URLs & Resources:")
                    if char_data.photoUrl:
                        print(f"   Photo URL: {char_data.photoUrl}")
                    if char_data.shareUrl:
                        print(f"   Share URL: {char_data.shareUrl}")

                    print("\n⚙️ Settings:")
                    print(f"   Adult Content: {'Yes' if char_data.adult else 'No'}")
                    print(f"   Web Enabled: {'Yes' if char_data.webEnabled else 'No'}")

                    print("\n🏷️ Tags:")
                    if char_data.tags:
                        print(
                            f"   {', '.join(char_data.tags[:10])}{'...' if len(char_data.tags) > 10 else ''}"
                        )
                    else:
                        print("   (No tags)")

                    print("\n📊 Statistics:")
                    if hasattr(char_data.stats, "imports"):
                        print(f"   Imports: {char_data.stats.imports}")
                    else:
                        print("   (No statistics available)")

                    print()

                except Exception as e:
                    # One character failing shouldn't abort the whole sample, but
                    # we track it so an all-failure run reports not-ok.
                    print(f"❌ Error getting details for {char_summary.slug}: {e}")
                    print()
                    detail_failures += 1

            if example_characters and detail_failures == len(example_characters):
                print("❌ Every sampled character lookup failed.")
                ok = False

        except Exception as e:
            print(f"❌ Error: {e}")
            ok = False

    return ok


async def character_metadata_showcase() -> bool:
    """Showcase the new character metadata fields and their uses.

    Returns ``True`` on success, ``False`` if the list call failed or every
    sampled character lookup failed.
    """
    print("🎨 Character Metadata Showcase")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            # Get characters and examine their metadata
            characters_list = await client.characters.list()

            if not characters_list.data:
                print("ℹ️ No characters available for metadata showcase")
                return True

            print("🆕 New Character Metadata Fields:")
            print("   • photoUrl: Direct access to character photos")
            print("   • shareUrl: Public character page URLs")
            print("   • modelId: Character's optimal model identifier")

            # Examine metadata from multiple characters
            metadata_summary: dict[str, object] = {
                "has_photo": 0,
                "has_share_url": 0,
                "unique_models": set(),
            }

            sample = characters_list.data[:5]
            print(
                f"\n📊 Metadata Analysis from {len(sample)} sampled characters "
                f"(of {len(characters_list.data)} total):"
            )

            detail_failures = 0
            for char_summary in sample:  # Check first 5 characters
                try:
                    char_details = await client.characters.get(char_summary.slug)
                    char_data = char_details.data

                    # Count metadata availability
                    if char_data.photoUrl:
                        metadata_summary["has_photo"] += 1  # type: ignore[operator]
                    if char_data.shareUrl:
                        metadata_summary["has_share_url"] += 1  # type: ignore[operator]

                    metadata_summary["unique_models"].add(char_data.modelId)  # type: ignore[attr-defined]

                    print(f"\n   🎭 {char_data.name}:")
                    print(f"      Model: {char_data.modelId}")
                    print(f"      Photo: {'✅' if char_data.photoUrl else '❌'}")
                    print(f"      Char URL: {'✅' if char_data.shareUrl else '❌'}")

                except Exception as e:
                    print(f"   ❌ Error getting details for {char_summary.slug}: {e}")
                    detail_failures += 1

            sampled = len(sample)
            print("\n📈 Metadata Summary:")
            print(f"   Characters with photos: {metadata_summary['has_photo']}/{sampled}")
            print(f"   Characters with share URLs: {metadata_summary['has_share_url']}/{sampled}")
            print(f"   Unique models used: {len(metadata_summary['unique_models'])}")  # type: ignore[arg-type]

            if metadata_summary["unique_models"]:
                print(
                    f"   Models: {', '.join(sorted(metadata_summary['unique_models']))}"  # type: ignore[arg-type]
                )

            if sample and detail_failures == len(sample):
                print("❌ Every sampled character lookup failed.")
                ok = False

        except Exception as e:
            print(f"❌ Error in metadata showcase: {e}")
            ok = False

    return ok


async def advanced_character_integration() -> bool:
    """Show advanced integration patterns using new character details.

    Returns ``True`` on success, ``False`` if the live calls backing the
    pattern walkthrough failed.
    """
    print("🚀 Advanced Character Integration")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            # Get characters
            characters_list = await client.characters.list()

            if not characters_list.data:
                print("ℹ️ No characters available for advanced integration")
                return True

            print("💡 Advanced Integration Patterns:")

            # Pattern 1: Model-Optimized Chat
            print("\n🎯 Pattern 1: Model-Optimized Character Chat")
            first_char = characters_list.data[0]
            char_details = await client.characters.get(first_char.slug)
            char_data = char_details.data

            print(f"   Character: {char_data.name}")
            print(f"   Optimized for model: {char_data.modelId}")
            print("   Implementation:")
            print("   ```python")
            print("   async def optimized_character_chat(client, char_slug, user_message):")
            print("       # Get character and use its optimal model")
            print("       char = await client.characters.get(char_slug)")
            print("       ")
            print("       response = await client.chat.completions.create(")
            print("           model=char.data.modelId,  # Use character's model")
            print("           venice_parameters={'character_slug': char.data.slug},")
            print("           messages=[")
            print("               {'role': 'user', 'content': user_message}")
            print("           ]")
            print("       )")
            print("       return response")
            print("   ```")

            # Pattern 2: Character Gallery Builder
            if len(characters_list.data) > 1:
                print("\n🖼️ Pattern 2: Character Gallery Builder")
                print("   Use Case: Build character selection interface")
                print("   Implementation:")
                print("   ```python")
                print("   async def build_character_gallery(client, char_slugs):")
                print("       gallery = []")
                print("       ")
                print("       for slug in char_slugs:")
                print("           char = await client.characters.get(slug)")
                print("           gallery.append({")
                print("               'id': char.data.slug,")
                print("               'name': char.data.name,")
                print("               'description': char.data.description,")
                print("               'photo': char.data.photoUrl,")
                print("               'url': char.data.shareUrl,")
                print("               'model': char.data.modelId,")
                print("               'tags': char.data.tags")
                print("           })")
                print("       ")
                print("       return gallery")
                print("   ```")

            # Pattern 3: Character Validation
            print("\n✅ Pattern 3: Character Validation & Error Handling")
            print("   Use Case: Robust character access with fallbacks")
            print("   Implementation:")
            print("   ```python")
            print("   from venice_ai.exceptions import NotFoundError")
            print("   ")
            print("   async def safe_character_access(client, char_slug):")
            print("       try:")
            print("           char = await client.characters.get(char_slug)")
            print("           ")
            print("           # Validate character is suitable")
            print("           if not char.data.webEnabled:")
            print("               raise ValueError(f'Character {char_slug} not web-enabled')")
            print("           ")
            print("           return char.data")
            print("       ")
            print("       except NotFoundError:")
            print("           print(f'Character {char_slug} not found')")
            print("           return None")
            print("       except Exception as e:")
            print("           print(f'Error accessing character: {e}')")
            print("           return None")
            print("   ```")

        except Exception as e:
            print(f"❌ Error in advanced integration: {e}")
            ok = False

    return ok


async def main() -> int:
    """Run all character details examples.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    print("🚀 Venice AI Character Details Examples")
    print("=" * 60)

    results: list[tuple[str, bool]] = [
        ("get_character_details", await get_character_details()),
        ("character_metadata_showcase", await character_metadata_showcase()),
        ("advanced_character_integration", await advanced_character_integration()),
    ]

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("\n✨ Character details examples completed!")

    print("\n💡 Key concepts demonstrated:")
    print("   - Individual character access via GET /characters/{slug}")
    print("   - New character metadata fields (photoUrl, shareUrl, modelId)")
    print("   - Character-model optimization patterns")
    print("   - Enhanced character integration workflows")
    print("   - Character validation and error handling")

    print("\n🔧 Integration Benefits:")
    print("   - Direct character metadata access")
    print("   - Model-optimized character interactions")
    print("   - Enhanced UI/UX with character photos and URLs")
    print("   - Robust character validation and error handling")

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print(
            "Check that your API key is valid and you have character access.",
            file=sys.stderr,
        )
        sys.exit(1)
