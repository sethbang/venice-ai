#!/usr/bin/env python3
"""
Venice AI SDK - Image Generation with Web Search
================================================

Demonstrates ``client.image.create(enable_web_search=True)`` — a new
optional body field on ``POST /image/generate`` that lets the image model
pull in recent web-search context before generating. Supported by models
whose ``model_spec.capabilities.supportsWebSearch`` is true.

When the flag is off (default), the model generates purely from the prompt.
When on, the model may first search the web for up-to-date references
matching the prompt (e.g. "a photo of the latest SpaceX Starship test").
"""

import asyncio
import sys
from pathlib import Path

from venice_ai import VeniceClient, detect_image_format

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


async def generate_with_web_search() -> None:
    """Side-by-side: generate the same prompt with and without web search."""
    print("🌐 Image Generation — enable_web_search")
    print("-" * 30)

    prompt = (
        "A photorealistic image of the latest NASA Artemis lunar mission "
        "with astronauts on the lunar surface"
    )

    async with VeniceClient() as client:
        model = await client.models.resolve_image()
        print(f"📍 Using model: {model}")

        for enable in (False, True):
            label = "with" if enable else "without"
            print(f"\n🎨 Generating {label} web search …")

            response = await client.image.create(
                model=model,
                prompt=prompt,
                width=768,
                height=768,
                steps=20,
                enable_web_search=enable,
            )

            assert not isinstance(response, bytes), "Expected JSON response"
            if not response.images:
                print("   (no image returned)")
                continue

            # Save the first image for visual comparison.
            payload = response.images[0]
            suffix = "on" if enable else "off"
            stem = RESULTS_DIR / f"image_web_search_{suffix}"
            saved = _save_image(payload, stem)
            if saved is not None:
                print(f"   💾 Saved: {saved}")


def _save_image(payload: str, stem: Path) -> Path | None:
    """Decode a base64 / data-URL / HTTP-URL payload to disk."""
    import base64

    if payload.startswith("data:"):
        _, b64 = payload.split(",", 1)
        data = base64.b64decode(b64)
    elif payload.startswith("http"):
        # The SDK returned a pre-signed URL rather than inline base64.
        print(f"   🔗 Download manually from: {payload}")
        return None
    else:
        # Bare base64
        data = base64.b64decode(payload)
    out_path = stem.with_suffix(f".{detect_image_format(data)[0]}")
    out_path.write_bytes(data)
    return out_path


async def main() -> None:
    print("🚀 Venice AI Image — enable_web_search Example")
    print("=" * 50)
    await generate_with_web_search()
    print("\n✨ Done.")
    print("\n💡 Compare the two saved images side-by-side — the web-search-")
    print("   enabled version should reflect the most recent public imagery.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
