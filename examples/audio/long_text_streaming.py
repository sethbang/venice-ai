#!/usr/bin/env python3
"""
Venice AI SDK — Long-text TTS via stream_long_text
===================================================

Demonstrates ``client.audio.stream_long_text``, the helper that splits long
input into parallel segments and yields concatenated audio in order.

Use this helper when either:

1. **Output truncation:** the ``tts-qwen3-*`` family caps output at exactly
   15.896875 s regardless of input length. A 3-stanza poem renders as the
   first stanza-and-a-half. Splitting the input keeps each segment under
   the cap so the full text gets voiced.

2. **Pseudo-streaming on buffering models:** ``tts-orpheus``,
   ``tts-chatterbox-hd``, ``tts-gemini-3-1-flash``, ``tts-inworld-1-5-max``,
   and the qwen3 family buffer the full response server-side before
   sending bytes. Splitting + parallel dispatch starts later segments
   while the first one is still generating, so you hear audio sooner.

For ``tts-kokoro`` and ``tts-xai-v1`` (which truly stream and don't truncate),
the helper short-circuits to ``create_speech`` when input fits the budget,
so it's safe to call regardless of the chosen model.

Run::

    poetry run python examples/audio/long_text_streaming.py

(Requires ``VENICE_API_KEY`` in the environment.)
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

from venice_ai import VeniceClient

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LONG_POEM = (
    "Beneath the silver moon's soft glow,\n"
    "Where rivers hum and willows sway,\n"
    "The night air holds a tender flow,\n"
    "As dreamers drift in twilight's play.\n"
    "\n"
    "A whispered breeze through pines aloft,\n"
    "Carries tales of stars long-burned,\n"
    "Their light, though distant, lingers soft,\n"
    "On every wave the ocean turned.\n"
    "\n"
    "And in the hush before the dawn,\n"
    "When shadows learn the language of light,\n"
    "A quiet hope is gently drawn,\n"
    "To carry day past every night."
)


async def qwen3_long_text() -> None:
    """Render the full 12-line poem on a model that would otherwise truncate.

    Without the helper, ``tts-qwen3-1-7b`` produces ~15.9 s of audio (the
    first stanza and a half) regardless of input length. With the helper,
    the splitter emits four ~12 s segments that stitch into the full poem.
    """
    print("\n🎙  qwen3-1-7b long text via stream_long_text")
    print("-" * 50)

    segment_log: list[tuple[int, float, int]] = []
    t0 = time.monotonic()

    async with VeniceClient() as client:
        chunks: list[bytes] = []
        stream = await client.audio.stream_long_text(
            input=LONG_POEM,
            # This demo pins tts-qwen3-1-7b specifically because that family
            # has a documented ~15.9 s output-truncation cap; a resolved
            # default model would not reliably reproduce the cap that the
            # stream_long_text helper is meant to work around.
            model="tts-qwen3-1-7b",
            voice="Serena",
            response_format="mp3",
            max_concurrency=4,
            on_segment_complete=lambda i, t, b: segment_log.append((i, t, b)),
            timeout=180.0,
        )
        async for chunk in stream:
            chunks.append(chunk)
        body = b"".join(chunks)

    output = RESULTS_DIR / "long_text_qwen3.mp3"
    output.write_bytes(body)
    elapsed = time.monotonic() - t0

    print(f"✅ Saved {len(body)} bytes to {output}")
    print(f"⏱  Wall time: {elapsed:.1f}s")
    print(f"🧩 Segments completed: {len(segment_log)}")
    for idx, lat, n_bytes in sorted(segment_log):
        print(f"   segment {idx}: {lat:.1f}s wall, {n_bytes} bytes")
    print(f"▶  Play with:  afplay {output}")


async def kokoro_under_budget_passthrough() -> None:
    """Same input on kokoro: helper passes through to a single call.

    The 74-word poem fits under the default 100-word budget, so the helper
    short-circuits to ``create_speech``. Output is byte-identical to calling
    ``create_speech`` directly. The helper is safe to use unconditionally.
    """
    print("\n🎙  kokoro long text via stream_long_text (passthrough)")
    print("-" * 50)

    segment_log: list[tuple[int, float, int]] = []
    t0 = time.monotonic()

    async with VeniceClient() as client:
        chunks: list[bytes] = []
        stream = await client.audio.stream_long_text(
            input=LONG_POEM,
            # Pinned to tts-kokoro because, unlike the qwen3 family, it truly
            # streams and does not truncate long input — so the helper passes
            # through to a single create_speech call (byte-identical output).
            model="tts-kokoro",
            voice="af_alloy",
            response_format="mp3",
            on_segment_complete=lambda i, t, b: segment_log.append((i, t, b)),
            timeout=60.0,
        )
        async for chunk in stream:
            chunks.append(chunk)
        body = b"".join(chunks)

    output = RESULTS_DIR / "long_text_kokoro.mp3"
    output.write_bytes(body)
    elapsed = time.monotonic() - t0

    print(f"✅ Saved {len(body)} bytes to {output}")
    print(f"⏱  Wall time: {elapsed:.1f}s")
    print(f"🧩 Splits used: {len(segment_log)}  ({'(passthrough)' if not segment_log else ''})")
    print(f"▶  Play with:  afplay {output}")


async def main() -> None:
    await qwen3_long_text()
    await kokoro_under_budget_passthrough()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
