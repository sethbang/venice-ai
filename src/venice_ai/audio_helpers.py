"""Helpers for Venice TTS that work around model-specific output limits.

The main entry point is :func:`stream_long_text`. It splits a long input into
sentence-aligned segments, dispatches them to ``client.audio.create_speech``
in parallel, and yields the concatenated audio bytes in input order.

Two server-side issues motivate this helper:

1. **tts-qwen3-0-6b and tts-qwen3-1-7b silently truncate output at exactly
   15.896875 s (664 MP3 frames at 24 kHz, identical for both sizes).**
   Past ~25 words of input, the model either speeds up unnaturally or stops
   mid-sentence. Reproduced across all four ``Accept-Encoding`` variants and
   both qwen3 sizes — the cap is server-side, not transport.

2. **Six of ten Venice TTS models buffer the full response before sending
   any bytes**, despite ``stream=True``: qwen3 family, orpheus, chatterbox,
   inworld, gemini. Only ``tts-xai-v1`` and ``tts-kokoro`` deliver chunks
   progressively. Parallel segment fan-out converts the buffering models
   into pseudo-streaming because chunk 1 can be in flight while chunk 0
   is still generating.

See :data:`MODEL_WORD_BUDGETS` for the per-model split thresholds. Update
that constant — not the helper's logic — when Venice ships fixes or new
models. This file is intentionally the only place that knows about the
qwen3 ceiling.

Known limitation: voice drift across segments
---------------------------------------------
On qwen3-family models, successive segments rendered through this helper
may exhibit subtle voice timbre / tone differences ("voice drift") even
with identical ``voice``, ``model``, and input style. Cause: ``/audio/speech``
does not currently accept a ``seed`` parameter, so each segment call samples
fresh RNG state on the server.

Empirical test (12-line poem on tts-qwen3-1-7b/Serena):
``temperature`` / ``top_p`` adjustments (0.2/0.8 and 0.05/0.5) and a stable
style ``prompt`` were tried as anchors. None produced an audibly-clear
improvement over Venice defaults; run-to-run variance dominated any
configuration delta. ``temperature``, ``top_p``, and ``prompt`` remain
caller-controlled on :func:`stream_long_text` for use cases where they
help, but no default is baked in. The real fix is a server-side seed
parameter on ``/audio/speech``.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._client import VeniceClient
    from .types.enums import ResponseFormat, Voice

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model-specific output limits
# ---------------------------------------------------------------------------
# Per-model word budget for :func:`split_text_for_tts`. Each segment is sized
# so its expected output stays well under the server-side cap.
#
# **tts-qwen3-0-6b / tts-qwen3-1-7b**: hard cap of 664 MP3 frames =
# **15.896875 s** of audio. Identical cap on both model sizes (strong signal
# this is a hardcoded inference-path limit, not a per-model artifact).
# Natural pace for these models is ~2.2 words/sec, so 25 words ≈ 11 s of
# audio — leaves ~5 s headroom for slower deliveries.
#
# **Other Venice TTS models**: no truncation observed up to ~3500 chars of
# input (kokoro tested to 177 s output, xai-v1 to 224 s output). The default
# budget below is a parallelism-vs-overhead tradeoff, not a correctness one.
#
# If a Venice fix removes the qwen3 cap, set the entries to
# ``DEFAULT_WORD_BUDGET`` (or remove them) — do not change the
# splitter logic itself.
MODEL_WORD_BUDGETS: dict[str, int] = {
    "tts-qwen3-0-6b": 25,
    "tts-qwen3-1-7b": 25,
}

#: Words per segment for any model not in :data:`MODEL_WORD_BUDGETS`.
#: Chosen to produce ~30-45 s segments at typical TTS pace, which is large
#: enough to amortize per-request overhead and small enough that ~4 in
#: parallel saturate a typical user's perceived audio playback rate.
DEFAULT_WORD_BUDGET: int = 100


# Splits on sentence-ending punctuation that's followed by whitespace.
# Newlines inside a paragraph (single \n) are treated as sentence separators
# because TTS-friendly inputs often have line-broken poetry / formatted text.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])\n|\n(?=[A-Z])")


def split_text_for_tts(
    text: str,
    model: str | None = None,
    *,
    max_words: int | None = None,
) -> list[str]:
    """Split text into TTS-friendly segments only when the budget is exceeded.

    For text that fits under ``max_words`` the result is ``[text]`` and the
    caller can pass it to ``create_speech`` unchanged — the helper is a
    no-op for short inputs.

    When the input exceeds the budget the splitter is greedy: packs
    sentences (respecting paragraph and sentence terminators as boundaries)
    into a segment until the next sentence would push word count over
    ``max_words``, then starts a new segment.

    A single sentence longer than ``max_words`` is emitted as its own
    over-budget segment — splitting mid-sentence produces worse audio than
    one over-budget chunk.

    Empirical note: paragraph-boundary forcing was tried earlier and removed.
    Several Venice TTS models render each segment faster than the
    corresponding portion of a longer call (kokoro shrinks ~30 % when given
    one stanza at a time instead of three together), so forcing extra splits
    when not needed costs audible audio with no upside.

    :param text: Input text. Empty / whitespace-only returns ``[""]``.
    :param model: Venice TTS model id. Used to look up
        :data:`MODEL_WORD_BUDGETS` when ``max_words`` is not given.
    :param max_words: Override the per-model word budget.
    :return: Segments in input order. Always at least one element.
    :raises ValueError: If ``max_words`` is given and not positive.
    """
    if max_words is not None and max_words <= 0:
        raise ValueError(f"max_words must be positive, got {max_words!r}")

    if max_words is None:
        max_words = MODEL_WORD_BUDGETS.get(model or "", DEFAULT_WORD_BUDGET)

    text = text.strip()
    if not text:
        return [""]

    # Fast path: total input under budget → no splitting at all.
    if len(text.split()) <= max_words:
        return [text]

    # Flatten paragraphs + sentences into an ordered sentence list, then
    # greedily pack. Paragraph boundaries are treated as sentence boundaries
    # for splitting purposes but otherwise carry no weight.
    sentences: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for raw in _SENTENCE_SPLIT_RE.split(paragraph):
            s = raw.strip()
            if s:
                sentences.append(s)

    segments: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if current:
            segments.append(" ".join(current))
            current = []
            current_words = 0

    for sentence in sentences:
        n_words = len(sentence.split())
        if n_words > max_words:
            flush()
            segments.append(sentence)
            continue
        if current_words + n_words > max_words:
            flush()
        current.append(sentence)
        current_words += n_words
    flush()

    return segments or [text]


# ID3v2 syncsafe size decoder. Tags 2..N in a concatenated mp3 stream produce
# a brief player-visible metadata blip on some players (afplay is fine; some
# JS Audio implementations re-emit the metadata event). Stripping is purely
# cosmetic; the audio plays correctly either way.
def _strip_leading_id3(data: bytes) -> bytes:
    if len(data) < 10 or data[:3] != b"ID3":
        return data
    size = (
        ((data[6] & 0x7F) << 21)
        | ((data[7] & 0x7F) << 14)
        | ((data[8] & 0x7F) << 7)
        | (data[9] & 0x7F)
    )
    header_total = 10 + size
    if header_total > len(data):
        return b""
    return data[header_total:]


# Type alias for the progress callback
SegmentCallback = Callable[[int, float, int], None]


async def stream_long_text(
    client: VeniceClient,
    *,
    input: str,
    model: str,
    voice: str | Voice,
    response_format: str | ResponseFormat = "mp3",
    speed: float | None = None,
    language: str | None = None,
    prompt: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_words_per_segment: int | None = None,
    max_concurrency: int = 4,
    on_segment_complete: SegmentCallback | None = None,
    timeout: float | None = None,
) -> AsyncIterator[bytes]:
    """Stream TTS audio for long input by splitting into parallel segments.

    Splits ``input`` via :func:`split_text_for_tts`, dispatches each segment
    to ``client.audio.create_speech(stream=True, ...)`` with bounded
    concurrency, and yields the resulting mp3 bytes in input order.

    Only ``response_format="mp3"`` is supported. Other formats require
    container-level demuxing to concatenate cleanly and would produce
    malformed output if naively appended.

    If ``input`` fits in a single segment, this short-circuits to
    ``create_speech`` directly — no extra task scheduling overhead.

    :param client: A connected ``VeniceClient``.
    :param input: Text to synthesize.
    :param model: Venice TTS model id (e.g. ``"tts-qwen3-1-7b"``).
    :param voice: Model-specific voice id.
    :param response_format: Only ``"mp3"`` supported in this helper.
    :param max_words_per_segment: Override the per-model budget.
    :param max_concurrency: Concurrent in-flight create_speech calls.
        Default 4 keeps headroom under Venice's 60 req/min limit for typical
        single-document use.
    :param on_segment_complete: Optional callback invoked as each segment
        finishes its stream, with ``(segment_index, latency_seconds, bytes)``.
    :param timeout: Per-segment timeout in seconds, forwarded to
        ``create_speech``.

    :raises NotImplementedError: If ``response_format`` is not ``"mp3"``.
    :raises ValueError: If ``max_concurrency < 1``.

    Exceptions raised inside any segment task are re-raised when that
    segment's bytes would have been yielded; later segments are cancelled.
    """
    if str(response_format).lower() != "mp3":
        raise NotImplementedError(
            f"stream_long_text currently only supports response_format='mp3'; "
            f"got {response_format!r}. wav/flac/opus/aac/pcm need container-"
            f"level concatenation and are not yet implemented."
        )
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")

    segments = split_text_for_tts(input, model, max_words=max_words_per_segment)

    # Forward only non-None per-segment kwargs so model defaults stay intact.
    common_kwargs: dict[str, Any] = {
        "model": model,
        "voice": voice,
        "response_format": response_format,
        "stream": True,
    }
    for k, v in (
        ("speed", speed),
        ("language", language),
        ("prompt", prompt),
        ("temperature", temperature),
        ("top_p", top_p),
        ("timeout", timeout),
    ):
        if v is not None:
            common_kwargs[k] = v

    # Fast path: single segment, no parallel orchestration overhead.
    if len(segments) == 1:
        stream = await client.audio.create_speech(input=segments[0], **common_kwargs)
        async for chunk in stream:
            yield chunk
        return

    logger.info(
        "stream_long_text: %d chars → %d segments (model=%s, max_concurrency=%d)",
        len(input),
        len(segments),
        model,
        max_concurrency,
    )

    semaphore = asyncio.Semaphore(max_concurrency)
    # Each queue carries chunks for one segment. Sentinel: None for clean
    # end-of-segment, a BaseException for segment failure.
    queues: list[asyncio.Queue[bytes | None | BaseException]] = [asyncio.Queue() for _ in segments]

    async def fetch_segment(idx: int, segment_text: str) -> None:
        t0 = time.monotonic()
        total_bytes = 0
        try:
            async with semaphore:
                stream = await client.audio.create_speech(input=segment_text, **common_kwargs)
                first_chunk = True
                async for chunk in stream:
                    if idx > 0 and first_chunk:
                        chunk = _strip_leading_id3(chunk)
                        first_chunk = False
                    if chunk:
                        total_bytes += len(chunk)
                        await queues[idx].put(chunk)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - re-raised via queue
            await queues[idx].put(exc)
        else:
            if on_segment_complete is not None:
                try:
                    on_segment_complete(idx, time.monotonic() - t0, total_bytes)
                except Exception:  # noqa: BLE001
                    logger.warning("on_segment_complete callback raised; ignoring")
        finally:
            await queues[idx].put(None)

    tasks = [asyncio.create_task(fetch_segment(i, seg)) for i, seg in enumerate(segments)]

    try:
        for q in queues:
            while True:
                item = await q.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
