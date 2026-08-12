"""
Streaming handler for Venice AI CLI with animated display options
"""

import asyncio
import time
from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text


class AnimationMode(StrEnum):
    """Animation modes for streaming display"""

    NONE = "none"  # No animation, instant display
    SMOOTH = "smooth"  # Default smooth markdown rendering
    WORD = "word"  # Word-by-word animation
    CHAR = "char"  # Character-by-character animation
    LINE = "line"  # Line-buffered animation
    TYPEWRITER = "typewriter"  # Classic typewriter effect


class StreamAccumulator:
    """Accumulates streaming chunks with timing statistics"""

    def __init__(self):
        self.content: str = ""
        self.chunk_count: int = 0
        self.first_token_time: float | None = None
        self.last_token_time: float | None = None
        self.start_time: float | None = None
        self.finish_reason: str | None = None

    def process_chunk(self, chunk: Any) -> str:
        """Process a single chunk and return new content"""
        self.chunk_count += 1

        # Track timing
        current_time = time.perf_counter()
        if self.start_time is None:
            self.start_time = current_time
        if self.first_token_time is None and chunk.choices and chunk.choices[0].delta.content:
            self.first_token_time = current_time
        self.last_token_time = current_time

        new_content = ""
        if chunk.choices:
            choice = chunk.choices[0]
            delta = choice.delta

            # Extract content
            if delta.content:
                self.content += delta.content
                new_content = delta.content

            # Extract finish reason
            if choice.finish_reason:
                self.finish_reason = choice.finish_reason

        return new_content

    def get_statistics(self) -> dict[str, Any]:
        """Get streaming statistics"""
        stats: dict[str, Any] = {
            "total_chunks": self.chunk_count,
            "content_length": len(self.content),
            "finish_reason": self.finish_reason,
        }

        if self.start_time and self.last_token_time:
            duration = self.last_token_time - self.start_time
            stats["stream_duration"] = round(duration, 3)

            if self.first_token_time:
                time_to_first = self.first_token_time - self.start_time
                stats["time_to_first_token"] = round(time_to_first, 3)

            if duration > 0.1:  # Only show chunks/sec for meaningful durations
                stats["chunks_per_second"] = round(self.chunk_count / duration, 1)

        return stats


class StreamHandler:
    """Handler for streaming responses from Venice AI with animation support"""

    def __init__(
        self,
        console: Console | None = None,
        animation_mode: AnimationMode = AnimationMode.SMOOTH,
        animation_speed: float = 0.03,  # Default delay between animations
        plain: bool = False,
    ):
        """Initialize the stream handler"""
        self.console = console or Console()
        self.animation_mode = animation_mode
        self.animation_speed = animation_speed
        self.plain = plain
        self.buffer = ""

    async def handle_chat_stream(
        self, stream: AsyncIterator, show_stats: bool = False
    ) -> tuple[str, dict[str, Any]]:
        """
        Handle streaming chat completions with animation

        Returns:
            Tuple of (complete_text, statistics)
        """
        if self.plain:
            return await self._handle_plain_stream(stream, show_stats)
        if self.animation_mode == AnimationMode.SMOOTH:
            return await self._handle_smooth_stream(stream, show_stats)
        elif self.animation_mode == AnimationMode.WORD:
            return await self._handle_word_stream(stream, show_stats)
        elif self.animation_mode == AnimationMode.CHAR:
            return await self._handle_char_stream(stream, show_stats)
        elif self.animation_mode == AnimationMode.LINE:
            return await self._handle_line_stream(stream, show_stats)
        elif self.animation_mode == AnimationMode.TYPEWRITER:
            return await self._handle_typewriter_stream(stream, show_stats)
        else:  # NONE
            return await self._handle_no_animation_stream(stream, show_stats)

    async def _handle_smooth_stream(
        self, stream: AsyncIterator, show_stats: bool
    ) -> tuple[str, dict[str, Any]]:
        """Handle smooth markdown rendering with proper animation"""
        accumulator = StreamAccumulator()
        last_update = time.perf_counter()
        min_update_interval = 0.05  # Update at most every 50ms for smooth animation

        with Live(
            Text("", style="dim"),
            console=self.console,
            refresh_per_second=20,  # Higher refresh rate for smoother display
        ) as live:
            async for chunk in stream:
                new_content = accumulator.process_chunk(chunk)
                if new_content:
                    current_time = time.perf_counter()
                    # Add a small delay to create smooth animation effect
                    if current_time - last_update >= min_update_interval:
                        live.update(Markdown(accumulator.content))
                        last_update = current_time
                    else:
                        # Still add small delay to prevent tight loop
                        await asyncio.sleep(0.01)

            # Final update to ensure all content is displayed
            live.update(Markdown(accumulator.content))

        stats = accumulator.get_statistics() if show_stats else {}
        return accumulator.content, stats

    async def _handle_word_stream(
        self, stream: AsyncIterator, show_stats: bool
    ) -> tuple[str, dict[str, Any]]:
        """Handle word-by-word animated streaming"""
        accumulator = StreamAccumulator()
        buffer = ""
        displayed_text = ""

        with Live(Text(""), console=self.console, refresh_per_second=30) as live:
            async for chunk in stream:
                new_content = accumulator.process_chunk(chunk)
                if new_content:
                    buffer += new_content

                    # Process complete words
                    while " " in buffer or "\n" in buffer:
                        # Find next delimiter
                        space_idx = buffer.find(" ")
                        newline_idx = buffer.find("\n")

                        if space_idx == -1:
                            delimiter_idx = newline_idx
                        elif newline_idx == -1:
                            delimiter_idx = space_idx
                        else:
                            delimiter_idx = min(space_idx, newline_idx)

                        word = buffer[: delimiter_idx + 1]
                        displayed_text += word
                        buffer = buffer[delimiter_idx + 1 :]

                        # Update display
                        live.update(Markdown(displayed_text + buffer))

                        # Consistent animation speed throughout
                        await asyncio.sleep(self.animation_speed)

            # Display remaining buffer
            if buffer:
                displayed_text += buffer
                live.update(Markdown(displayed_text))

        stats = accumulator.get_statistics() if show_stats else {}
        return accumulator.content, stats

    async def _handle_char_stream(
        self, stream: AsyncIterator, show_stats: bool
    ) -> tuple[str, dict[str, Any]]:
        """Handle character-by-character animated streaming"""
        accumulator = StreamAccumulator()
        displayed_text = ""

        with Live(Text(""), console=self.console, refresh_per_second=60) as live:
            async for chunk in stream:
                new_content = accumulator.process_chunk(chunk)
                if new_content:
                    for char in new_content:
                        displayed_text += char

                        # Update display
                        live.update(Markdown(displayed_text))

                        # Consistent animation speed throughout
                        await asyncio.sleep(self.animation_speed / 3)

        stats = accumulator.get_statistics() if show_stats else {}
        return accumulator.content, stats

    async def _handle_line_stream(
        self, stream: AsyncIterator, show_stats: bool
    ) -> tuple[str, dict[str, Any]]:
        """Handle line-buffered animated streaming"""
        accumulator = StreamAccumulator()
        line_buffer = ""
        displayed_text = ""

        with Live(Text(""), console=self.console, refresh_per_second=10) as live:
            async for chunk in stream:
                new_content = accumulator.process_chunk(chunk)
                if new_content:
                    line_buffer += new_content

                    # Process complete lines
                    while "\n" in line_buffer:
                        line, line_buffer = line_buffer.split("\n", 1)
                        displayed_text += line + "\n"

                        # Update display
                        live.update(Markdown(displayed_text + line_buffer))

                        # Consistent animation speed throughout
                        await asyncio.sleep(self.animation_speed * 3)

            # Display remaining buffer
            if line_buffer:
                displayed_text += line_buffer
                live.update(Markdown(displayed_text))

        stats = accumulator.get_statistics() if show_stats else {}
        return accumulator.content, stats

    async def _handle_typewriter_stream(
        self, stream: AsyncIterator, show_stats: bool
    ) -> tuple[str, dict[str, Any]]:
        """Handle typewriter effect with character sounds (visual only)"""
        accumulator = StreamAccumulator()
        displayed_text = ""

        # Typewriter cursor
        cursor_states = ["▊", "▌", "▍", "▎", " "]
        cursor_idx = 0

        with Live(Text(""), console=self.console, refresh_per_second=30) as live:
            async for chunk in stream:
                new_content = accumulator.process_chunk(chunk)
                if new_content:
                    for char in new_content:
                        displayed_text += char

                        # Show with blinking cursor effect
                        cursor = cursor_states[cursor_idx % len(cursor_states)]
                        live.update(Text(displayed_text + cursor))
                        cursor_idx += 1

                        # Variable speed for more realistic typing
                        if char == " ":
                            await asyncio.sleep(self.animation_speed * 0.5)
                        elif char in ".!?":
                            await asyncio.sleep(self.animation_speed * 3)
                        elif char == "\n":
                            await asyncio.sleep(self.animation_speed * 2)
                        else:
                            await asyncio.sleep(self.animation_speed)

            # Final display without cursor
            live.update(Markdown(displayed_text))

        stats = accumulator.get_statistics() if show_stats else {}
        return accumulator.content, stats

    async def _handle_no_animation_stream(
        self, stream: AsyncIterator, show_stats: bool
    ) -> tuple[str, dict[str, Any]]:
        """Handle streaming without animation - instant display"""
        accumulator = StreamAccumulator()

        with Live(Text(""), console=self.console, refresh_per_second=30) as live:
            async for chunk in stream:
                new_content = accumulator.process_chunk(chunk)
                if new_content:
                    # Instant update
                    live.update(Markdown(accumulator.content))

        stats = accumulator.get_statistics() if show_stats else {}
        return accumulator.content, stats

    async def _handle_plain_stream(
        self, stream: AsyncIterator, show_stats: bool
    ) -> tuple[str, dict[str, Any]]:
        """Handle streaming in plain mode — prints raw text chunks to stdout"""
        import sys

        accumulator = StreamAccumulator()

        async for chunk in stream:
            new_content = accumulator.process_chunk(chunk)
            if new_content:
                sys.stdout.write(new_content)
                sys.stdout.flush()

        sys.stdout.write("\n")
        sys.stdout.flush()

        stats = accumulator.get_statistics() if show_stats else {}
        return accumulator.content, stats

    async def handle_simple_stream(self, stream: AsyncIterator) -> str:
        """Handle simple text streaming without rich formatting"""
        complete_text = ""

        async for chunk in stream:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]

                # Extract the content from the delta
                if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                    content = choice.delta.content
                    if content:
                        complete_text += content
                        # Simple print without newline
                        print(content, end="", flush=True)

        print()  # Final newline
        return complete_text

    def display_progress(self, message: str) -> "Live | _PlainNoOpContext":
        """Create a progress display (no-op in plain mode)"""
        if self.plain:
            return _PlainNoOpContext()
        return Live(
            Text(f"⏳ {message}", style="yellow"),
            console=self.console,
            refresh_per_second=4,
        )


class _PlainNoOpContext:
    """No-op context manager used in plain mode to skip Rich Live display."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, *args, **kwargs):
        pass
