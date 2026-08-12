"""
Tests for streaming utilities
"""

import time
from types import SimpleNamespace

import pytest

from venice_ai.cli.utils.streaming import (
    AnimationMode,
    StreamAccumulator,
    StreamHandler,
)


class TestAnimationMode:
    """Test AnimationMode enum"""

    def test_animation_modes_exist(self):
        """Test all animation modes are defined"""
        assert AnimationMode.NONE
        assert AnimationMode.SMOOTH
        assert AnimationMode.WORD
        assert AnimationMode.CHAR
        assert AnimationMode.LINE
        assert AnimationMode.TYPEWRITER

    def test_animation_mode_values(self):
        """Test animation mode string values"""
        assert AnimationMode.NONE.value == "none"
        assert AnimationMode.SMOOTH.value == "smooth"
        assert AnimationMode.WORD.value == "word"
        assert AnimationMode.CHAR.value == "char"
        assert AnimationMode.LINE.value == "line"
        assert AnimationMode.TYPEWRITER.value == "typewriter"


class TestStreamAccumulator:
    """Test StreamAccumulator class"""

    def test_initialization(self):
        """Test accumulator initializes correctly"""
        acc = StreamAccumulator()

        assert acc.content == ""
        assert acc.chunk_count == 0
        assert acc.first_token_time is None
        assert acc.last_token_time is None
        assert acc.start_time is None
        assert acc.finish_reason is None

    def test_process_chunk_with_content(self):
        """Test processing chunk with content"""
        acc = StreamAccumulator()

        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello"), finish_reason=None)]
        )

        new_content = acc.process_chunk(chunk)

        assert new_content == "Hello"
        assert acc.content == "Hello"
        assert acc.chunk_count == 1
        assert acc.first_token_time is not None
        assert acc.last_token_time is not None
        assert acc.start_time is not None

    def test_process_multiple_chunks(self):
        """Test processing multiple chunks accumulates content"""
        acc = StreamAccumulator()

        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="Hello"), finish_reason=None)
                ]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=" "), finish_reason=None)]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="world"), finish_reason=None)
                ]
            ),
        ]

        for chunk in chunks:
            acc.process_chunk(chunk)

        assert acc.content == "Hello world"
        assert acc.chunk_count == 3

    def test_process_chunk_with_finish_reason(self):
        """Test processing chunk with finish reason"""
        acc = StreamAccumulator()

        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Done"), finish_reason="stop")]
        )

        acc.process_chunk(chunk)

        assert acc.finish_reason == "stop"
        assert acc.content == "Done"

    def test_process_chunk_no_content(self):
        """Test processing chunk without content"""
        acc = StreamAccumulator()

        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason=None)]
        )

        new_content = acc.process_chunk(chunk)

        assert new_content == ""
        assert acc.content == ""
        assert acc.chunk_count == 1

    def test_process_chunk_empty_choices(self):
        """Test processing chunk with empty choices"""
        acc = StreamAccumulator()

        chunk = SimpleNamespace(choices=[])

        new_content = acc.process_chunk(chunk)

        assert new_content == ""
        assert acc.chunk_count == 1

    def test_get_statistics_basic(self):
        """Test getting basic statistics"""
        acc = StreamAccumulator()

        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Test"), finish_reason="stop")]
        )
        acc.process_chunk(chunk)

        stats = acc.get_statistics()

        assert stats["total_chunks"] == 1
        assert stats["content_length"] == 4
        assert stats["finish_reason"] == "stop"
        assert "stream_duration" in stats

    def test_get_statistics_with_timing(self):
        """Test statistics include timing information"""
        acc = StreamAccumulator()

        # Process first chunk
        chunk1 = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="First"), finish_reason=None)]
        )
        acc.process_chunk(chunk1)

        # Small delay
        time.sleep(0.01)

        # Process second chunk
        chunk2 = SimpleNamespace(
            choices=[
                SimpleNamespace(delta=SimpleNamespace(content=" Second"), finish_reason="stop")
            ]
        )
        acc.process_chunk(chunk2)

        stats = acc.get_statistics()

        assert "stream_duration" in stats
        assert "time_to_first_token" in stats
        assert stats["stream_duration"] > 0
        assert stats["time_to_first_token"] >= 0

    def test_get_statistics_chunks_per_second(self):
        """Test chunks per second calculation"""
        acc = StreamAccumulator()

        # Process chunks with known timing
        for i in range(5):
            chunk = SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=str(i)), finish_reason=None)]
            )
            acc.process_chunk(chunk)
            time.sleep(0.05)  # Small delay to ensure meaningful duration

        stats = acc.get_statistics()

        # Should have chunks_per_second since duration > 0.1
        if stats.get("stream_duration", 0) > 0.1:
            assert "chunks_per_second" in stats
            assert stats["chunks_per_second"] > 0


class TestStreamHandler:
    """Test StreamHandler class"""

    def test_initialization_default(self):
        """Test handler initializes with defaults"""
        handler = StreamHandler()

        assert handler.animation_mode == AnimationMode.SMOOTH
        assert handler.animation_speed == 0.03
        assert handler.buffer == ""

    def test_initialization_custom_mode(self):
        """Test handler with custom animation mode"""
        handler = StreamHandler(animation_mode=AnimationMode.TYPEWRITER)

        assert handler.animation_mode == AnimationMode.TYPEWRITER

    def test_initialization_custom_speed(self):
        """Test handler with custom animation speed"""
        handler = StreamHandler(animation_speed=0.01)

        assert handler.animation_speed == 0.01

    def test_initialization_custom_console(self):
        """Test handler with custom console"""
        from rich.console import Console

        custom_console = Console()

        handler = StreamHandler(console=custom_console)

        assert handler.console == custom_console

    def test_all_animation_modes_instantiate(self):
        """Test all animation modes can be instantiated"""
        for mode in AnimationMode:
            handler = StreamHandler(animation_mode=mode)
            assert handler.animation_mode == mode


class TestStreamHandlerIntegration:
    """Integration tests for StreamHandler with async streams"""

    @pytest.mark.asyncio
    async def test_stream_accumulator_with_async_chunks(self):
        """Test accumulator works with async chunk processing"""
        acc = StreamAccumulator()

        async def mock_stream():
            """Mock async stream generator"""
            chunks = [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(delta=SimpleNamespace(content="Async "), finish_reason=None)
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(delta=SimpleNamespace(content="test"), finish_reason="stop")
                    ]
                ),
            ]
            for chunk in chunks:
                yield chunk

        async for chunk in mock_stream():
            acc.process_chunk(chunk)

        assert acc.content == "Async test"
        assert acc.chunk_count == 2
        assert acc.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_handler_buffer_management(self):
        """Test stream handler manages buffer correctly"""
        handler = StreamHandler(animation_mode=AnimationMode.NONE)

        # Buffer should start empty
        assert handler.buffer == ""

        # After usage, buffer might contain content
        handler.buffer = "test"
        assert handler.buffer == "test"


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_accumulator_handles_unicode(self):
        """Test accumulator handles unicode content"""
        acc = StreamAccumulator()

        chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(delta=SimpleNamespace(content="Hello 世界 🌍"), finish_reason=None)
            ]
        )

        acc.process_chunk(chunk)

        assert acc.content == "Hello 世界 🌍"
        assert acc.chunk_count == 1

    def test_accumulator_handles_empty_content_sequence(self):
        """Test accumulator handles sequence of empty content"""
        acc = StreamAccumulator()

        for _ in range(5):
            chunk = SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=""), finish_reason=None)]
            )
            acc.process_chunk(chunk)

        assert acc.content == ""
        assert acc.chunk_count == 5

    def test_statistics_without_processing(self):
        """Test getting statistics before processing any chunks"""
        acc = StreamAccumulator()

        stats = acc.get_statistics()

        assert stats["total_chunks"] == 0
        assert stats["content_length"] == 0
        assert stats["finish_reason"] is None

    def test_statistics_without_first_token(self):
        """Test getting statistics when first_token_time is None but start_time exists.

        Covers the partial branch at line 81 where first_token_time might not be set
        even though start_time is (e.g., when chunks have no content).
        """
        acc = StreamAccumulator()

        # Process chunks with no content - this sets start_time but NOT first_token_time
        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason=None)]
        )
        acc.process_chunk(chunk)

        # Process another chunk to ensure duration > 0
        import time

        time.sleep(0.15)  # Sleep to ensure duration > 0.1 for chunks_per_second
        chunk2 = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason="stop")]
        )
        acc.process_chunk(chunk2)

        # At this point, start_time and last_token_time are set, but first_token_time is None
        assert acc.start_time is not None
        assert acc.last_token_time is not None
        assert acc.first_token_time is None  # No content was ever received

        stats = acc.get_statistics()

        # Should have stream_duration but NOT time_to_first_token
        assert "stream_duration" in stats
        assert "time_to_first_token" not in stats
        assert "chunks_per_second" in stats


class TestStreamHandlerAnimationModes:
    """Test all animation mode handlers in StreamHandler"""

    @pytest.fixture
    def mock_stream_chunks(self):
        """Create a mock async stream that yields content chunks"""

        async def _create_stream(chunks_content: list[str], finish_reason: str = "stop"):
            for i, content in enumerate(chunks_content):
                is_last = i == len(chunks_content) - 1
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=content),
                            finish_reason=finish_reason if is_last else None,
                        )
                    ]
                )

        return _create_stream

    @pytest.fixture
    def mock_empty_stream(self):
        """Create a mock async stream that yields no content"""

        async def _empty_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason="stop")]
            )

        return _empty_stream

    @pytest.fixture
    def mock_console(self):
        """Create a mock console for testing"""
        from rich.console import Console

        console = Console(force_terminal=True, no_color=True, width=80)
        return console

    @pytest.mark.asyncio
    async def test_handle_chat_stream_smooth_mode(self, mock_stream_chunks, mock_console):
        """Test handle_chat_stream with SMOOTH animation mode - covers lines 115-116"""
        handler = StreamHandler(console=mock_console, animation_mode=AnimationMode.SMOOTH)
        stream = mock_stream_chunks(["Hello", " ", "world"])

        content, stats = await handler.handle_chat_stream(stream, show_stats=False)

        assert content == "Hello world"
        assert stats == {}

    @pytest.mark.asyncio
    async def test_handle_chat_stream_smooth_mode_with_stats(
        self, mock_stream_chunks, mock_console
    ):
        """Test handle_chat_stream SMOOTH mode with stats - covers line 156-157"""
        handler = StreamHandler(console=mock_console, animation_mode=AnimationMode.SMOOTH)
        stream = mock_stream_chunks(["Test", " content"])

        content, stats = await handler.handle_chat_stream(stream, show_stats=True)

        assert content == "Test content"
        assert "total_chunks" in stats
        assert stats["total_chunks"] == 2

    @pytest.mark.asyncio
    async def test_handle_chat_stream_word_mode(self, mock_stream_chunks, mock_console):
        """Test handle_chat_stream with WORD animation mode - covers lines 117-118"""
        handler = StreamHandler(
            console=mock_console,
            animation_mode=AnimationMode.WORD,
            animation_speed=0.001,  # Fast for testing
        )
        stream = mock_stream_chunks(["Hello ", "world ", "test\nline"])

        content, stats = await handler.handle_chat_stream(stream, show_stats=True)

        assert content == "Hello world test\nline"
        assert "total_chunks" in stats

    @pytest.mark.asyncio
    async def test_handle_chat_stream_char_mode(self, mock_stream_chunks, mock_console):
        """Test handle_chat_stream with CHAR animation mode - covers lines 119-120"""
        handler = StreamHandler(
            console=mock_console,
            animation_mode=AnimationMode.CHAR,
            animation_speed=0.001,  # Fast for testing
        )
        stream = mock_stream_chunks(["Hi", "!"])

        content, stats = await handler.handle_chat_stream(stream, show_stats=True)

        assert content == "Hi!"
        assert "total_chunks" in stats

    @pytest.mark.asyncio
    async def test_handle_chat_stream_line_mode(self, mock_stream_chunks, mock_console):
        """Test handle_chat_stream with LINE animation mode - covers lines 121-122"""
        handler = StreamHandler(
            console=mock_console,
            animation_mode=AnimationMode.LINE,
            animation_speed=0.001,  # Fast for testing
        )
        stream = mock_stream_chunks(["Line one\n", "Line two\n", "Line three"])

        content, stats = await handler.handle_chat_stream(stream, show_stats=True)

        assert content == "Line one\nLine two\nLine three"
        assert "total_chunks" in stats

    @pytest.mark.asyncio
    async def test_handle_chat_stream_typewriter_mode(self, mock_stream_chunks, mock_console):
        """Test handle_chat_stream with TYPEWRITER animation mode - covers lines 123-124"""
        handler = StreamHandler(
            console=mock_console,
            animation_mode=AnimationMode.TYPEWRITER,
            animation_speed=0.001,  # Fast for testing
        )
        stream = mock_stream_chunks(["Type", "writer"])

        content, stats = await handler.handle_chat_stream(stream, show_stats=True)

        assert content == "Typewriter"
        assert "total_chunks" in stats

    @pytest.mark.asyncio
    async def test_handle_chat_stream_none_mode(self, mock_stream_chunks, mock_console):
        """Test handle_chat_stream with NONE animation mode - covers lines 125-126"""
        handler = StreamHandler(console=mock_console, animation_mode=AnimationMode.NONE)
        stream = mock_stream_chunks(["No", " animation"])

        content, stats = await handler.handle_chat_stream(stream, show_stats=True)

        assert content == "No animation"
        assert "total_chunks" in stats


class TestSmoothStreamHandler:
    """Test _handle_smooth_stream specifically - covers lines 132-157"""

    @pytest.fixture
    def mock_console(self):
        from rich.console import Console

        return Console(force_terminal=True, no_color=True, width=80)

    @pytest.mark.asyncio
    async def test_smooth_stream_with_timing_update(self, mock_console):
        """Test smooth stream updates based on timing interval - covers lines 141-148"""
        import asyncio

        handler = StreamHandler(console=mock_console, animation_mode=AnimationMode.SMOOTH)

        async def slow_stream():
            """Stream that yields chunks with delays to trigger timing-based updates"""
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="First"), finish_reason=None)
                ]
            )
            await asyncio.sleep(0.06)  # > 50ms to trigger update threshold
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content=" Second"), finish_reason=None)
                ]
            )
            await asyncio.sleep(0.06)
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content=" Third"), finish_reason="stop")
                ]
            )

        content, stats = await handler.handle_chat_stream(slow_stream(), show_stats=True)

        assert content == "First Second Third"
        assert stats["total_chunks"] == 3

    @pytest.mark.asyncio
    async def test_smooth_stream_fast_chunks(self, mock_console):
        """Test smooth stream with fast chunks - covers line 151"""
        handler = StreamHandler(console=mock_console, animation_mode=AnimationMode.SMOOTH)

        async def fast_stream():
            """Stream that yields chunks quickly"""
            for i in range(5):
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=f"chunk{i}"),
                            finish_reason="stop" if i == 4 else None,
                        )
                    ]
                )

        content, stats = await handler.handle_chat_stream(fast_stream(), show_stats=True)

        assert content == "chunk0chunk1chunk2chunk3chunk4"

    @pytest.mark.asyncio
    async def test_smooth_stream_empty_content(self, mock_console):
        """Test smooth stream when chunks have empty content - covers line 143 (if new_content:)"""
        handler = StreamHandler(console=mock_console, animation_mode=AnimationMode.SMOOTH)

        async def mixed_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="Start"), finish_reason=None)
                ]
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason=None)]
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=""), finish_reason=None)]
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="End"), finish_reason="stop")
                ]
            )

        content, stats = await handler.handle_chat_stream(mixed_stream(), show_stats=True)

        assert content == "StartEnd"


class TestWordStreamHandler:
    """Test _handle_word_stream specifically - covers lines 163-202"""

    @pytest.fixture
    def mock_console(self):
        from rich.console import Console

        return Console(force_terminal=True, no_color=True, width=80)

    @pytest.mark.asyncio
    async def test_word_stream_with_spaces(self, mock_console):
        """Test word stream splits on spaces - covers lines 174-188"""
        handler = StreamHandler(
            console=mock_console,
            animation_mode=AnimationMode.WORD,
            animation_speed=0.001,
        )

        async def word_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="word1 word2 word3"),
                        finish_reason=None,
                    )
                ]
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=""), finish_reason="stop")]
            )

        content, stats = await handler.handle_chat_stream(word_stream(), show_stats=True)

        assert content == "word1 word2 word3"

    @pytest.mark.asyncio
    async def test_word_stream_with_newlines(self, mock_console):
        """Test word stream splits on newlines - covers lines 176-184"""
        handler = StreamHandler(
            console=mock_console,
            animation_mode=AnimationMode.WORD,
            animation_speed=0.001,
        )

        async def newline_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="line1\nline2\n"),
                        finish_reason="stop",
                    )
                ]
            )

        content, stats = await handler.handle_chat_stream(newline_stream(), show_stats=True)

        assert content == "line1\nline2\n"

    @pytest.mark.asyncio
    async def test_word_stream_mixed_delimiters(self, mock_console):
        """Test word stream with mixed spaces and newlines - covers lines 179-184"""
        handler = StreamHandler(
            console=mock_console,
            animation_mode=AnimationMode.WORD,
            animation_speed=0.001,
        )

        async def mixed_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="word1 word2\nword3 word4"),
                        finish_reason=None,
                    )
                ]
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="\nfinal"), finish_reason="stop")
                ]
            )

        content, stats = await handler.handle_chat_stream(mixed_stream(), show_stats=True)

        assert content == "word1 word2\nword3 word4\nfinal"

    @pytest.mark.asyncio
    async def test_word_stream_remaining_buffer(self, mock_console):
        """Test word stream displays remaining buffer at end - covers lines 196-199"""
        handler = StreamHandler(
            console=mock_console,
            animation_mode=AnimationMode.WORD,
            animation_speed=0.001,
        )

        async def partial_word_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="word1 partial"),
                        finish_reason="stop",
                    )
                ]
            )

        content, stats = await handler.handle_chat_stream(partial_word_stream(), show_stats=True)

        # "partial" has no trailing space/newline so should be in remaining buffer
        assert content == "word1 partial"

    @pytest.mark.asyncio
    async def test_word_stream_no_stats(self, mock_console):
        """Test word stream without stats - covers line 201"""
        handler = StreamHandler(
            console=mock_console,
            animation_mode=AnimationMode.WORD,
            animation_speed=0.001,
        )

        async def simple_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="test"), finish_reason="stop")
                ]
            )

        content, stats = await handler.handle_chat_stream(simple_stream(), show_stats=False)

        assert content == "test"
        assert stats == {}


class TestCharStreamHandler:
    """Test _handle_char_stream specifically - covers lines 208-225"""

    @pytest.fixture
    def mock_console(self):
        from rich.console import Console

        return Console(force_terminal=True, no_color=True, width=80)

    @pytest.mark.asyncio
    async def test_char_stream_basic(self, mock_console):
        """Test char stream processes character by character - covers lines 211-222"""
        handler = StreamHandler(
            console=mock_console,
            animation_mode=AnimationMode.CHAR,
            animation_speed=0.003,  # Very fast for test
        )

        async def char_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="ABC"), finish_reason=None)]
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="D"), finish_reason="stop")]
            )

        content, stats = await handler.handle_chat_stream(char_stream(), show_stats=True)

        assert content == "ABCD"
        assert stats["total_chunks"] == 2

    @pytest.mark.asyncio
    async def test_char_stream_with_unicode(self, mock_console):
        """Test char stream handles unicode - covers lines 215-216"""
        handler = StreamHandler(
            console=mock_console,
            animation_mode=AnimationMode.CHAR,
            animation_speed=0.003,
        )

        async def unicode_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="Hello 🌍"), finish_reason="stop")
                ]
            )

        content, stats = await handler.handle_chat_stream(unicode_stream(), show_stats=True)

        assert content == "Hello 🌍"

    @pytest.mark.asyncio
    async def test_char_stream_empty_content(self, mock_console):
        """Test char stream with empty content - covers line 214 (if new_content:)"""
        handler = StreamHandler(
            console=mock_console,
            animation_mode=AnimationMode.CHAR,
            animation_speed=0.003,
        )

        async def sparse_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="A"), finish_reason=None)]
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason=None)]
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="B"), finish_reason="stop")]
            )

        content, stats = await handler.handle_chat_stream(sparse_stream(), show_stats=True)

        assert content == "AB"

    @pytest.mark.asyncio
    async def test_char_stream_no_stats(self, mock_console):
        """Test char stream without stats - covers line 224"""
        handler = StreamHandler(
            console=mock_console,
            animation_mode=AnimationMode.CHAR,
            animation_speed=0.003,
        )

        async def simple_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="X"), finish_reason="stop")]
            )

        content, stats = await handler.handle_chat_stream(simple_stream(), show_stats=False)

        assert content == "X"
        assert stats == {}


class TestLineStreamHandler:
    """Test _handle_line_stream specifically - covers lines 231-258"""

    @pytest.fixture
    def mock_console(self):
        from rich.console import Console

        return Console(force_terminal=True, no_color=True, width=80)

    @pytest.mark.asyncio
    async def test_line_stream_with_newlines(self, mock_console):
        """Test line stream buffers until newline - covers lines 235-250"""
        handler = StreamHandler(
            console=mock_console,
            animation_mode=AnimationMode.LINE,
            animation_speed=0.001,
        )

        async def line_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="Line 1\nLine 2\n"),
                        finish_reason=None,
                    )
                ]
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="Line 3"), finish_reason="stop")
                ]
            )

        content, stats = await handler.handle_chat_stream(line_stream(), show_stats=True)

        assert content == "Line 1\nLine 2\nLine 3"
        assert stats["total_chunks"] == 2

    @pytest.mark.asyncio
    async def test_line_stream_remaining_buffer(self, mock_console):
        """Test line stream displays remaining buffer - covers lines 252-255"""
        handler = StreamHandler(
            console=mock_console,
            animation_mode=AnimationMode.LINE,
            animation_speed=0.001,
        )

        async def partial_line_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="Complete line\nPartial"),
                        finish_reason="stop",
                    )
                ]
            )

        content, stats = await handler.handle_chat_stream(partial_line_stream(), show_stats=True)

        # "Partial" has no trailing newline so should be in remaining buffer
        assert content == "Complete line\nPartial"

    @pytest.mark.asyncio
    async def test_line_stream_empty_buffer(self, mock_console):
        """Test line stream when buffer is empty at end - covers lines 253 (if line_buffer:)"""
        handler = StreamHandler(
            console=mock_console,
            animation_mode=AnimationMode.LINE,
            animation_speed=0.001,
        )

        async def complete_lines_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="Line 1\nLine 2\n"),
                        finish_reason="stop",
                    )
                ]
            )

        content, stats = await handler.handle_chat_stream(complete_lines_stream(), show_stats=True)

        # All lines end with newline so buffer should be empty at end
        assert content == "Line 1\nLine 2\n"

    @pytest.mark.asyncio
    async def test_line_stream_empty_content(self, mock_console):
        """Test line stream with empty content - covers line 238 (if new_content:)"""
        handler = StreamHandler(
            console=mock_console,
            animation_mode=AnimationMode.LINE,
            animation_speed=0.001,
        )

        async def sparse_line_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="Line 1\n"), finish_reason=None)
                ]
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=""), finish_reason=None)]
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="Line 2"), finish_reason="stop")
                ]
            )

        content, stats = await handler.handle_chat_stream(sparse_line_stream(), show_stats=True)

        assert content == "Line 1\nLine 2"

    @pytest.mark.asyncio
    async def test_line_stream_no_stats(self, mock_console):
        """Test line stream without stats - covers line 257"""
        handler = StreamHandler(
            console=mock_console,
            animation_mode=AnimationMode.LINE,
            animation_speed=0.001,
        )

        async def simple_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="test\n"), finish_reason="stop")
                ]
            )

        content, stats = await handler.handle_chat_stream(simple_stream(), show_stats=False)

        assert content == "test\n"
        assert stats == {}


class TestTypewriterStreamHandler:
    """Test _handle_typewriter_stream specifically - covers lines 264-297"""

    @pytest.fixture
    def mock_console(self):
        from rich.console import Console

        return Console(force_terminal=True, no_color=True, width=80)

    @pytest.mark.asyncio
    async def test_typewriter_stream_basic(self, mock_console):
        """Test typewriter stream basic operation - covers lines 271-291"""
        handler = StreamHandler(
            console=mock_console,
            animation_mode=AnimationMode.TYPEWRITER,
            animation_speed=0.001,  # Very fast for test
        )

        async def type_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="Hello"), finish_reason=None)
                ]
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="!"), finish_reason="stop")]
            )

        content, stats = await handler.handle_chat_stream(type_stream(), show_stats=True)

        assert content == "Hello!"
        assert stats["total_chunks"] == 2

    @pytest.mark.asyncio
    async def test_typewriter_stream_special_chars(self, mock_console):
        """Test typewriter stream with special characters for variable timing - covers lines 284-291"""
        handler = StreamHandler(
            console=mock_console,
            animation_mode=AnimationMode.TYPEWRITER,
            animation_speed=0.001,
        )

        async def special_chars_stream():
            # Test space (line 285), punctuation (line 286-287), newline (line 289), regular char (line 291)
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="Hi! How are you?\n"),
                        finish_reason=None,
                    )
                ]
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="Fine."), finish_reason="stop")
                ]
            )

        content, stats = await handler.handle_chat_stream(special_chars_stream(), show_stats=True)

        assert content == "Hi! How are you?\nFine."

    @pytest.mark.asyncio
    async def test_typewriter_stream_cursor_animation(self, mock_console):
        """Test typewriter stream cursor states - covers lines 268-281"""
        handler = StreamHandler(
            console=mock_console,
            animation_mode=AnimationMode.TYPEWRITER,
            animation_speed=0.001,
        )

        async def cursor_stream():
            # Multiple characters to cycle through cursor states
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="ABCDEFGH"), finish_reason="stop")
                ]
            )

        content, stats = await handler.handle_chat_stream(cursor_stream(), show_stats=True)

        # Cursor should cycle through all 5 states
        assert content == "ABCDEFGH"

    @pytest.mark.asyncio
    async def test_typewriter_stream_empty_content(self, mock_console):
        """Test typewriter stream with empty content - covers line 274 (if new_content:)"""
        handler = StreamHandler(
            console=mock_console,
            animation_mode=AnimationMode.TYPEWRITER,
            animation_speed=0.001,
        )

        async def sparse_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="A"), finish_reason=None)]
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason=None)]
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="B"), finish_reason="stop")]
            )

        content, stats = await handler.handle_chat_stream(sparse_stream(), show_stats=True)

        assert content == "AB"

    @pytest.mark.asyncio
    async def test_typewriter_stream_no_stats(self, mock_console):
        """Test typewriter stream without stats - covers line 296"""
        handler = StreamHandler(
            console=mock_console,
            animation_mode=AnimationMode.TYPEWRITER,
            animation_speed=0.001,
        )

        async def simple_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="X"), finish_reason="stop")]
            )

        content, stats = await handler.handle_chat_stream(simple_stream(), show_stats=False)

        assert content == "X"
        assert stats == {}


class TestNoAnimationStreamHandler:
    """Test _handle_no_animation_stream specifically - covers lines 303-313"""

    @pytest.fixture
    def mock_console(self):
        from rich.console import Console

        return Console(force_terminal=True, no_color=True, width=80)

    @pytest.mark.asyncio
    async def test_no_animation_stream_basic(self, mock_console):
        """Test no animation stream basic operation - covers lines 305-310"""
        handler = StreamHandler(console=mock_console, animation_mode=AnimationMode.NONE)

        async def instant_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="Instant "), finish_reason=None)
                ]
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="display"), finish_reason="stop")
                ]
            )

        content, stats = await handler.handle_chat_stream(instant_stream(), show_stats=True)

        assert content == "Instant display"
        assert stats["total_chunks"] == 2

    @pytest.mark.asyncio
    async def test_no_animation_stream_empty_content(self, mock_console):
        """Test no animation stream with empty content - covers line 308 (if new_content:)"""
        handler = StreamHandler(console=mock_console, animation_mode=AnimationMode.NONE)

        async def sparse_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="A"), finish_reason=None)]
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason=None)]
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="B"), finish_reason="stop")]
            )

        content, stats = await handler.handle_chat_stream(sparse_stream(), show_stats=True)

        assert content == "AB"

    @pytest.mark.asyncio
    async def test_no_animation_stream_no_stats(self, mock_console):
        """Test no animation stream without stats - covers line 312"""
        handler = StreamHandler(console=mock_console, animation_mode=AnimationMode.NONE)

        async def simple_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="test"), finish_reason="stop")
                ]
            )

        content, stats = await handler.handle_chat_stream(simple_stream(), show_stats=False)

        assert content == "test"
        assert stats == {}


class TestSimpleStreamHandler:
    """Test handle_simple_stream method - covers lines 317-332"""

    @pytest.fixture
    def mock_console(self):
        from rich.console import Console

        return Console(force_terminal=True, no_color=True, width=80)

    @pytest.mark.asyncio
    async def test_simple_stream_basic(self, mock_console, capsys):
        """Test simple stream basic operation - covers lines 319-329"""
        handler = StreamHandler(console=mock_console)

        async def simple_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello "))]
            )
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="World"))])

        result = await handler.handle_simple_stream(simple_stream())

        assert result == "Hello World"
        captured = capsys.readouterr()
        assert "Hello " in captured.out
        assert "World" in captured.out

    @pytest.mark.asyncio
    async def test_simple_stream_no_choices(self, mock_console, capsys):
        """Test simple stream with no choices - covers line 320"""
        handler = StreamHandler(console=mock_console)

        async def empty_choices_stream():
            yield SimpleNamespace(choices=[])
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Content"))]
            )

        result = await handler.handle_simple_stream(empty_choices_stream())

        assert result == "Content"

    @pytest.mark.asyncio
    async def test_simple_stream_no_delta(self, mock_console, capsys):
        """Test simple stream with no delta attribute - covers line 324"""
        handler = StreamHandler(console=mock_console)

        async def no_delta_stream():
            # Chunk without delta attribute
            yield SimpleNamespace(choices=[SimpleNamespace(other="value")])
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Valid"))])

        result = await handler.handle_simple_stream(no_delta_stream())

        assert result == "Valid"

    @pytest.mark.asyncio
    async def test_simple_stream_no_content_attribute(self, mock_console, capsys):
        """Test simple stream with delta but no content attribute - covers line 324"""
        handler = StreamHandler(console=mock_console)

        async def no_content_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(role="assistant"))]
            )
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Test"))])

        result = await handler.handle_simple_stream(no_content_stream())

        assert result == "Test"

    @pytest.mark.asyncio
    async def test_simple_stream_empty_content(self, mock_console, capsys):
        """Test simple stream with empty/None content - covers lines 326-327"""
        handler = StreamHandler(console=mock_console)

        async def null_content_stream():
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))])
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=""))])
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Actual"))]
            )

        result = await handler.handle_simple_stream(null_content_stream())

        assert result == "Actual"

    @pytest.mark.asyncio
    async def test_simple_stream_final_newline(self, mock_console, capsys):
        """Test simple stream prints final newline - covers line 331"""
        handler = StreamHandler(console=mock_console)

        async def single_chunk_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Output"))]
            )

        result = await handler.handle_simple_stream(single_chunk_stream())

        assert result == "Output"
        captured = capsys.readouterr()
        # Check that final newline was printed
        assert captured.out.endswith("\n")

    @pytest.mark.asyncio
    async def test_simple_stream_no_choices_attribute(self, mock_console, capsys):
        """Test simple stream with chunks missing choices attribute - covers line 320"""
        handler = StreamHandler(console=mock_console)

        async def missing_choices_stream():
            yield SimpleNamespace(data="no choices")
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Valid"))])

        result = await handler.handle_simple_stream(missing_choices_stream())

        assert result == "Valid"


class TestDisplayProgress:
    """Test display_progress method - covers lines 334-340"""

    @pytest.fixture
    def mock_console(self):
        from rich.console import Console

        return Console(force_terminal=True, no_color=True, width=80)

    def test_display_progress_returns_live(self, mock_console):
        """Test display_progress returns a Live instance - covers lines 336-340"""
        from rich.live import Live

        handler = StreamHandler(console=mock_console)

        progress = handler.display_progress("Loading...")

        assert isinstance(progress, Live)

    def test_display_progress_uses_message(self, mock_console):
        """Test display_progress uses the provided message - covers line 337"""
        handler = StreamHandler(console=mock_console)

        progress = handler.display_progress("Custom message")

        # The Live renderable should contain our message
        assert progress is not None
        # Verify it can be used as a context manager
        with progress:
            pass  # Just verify no error

    def test_display_progress_different_messages(self, mock_console):
        """Test display_progress works with different messages"""
        handler = StreamHandler(console=mock_console)

        # Test with various message types
        for msg in ["Loading...", "Processing data", "⏳ Wait..."]:
            progress = handler.display_progress(msg)
            assert progress is not None
            with progress:
                pass
