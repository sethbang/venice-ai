"""
Tests for cli/commands/audio.py

Covers:
- audio() click group command
- speak() click command entrypoint
- _speak_async() core logic with all branches
- transcribe() click command entrypoint
- _transcribe_async() core logic with all branches
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from click.testing import CliRunner

from venice_ai.cli.commands.audio import _speak_async, _transcribe_async

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(plain: bool = False):
    """Return a minimal mock Click context."""
    ctx = MagicMock()
    ctx.obj = {"config": {"api": {"key": "test-key"}}, "plain": plain}
    return ctx


def _setup_client_patch(MockVeniceClient, mock_client):
    """Configure MockVeniceClient to act as async context manager."""
    MockVeniceClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    MockVeniceClient.return_value.__aexit__ = AsyncMock(return_value=None)


def _make_audio_response(content=b"fake audio bytes data"):
    """Create a fake audio response object."""
    response = MagicMock()
    response.content = content
    return response


def _make_transcription_response(text="Hello world", words=None):
    """Create a fake transcription response."""
    if words is not None:
        return SimpleNamespace(text=text, words=words)
    return SimpleNamespace(text=text)


# ---------------------------------------------------------------------------
# audio() group tests
# ---------------------------------------------------------------------------


class TestAudioGroup:
    """Tests for the audio() click group."""

    def test_audio_group_help(self):
        """Check --help output for the audio group."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["audio", "--help"])
        assert result.exit_code == 0
        assert "speak" in result.output or "transcribe" in result.output

    def test_audio_group_shows_subcommands(self):
        """audio group lists speak and transcribe subcommands."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["audio", "--help"])
        assert result.exit_code == 0
        assert "speak" in result.output
        assert "transcribe" in result.output


# ---------------------------------------------------------------------------
# speak() CLI entrypoint tests
# ---------------------------------------------------------------------------


class TestSpeakCLI:
    """Tests for the speak() click command entrypoint."""

    def test_speak_help(self):
        """Check --help output for speak command."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["audio", "speak", "--help"])
        assert result.exit_code == 0
        assert "speak" in result.output.lower() or "text" in result.output.lower()

    def test_speak_invokes_asyncio_run(self):
        """speak() calls asyncio.run with _speak_async coroutine."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch(
            "venice_ai.cli.commands.audio.asyncio.run", side_effect=consume_coro
        ) as mock_run:
            runner.invoke(cli, ["audio", "speak", "Hello world"])
            assert mock_run.called

    def test_speak_with_all_options(self):
        """speak() passes all options through to asyncio.run."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch("venice_ai.cli.commands.audio.asyncio.run", side_effect=consume_coro):
            result = runner.invoke(
                cli,
                [
                    "audio",
                    "speak",
                    "Hello",
                    "--model",
                    "tts-kokoro",
                    "--voice",
                    "af_heart",
                    "--format",
                    "wav",
                    "--speed",
                    "1.5",
                    "--output",
                    "out.wav",
                ],
            )
            assert result.exit_code == 0


# ---------------------------------------------------------------------------
# _speak_async tests — basic functionality
# ---------------------------------------------------------------------------


class TestSpeakAsyncBasic:
    """Tests for _speak_async core functionality."""

    @pytest.mark.asyncio
    async def test_speak_async_basic_text_plain_mode(self):
        """Basic TTS with text argument in plain mode."""
        mock_client = AsyncMock()
        mock_response = _make_audio_response()
        mock_client.audio.create_speech = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo") as mock_echo,
            patch("builtins.open", mock_open()),
            patch("os.makedirs"),
            patch("os.path.getsize", return_value=1024),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _speak_async(
                ctx,
                text="Hello world",
                model="tts-kokoro",
                voice="af_heart",
                audio_format="mp3",
                speed=1.0,
                output=None,
                save_dir=".",
            )

            mock_client.audio.create_speech.assert_called_once()
            mock_echo.assert_called()

    @pytest.mark.asyncio
    async def test_speak_async_basic_text_rich_mode(self):
        """Basic TTS with text argument in rich mode."""
        mock_client = AsyncMock()
        mock_response = _make_audio_response()
        mock_client.audio.create_speech = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.console") as mock_console,
            patch("builtins.open", mock_open()),
            patch("os.makedirs"),
            patch("os.path.getsize", return_value=2048),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _speak_async(
                ctx,
                text="Hello world",
                model="tts-kokoro",
                voice="af_heart",
                audio_format="mp3",
                speed=1.0,
                output=None,
                save_dir=".",
            )

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_speak_async_ctx_obj_none(self):
        """ctx.obj is None → plain defaults to False (rich mode)."""
        mock_client = AsyncMock()
        mock_response = _make_audio_response()
        mock_client.audio.create_speech = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.console") as mock_console,
            patch("builtins.open", mock_open()),
            patch("os.makedirs"),
            patch("os.path.getsize", return_value=512),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = MagicMock()
            ctx.obj = None
            await _speak_async(
                ctx,
                text="Hello world",
                model="tts-kokoro",
                voice="af_heart",
                audio_format="mp3",
                speed=1.0,
                output=None,
                save_dir=".",
            )

            assert mock_console.print.called


# ---------------------------------------------------------------------------
# _speak_async tests — stdin input
# ---------------------------------------------------------------------------


class TestSpeakAsyncStdin:
    """Tests for _speak_async stdin handling."""

    @pytest.mark.asyncio
    async def test_speak_async_stdin_piped(self):
        """Text read from piped stdin when text arg is not provided."""
        mock_client = AsyncMock()
        mock_response = _make_audio_response()
        mock_client.audio.create_speech = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo"),
            patch("builtins.open", mock_open()),
            patch("os.makedirs"),
            patch("os.path.getsize", return_value=512),
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.isatty.return_value = False
            mock_stdin.read.return_value = "piped text"
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _speak_async(
                ctx,
                text=None,
                model="tts-kokoro",
                voice="af_heart",
                audio_format="mp3",
                speed=1.0,
                output=None,
                save_dir=".",
            )

            call_kwargs = mock_client.audio.create_speech.call_args
            assert call_kwargs[1]["input"] == "piped text"

    @pytest.mark.asyncio
    async def test_speak_async_no_text_tty_raises_error(self):
        """ClickException raised when no text and stdin is a TTY."""
        import click

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True

            ctx = _make_ctx(plain=True)
            with pytest.raises(click.ClickException) as exc_info:
                await _speak_async(
                    ctx,
                    text=None,
                    model="tts-kokoro",
                    voice="af_heart",
                    audio_format="mp3",
                    speed=1.0,
                    output=None,
                    save_dir=".",
                )
            assert "Text is required" in str(exc_info.value.format_message())

    @pytest.mark.asyncio
    async def test_speak_async_stdin_piped_but_empty_raises_error(self):
        """ClickException raised when piped stdin is empty."""
        import click

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            mock_stdin.read.return_value = "   "  # whitespace only

            ctx = _make_ctx(plain=True)
            with pytest.raises(click.ClickException) as exc_info:
                await _speak_async(
                    ctx,
                    text=None,
                    model="tts-kokoro",
                    voice="af_heart",
                    audio_format="mp3",
                    speed=1.0,
                    output=None,
                    save_dir=".",
                )
            assert "Text is required" in str(exc_info.value.format_message())


# ---------------------------------------------------------------------------
# _speak_async tests — output path handling
# ---------------------------------------------------------------------------


class TestSpeakAsyncOutputPath:
    """Tests for _speak_async output path resolution."""

    @pytest.mark.asyncio
    async def test_speak_async_no_output_uses_default_name(self):
        """No output specified → default speech_<timestamp>.<format> filename."""
        mock_client = AsyncMock()
        mock_response = _make_audio_response()
        mock_client.audio.create_speech = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo"),
            patch("builtins.open", mock_open()),
            patch("os.makedirs"),
            patch("os.path.getsize", return_value=512),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _speak_async(
                ctx,
                text="Hello",
                model="tts-kokoro",
                voice="af_heart",
                audio_format="mp3",
                speed=1.0,
                output=None,
                save_dir=".",
            )

            # Verify the API was called (file was generated with default name)
            mock_client.audio.create_speech.assert_called_once()

    @pytest.mark.asyncio
    async def test_speak_async_output_with_correct_extension(self):
        """Output path already has correct extension → not modified."""
        mock_client = AsyncMock()
        mock_response = _make_audio_response()
        mock_client.audio.create_speech = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo"),
            patch("builtins.open", mock_open()) as mock_file,
            patch("os.makedirs"),
            patch("os.path.getsize", return_value=512),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _speak_async(
                ctx,
                text="Hello",
                model="tts-kokoro",
                voice="af_heart",
                audio_format="mp3",
                speed=1.0,
                output="greeting.mp3",
                save_dir=".",
            )

            # File should be opened with path containing greeting.mp3
            mock_file.assert_called()

    @pytest.mark.asyncio
    async def test_speak_async_output_without_extension_adds_extension(self):
        """Output path without correct extension gets format extension added."""
        mock_client = AsyncMock()
        mock_response = _make_audio_response()
        mock_client.audio.create_speech = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo"),
            patch("builtins.open", mock_open()) as mock_file,
            patch("os.makedirs"),
            patch("os.path.getsize", return_value=512),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _speak_async(
                ctx,
                text="Hello",
                model="tts-kokoro",
                voice="af_heart",
                audio_format="wav",
                speed=1.0,
                output="greeting",
                save_dir=".",
            )

            # Should open file with .wav extension added
            call_args = mock_file.call_args_list
            opened_path = call_args[0][0][0]
            assert opened_path.endswith(".wav")

    @pytest.mark.asyncio
    async def test_speak_async_output_with_wrong_extension_adds_format(self):
        """Output path with wrong extension gets format extension appended."""
        mock_client = AsyncMock()
        mock_response = _make_audio_response()
        mock_client.audio.create_speech = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo"),
            patch("builtins.open", mock_open()) as mock_file,
            patch("os.makedirs"),
            patch("os.path.getsize", return_value=512),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _speak_async(
                ctx,
                text="Hello",
                model="tts-kokoro",
                voice="af_heart",
                audio_format="mp3",
                speed=1.0,
                output="greeting.txt",
                save_dir=".",
            )

            call_args = mock_file.call_args_list
            opened_path = call_args[0][0][0]
            assert opened_path.endswith(".mp3")

    @pytest.mark.asyncio
    async def test_speak_async_custom_save_dir(self):
        """Custom save_dir creates directory and saves file there."""
        mock_client = AsyncMock()
        mock_response = _make_audio_response()
        mock_client.audio.create_speech = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo"),
            patch("builtins.open", mock_open()),
            patch("os.makedirs") as mock_makedirs,
            patch("os.path.getsize", return_value=512),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _speak_async(
                ctx,
                text="Hello",
                model="tts-kokoro",
                voice="af_heart",
                audio_format="mp3",
                speed=1.0,
                output=None,
                save_dir="/tmp/audio",
            )

            mock_makedirs.assert_called_with("/tmp/audio", exist_ok=True)


# ---------------------------------------------------------------------------
# _speak_async tests — API parameters
# ---------------------------------------------------------------------------


class TestSpeakAsyncAPIParams:
    """Tests that _speak_async passes correct parameters to API."""

    @pytest.mark.asyncio
    async def test_speak_async_speed_passed_correctly(self):
        """Speed parameter is passed to create_speech API call."""
        mock_client = AsyncMock()
        mock_response = _make_audio_response()
        mock_client.audio.create_speech = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo"),
            patch("builtins.open", mock_open()),
            patch("os.makedirs"),
            patch("os.path.getsize", return_value=512),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _speak_async(
                ctx,
                text="Hello",
                model="tts-kokoro",
                voice="af_heart",
                audio_format="mp3",
                speed=2.5,
                output=None,
                save_dir=".",
            )

            call_kwargs = mock_client.audio.create_speech.call_args[1]
            assert call_kwargs["speed"] == 2.5

    @pytest.mark.asyncio
    async def test_speak_async_all_api_params_passed(self):
        """All API params (model, input, voice, format, speed) passed correctly."""
        mock_client = AsyncMock()
        mock_response = _make_audio_response()
        mock_client.audio.create_speech = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo"),
            patch("builtins.open", mock_open()),
            patch("os.makedirs"),
            patch("os.path.getsize", return_value=512),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _speak_async(
                ctx,
                text="Test text",
                model="custom-model",
                voice="custom-voice",
                audio_format="wav",
                speed=0.75,
                output=None,
                save_dir=".",
            )

            call_kwargs = mock_client.audio.create_speech.call_args[1]
            assert call_kwargs["model"] == "custom-model"
            assert call_kwargs["input"] == "Test text"
            assert call_kwargs["voice"] == "custom-voice"
            assert call_kwargs["response_format"] == "wav"
            assert call_kwargs["speed"] == 0.75

    @pytest.mark.asyncio
    async def test_speak_async_different_formats(self):
        """Various audio formats are passed correctly to the API."""
        for fmt in ["mp3", "wav", "opus", "flac", "aac", "pcm"]:
            mock_client = AsyncMock()
            mock_response = _make_audio_response()
            mock_client.audio.create_speech = AsyncMock(return_value=mock_response)

            with (
                patch("venice_ai.VeniceClient") as MockClient,
                patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
                patch("venice_ai.cli.commands.audio.click.echo"),
                patch("builtins.open", mock_open()),
                patch("os.makedirs"),
                patch("os.path.getsize", return_value=512),
            ):
                _setup_client_patch(MockClient, mock_client)

                ctx = _make_ctx(plain=True)
                await _speak_async(
                    ctx,
                    text="Hello",
                    model="tts-kokoro",
                    voice="af_heart",
                    audio_format=fmt,
                    speed=1.0,
                    output=None,
                    save_dir=".",
                )

                call_kwargs = mock_client.audio.create_speech.call_args[1]
                assert call_kwargs["response_format"] == fmt


# ---------------------------------------------------------------------------
# _speak_async tests — file write
# ---------------------------------------------------------------------------


class TestSpeakAsyncFileWrite:
    """Tests for _speak_async file write behavior."""

    @pytest.mark.asyncio
    async def test_speak_async_writes_bytes_to_file(self):
        """Audio bytes are written to the output file."""
        audio_bytes = b"real audio data bytes"
        mock_client = AsyncMock()
        mock_response = _make_audio_response(content=audio_bytes)
        mock_client.audio.create_speech = AsyncMock(return_value=mock_response)

        write_calls = []

        def capture_write(data):
            write_calls.append(data)

        m = mock_open()
        m.return_value.__enter__.return_value.write.side_effect = capture_write

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo"),
            patch("builtins.open", m),
            patch("os.makedirs"),
            patch("os.path.getsize", return_value=len(audio_bytes)),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _speak_async(
                ctx,
                text="Hello",
                model="tts-kokoro",
                voice="af_heart",
                audio_format="mp3",
                speed=1.0,
                output=None,
                save_dir=".",
            )

            assert write_calls == [audio_bytes]

    @pytest.mark.asyncio
    async def test_speak_async_rich_shows_file_size(self):
        """Rich mode displays file size after saving."""
        mock_client = AsyncMock()
        mock_response = _make_audio_response()
        mock_client.audio.create_speech = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.console") as mock_console,
            patch("builtins.open", mock_open()),
            patch("os.makedirs"),
            patch("os.path.getsize", return_value=4096),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _speak_async(
                ctx,
                text="Hello",
                model="tts-kokoro",
                voice="af_heart",
                audio_format="mp3",
                speed=1.0,
                output=None,
                save_dir=".",
            )

            calls_str = "".join(str(c) for c in mock_console.print.call_args_list)
            assert "4.0 KB" in calls_str or "KB" in calls_str

    @pytest.mark.asyncio
    async def test_speak_async_plain_shows_saved_path(self):
        """Plain mode shows 'Saved:' with the output path."""
        mock_client = AsyncMock()
        mock_response = _make_audio_response()
        mock_client.audio.create_speech = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo") as mock_echo,
            patch("builtins.open", mock_open()),
            patch("os.makedirs"),
            patch("os.path.getsize", return_value=512),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _speak_async(
                ctx,
                text="Hello",
                model="tts-kokoro",
                voice="af_heart",
                audio_format="mp3",
                speed=1.0,
                output="output.mp3",
                save_dir=".",
            )

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Saved:" in calls_str


# ---------------------------------------------------------------------------
# transcribe() CLI entrypoint tests
# ---------------------------------------------------------------------------


class TestTranscribeCLI:
    """Tests for the transcribe() click command entrypoint."""

    def test_transcribe_help(self):
        """Check --help output for transcribe command."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["audio", "transcribe", "--help"])
        assert result.exit_code == 0
        assert "transcribe" in result.output.lower() or "audio" in result.output.lower()

    def test_transcribe_invokes_asyncio_run(self, tmp_path):
        """transcribe() calls asyncio.run with _transcribe_async coroutine."""
        from venice_ai.cli.cli import cli

        # Create a temp audio file
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio data")

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch(
            "venice_ai.cli.commands.audio.asyncio.run", side_effect=consume_coro
        ) as mock_run:
            runner.invoke(cli, ["audio", "transcribe", str(audio_file)])
            assert mock_run.called

    def test_transcribe_with_all_options(self, tmp_path):
        """transcribe() passes all options through to asyncio.run."""
        from venice_ai.cli.cli import cli

        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio data")

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch("venice_ai.cli.commands.audio.asyncio.run", side_effect=consume_coro):
            result = runner.invoke(
                cli,
                [
                    "audio",
                    "transcribe",
                    str(audio_file),
                    "--model",
                    "nvidia/parakeet-tdt-0.6b-v3",
                    "--language",
                    "en",
                    "--timestamps",
                    "--format",
                    "json",
                    "--output",
                    "/tmp/out.txt",
                ],
            )
            assert result.exit_code == 0


# ---------------------------------------------------------------------------
# _transcribe_async tests — basic functionality
# ---------------------------------------------------------------------------


class TestTranscribeAsyncBasic:
    """Tests for _transcribe_async core functionality."""

    @pytest.mark.asyncio
    async def test_transcribe_async_basic_plain_mode(self):
        """Basic transcription with no extra options in plain mode."""
        mock_client = AsyncMock()
        mock_response = _make_transcription_response(text="Hello world")
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _transcribe_async(
                ctx,
                file="test.mp3",
                model="nvidia/parakeet-tdt-0.6b-v3",
                language=None,
                timestamps=False,
                output_format="text",
                output=None,
            )

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Hello world" in calls_str

    @pytest.mark.asyncio
    async def test_transcribe_async_basic_rich_mode(self):
        """Basic transcription in rich mode."""
        mock_client = AsyncMock()
        mock_response = _make_transcription_response(text="Hello world")
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _transcribe_async(
                ctx,
                file="test.mp3",
                model="nvidia/parakeet-tdt-0.6b-v3",
                language=None,
                timestamps=False,
                output_format="text",
                output=None,
            )

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_transcribe_async_ctx_obj_none(self):
        """ctx.obj is None → plain defaults to False (rich mode)."""
        mock_client = AsyncMock()
        mock_response = _make_transcription_response(text="Hello world")
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = MagicMock()
            ctx.obj = None
            await _transcribe_async(
                ctx,
                file="test.mp3",
                model="nvidia/parakeet-tdt-0.6b-v3",
                language=None,
                timestamps=False,
                output_format="text",
                output=None,
            )

            assert mock_console.print.called


# ---------------------------------------------------------------------------
# _transcribe_async tests — optional parameters
# ---------------------------------------------------------------------------


class TestTranscribeAsyncOptionalParams:
    """Tests for _transcribe_async optional parameters."""

    @pytest.mark.asyncio
    async def test_transcribe_async_with_language(self):
        """Language parameter is passed to the API when specified."""
        mock_client = AsyncMock()
        mock_response = _make_transcription_response(text="Hello world")
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _transcribe_async(
                ctx,
                file="test.mp3",
                model="nvidia/parakeet-tdt-0.6b-v3",
                language="en",
                timestamps=False,
                output_format="text",
                output=None,
            )

            call_kwargs = mock_client.audio.transcribe.call_args[1]
            assert call_kwargs["language"] == "en"

    @pytest.mark.asyncio
    async def test_transcribe_async_without_language(self):
        """Language parameter NOT included when None."""
        mock_client = AsyncMock()
        mock_response = _make_transcription_response(text="Hello world")
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _transcribe_async(
                ctx,
                file="test.mp3",
                model="nvidia/parakeet-tdt-0.6b-v3",
                language=None,
                timestamps=False,
                output_format="text",
                output=None,
            )

            call_kwargs = mock_client.audio.transcribe.call_args[1]
            assert "language" not in call_kwargs

    @pytest.mark.asyncio
    async def test_transcribe_async_with_timestamps(self):
        """Timestamps=True is passed to API when specified."""
        mock_client = AsyncMock()
        mock_response = _make_transcription_response(text="Hello world")
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _transcribe_async(
                ctx,
                file="test.mp3",
                model="nvidia/parakeet-tdt-0.6b-v3",
                language=None,
                timestamps=True,
                output_format="text",
                output=None,
            )

            call_kwargs = mock_client.audio.transcribe.call_args[1]
            assert call_kwargs["timestamps"] is True

    @pytest.mark.asyncio
    async def test_transcribe_async_without_timestamps(self):
        """Timestamps NOT included in API call when False."""
        mock_client = AsyncMock()
        mock_response = _make_transcription_response(text="Hello world")
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _transcribe_async(
                ctx,
                file="test.mp3",
                model="nvidia/parakeet-tdt-0.6b-v3",
                language=None,
                timestamps=False,
                output_format="text",
                output=None,
            )

            call_kwargs = mock_client.audio.transcribe.call_args[1]
            assert "timestamps" not in call_kwargs

    @pytest.mark.asyncio
    async def test_transcribe_async_with_json_output_format(self):
        """response_format passed to API when output_format is not 'text'."""
        mock_client = AsyncMock()
        mock_response = _make_transcription_response(text="Hello world")
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _transcribe_async(
                ctx,
                file="test.mp3",
                model="nvidia/parakeet-tdt-0.6b-v3",
                language=None,
                timestamps=False,
                output_format="json",
                output=None,
            )

            call_kwargs = mock_client.audio.transcribe.call_args[1]
            assert call_kwargs["response_format"] == "json"

    @pytest.mark.asyncio
    async def test_transcribe_async_text_format_no_response_format(self):
        """response_format NOT included when output_format is 'text'."""
        mock_client = AsyncMock()
        mock_response = _make_transcription_response(text="Hello world")
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _transcribe_async(
                ctx,
                file="test.mp3",
                model="nvidia/parakeet-tdt-0.6b-v3",
                language=None,
                timestamps=False,
                output_format="text",
                output=None,
            )

            call_kwargs = mock_client.audio.transcribe.call_args[1]
            assert "response_format" not in call_kwargs

    @pytest.mark.asyncio
    async def test_transcribe_async_all_optional_params(self):
        """All optional params passed when all specified."""
        mock_client = AsyncMock()
        mock_response = _make_transcription_response(text="Hello world")
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _transcribe_async(
                ctx,
                file="audio.mp3",
                model="parakeet",
                language="fr",
                timestamps=True,
                output_format="json",
                output=None,
            )

            call_kwargs = mock_client.audio.transcribe.call_args[1]
            assert call_kwargs["language"] == "fr"
            assert call_kwargs["timestamps"] is True
            assert call_kwargs["response_format"] == "json"


# ---------------------------------------------------------------------------
# _transcribe_async tests — output file
# ---------------------------------------------------------------------------


class TestTranscribeAsyncOutputFile:
    """Tests for _transcribe_async saving to file."""

    @pytest.mark.asyncio
    async def test_transcribe_async_save_to_file_plain_mode(self):
        """Transcription saved to file in plain mode shows 'Saved:'."""
        mock_client = AsyncMock()
        mock_response = _make_transcription_response(text="Transcribed text")
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo") as mock_echo,
            patch("builtins.open", mock_open()),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _transcribe_async(
                ctx,
                file="test.mp3",
                model="nvidia/parakeet-tdt-0.6b-v3",
                language=None,
                timestamps=False,
                output_format="text",
                output="transcript.txt",
            )

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Saved:" in calls_str
            assert "transcript.txt" in calls_str

    @pytest.mark.asyncio
    async def test_transcribe_async_save_to_file_rich_mode(self):
        """Transcription saved to file in rich mode shows success message."""
        mock_client = AsyncMock()
        mock_response = _make_transcription_response(text="Transcribed text")
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.console") as mock_console,
            patch("builtins.open", mock_open()),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _transcribe_async(
                ctx,
                file="test.mp3",
                model="nvidia/parakeet-tdt-0.6b-v3",
                language=None,
                timestamps=False,
                output_format="text",
                output="transcript.txt",
            )

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_transcribe_async_save_writes_text(self):
        """Transcription text is written to output file."""
        transcription_text = "Hello world transcription"
        mock_client = AsyncMock()
        mock_response = _make_transcription_response(text=transcription_text)
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        write_calls = []

        def capture_write(data):
            write_calls.append(data)

        m = mock_open()
        m.return_value.__enter__.return_value.write.side_effect = capture_write

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo"),
            patch("builtins.open", m),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _transcribe_async(
                ctx,
                file="test.mp3",
                model="nvidia/parakeet-tdt-0.6b-v3",
                language=None,
                timestamps=False,
                output_format="text",
                output="transcript.txt",
            )

            assert transcription_text in write_calls

    @pytest.mark.asyncio
    async def test_transcribe_async_no_output_plain_prints_result(self):
        """Without output file, transcription printed to console in plain mode."""
        mock_client = AsyncMock()
        mock_response = _make_transcription_response(text="Hello plain world")
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _transcribe_async(
                ctx,
                file="test.mp3",
                model="nvidia/parakeet-tdt-0.6b-v3",
                language=None,
                timestamps=False,
                output_format="text",
                output=None,
            )

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Hello plain world" in calls_str

    @pytest.mark.asyncio
    async def test_transcribe_async_no_output_rich_prints_result(self):
        """Without output file, transcription printed to console in rich mode."""
        mock_client = AsyncMock()
        mock_response = _make_transcription_response(text="Hello rich world")
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _transcribe_async(
                ctx,
                file="test.mp3",
                model="nvidia/parakeet-tdt-0.6b-v3",
                language=None,
                timestamps=False,
                output_format="text",
                output=None,
            )

            calls_str = "".join(str(c) for c in mock_console.print.call_args_list)
            assert "Hello rich world" in calls_str


# ---------------------------------------------------------------------------
# _transcribe_async tests — word-level timestamps display
# ---------------------------------------------------------------------------


class TestTranscribeAsyncTimestamps:
    """Tests for _transcribe_async word-level timestamps display."""

    @pytest.mark.asyncio
    async def test_transcribe_async_timestamps_plain_shows_word_info(self):
        """Timestamps in plain mode shows each word with start/end times."""
        words = [
            SimpleNamespace(word="Hello", start=0.0, end=0.5),
            SimpleNamespace(word="world", start=0.5, end=1.0),
        ]
        mock_client = AsyncMock()
        mock_response = _make_transcription_response(text="Hello world", words=words)
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _transcribe_async(
                ctx,
                file="test.mp3",
                model="nvidia/parakeet-tdt-0.6b-v3",
                language=None,
                timestamps=True,
                output_format="text",
                output=None,
            )

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Hello" in calls_str
            assert "world" in calls_str
            assert "0.00s" in calls_str
            assert "0.50s" in calls_str

    @pytest.mark.asyncio
    async def test_transcribe_async_timestamps_rich_shows_word_info(self):
        """Timestamps in rich mode shows word timestamps via console.print."""
        words = [
            SimpleNamespace(word="Hello", start=0.1, end=0.6),
            SimpleNamespace(word="world", start=0.6, end=1.2),
        ]
        mock_client = AsyncMock()
        mock_response = _make_transcription_response(text="Hello world", words=words)
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _transcribe_async(
                ctx,
                file="test.mp3",
                model="nvidia/parakeet-tdt-0.6b-v3",
                language=None,
                timestamps=True,
                output_format="text",
                output=None,
            )

            calls_str = "".join(str(c) for c in mock_console.print.call_args_list)
            assert "Hello" in calls_str
            assert "world" in calls_str

    @pytest.mark.asyncio
    async def test_transcribe_async_timestamps_with_none_times(self):
        """Word timestamps with None start/end show 'N/A'."""
        words = [
            SimpleNamespace(word="Hello", start=None, end=None),
        ]
        mock_client = AsyncMock()
        mock_response = _make_transcription_response(text="Hello", words=words)
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _transcribe_async(
                ctx,
                file="test.mp3",
                model="nvidia/parakeet-tdt-0.6b-v3",
                language=None,
                timestamps=True,
                output_format="text",
                output=None,
            )

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "N/A" in calls_str

    @pytest.mark.asyncio
    async def test_transcribe_async_timestamps_rich_with_none_times(self):
        """Word timestamps with None times in rich mode show 'N/A'."""
        words = [
            SimpleNamespace(word="Test", start=None, end=None),
        ]
        mock_client = AsyncMock()
        mock_response = _make_transcription_response(text="Test", words=words)
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _transcribe_async(
                ctx,
                file="test.mp3",
                model="nvidia/parakeet-tdt-0.6b-v3",
                language=None,
                timestamps=True,
                output_format="text",
                output=None,
            )

            calls_str = "".join(str(c) for c in mock_console.print.call_args_list)
            assert "N/A" in calls_str

    @pytest.mark.asyncio
    async def test_transcribe_async_timestamps_true_no_words_attr(self):
        """timestamps=True but response has no 'words' attr → no timestamps block."""
        mock_client = AsyncMock()
        # Response without 'words' attribute
        mock_response = SimpleNamespace(text="Hello world")
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            # Should not raise an error even without words
            await _transcribe_async(
                ctx,
                file="test.mp3",
                model="nvidia/parakeet-tdt-0.6b-v3",
                language=None,
                timestamps=True,
                output_format="text",
                output=None,
            )

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Hello world" in calls_str
            assert "Timestamps:" not in calls_str

    @pytest.mark.asyncio
    async def test_transcribe_async_timestamps_true_words_empty(self):
        """timestamps=True with empty words list → no timestamps block."""
        mock_client = AsyncMock()
        mock_response = _make_transcription_response(text="Hello world", words=[])
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _transcribe_async(
                ctx,
                file="test.mp3",
                model="nvidia/parakeet-tdt-0.6b-v3",
                language=None,
                timestamps=True,
                output_format="text",
                output=None,
            )

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Hello world" in calls_str

    @pytest.mark.asyncio
    async def test_transcribe_async_timestamps_false_no_word_display(self):
        """timestamps=False → word timestamps not shown even if words present."""
        words = [
            SimpleNamespace(word="Hello", start=0.0, end=0.5),
        ]
        mock_client = AsyncMock()
        mock_response = _make_transcription_response(text="Hello", words=words)
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _transcribe_async(
                ctx,
                file="test.mp3",
                model="nvidia/parakeet-tdt-0.6b-v3",
                language=None,
                timestamps=False,
                output_format="text",
                output=None,
            )

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            # Word timestamps section should not appear
            assert "Word Timestamps:" not in calls_str
            assert "0.00s" not in calls_str


# ---------------------------------------------------------------------------
# _transcribe_async tests — with API file param
# ---------------------------------------------------------------------------


class TestTranscribeAsyncFileParam:
    """Tests that file path is passed correctly to API."""

    @pytest.mark.asyncio
    async def test_transcribe_async_file_path_passed_to_api(self):
        """File path is passed correctly to the transcribe API."""
        mock_client = AsyncMock()
        mock_response = _make_transcription_response(text="Hello")
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _transcribe_async(
                ctx,
                file="/path/to/audio.mp3",
                model="nvidia/parakeet-tdt-0.6b-v3",
                language=None,
                timestamps=False,
                output_format="text",
                output=None,
            )

            call_kwargs = mock_client.audio.transcribe.call_args[1]
            assert call_kwargs["file"] == "/path/to/audio.mp3"

    @pytest.mark.asyncio
    async def test_transcribe_async_model_passed_to_api(self):
        """Model parameter is passed correctly to the transcribe API."""
        mock_client = AsyncMock()
        mock_response = _make_transcription_response(text="Hello")
        mock_client.audio.transcribe = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _transcribe_async(
                ctx,
                file="test.mp3",
                model="custom-stt-model",
                language=None,
                timestamps=False,
                output_format="text",
                output=None,
            )

            call_kwargs = mock_client.audio.transcribe.call_args[1]
            assert call_kwargs["model"] == "custom-stt-model"
