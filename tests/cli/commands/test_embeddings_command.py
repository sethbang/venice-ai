"""
Tests for cli/commands/embeddings.py

Covers:
- embeddings() click command entrypoint
- _embeddings_async() core logic with all branches:
  - Text provided as argument
  - Text via stdin (piped)
  - No text and stdin is TTY (error)
  - JSON output mode (save to file / print to stdout)
  - Non-JSON output (save to file / display summary)
  - With/without dimensions parameter
  - Rich mode vs plain mode
  - Embedding preview (> 5 values shows "..." truncation, <= 5 shows all)
  - Multiple embeddings in response
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from click.testing import CliRunner

from venice_ai.cli.commands.embeddings import _embeddings_async

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


def _make_embedding_response(
    embedding=None,
    model="text-embedding-ada-002",
    object_type="list",
    prompt_tokens=5,
    total_tokens=5,
):
    """Create a mock embeddings response with a single embedding."""
    if embedding is None:
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    return SimpleNamespace(
        data=[
            SimpleNamespace(
                embedding=embedding,
                index=0,
                object="embedding",
            )
        ],
        model=model,
        object=object_type,
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, total_tokens=total_tokens),
    )


def _make_multi_embedding_response(embeddings_list, model="text-embedding-ada-002"):
    """Create a mock response with multiple embeddings."""
    data = [
        SimpleNamespace(
            embedding=emb,
            index=idx,
            object="embedding",
        )
        for idx, emb in enumerate(embeddings_list)
    ]
    return SimpleNamespace(
        data=data,
        model=model,
        object="list",
        usage=SimpleNamespace(prompt_tokens=10, total_tokens=10),
    )


# ---------------------------------------------------------------------------
# embeddings() CLI entrypoint tests
# ---------------------------------------------------------------------------


class TestEmbeddingsCLI:
    """Tests for the embeddings() click command entrypoint."""

    def test_embeddings_help(self):
        """Check --help output for the embeddings command."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["embeddings", "--help"])
        assert result.exit_code == 0
        assert "embedding" in result.output.lower()

    def test_embeddings_invokes_asyncio_run(self):
        """embeddings() calls asyncio.run with _embeddings_async coroutine."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch(
            "venice_ai.cli.commands.embeddings.asyncio.run", side_effect=consume_coro
        ) as mock_run:
            runner.invoke(cli, ["embeddings", "Hello world"])
            assert mock_run.called

    def test_embeddings_with_all_options(self):
        """embeddings() passes all options through to asyncio.run."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch("venice_ai.cli.commands.embeddings.asyncio.run", side_effect=consume_coro):
            result = runner.invoke(
                cli,
                [
                    "embeddings",
                    "Test text",
                    "--model",
                    "text-embedding-bge-m3",
                    "--dimensions",
                    "256",
                    "--json",
                    "--output",
                    "out.json",
                ],
            )
            assert result.exit_code == 0

    def test_embeddings_command_encoding_format_option(self):
        """embeddings() accepts --encoding-format option."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch("venice_ai.cli.commands.embeddings.asyncio.run", side_effect=consume_coro):
            result = runner.invoke(cli, ["embeddings", "Test text", "--encoding-format", "float"])
            assert result.exit_code == 0


# ---------------------------------------------------------------------------
# _embeddings_async — text input handling
# ---------------------------------------------------------------------------


class TestEmbeddingsAsyncTextInput:
    """Tests for text input handling in _embeddings_async."""

    @pytest.mark.asyncio
    async def test_text_provided_as_argument_plain_mode(self):
        """Text provided as argument is used directly in plain mode."""
        mock_client = AsyncMock()
        mock_response = _make_embedding_response()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _embeddings_async(
                ctx,
                text="hello world",
                model="text-embedding-bge-m3",
                encoding_format="float",
                dimensions=None,
                output_json=False,
                output=None,
            )

            mock_client.embeddings.create.assert_called_once()
            call_kwargs = mock_client.embeddings.create.call_args[1]
            assert call_kwargs["input"] == "hello world"

    @pytest.mark.asyncio
    async def test_text_provided_as_argument_rich_mode(self):
        """Text provided as argument is used in rich mode."""
        mock_client = AsyncMock()
        mock_response = _make_embedding_response()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _embeddings_async(
                ctx,
                text="hello world",
                model="text-embedding-bge-m3",
                encoding_format="float",
                dimensions=None,
                output_json=False,
                output=None,
            )

            mock_client.embeddings.create.assert_called_once()
            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_text_via_stdin_piped(self):
        """Text is read from stdin when not provided as argument and stdin is piped."""
        mock_client = AsyncMock()
        mock_response = _make_embedding_response()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.click.echo"),
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.isatty.return_value = False
            mock_stdin.read.return_value = "piped text input"
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _embeddings_async(
                ctx,
                text=None,
                model="text-embedding-bge-m3",
                encoding_format="float",
                dimensions=None,
                output_json=False,
                output=None,
            )

            call_kwargs = mock_client.embeddings.create.call_args[1]
            assert call_kwargs["input"] == "piped text input"

    @pytest.mark.asyncio
    async def test_text_via_stdin_strips_whitespace(self):
        """Stdin text is stripped of surrounding whitespace."""
        mock_client = AsyncMock()
        mock_response = _make_embedding_response()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.click.echo"),
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.isatty.return_value = False
            mock_stdin.read.return_value = "  trimmed text  \n"
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _embeddings_async(
                ctx,
                text=None,
                model="text-embedding-bge-m3",
                encoding_format="float",
                dimensions=None,
                output_json=False,
                output=None,
            )

            call_kwargs = mock_client.embeddings.create.call_args[1]
            assert call_kwargs["input"] == "trimmed text"

    @pytest.mark.asyncio
    async def test_no_text_stdin_is_tty_raises_error(self):
        """ClickException raised when no text and stdin is a TTY."""
        import click

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True

            ctx = _make_ctx(plain=True)
            with pytest.raises(click.ClickException) as exc_info:
                await _embeddings_async(
                    ctx,
                    text=None,
                    model="text-embedding-bge-m3",
                    encoding_format="float",
                    dimensions=None,
                    output_json=False,
                    output=None,
                )
            assert "Text is required" in str(exc_info.value.format_message())

    @pytest.mark.asyncio
    async def test_no_text_stdin_piped_but_empty_raises_error(self):
        """ClickException raised when piped stdin results in empty string after strip."""
        import click

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            mock_stdin.read.return_value = "   "  # whitespace only

            ctx = _make_ctx(plain=True)
            with pytest.raises(click.ClickException) as exc_info:
                await _embeddings_async(
                    ctx,
                    text=None,
                    model="text-embedding-bge-m3",
                    encoding_format="float",
                    dimensions=None,
                    output_json=False,
                    output=None,
                )
            assert "Text is required" in str(exc_info.value.format_message())

    @pytest.mark.asyncio
    async def test_ctx_obj_none_defaults_to_non_plain(self):
        """ctx.obj is None → plain defaults to False (rich mode)."""
        mock_client = AsyncMock()
        mock_response = _make_embedding_response()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = MagicMock()
            ctx.obj = None
            await _embeddings_async(
                ctx,
                text="hello",
                model="text-embedding-bge-m3",
                encoding_format="float",
                dimensions=None,
                output_json=False,
                output=None,
            )

            # Rich (non-plain) mode should use console.print
            assert mock_console.print.called


# ---------------------------------------------------------------------------
# _embeddings_async — API parameters
# ---------------------------------------------------------------------------


class TestEmbeddingsAsyncAPIParams:
    """Tests that _embeddings_async passes correct parameters to API."""

    @pytest.mark.asyncio
    async def test_with_dimensions_parameter(self):
        """Dimensions parameter is passed to the API when specified."""
        mock_client = AsyncMock()
        mock_response = _make_embedding_response()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _embeddings_async(
                ctx,
                text="test text",
                model="text-embedding-bge-m3",
                encoding_format="float",
                dimensions=256,
                output_json=False,
                output=None,
            )

            call_kwargs = mock_client.embeddings.create.call_args[1]
            assert call_kwargs["dimensions"] == 256

    @pytest.mark.asyncio
    async def test_without_dimensions_parameter(self):
        """Dimensions parameter is NOT included when None."""
        mock_client = AsyncMock()
        mock_response = _make_embedding_response()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _embeddings_async(
                ctx,
                text="test text",
                model="text-embedding-bge-m3",
                encoding_format="float",
                dimensions=None,
                output_json=False,
                output=None,
            )

            call_kwargs = mock_client.embeddings.create.call_args[1]
            assert "dimensions" not in call_kwargs

    @pytest.mark.asyncio
    async def test_model_and_encoding_format_passed(self):
        """Model and encoding_format are passed to the API."""
        mock_client = AsyncMock()
        mock_response = _make_embedding_response()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _embeddings_async(
                ctx,
                text="test text",
                model="custom-embedding-model",
                encoding_format="base64",
                dimensions=None,
                output_json=False,
                output=None,
            )

            call_kwargs = mock_client.embeddings.create.call_args[1]
            assert call_kwargs["model"] == "custom-embedding-model"
            assert call_kwargs["encoding_format"] == "base64"
            assert call_kwargs["input"] == "test text"


# ---------------------------------------------------------------------------
# _embeddings_async — JSON output mode
# ---------------------------------------------------------------------------


class TestEmbeddingsAsyncJSONOutput:
    """Tests for JSON output mode in _embeddings_async."""

    @pytest.mark.asyncio
    async def test_json_output_prints_to_stdout(self):
        """JSON output mode prints JSON string to stdout via click.echo."""
        mock_client = AsyncMock()
        mock_response = _make_embedding_response(embedding=[0.1, 0.2, 0.3])
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _embeddings_async(
                ctx,
                text="test",
                model="text-embedding-bge-m3",
                encoding_format="float",
                dimensions=None,
                output_json=True,
                output=None,
            )

            mock_echo.assert_called_once()
            call_arg = mock_echo.call_args[0][0]
            import json

            parsed = json.loads(call_arg)
            assert "data" in parsed
            assert "model" in parsed
            assert parsed["model"] == "text-embedding-ada-002"

    @pytest.mark.asyncio
    async def test_json_output_save_to_file_plain_mode(self):
        """JSON output saved to file in plain mode shows 'Saved:' message."""
        mock_client = AsyncMock()
        mock_response = _make_embedding_response()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.click.echo") as mock_echo,
            patch("builtins.open", mock_open()),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _embeddings_async(
                ctx,
                text="test",
                model="text-embedding-bge-m3",
                encoding_format="float",
                dimensions=None,
                output_json=True,
                output="embeddings.json",
            )

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Saved:" in calls_str
            assert "embeddings.json" in calls_str

    @pytest.mark.asyncio
    async def test_json_output_save_to_file_rich_mode(self):
        """JSON output saved to file in rich mode shows success console message."""
        mock_client = AsyncMock()
        mock_response = _make_embedding_response()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.console") as mock_console,
            patch("builtins.open", mock_open()),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _embeddings_async(
                ctx,
                text="test",
                model="text-embedding-bge-m3",
                encoding_format="float",
                dimensions=None,
                output_json=True,
                output="embeddings.json",
            )

            calls_str = "".join(str(c) for c in mock_console.print.call_args_list)
            assert "embeddings.json" in calls_str

    @pytest.mark.asyncio
    async def test_json_output_contains_usage_info(self):
        """JSON output includes usage (prompt_tokens, total_tokens)."""
        mock_client = AsyncMock()
        mock_response = _make_embedding_response(prompt_tokens=7, total_tokens=7)
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _embeddings_async(
                ctx,
                text="test",
                model="text-embedding-bge-m3",
                encoding_format="float",
                dimensions=None,
                output_json=True,
                output=None,
            )

            call_arg = mock_echo.call_args[0][0]
            import json

            parsed = json.loads(call_arg)
            assert parsed["usage"]["prompt_tokens"] == 7
            assert parsed["usage"]["total_tokens"] == 7

    @pytest.mark.asyncio
    async def test_json_output_contains_embedding_vector(self):
        """JSON output includes the embedding vector in data."""
        embedding_vec = [0.11, 0.22, 0.33]
        mock_client = AsyncMock()
        mock_response = _make_embedding_response(embedding=embedding_vec)
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _embeddings_async(
                ctx,
                text="test",
                model="text-embedding-bge-m3",
                encoding_format="float",
                dimensions=None,
                output_json=True,
                output=None,
            )

            call_arg = mock_echo.call_args[0][0]
            import json

            parsed = json.loads(call_arg)
            assert parsed["data"][0]["embedding"] == embedding_vec

    @pytest.mark.asyncio
    async def test_json_output_writes_to_file(self):
        """JSON output writes the JSON string to the file."""
        mock_client = AsyncMock()
        mock_response = _make_embedding_response()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        write_calls = []

        def capture_write(data):
            write_calls.append(data)

        m = mock_open()
        m.return_value.__enter__.return_value.write.side_effect = capture_write

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.click.echo"),
            patch("builtins.open", m),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _embeddings_async(
                ctx,
                text="test",
                model="text-embedding-bge-m3",
                encoding_format="float",
                dimensions=None,
                output_json=True,
                output="out.json",
            )

        assert len(write_calls) == 1
        import json

        parsed = json.loads(write_calls[0])
        assert "data" in parsed


# ---------------------------------------------------------------------------
# _embeddings_async — default (non-JSON) summary output
# ---------------------------------------------------------------------------


class TestEmbeddingsAsyncSummaryOutput:
    """Tests for default summary output in _embeddings_async."""

    @pytest.mark.asyncio
    async def test_summary_display_rich_mode_no_file(self):
        """Default summary in rich mode prints via console.print."""
        mock_client = AsyncMock()
        mock_response = _make_embedding_response()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _embeddings_async(
                ctx,
                text="test",
                model="text-embedding-ada-002",
                encoding_format="float",
                dimensions=None,
                output_json=False,
                output=None,
            )

            calls_str = "".join(str(c) for c in mock_console.print.call_args_list)
            assert "text-embedding-ada-002" in calls_str

    @pytest.mark.asyncio
    async def test_summary_display_plain_mode_no_file(self):
        """Default summary in plain mode uses click.echo."""
        mock_client = AsyncMock()
        mock_response = _make_embedding_response(model="my-model")
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _embeddings_async(
                ctx,
                text="test",
                model="my-model",
                encoding_format="float",
                dimensions=None,
                output_json=False,
                output=None,
            )

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "my-model" in calls_str

    @pytest.mark.asyncio
    async def test_summary_shows_dimensions_when_list_embedding(self):
        """Summary displays dimensions when embedding is a list."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_client = AsyncMock()
        mock_response = _make_embedding_response(embedding=embedding)
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _embeddings_async(
                ctx,
                text="test",
                model="test-model",
                encoding_format="float",
                dimensions=None,
                output_json=False,
                output=None,
            )

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "5" in calls_str  # 5 dimensions

    @pytest.mark.asyncio
    async def test_summary_shows_total_tokens(self):
        """Summary displays total tokens used."""
        mock_client = AsyncMock()
        mock_response = _make_embedding_response(total_tokens=42)
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _embeddings_async(
                ctx,
                text="test",
                model="test-model",
                encoding_format="float",
                dimensions=None,
                output_json=False,
                output=None,
            )

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "42" in calls_str

    @pytest.mark.asyncio
    async def test_summary_shows_preview_more_than_5_values(self):
        """Summary shows first 5 values for embedding with more than 5 values."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        mock_client = AsyncMock()
        mock_response = _make_embedding_response(embedding=embedding)
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _embeddings_async(
                ctx,
                text="test",
                model="test-model",
                encoding_format="float",
                dimensions=None,
                output_json=False,
                output=None,
            )

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            # Should show the first 5 values
            assert "0.1" in calls_str
            assert "0.5" in calls_str

    @pytest.mark.asyncio
    async def test_summary_shows_preview_exactly_5_values(self):
        """Summary for embedding with exactly 5 values shows all values as preview."""
        embedding = [0.11, 0.22, 0.33, 0.44, 0.55]
        mock_client = AsyncMock()
        mock_response = _make_embedding_response(embedding=embedding)
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _embeddings_async(
                ctx,
                text="test",
                model="test-model",
                encoding_format="float",
                dimensions=None,
                output_json=False,
                output=None,
            )

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "0.55" in calls_str

    @pytest.mark.asyncio
    async def test_summary_plain_shows_dimensions(self):
        """Summary in plain mode explicitly shows 'Dimensions:' line."""
        embedding = [0.1] * 128
        mock_client = AsyncMock()
        mock_response = _make_embedding_response(embedding=embedding)
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _embeddings_async(
                ctx,
                text="test",
                model="test-model",
                encoding_format="float",
                dimensions=None,
                output_json=False,
                output=None,
            )

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Dimensions:" in calls_str
            assert "128" in calls_str

    @pytest.mark.asyncio
    async def test_summary_rich_shows_dimensions(self):
        """Summary in rich mode shows Dimensions via console.print."""
        embedding = [0.1] * 64
        mock_client = AsyncMock()
        mock_response = _make_embedding_response(embedding=embedding)
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _embeddings_async(
                ctx,
                text="test",
                model="test-model",
                encoding_format="float",
                dimensions=None,
                output_json=False,
                output=None,
            )

            calls_str = "".join(str(c) for c in mock_console.print.call_args_list)
            assert "64" in calls_str


# ---------------------------------------------------------------------------
# _embeddings_async — save to file (non-JSON mode)
# ---------------------------------------------------------------------------


class TestEmbeddingsAsyncSaveToFile:
    """Tests for saving embeddings to file in non-JSON mode."""

    @pytest.mark.asyncio
    async def test_save_to_file_plain_mode(self):
        """Save to file in plain mode shows 'Saved:' message."""
        mock_client = AsyncMock()
        mock_response = _make_embedding_response()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.click.echo") as mock_echo,
            patch("builtins.open", mock_open()),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _embeddings_async(
                ctx,
                text="test",
                model="test-model",
                encoding_format="float",
                dimensions=None,
                output_json=False,
                output="embeddings.json",
            )

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Saved:" in calls_str
            assert "embeddings.json" in calls_str

    @pytest.mark.asyncio
    async def test_save_to_file_rich_mode(self):
        """Save to file in rich mode shows success console message."""
        mock_client = AsyncMock()
        mock_response = _make_embedding_response()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.console") as mock_console,
            patch("builtins.open", mock_open()),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _embeddings_async(
                ctx,
                text="test",
                model="test-model",
                encoding_format="float",
                dimensions=None,
                output_json=False,
                output="embeddings.json",
            )

            calls_str = "".join(str(c) for c in mock_console.print.call_args_list)
            assert "embeddings.json" in calls_str

    @pytest.mark.asyncio
    async def test_save_to_file_writes_json(self):
        """Save to file writes valid JSON with embedding vector."""
        embedding = [0.1, 0.2, 0.3]
        mock_client = AsyncMock()
        mock_response = _make_embedding_response(embedding=embedding, total_tokens=3)
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        m = mock_open()
        # Capture the json.dump output by checking open was called with the right file
        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.click.echo"),
            patch("builtins.open", m),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _embeddings_async(
                ctx,
                text="test",
                model="test-model",
                encoding_format="float",
                dimensions=None,
                output_json=False,
                output="out.json",
            )

            # json.dump writes incrementally so check the file was opened
            m.assert_called()
            open_path = m.call_args[0][0]
            assert open_path == "out.json"

    @pytest.mark.asyncio
    async def test_save_to_file_plain_shows_dimensions(self):
        """Save to file in plain mode also shows dimensions."""
        embedding = [0.1] * 32
        mock_client = AsyncMock()
        mock_response = _make_embedding_response(embedding=embedding)
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.click.echo") as mock_echo,
            patch("builtins.open", mock_open()),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _embeddings_async(
                ctx,
                text="test",
                model="test-model",
                encoding_format="float",
                dimensions=None,
                output_json=False,
                output="out.json",
            )

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "32" in calls_str

    @pytest.mark.asyncio
    async def test_save_to_file_rich_shows_model_and_dimensions(self):
        """Save to file in rich mode shows model and dimensions."""
        embedding = [0.1] * 16
        mock_client = AsyncMock()
        mock_response = _make_embedding_response(embedding=embedding, model="my-embed-model")
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.console") as mock_console,
            patch("builtins.open", mock_open()),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _embeddings_async(
                ctx,
                text="test",
                model="my-embed-model",
                encoding_format="float",
                dimensions=None,
                output_json=False,
                output="out.json",
            )

            calls_str = "".join(str(c) for c in mock_console.print.call_args_list)
            assert "my-embed-model" in calls_str
            assert "16" in calls_str


# ---------------------------------------------------------------------------
# _embeddings_async — multiple embeddings in response
# ---------------------------------------------------------------------------


class TestEmbeddingsAsyncMultipleEmbeddings:
    """Tests for responses with multiple embeddings."""

    @pytest.mark.asyncio
    async def test_json_output_with_multiple_embeddings(self):
        """JSON output includes all embeddings when response has multiple data items."""
        embeddings_list = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
        mock_client = AsyncMock()
        mock_response = _make_multi_embedding_response(embeddings_list)
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _embeddings_async(
                ctx,
                text="test",
                model="test-model",
                encoding_format="float",
                dimensions=None,
                output_json=True,
                output=None,
            )

            call_arg = mock_echo.call_args[0][0]
            import json

            parsed = json.loads(call_arg)
            assert len(parsed["data"]) == 2
            assert parsed["data"][0]["embedding"] == [0.1, 0.2, 0.3]
            assert parsed["data"][1]["embedding"] == [0.4, 0.5, 0.6]


# ---------------------------------------------------------------------------
# _embeddings_async — non-list embedding (edge case)
# ---------------------------------------------------------------------------


class TestEmbeddingsAsyncNonListEmbedding:
    """Tests for non-list embedding values (e.g., base64 string)."""

    @pytest.mark.asyncio
    async def test_non_list_embedding_dims_is_none(self):
        """When embedding is not a list (e.g. base64 string), dims is None and no preview."""
        mock_client = AsyncMock()
        mock_response = SimpleNamespace(
            data=[
                SimpleNamespace(
                    embedding="base64EncodedString==",
                    index=0,
                    object="embedding",
                )
            ],
            model="text-embedding-ada-002",
            object="list",
            usage=SimpleNamespace(prompt_tokens=5, total_tokens=5),
        )
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _embeddings_async(
                ctx,
                text="test",
                model="text-embedding-ada-002",
                encoding_format="base64",
                dimensions=None,
                output_json=False,
                output=None,
            )

            # Should not crash; dimensions not shown since it's not a list
            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Dimensions:" not in calls_str

    @pytest.mark.asyncio
    async def test_non_list_embedding_in_json_mode(self):
        """Non-list (base64) embedding in JSON mode is included as-is."""
        mock_client = AsyncMock()
        mock_response = SimpleNamespace(
            data=[
                SimpleNamespace(
                    embedding="base64EncodedString==",
                    index=0,
                    object="embedding",
                )
            ],
            model="text-embedding-ada-002",
            object="list",
            usage=SimpleNamespace(prompt_tokens=5, total_tokens=5),
        )
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _embeddings_async(
                ctx,
                text="test",
                model="text-embedding-ada-002",
                encoding_format="base64",
                dimensions=None,
                output_json=True,
                output=None,
            )

            import json

            call_arg = mock_echo.call_args[0][0]
            parsed = json.loads(call_arg)
            assert parsed["data"][0]["embedding"] == "base64EncodedString=="


# ---------------------------------------------------------------------------
# _embeddings_async — rich mode startup banner suppression
# ---------------------------------------------------------------------------


class TestEmbeddingsAsyncBanner:
    """Tests for startup banner behavior in _embeddings_async."""

    @pytest.mark.asyncio
    async def test_rich_mode_and_no_json_shows_banner(self):
        """In rich mode (plain=False) and no json_output, shows 'Generating embeddings' banner."""
        mock_client = AsyncMock()
        mock_response = _make_embedding_response()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _embeddings_async(
                ctx,
                text="test",
                model="text-embedding-bge-m3",
                encoding_format="float",
                dimensions=None,
                output_json=False,
                output=None,
            )

            calls_str = "".join(str(c) for c in mock_console.print.call_args_list)
            assert "Generating embeddings" in calls_str or "🔢" in calls_str

    @pytest.mark.asyncio
    async def test_plain_mode_no_banner(self):
        """In plain mode (plain=True), no 'Generating embeddings' banner via console."""
        mock_client = AsyncMock()
        mock_response = _make_embedding_response()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.console") as mock_console,
            patch("venice_ai.cli.commands.embeddings.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _embeddings_async(
                ctx,
                text="test",
                model="text-embedding-bge-m3",
                encoding_format="float",
                dimensions=None,
                output_json=False,
                output=None,
            )

            # console.print should not be called for banner in plain mode
            assert not mock_console.print.called

    @pytest.mark.asyncio
    async def test_json_output_mode_no_banner(self):
        """In json_output mode, no 'Generating embeddings' banner is printed."""
        mock_client = AsyncMock()
        mock_response = _make_embedding_response()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.embeddings.console") as mock_console,
            patch("venice_ai.cli.commands.embeddings.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _embeddings_async(
                ctx,
                text="test",
                model="text-embedding-bge-m3",
                encoding_format="float",
                dimensions=None,
                output_json=True,
                output=None,
            )

            # Startup banner should not be printed when output_json=True
            calls_str = "".join(str(c) for c in mock_console.print.call_args_list)
            assert "Generating embeddings" not in calls_str
