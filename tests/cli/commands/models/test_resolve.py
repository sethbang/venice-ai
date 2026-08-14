"""Tests for ``venice-py models resolve`` (cli/commands/models/resolve.py)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from venice_ai.cli.commands.models.resolve import _resolve_async


def _make_ctx():
    ctx = MagicMock()
    ctx.obj = {"config": {"api": {"key": "test-key"}}, "plain": False}
    ctx.exit = MagicMock(side_effect=SystemExit(1))
    return ctx


def _setup_client(MockVeniceClient, mock_client):
    MockVeniceClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    MockVeniceClient.return_value.__aexit__ = AsyncMock(return_value=None)


# ---------------------------------------------------------------------------
# CLI entrypoint smoke tests
# ---------------------------------------------------------------------------


class TestResolveCLI:
    def test_resolve_help(self):
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["models", "resolve", "--help"])
        assert result.exit_code == 0
        assert "--type" in result.output
        # Each supported type should appear in the choice list.
        for t in [
            "chat",
            "embedding",
            "image",
            "video",
            "tts",
            "asr",
            "inpaint",
            "music",
            "video-upscale",
            "cheapest-video",
        ]:
            assert t in result.output

    def test_resolve_requires_type(self):
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["models", "resolve"])
        # Click exits with code 2 on missing required option.
        assert result.exit_code == 2

    def test_resolve_invokes_asyncio_run(self):
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch(
            "venice_ai.cli.commands.models.resolve.asyncio.run", side_effect=consume_coro
        ) as mock_run:
            runner.invoke(cli, ["models", "resolve", "--type", "chat"])
            assert mock_run.called


# ---------------------------------------------------------------------------
# _resolve_async — chat
# ---------------------------------------------------------------------------


class TestResolveAsyncChat:
    @pytest.mark.asyncio
    async def test_resolve_chat_basic_calls_resolve_with_chat_kwargs(self):
        mock_client = AsyncMock()
        mock_client.models.resolve = AsyncMock(return_value="llama-3.3-70b")

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.models.resolve.console"),
        ):
            _setup_client(MockClient, mock_client)
            await _resolve_async(
                _make_ctx(),
                model_type="chat",
                function_calling=True,
                vision=True,
                reasoning=False,
                code=False,
                response_schema=False,
                min_context_tokens=8000,
                require_private=False,
                video_type=None,
                audio=False,
                min_resolution=None,
                min_duration=None,
                duration="5s",
                resolution=None,
                aspect_ratio=None,
                preferred_models=None,
                exclude_models=None,
                include_beta=False,
                output_json=False,
            )

            mock_client.models.resolve.assert_awaited_once()
            kwargs = mock_client.models.resolve.await_args.kwargs
            assert kwargs["type"] == "chat"
            assert kwargs["require_function_calling"] is True
            assert kwargs["require_vision"] is True
            assert kwargs["min_context_tokens"] == 8000
            assert kwargs["exclude_beta"] is True

    @pytest.mark.asyncio
    async def test_resolve_chat_json_output_includes_model(self):
        mock_client = AsyncMock()
        mock_client.models.resolve = AsyncMock(return_value="resolved-id")

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.models.resolve.click.echo") as mock_echo,
        ):
            _setup_client(MockClient, mock_client)
            await _resolve_async(
                _make_ctx(),
                model_type="chat",
                function_calling=False,
                vision=False,
                reasoning=False,
                code=False,
                response_schema=False,
                min_context_tokens=None,
                require_private=False,
                video_type=None,
                audio=False,
                min_resolution=None,
                min_duration=None,
                duration="5s",
                resolution=None,
                aspect_ratio=None,
                preferred_models=None,
                exclude_models=None,
                include_beta=True,
                output_json=True,
            )

            mock_echo.assert_called_once()
            payload = json.loads(mock_echo.call_args[0][0])
            assert payload == {"type": "chat", "model": "resolved-id"}
            kwargs = mock_client.models.resolve.await_args.kwargs
            assert kwargs["exclude_beta"] is False  # --include-beta toggles it off


# ---------------------------------------------------------------------------
# _resolve_async — video / cheapest-video / video-upscale
# ---------------------------------------------------------------------------


class TestResolveAsyncVideo:
    @pytest.mark.asyncio
    async def test_resolve_video_passes_video_kwargs(self):
        mock_client = AsyncMock()
        mock_client.models.resolve = AsyncMock(return_value="some-video-model")

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.models.resolve.console"),
        ):
            _setup_client(MockClient, mock_client)
            await _resolve_async(
                _make_ctx(),
                model_type="video",
                function_calling=False,
                vision=False,
                reasoning=False,
                code=False,
                response_schema=False,
                min_context_tokens=None,
                require_private=False,
                video_type="text-to-video",
                audio=True,
                min_resolution="720p",
                min_duration="5s",
                duration="5s",
                resolution=None,
                aspect_ratio=None,
                preferred_models=None,
                exclude_models=None,
                include_beta=False,
                output_json=False,
            )

            kwargs = mock_client.models.resolve.await_args.kwargs
            assert kwargs["video_type"] == "text-to-video"
            assert kwargs["require_audio"] is True
            assert kwargs["min_resolution"] == "720p"
            assert kwargs["min_duration"] == "5s"

    @pytest.mark.asyncio
    async def test_resolve_cheapest_video_uses_dedicated_method(self):
        mock_client = AsyncMock()
        mock_client.models.resolve_cheapest_video = AsyncMock(
            return_value=SimpleNamespace(
                model="topaz-vid", quote_usd=0.25, all_quotes={"topaz-vid": 0.25, "x": 1.0}
            )
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.models.resolve.click.echo") as mock_echo,
        ):
            _setup_client(MockClient, mock_client)
            await _resolve_async(
                _make_ctx(),
                model_type="cheapest-video",
                function_calling=False,
                vision=False,
                reasoning=False,
                code=False,
                response_schema=False,
                min_context_tokens=None,
                require_private=False,
                video_type=None,
                audio=False,
                min_resolution=None,
                min_duration=None,
                duration="10s",
                resolution="720p",
                aspect_ratio="16:9",
                preferred_models=None,
                exclude_models=None,
                include_beta=False,
                output_json=True,
            )

            mock_client.models.resolve_cheapest_video.assert_awaited_once()
            kwargs = mock_client.models.resolve_cheapest_video.await_args.kwargs
            assert kwargs["duration"] == "10s"
            assert kwargs["resolution"] == "720p"
            assert kwargs["aspect_ratio"] == "16:9"
            mock_echo.assert_called_once()
            payload = json.loads(mock_echo.call_args[0][0])
            assert payload["model"] == "topaz-vid"
            assert payload["quote_usd"] == 0.25

    @pytest.mark.asyncio
    async def test_resolve_video_upscale_uses_dedicated_method(self):
        mock_client = AsyncMock()
        mock_client.models.resolve_video_upscale = AsyncMock(return_value="topaz-video-upscale")

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.models.resolve.console"),
        ):
            _setup_client(MockClient, mock_client)
            await _resolve_async(
                _make_ctx(),
                model_type="video-upscale",
                function_calling=False,
                vision=False,
                reasoning=False,
                code=False,
                response_schema=False,
                min_context_tokens=None,
                require_private=False,
                video_type=None,
                audio=False,
                min_resolution=None,
                min_duration=None,
                duration="5s",
                resolution=None,
                aspect_ratio=None,
                preferred_models=("topaz-video-upscale",),
                exclude_models=None,
                include_beta=False,
                output_json=False,
            )

            mock_client.models.resolve_video_upscale.assert_awaited_once()


# ---------------------------------------------------------------------------
# _resolve_async — error handling
# ---------------------------------------------------------------------------


class TestResolveAsyncErrors:
    @pytest.mark.asyncio
    async def test_resolve_value_error_exits_nonzero(self):
        mock_client = AsyncMock()
        mock_client.models.resolve = AsyncMock(side_effect=ValueError("no match"))

        ctx = _make_ctx()
        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.models.resolve.console"),
            pytest.raises(SystemExit),
        ):
            _setup_client(MockClient, mock_client)
            await _resolve_async(
                ctx,
                model_type="chat",
                function_calling=False,
                vision=False,
                reasoning=False,
                code=False,
                response_schema=False,
                min_context_tokens=None,
                require_private=False,
                video_type=None,
                audio=False,
                min_resolution=None,
                min_duration=None,
                duration="5s",
                resolution=None,
                aspect_ratio=None,
                preferred_models=None,
                exclude_models=None,
                include_beta=False,
                output_json=False,
            )

        ctx.exit.assert_called_with(1)

    @pytest.mark.asyncio
    async def test_resolve_value_error_json_output_emits_error_payload(self):
        mock_client = AsyncMock()
        mock_client.models.resolve = AsyncMock(side_effect=ValueError("no match"))

        ctx = _make_ctx()
        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.models.resolve.click.echo") as mock_echo,
            pytest.raises(SystemExit),
        ):
            _setup_client(MockClient, mock_client)
            await _resolve_async(
                ctx,
                model_type="chat",
                function_calling=False,
                vision=False,
                reasoning=False,
                code=False,
                response_schema=False,
                min_context_tokens=None,
                require_private=False,
                video_type=None,
                audio=False,
                min_resolution=None,
                min_duration=None,
                duration="5s",
                resolution=None,
                aspect_ratio=None,
                preferred_models=None,
                exclude_models=None,
                include_beta=False,
                output_json=True,
            )

        mock_echo.assert_called_once()
        payload = json.loads(mock_echo.call_args[0][0])
        assert "error" in payload
