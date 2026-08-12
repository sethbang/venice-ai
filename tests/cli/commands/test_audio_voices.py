"""Tests for ``venice audio voices`` (cli/commands/audio.py)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from venice_ai.cli.commands.audio import _voices_async


def _make_ctx(plain: bool = False):
    ctx = MagicMock()
    ctx.obj = {"config": {"api": {"key": "test-key"}}, "plain": plain}
    return ctx


def _setup_client(MockVeniceClient, mock_client):
    MockVeniceClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    MockVeniceClient.return_value.__aexit__ = AsyncMock(return_value=None)


def _make_voice(
    voice_id="af_heart",
    model_id="tts-kokoro",
    gender="female",
    region="af",
    language="American English",
    accent="US",
):
    return SimpleNamespace(
        id=voice_id,
        model_id=model_id,
        gender=gender,
        region_code=region,
        language=language,
        accent=accent,
    )


def _make_voice_list(voices):
    obj = SimpleNamespace(
        object="list",
        data=voices,
        model_id_filter=None,
        gender_filter=None,
        region_code_filter=None,
    )
    obj.model_dump = lambda: {
        "object": "list",
        "data": [
            {
                "id": v.id,
                "model_id": v.model_id,
                "gender": v.gender,
                "region_code": v.region_code,
                "language": v.language,
                "accent": v.accent,
            }
            for v in voices
        ],
    }
    return obj


class TestVoicesCLI:
    def test_voices_help(self):
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["audio", "voices", "--help"])
        assert result.exit_code == 0
        assert "voice" in result.output.lower()

    def test_voices_invokes_asyncio_run(self):
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch(
            "venice_ai.cli.commands.audio.asyncio.run", side_effect=consume_coro
        ) as mock_run:
            runner.invoke(cli, ["audio", "voices"])
            assert mock_run.called


class TestVoicesAsync:
    @pytest.mark.asyncio
    async def test_voices_invokes_get_voices_with_filters(self):
        mock_client = AsyncMock()
        mock_client.audio.get_voices = AsyncMock(return_value=_make_voice_list([_make_voice()]))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.console"),
        ):
            _setup_client(MockClient, mock_client)
            await _voices_async(
                _make_ctx(),
                model_id="tts-kokoro",
                gender="female",
                region_code="af",
                output_json=False,
            )

            mock_client.audio.get_voices.assert_awaited_once_with(
                model_id="tts-kokoro", gender="female", region_code="af"
            )

    @pytest.mark.asyncio
    async def test_voices_json_output_dumps_payload(self):
        mock_client = AsyncMock()
        mock_client.audio.get_voices = AsyncMock(
            return_value=_make_voice_list([_make_voice(), _make_voice("am_adam", gender="male")])
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo") as mock_echo,
        ):
            _setup_client(MockClient, mock_client)
            await _voices_async(
                _make_ctx(),
                model_id=None,
                gender=None,
                region_code=None,
                output_json=True,
            )

            mock_echo.assert_called_once()
            payload = json.loads(mock_echo.call_args[0][0])
            assert len(payload["data"]) == 2

    @pytest.mark.asyncio
    async def test_voices_empty_list_message_rich_mode(self):
        mock_client = AsyncMock()
        mock_client.audio.get_voices = AsyncMock(return_value=_make_voice_list([]))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.console") as mock_console,
        ):
            _setup_client(MockClient, mock_client)
            await _voices_async(
                _make_ctx(),
                model_id=None,
                gender=None,
                region_code=None,
                output_json=False,
            )
            mock_console.print.assert_called()

    @pytest.mark.asyncio
    async def test_voices_plain_table_renders(self):
        mock_client = AsyncMock()
        mock_client.audio.get_voices = AsyncMock(return_value=_make_voice_list([_make_voice()]))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.audio.click.echo") as mock_echo,
        ):
            _setup_client(MockClient, mock_client)
            await _voices_async(
                _make_ctx(plain=True),
                model_id=None,
                gender=None,
                region_code=None,
                output_json=False,
            )
            assert mock_echo.called
