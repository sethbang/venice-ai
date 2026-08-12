"""Tests for ``venice models capabilities`` (cli/commands/models/capabilities.py)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from venice_ai.cli.commands.models.capabilities import _capabilities_async


def _make_ctx():
    ctx = MagicMock()
    ctx.obj = {"config": {"api": {"key": "test-key"}}, "plain": False}
    ctx.exit = MagicMock(side_effect=SystemExit(1))
    return ctx


def _setup_client(MockVeniceClient, mock_client):
    MockVeniceClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    MockVeniceClient.return_value.__aexit__ = AsyncMock(return_value=None)


class TestCapabilitiesCLI:
    def test_capabilities_help(self):
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["models", "capabilities", "--help"])
        assert result.exit_code == 0
        assert "MODEL_ID" in result.output

    def test_capabilities_invokes_asyncio_run(self):
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch(
            "venice_ai.cli.commands.models.capabilities.asyncio.run",
            side_effect=consume_coro,
        ) as mock_run:
            runner.invoke(cli, ["models", "capabilities", "llama-3.3-70b"])
            assert mock_run.called


class TestCapabilitiesAsync:
    @pytest.mark.asyncio
    async def test_capabilities_invokes_get_capabilities(self):
        from venice_ai.types.api.capabilities import ChatCapabilities

        caps = ChatCapabilities(
            context_window=131072,
            supports_function_calling=True,
            supports_vision=False,
            supports_reasoning=True,
            supports_response_schema=True,
            supports_web_search=True,
            supports_logprobs=True,
            supports_audio_input=False,
            supports_video_input=False,
            supports_multiple_images=False,
            supports_reasoning_effort=False,
            supports_tee_attestation=False,
            supports_e2ee=False,
            supports_x_search=False,
            optimized_for_code=False,
            quantization="fp8",
            privacy=None,
        )
        mock_client = AsyncMock()
        mock_client.models.get_capabilities = AsyncMock(return_value=caps)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.models.capabilities.console"),
        ):
            _setup_client(MockClient, mock_client)
            await _capabilities_async(_make_ctx(), model_id="llama-3.3-70b", output_json=False)

            mock_client.models.get_capabilities.assert_awaited_once_with(model_id="llama-3.3-70b")

    @pytest.mark.asyncio
    async def test_capabilities_json_output_emits_payload(self):
        from venice_ai.types.api.capabilities import GenericCapabilities

        caps = GenericCapabilities(type="embedding", privacy=None)
        mock_client = AsyncMock()
        mock_client.models.get_capabilities = AsyncMock(return_value=caps)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.models.capabilities.click.echo") as mock_echo,
        ):
            _setup_client(MockClient, mock_client)
            await _capabilities_async(
                _make_ctx(), model_id="text-embedding-3-large", output_json=True
            )

            mock_echo.assert_called_once()
            payload = json.loads(mock_echo.call_args[0][0])
            assert payload["type"] == "embedding"

    @pytest.mark.asyncio
    async def test_capabilities_value_error_exits_nonzero(self):
        mock_client = AsyncMock()
        mock_client.models.get_capabilities = AsyncMock(side_effect=ValueError("nope"))

        ctx = _make_ctx()
        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.models.capabilities.console"),
            pytest.raises(SystemExit),
        ):
            _setup_client(MockClient, mock_client)
            await _capabilities_async(ctx, model_id="missing", output_json=False)

        ctx.exit.assert_called_with(1)
