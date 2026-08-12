"""
Tests for cli/commands/account.py

Covers:
- account() click group command
- balance() click command entrypoint
- _balance_async() core logic with all branches
- usage() click command entrypoint
- _normalize_date() date conversion utility
- _usage_async() core logic with all branches
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from venice_ai.cli.commands.account import _balance_async, _normalize_date, _usage_async

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(plain: bool = False):
    """Return a minimal mock Click context."""
    ctx = MagicMock()
    ctx.obj = {"config": {"api": {"key": "test-key"}}, "plain": plain}
    return ctx


def _setup_client_patch(MockVeniceClient, mock_client):
    """Configure the MockVeniceClient to act as an async context manager returning mock_client."""
    MockVeniceClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    MockVeniceClient.return_value.__aexit__ = AsyncMock(return_value=None)


# ---------------------------------------------------------------------------
# _normalize_date tests
# ---------------------------------------------------------------------------


class TestNormalizeDate:
    """Tests for the _normalize_date helper."""

    def test_none_returns_none(self):
        """None input should return None."""
        assert _normalize_date(None) is None

    def test_simple_date_appends_time(self):
        """YYYY-MM-DD format should get T00:00:00Z appended."""
        result = _normalize_date("2024-01-15")
        assert result == "2024-01-15T00:00:00Z"

    def test_already_has_time_component(self):
        """ISO format with T should pass through unchanged."""
        iso = "2024-01-15T12:30:00Z"
        assert _normalize_date(iso) == iso

    def test_full_iso_format_unchanged(self):
        """Full ISO 8601 YYYY-MM-DDTHH:MM:SS+00:00 should pass through."""
        iso = "2025-03-01T08:00:00+00:00"
        assert _normalize_date(iso) == iso


# ---------------------------------------------------------------------------
# _balance_async tests — object-style response
# ---------------------------------------------------------------------------


class TestBalanceAsyncObjectResponse:
    """Tests for _balance_async when response has attributes (object format)."""

    @pytest.mark.asyncio
    async def test_balance_async_object_json_output(self):
        """JSON output mode with object-style response."""
        mock_client = AsyncMock()
        mock_response = SimpleNamespace(
            balances=SimpleNamespace(diem=500.0, usd=25.50),
            diem_epoch_allocation=1000.0,
            can_consume=True,
            consumption_currency="DIEM",
        )
        mock_client.billing.get_balance = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _balance_async(ctx, output_json=True)

            mock_echo.assert_called_once()
            output = mock_echo.call_args[0][0]
            data = json.loads(output)
            assert data["balances"]["diem"] == 500.0
            assert data["balances"]["usd"] == 25.50
            assert data["diemEpochAllocation"] == 1000.0
            assert data["canConsume"] is True
            assert data["consumptionCurrency"] == "DIEM"

    @pytest.mark.asyncio
    async def test_balance_async_object_json_output_no_epoch(self):
        """JSON output without epoch allocation (getattr returns None)."""
        mock_client = AsyncMock()
        mock_response = SimpleNamespace(
            balances=SimpleNamespace(diem=500.0, usd=25.50),
            diem_epoch_allocation=None,
            can_consume=None,
            consumption_currency=None,
        )
        mock_client.billing.get_balance = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _balance_async(ctx, output_json=True)

            output = mock_echo.call_args[0][0]
            data = json.loads(output)
            assert "diemEpochAllocation" not in data

    @pytest.mark.asyncio
    async def test_balance_async_object_plain_mode(self):
        """Plain text output with object-style response."""
        mock_client = AsyncMock()
        mock_response = SimpleNamespace(
            balances=SimpleNamespace(diem=500.0, usd=25.50),
            diem_epoch_allocation=1000.0,
            can_consume=True,
            consumption_currency="DIEM",
        )
        mock_client.billing.get_balance = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.utils.output.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _balance_async(ctx, output_json=False)

            mock_om.panel.assert_called_once()
            content = mock_om.panel.call_args[0][0]
            assert "DIEM Balance" in content
            assert "USD Balance" in content
            assert "DIEM Epoch Allocation" in content

    @pytest.mark.asyncio
    async def test_balance_async_object_rich_mode(self):
        """Rich console output with object-style response."""
        mock_client = AsyncMock()
        mock_response = SimpleNamespace(
            balances=SimpleNamespace(diem=500.0, usd=25.50),
            diem_epoch_allocation=1000.0,
            can_consume=True,
            consumption_currency="DIEM",
        )
        mock_client.billing.get_balance = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.utils.output.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _balance_async(ctx, output_json=False)

            mock_om.panel.assert_called_once()

    @pytest.mark.asyncio
    async def test_balance_async_object_plain_only_diem(self):
        """Plain text with only DIEM balance (USD is None)."""
        mock_client = AsyncMock()
        mock_response = SimpleNamespace(
            balances=SimpleNamespace(diem=300.0, usd=None),
            diem_epoch_allocation=None,
            can_consume=None,
            consumption_currency=None,
        )
        mock_client.billing.get_balance = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.utils.output.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _balance_async(ctx, output_json=False)

            content = mock_om.panel.call_args[0][0]
            assert "DIEM Balance" in content

    @pytest.mark.asyncio
    async def test_balance_async_object_plain_only_usd(self):
        """Plain text with only USD balance (DIEM is None)."""
        mock_client = AsyncMock()
        mock_response = SimpleNamespace(
            balances=SimpleNamespace(diem=None, usd=10.00),
            diem_epoch_allocation=None,
            can_consume=None,
            consumption_currency=None,
        )
        mock_client.billing.get_balance = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.utils.output.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _balance_async(ctx, output_json=False)

            content = mock_om.panel.call_args[0][0]
            assert "USD Balance" in content


# ---------------------------------------------------------------------------
# _balance_async tests — dict response
# ---------------------------------------------------------------------------


class TestBalanceAsyncDictResponse:
    """Tests for _balance_async when response is a dict."""

    @pytest.mark.asyncio
    async def test_balance_async_dict_nested(self):
        """Dict response with nested balances matching live API shape."""
        mock_client = AsyncMock()
        mock_response = {
            "canConsume": True,
            "consumptionCurrency": "DIEM",
            "balances": {"diem": 200.0, "usd": 50.0},
            "diemEpochAllocation": 2000.0,
        }
        mock_client.billing.get_balance = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _balance_async(ctx, output_json=True)

            output = mock_echo.call_args[0][0]
            data = json.loads(output)
            assert data["balances"]["diem"] == 200.0
            assert data["balances"]["usd"] == 50.0
            assert data["diemEpochAllocation"] == 2000.0
            assert data["canConsume"] is True
            assert data["consumptionCurrency"] == "DIEM"

    @pytest.mark.asyncio
    async def test_balance_async_dict_plain_mode(self):
        """Dict response in plain mode."""
        mock_client = AsyncMock()
        mock_response = {
            "canConsume": True,
            "consumptionCurrency": "DIEM",
            "balances": {"diem": 200.0, "usd": 50.0},
        }
        mock_client.billing.get_balance = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.utils.output.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _balance_async(ctx, output_json=False)

            content = mock_om.panel.call_args[0][0]
            assert "DIEM Balance" in content


# ---------------------------------------------------------------------------
# _balance_async tests — fallback / edge cases
# ---------------------------------------------------------------------------


class TestBalanceAsyncFallbackResponse:
    """Tests when response doesn't match object or dict patterns."""

    @pytest.mark.asyncio
    async def test_balance_async_unknown_response_json_output(self):
        """Unknown response type → all None → JSON still output."""
        mock_client = AsyncMock()
        # A plain integer has no balances attr and isn't a dict
        mock_client.billing.get_balance = AsyncMock(return_value=42)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _balance_async(ctx, output_json=True)

            output = mock_echo.call_args[0][0]
            data = json.loads(output)
            assert data["balances"]["diem"] is None
            assert data["balances"]["usd"] is None
            assert data["canConsume"] is None

    @pytest.mark.asyncio
    async def test_balance_async_ctx_obj_none(self):
        """ctx.obj is None → plain defaults to False (rich mode)."""
        mock_client = AsyncMock()
        mock_response = SimpleNamespace(
            balances=SimpleNamespace(diem=10.0, usd=5.0),
            diem_epoch_allocation=None,
            can_consume=None,
            consumption_currency=None,
        )
        mock_client.billing.get_balance = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.utils.output.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = MagicMock()
            ctx.obj = None
            await _balance_async(ctx, output_json=False)

            # Should call OutputManager.panel (rich mode handled internally)
            mock_om.panel.assert_called_once()

    @pytest.mark.asyncio
    async def test_balance_async_rich_no_epoch_allocation(self):
        """Rich mode with no epoch allocation (branch not taken)."""
        mock_client = AsyncMock()
        mock_response = SimpleNamespace(
            balances=SimpleNamespace(diem=500.0, usd=25.50),
            diem_epoch_allocation=None,
            can_consume=None,
            consumption_currency=None,
        )
        mock_client.billing.get_balance = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.utils.output.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _balance_async(ctx, output_json=False)

            mock_om.panel.assert_called_once()

    @pytest.mark.asyncio
    async def test_balance_async_rich_no_diem_no_usd(self):
        """Rich mode with both None balances (panel still printed)."""
        mock_client = AsyncMock()
        mock_response = SimpleNamespace(
            balances=SimpleNamespace(diem=None, usd=None),
            diem_epoch_allocation=None,
            can_consume=None,
            consumption_currency=None,
        )
        mock_client.billing.get_balance = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.utils.output.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _balance_async(ctx, output_json=False)

            mock_om.panel.assert_called_once()


# ---------------------------------------------------------------------------
# balance() CLI entrypoint tests
# ---------------------------------------------------------------------------


class TestBalanceCLI:
    """Tests for the balance() click command entrypoint."""

    def test_balance_help(self):
        """Check --help output for balance command."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["account", "balance", "--help"])
        assert result.exit_code == 0
        assert "balance" in result.output.lower() or "credits" in result.output.lower()

    def test_balance_invokes_asyncio_run(self):
        """balance() calls asyncio.run with _balance_async coroutine."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch(
            "venice_ai.cli.commands.account.asyncio.run", side_effect=consume_coro
        ) as mock_run:
            runner.invoke(cli, ["account", "balance"])
            assert mock_run.called

    def test_balance_json_flag(self):
        """balance --json passes output_json=True."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch("venice_ai.cli.commands.account.asyncio.run", side_effect=consume_coro):
            result = runner.invoke(cli, ["account", "balance", "--json"])
            assert result.exit_code == 0


# ---------------------------------------------------------------------------
# CLI-B-04: _usage_async --currency forwarding
# ---------------------------------------------------------------------------


class TestUsageAsyncCurrency:
    """Tests that --currency maps to billing.get_usage_history(currency=...)."""

    @pytest.mark.asyncio
    async def test_currency_forwarded(self):
        """--currency USD is forwarded to get_usage_history()."""
        mock_client = AsyncMock()
        mock_response = {"data": [], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _usage_async(
                ctx, output_json=False, start_date=None, end_date=None, currency="USD"
            )

            kwargs = mock_client.billing.get_usage_history.call_args.kwargs
            assert kwargs["currency"] == "USD"

    @pytest.mark.asyncio
    async def test_currency_default_none(self):
        """No --currency → currency=None forwarded (SDK omits it via exclude_none)."""
        mock_client = AsyncMock()
        mock_response = {"data": [], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _usage_async(ctx, output_json=False, start_date=None, end_date=None)

            kwargs = mock_client.billing.get_usage_history.call_args.kwargs
            assert kwargs["currency"] is None

    def test_currency_cli_flag_forwarded(self):
        """End-to-end CliRunner: --currency reaches get_usage_history()."""
        from venice_ai.cli.cli import cli

        mock_client = AsyncMock()
        mock_response = {"data": [], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
        ):
            _setup_client_patch(MockClient, mock_client)

            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["account", "usage", "--currency", "DIEM"],
            )

        assert result.exit_code == 0, result.output
        kwargs = mock_client.billing.get_usage_history.call_args.kwargs
        assert kwargs["currency"] == "DIEM"

    def test_currency_cli_rejects_invalid(self):
        """--currency is a Choice; an unsupported value errors."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["account", "usage", "--currency", "EUR"],
        )
        assert result.exit_code != 0
        assert "EUR" in result.output or "Invalid value" in result.output


# ---------------------------------------------------------------------------
# _usage_async tests — no entries
# ---------------------------------------------------------------------------


class TestUsageAsyncNoEntries:
    """Tests for _usage_async with no entries."""

    @pytest.mark.asyncio
    async def test_usage_async_no_entries_plain(self):
        """No entries in plain mode → print 'No usage data found.'"""
        mock_client = AsyncMock()
        mock_response = {"data": [], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _usage_async(ctx, output_json=False, start_date=None, end_date=None)

            calls = [str(c) for c in mock_echo.call_args_list]
            assert any("No usage data found" in c for c in calls)

    @pytest.mark.asyncio
    async def test_usage_async_no_entries_rich(self):
        """No entries in rich mode → console.print warning."""
        mock_client = AsyncMock()
        mock_response = {"data": [], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _usage_async(ctx, output_json=False, start_date=None, end_date=None)

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_usage_async_no_entries_with_start_date_plain(self):
        """No entries with start date filter in plain mode."""
        mock_client = AsyncMock()
        mock_response = {"data": [], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _usage_async(ctx, output_json=False, start_date="2025-01-01", end_date=None)

            calls = [str(c) for c in mock_echo.call_args_list]
            assert any("from 2025-01-01" in c for c in calls)

    @pytest.mark.asyncio
    async def test_usage_async_no_entries_with_end_date_plain(self):
        """No entries with end date filter in plain mode."""
        mock_client = AsyncMock()
        mock_response = {"data": [], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _usage_async(ctx, output_json=False, start_date=None, end_date="2025-02-01")

            calls = [str(c) for c in mock_echo.call_args_list]
            assert any("to 2025-02-01" in c for c in calls)

    @pytest.mark.asyncio
    async def test_usage_async_no_entries_with_both_dates_plain(self):
        """No entries with both start and end dates in plain mode."""
        mock_client = AsyncMock()
        mock_response = {"data": [], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _usage_async(
                ctx, output_json=False, start_date="2025-01-01", end_date="2025-02-01"
            )

            calls = [str(c) for c in mock_echo.call_args_list]
            assert any("from 2025-01-01" in c for c in calls)
            assert any("to 2025-02-01" in c for c in calls)


# ---------------------------------------------------------------------------
# _usage_async tests — JSON output
# ---------------------------------------------------------------------------


class TestUsageAsyncJsonOutput:
    """Tests for _usage_async JSON output mode."""

    @pytest.mark.asyncio
    async def test_usage_async_json_entries_with_model_dump(self):
        """JSON output with entries having model_dump method."""
        mock_client = AsyncMock()
        entry = MagicMock()
        entry.model_dump = MagicMock(
            return_value={
                "sku": "test-sku",
                "amount": 0.001,
                "currency": "USD",
                "units": 100.0,
            }
        )
        mock_response = {"data": [entry], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _usage_async(ctx, output_json=True, start_date=None, end_date=None)

            mock_echo.assert_called_once()
            data = json.loads(mock_echo.call_args[0][0])
            assert data["count"] == 1
            assert len(data["data"]) == 1
            assert data["data"][0]["sku"] == "test-sku"

    @pytest.mark.asyncio
    async def test_usage_async_json_entries_as_dicts(self):
        """JSON output with entries that are plain dicts."""
        mock_client = AsyncMock()
        entry = {"sku": "chat-sku", "amount": 0.002, "currency": "USD", "units": 200.0}
        mock_response = {"data": [entry], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _usage_async(ctx, output_json=True, start_date=None, end_date=None)

            output = mock_echo.call_args[0][0]
            data = json.loads(output)
            assert data["count"] == 1
            assert data["data"][0]["sku"] == "chat-sku"

    @pytest.mark.asyncio
    async def test_usage_async_json_entries_getattr_fallback(self):
        """JSON output with entries using getattr fallback (no model_dump, not dict)."""
        mock_client = AsyncMock()
        entry = SimpleNamespace(
            timestamp="2025-01-01T00:00:00Z",
            sku="img-sku",
            amount=0.01,
            currency="USD",
            units=1.0,
            pricePerUnitUsd=0.01,
            notes="test note",
        )
        mock_response = {"data": [entry], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _usage_async(ctx, output_json=True, start_date=None, end_date=None)

            output = mock_echo.call_args[0][0]
            data = json.loads(output)
            assert data["count"] == 1
            assert data["data"][0]["sku"] == "img-sku"
            assert data["data"][0]["notes"] == "test note"

    @pytest.mark.asyncio
    async def test_usage_async_json_has_more_true(self):
        """JSON output sets hasMore=True when a nextCursor is returned."""
        mock_client = AsyncMock()
        mock_response = {"data": [], "nextCursor": "CURSOR_2"}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _usage_async(ctx, output_json=True, start_date=None, end_date=None)

            output = mock_echo.call_args[0][0]
            data = json.loads(output)
            assert data["hasMore"] is True

    @pytest.mark.asyncio
    async def test_usage_async_json_has_more_false(self):
        """JSON output sets hasMore=False on the last page (nextCursor=None)."""
        mock_client = AsyncMock()
        mock_response = {"data": [], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _usage_async(ctx, output_json=True, start_date=None, end_date=None)

            output = mock_echo.call_args[0][0]
            data = json.loads(output)
            assert data["hasMore"] is False

    @pytest.mark.asyncio
    async def test_usage_async_json_unknown_response_type(self):
        """JSON output with completely unknown response (else branch for entries)."""
        mock_client = AsyncMock()
        # Response is neither BillingUsageHistoryResponse nor dict
        mock_client.billing.get_usage_history = AsyncMock(return_value=None)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _usage_async(ctx, output_json=True, start_date=None, end_date=None)

            output = mock_echo.call_args[0][0]
            data = json.loads(output)
            assert data["count"] == 0
            assert data["data"] == []

    @pytest.mark.asyncio
    async def test_usage_async_json_billing_usage_response_instance(self):
        """BillingUsageHistoryResponse instance is handled in JSON output."""
        from venice_ai.types.api.billing import (
            BillingUsageEntry,
            BillingUsageHistoryResponse,
        )

        mock_client = AsyncMock()
        entry = BillingUsageEntry(
            sku="text-completion",
            amount=0.0005,
            currency="USD",
            units=50.0,
            pricePerUnitUsd=0.00001,
            notes="inference",
            timestamp="2025-01-15T10:00:00Z",
            inferenceDetails=None,
        )
        response = BillingUsageHistoryResponse(data=[entry], nextCursor=None)
        mock_client.billing.get_usage_history = AsyncMock(return_value=response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _usage_async(ctx, output_json=True, start_date=None, end_date=None)

            output = mock_echo.call_args[0][0]
            data = json.loads(output)
            assert data["count"] == 1


# ---------------------------------------------------------------------------
# _usage_async tests — with entries (display path)
# ---------------------------------------------------------------------------


class TestUsageAsyncWithEntries:
    """Tests for _usage_async with entries (display paths)."""

    def _make_entry(
        self,
        sku="test-sku",
        amount=0.001,
        currency="USD",
        units=100.0,
        timestamp="2025-01-01T00:00:00Z",
    ):
        return SimpleNamespace(
            sku=sku,
            amount=amount,
            currency=currency,
            units=units,
            timestamp=timestamp,
            pricePerUnitUsd=0.00001,
            notes="",
        )

    @pytest.mark.asyncio
    async def test_usage_async_plain_with_usd_entries(self):
        """Plain mode with USD entries shows totals and SKU breakdown."""
        mock_client = AsyncMock()
        entry = self._make_entry(sku="chat-completions", amount=0.001, currency="USD", units=1000.0)
        mock_response = {"data": [entry], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _usage_async(ctx, output_json=False, start_date=None, end_date=None)

            calls = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Usage Summary" in calls
            assert "Total USD" in calls
            assert "chat-completions" in calls

    @pytest.mark.asyncio
    async def test_usage_async_plain_with_diem_entries(self):
        """Plain mode with DIEM entries shows DIEM totals."""
        mock_client = AsyncMock()
        entry = self._make_entry(sku="diem-sku", amount=-50.0, currency="DIEM", units=500.0)
        mock_response = {"data": [entry], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _usage_async(ctx, output_json=False, start_date=None, end_date=None)

            calls = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Total DIEM" in calls

    @pytest.mark.asyncio
    async def test_usage_async_rich_with_entries(self):
        """Rich mode with entries prints panel and tables."""
        mock_client = AsyncMock()
        entry1 = self._make_entry(sku="chat", amount=0.002, currency="USD", units=200.0)
        entry2 = self._make_entry(sku="image", amount=0.05, currency="USD", units=5.0)
        mock_response = {"data": [entry1, entry2], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _usage_async(ctx, output_json=False, start_date=None, end_date=None)

            # Panel + SKU table + recent table
            assert mock_console.print.call_count >= 2

    @pytest.mark.asyncio
    async def test_usage_async_rich_with_date_range_title(self):
        """Rich mode with both dates shows date range in title."""
        mock_client = AsyncMock()
        entry = self._make_entry()
        mock_response = {"data": [entry], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _usage_async(
                ctx, output_json=False, start_date="2025-01-01", end_date="2025-02-01"
            )

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_usage_async_rich_only_start_date(self):
        """Rich mode with only start_date."""
        mock_client = AsyncMock()
        entry = self._make_entry()
        mock_response = {"data": [entry], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _usage_async(ctx, output_json=False, start_date="2025-01-01", end_date=None)

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_usage_async_rich_only_end_date(self):
        """Rich mode with only end_date."""
        mock_client = AsyncMock()
        entry = self._make_entry()
        mock_response = {"data": [entry], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _usage_async(ctx, output_json=False, start_date=None, end_date="2025-02-01")

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_usage_async_plain_with_many_entries(self):
        """Plain mode with more than 5 entries only shows last 5 in recent."""
        mock_client = AsyncMock()
        entries = [
            self._make_entry(sku=f"sku-{i}", amount=0.001 * i, units=float(i))
            for i in range(1, 8)  # 7 entries
        ]
        mock_response = {"data": entries, "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _usage_async(ctx, output_json=False, start_date=None, end_date=None)

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Usage Summary (7 entries)" in calls_str

    @pytest.mark.asyncio
    async def test_usage_async_rich_with_diem_entries(self):
        """Rich mode shows DIEM totals when DIEM entries present."""
        mock_client = AsyncMock()
        entry = self._make_entry(sku="diem-sku", amount=-100.0, currency="DIEM", units=1000.0)
        mock_response = {"data": [entry], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _usage_async(ctx, output_json=False, start_date=None, end_date=None)

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_usage_async_ctx_obj_none(self):
        """ctx.obj is None → plain defaults to False (rich mode)."""
        mock_client = AsyncMock()
        entry = self._make_entry()
        mock_response = {"data": [entry], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = MagicMock()
            ctx.obj = None
            await _usage_async(ctx, output_json=False, start_date=None, end_date=None)

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_usage_async_date_normalization_called(self):
        """Start and end dates get normalized via _normalize_date."""
        mock_client = AsyncMock()
        mock_response = {"data": [], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.console"),
            patch(
                "venice_ai.cli.commands.account._normalize_date", wraps=_normalize_date
            ) as mock_norm,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _usage_async(
                ctx, output_json=False, start_date="2025-01-01", end_date="2025-02-01"
            )

            assert mock_norm.call_count == 2

    @pytest.mark.asyncio
    async def test_usage_async_billing_usage_response_instance_plain(self):
        """BillingUsageHistoryResponse object is handled directly in plain mode."""
        from venice_ai.types.api.billing import (
            BillingUsageEntry,
            BillingUsageHistoryResponse,
        )

        mock_client = AsyncMock()
        entry = BillingUsageEntry(
            sku="text-completion",
            amount=0.0005,
            currency="USD",
            units=50.0,
            pricePerUnitUsd=0.00001,
            notes="inference",
            timestamp="2025-01-15T10:00:00Z",
            inferenceDetails=None,
        )
        response = BillingUsageHistoryResponse(data=[entry], nextCursor=None)
        mock_client.billing.get_usage_history = AsyncMock(return_value=response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _usage_async(ctx, output_json=False, start_date=None, end_date=None)

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Usage Summary" in calls_str


# ---------------------------------------------------------------------------
# _usage_async tests — SKU grouping
# ---------------------------------------------------------------------------


class TestUsageAsyncSKUGrouping:
    """Tests for SKU grouping and sorting logic."""

    @pytest.mark.asyncio
    async def test_multiple_skus_sorted_by_cost(self):
        """Multiple SKUs are sorted by total_cost descending."""
        mock_client = AsyncMock()
        entries = [
            SimpleNamespace(
                sku="cheap-sku",
                amount=0.001,
                currency="USD",
                units=10.0,
                timestamp="2025-01-01T00:00:00Z",
                pricePerUnitUsd=0.0001,
                notes="",
            ),
            SimpleNamespace(
                sku="expensive-sku",
                amount=10.0,
                currency="USD",
                units=100.0,
                timestamp="2025-01-01T00:00:00Z",
                pricePerUnitUsd=0.1,
                notes="",
            ),
        ]
        mock_response = {"data": entries, "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _usage_async(ctx, output_json=False, start_date=None, end_date=None)

            output = "".join(str(c) for c in mock_echo.call_args_list)
            pos_expensive = output.find("expensive-sku")
            pos_cheap = output.find("cheap-sku")
            assert pos_expensive < pos_cheap

    @pytest.mark.asyncio
    async def test_entry_with_none_sku_uses_unknown(self):
        """Entry with None SKU is grouped under 'unknown'."""
        mock_client = AsyncMock()
        entry = SimpleNamespace(
            sku=None,
            amount=0.001,
            currency="USD",
            units=10.0,
            timestamp="2025-01-01T00:00:00Z",
            pricePerUnitUsd=0.0001,
            notes="",
        )
        mock_response = {"data": [entry], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _usage_async(ctx, output_json=False, start_date=None, end_date=None)

            output = "".join(str(c) for c in mock_echo.call_args_list)
            assert "unknown" in output

    @pytest.mark.asyncio
    async def test_rich_multiple_skus(self):
        """Rich mode with multiple SKUs → SKU table is shown."""
        mock_client = AsyncMock()
        entries = [
            SimpleNamespace(
                sku="chat",
                amount=0.002,
                currency="USD",
                units=200.0,
                timestamp="2025-01-01T00:00:00Z",
                pricePerUnitUsd=0.01,
                notes="",
            ),
            SimpleNamespace(
                sku="image",
                amount=0.05,
                currency="USD",
                units=5.0,
                timestamp="2025-01-01T00:00:00Z",
                pricePerUnitUsd=0.01,
                notes="",
            ),
        ]
        mock_response = {"data": entries, "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _usage_async(ctx, output_json=False, start_date=None, end_date=None)

            # Summary panel + SKU breakdown table + recent entries table = 3 calls
            assert mock_console.print.call_count >= 3


# ---------------------------------------------------------------------------
# usage() CLI entrypoint tests
# ---------------------------------------------------------------------------


class TestUsageCLI:
    """Tests for the usage() click command entrypoint."""

    def test_usage_help(self):
        """Check --help output for usage command."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["account", "usage", "--help"])
        assert result.exit_code == 0
        assert "usage" in result.output.lower() or "Usage" in result.output

    def test_usage_invokes_asyncio_run(self):
        """usage() calls asyncio.run with _usage_async coroutine."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch(
            "venice_ai.cli.commands.account.asyncio.run", side_effect=consume_coro
        ) as mock_run:
            runner.invoke(cli, ["account", "usage"])
            assert mock_run.called

    def test_usage_json_flag(self):
        """usage --json passes output_json=True."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch("venice_ai.cli.commands.account.asyncio.run", side_effect=consume_coro):
            result = runner.invoke(cli, ["account", "usage", "--json"])
            assert result.exit_code == 0

    def test_usage_with_dates(self):
        """usage --start-date --end-date passes dates correctly."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch("venice_ai.cli.commands.account.asyncio.run", side_effect=consume_coro):
            result = runner.invoke(
                cli,
                [
                    "account",
                    "usage",
                    "--start-date",
                    "2025-01-01",
                    "--end-date",
                    "2025-02-01",
                ],
            )
            assert result.exit_code == 0


# ---------------------------------------------------------------------------
# account() group tests
# ---------------------------------------------------------------------------


class TestAccountGroup:
    """Tests for the account() click group."""

    def test_account_group_help(self):
        """Check --help output for the account group."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["account", "--help"])
        assert result.exit_code == 0
        assert "balance" in result.output
        assert "usage" in result.output


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


class TestUsageAsyncEdgeCases:
    """Additional edge cases for _usage_async."""

    @pytest.mark.asyncio
    async def test_usage_async_no_entries_rich_both_dates(self):
        """No entries with both dates in rich mode."""
        mock_client = AsyncMock()
        mock_response = {"data": [], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _usage_async(
                ctx, output_json=False, start_date="2025-01-01", end_date="2025-02-01"
            )

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_usage_async_plain_no_usd_no_diem(self):
        """Plain mode with entries of unknown currency (no USD, no DIEM totals shown)."""
        mock_client = AsyncMock()
        entry = SimpleNamespace(
            sku="other",
            amount=5.0,
            currency="BUNDLED_CREDITS",
            units=50.0,
            timestamp="2025-01-01T00:00:00Z",
            pricePerUnitUsd=0.1,
            notes="",
        )
        mock_response = {"data": [entry], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _usage_async(ctx, output_json=False, start_date=None, end_date=None)

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Total Units" in calls_str
            assert "Total USD" not in calls_str
            assert "Total DIEM" not in calls_str

    @pytest.mark.asyncio
    async def test_usage_async_rich_no_usd_no_diem(self):
        """Rich mode with non-USD/DIEM entries doesn't show USD/DIEM totals."""
        mock_client = AsyncMock()
        entry = SimpleNamespace(
            sku="bundled-sku",
            amount=5.0,
            currency="BUNDLED_CREDITS",
            units=50.0,
            timestamp="2025-01-01T00:00:00Z",
            pricePerUnitUsd=0.1,
            notes="",
        )
        mock_response = {"data": [entry], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _usage_async(ctx, output_json=False, start_date=None, end_date=None)

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_usage_async_billing_usage_history_response_data_empty(self):
        """BillingUsageHistoryResponse with empty data → no entries path."""
        from venice_ai.types.api.billing import BillingUsageHistoryResponse

        mock_client = AsyncMock()
        response = BillingUsageHistoryResponse(data=[], nextCursor=None)
        mock_client.billing.get_usage_history = AsyncMock(return_value=response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _usage_async(ctx, output_json=False, start_date=None, end_date=None)

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "No usage data found" in calls_str

    @pytest.mark.asyncio
    async def test_usage_async_entry_with_none_amount_and_units(self):
        """Entries with None amount/units should not cause errors (defaults to 0)."""
        mock_client = AsyncMock()
        entry = SimpleNamespace(
            sku="test",
            amount=None,
            currency="USD",
            units=None,
            timestamp="2025-01-01T00:00:00Z",
            pricePerUnitUsd=0.01,
            notes="",
        )
        mock_response = {"data": [entry], "nextCursor": None}
        mock_client.billing.get_usage_history = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.account.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _usage_async(ctx, output_json=False, start_date=None, end_date=None)

            calls_str = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Usage Summary" in calls_str
