"""Tests for the ``venice-py health`` CLI subcommand.

Mocks every network-touching call (`models.list`, `billing.get_balance`,
`embeddings.create`, `x402.balance`) so the tests run offline. Asserts on
exit codes and key output fragments.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from venice_ai.cli.cli import cli
from venice_ai.types.api.billing import BillingBalanceResponse, BillingBalances
from venice_ai.types.api.models import ModelsListResponse


@pytest.fixture(autouse=True)
def _reset_plain_mode():
    """Reset console plain-mode state between tests.

    ``--plain`` flips a module global in ``utils.console`` that persists within
    an xdist worker; leaving it set would pollute non-plain tests. The
    ``utils`` package re-exports the ``console`` instance, so reach the actual
    submodule via ``sys.modules`` rather than an attribute import.
    """
    import importlib

    console_mod = importlib.import_module("venice_ai.cli.utils.console")

    saved_flag = console_mod._plain_mode
    saved_console = console_mod.console
    yield
    console_mod._plain_mode = saved_flag
    console_mod.console = saved_console


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def fake_models_response() -> ModelsListResponse:
    return ModelsListResponse(object="list", data=[], type="text")


@pytest.fixture
def fake_billing_response() -> BillingBalanceResponse:
    return BillingBalanceResponse(
        balances=BillingBalances(usd=42.5, diem=0.75),
        canConsume=True,
    )


def _patch_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    models_resp,
    billing_resp,
    embeddings_resp=None,
    x402_balance_resp=None,
    embeddings_resolve: str | None = None,
) -> MagicMock:
    """Patch VeniceClient so it returns the supplied fakes."""
    mock_client = MagicMock()
    # __aenter__ / __aexit__ return mock_client itself
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.models.list = AsyncMock(return_value=models_resp)
    mock_client.billing.get_balance = AsyncMock(return_value=billing_resp)
    if embeddings_resp is not None:
        mock_client.embeddings.create = AsyncMock(return_value=embeddings_resp)
    if embeddings_resolve is not None:
        mock_client.models.resolve_embedding = AsyncMock(return_value=embeddings_resolve)
    if x402_balance_resp is not None:
        mock_client.x402.balance = AsyncMock(return_value=x402_balance_resp)

    # Patch the VeniceClient constructor — return our mock when instantiated.
    def _factory(*args, **kwargs):
        return mock_client

    monkeypatch.setattr(
        "venice_ai.cli.commands.health.VeniceClient", MagicMock(side_effect=_factory)
    )
    return mock_client


def _ensure_api_key_returns(value: str = "vn_test_key_abc12345"):
    """Patch ensure_api_key to bypass real config lookup."""
    return patch("venice_ai.cli.commands.health.ensure_api_key", return_value=value)


def test_health_default_all_pass(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    fake_models_response: ModelsListResponse,
    fake_billing_response: BillingBalanceResponse,
) -> None:
    _patch_client(
        monkeypatch,
        models_resp=fake_models_response,
        billing_resp=fake_billing_response,
    )
    with _ensure_api_key_returns():
        result = cli_runner.invoke(cli, ["health"])
    assert result.exit_code == 0, result.output
    assert "api_key" in result.output
    assert "models.list" in result.output
    assert "billing.get_balance" in result.output
    assert "All 3 checks passed" in result.output


def test_health_default_does_not_run_full_or_wallet_checks(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    fake_models_response: ModelsListResponse,
    fake_billing_response: BillingBalanceResponse,
) -> None:
    _patch_client(
        monkeypatch,
        models_resp=fake_models_response,
        billing_resp=fake_billing_response,
    )
    with _ensure_api_key_returns():
        result = cli_runner.invoke(cli, ["health"])
    assert result.exit_code == 0
    assert "embeddings.create" not in result.output
    assert "x402.balance" not in result.output


def test_health_full_runs_embedding(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    fake_models_response: ModelsListResponse,
    fake_billing_response: BillingBalanceResponse,
) -> None:
    fake_emb = MagicMock()
    fake_emb.usage = MagicMock(total_tokens=3)
    _patch_client(
        monkeypatch,
        models_resp=fake_models_response,
        billing_resp=fake_billing_response,
        embeddings_resp=fake_emb,
        embeddings_resolve="text-embedding-bge-m3",
    )
    with _ensure_api_key_returns():
        result = cli_runner.invoke(cli, ["health", "--full"])
    assert result.exit_code == 0, result.output
    assert "embeddings.create" in result.output
    assert "text-embedding-bge-m3" in result.output
    assert "All 4 checks passed" in result.output


def test_health_models_list_failure_exits_one(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    fake_billing_response: BillingBalanceResponse,
) -> None:
    """When models.list raises, the check fails and exit code is 1."""
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.models.list = AsyncMock(side_effect=RuntimeError("upstream is down"))
    mock_client.billing.get_balance = AsyncMock(return_value=fake_billing_response)
    monkeypatch.setattr(
        "venice_ai.cli.commands.health.VeniceClient",
        MagicMock(side_effect=lambda *a, **k: mock_client),
    )
    with _ensure_api_key_returns():
        result = cli_runner.invoke(cli, ["health"])
    assert result.exit_code == 1
    assert "checks failed" in result.output
    assert "models.list" in result.output


def test_health_wallet_missing_env_var_fails(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    fake_models_response: ModelsListResponse,
    fake_billing_response: BillingBalanceResponse,
) -> None:
    """--wallet without the env var set fails the wallet check (exit 1)."""
    _patch_client(
        monkeypatch,
        models_resp=fake_models_response,
        billing_resp=fake_billing_response,
    )
    monkeypatch.delenv("VENICE_X402_TEST_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("MY_TEST_WALLET", raising=False)
    with _ensure_api_key_returns():
        result = cli_runner.invoke(cli, ["health", "--wallet", "--wallet-env", "MY_TEST_WALLET"])
    assert result.exit_code == 1
    assert "MY_TEST_WALLET" in result.output
    assert "x402.balance" in result.output


def test_health_wallet_with_env_var_set(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    fake_models_response: ModelsListResponse,
    fake_billing_response: BillingBalanceResponse,
) -> None:
    """--wallet with a valid private key in env runs the x402.balance check."""
    pytest.importorskip("eth_account")
    pytest.importorskip("siwe")

    fake_balance = MagicMock()
    fake_balance.data = MagicMock(balanceUsd=12.0)
    _patch_client(
        monkeypatch,
        models_resp=fake_models_response,
        billing_resp=fake_billing_response,
        x402_balance_resp=fake_balance,
    )
    monkeypatch.setenv("MY_TEST_WALLET", "0x" + "a" * 63 + "b")
    with _ensure_api_key_returns():
        result = cli_runner.invoke(cli, ["health", "--wallet", "--wallet-env", "MY_TEST_WALLET"])
    assert result.exit_code == 0, result.output
    assert "x402.balance" in result.output
    assert "$12.0" in result.output


def test_health_plain_mode_uses_ascii_markers(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    fake_models_response: ModelsListResponse,
    fake_billing_response: BillingBalanceResponse,
) -> None:
    """In --plain mode, output uses ASCII markers, never ✓/✗ glyphs."""
    _patch_client(
        monkeypatch,
        models_resp=fake_models_response,
        billing_resp=fake_billing_response,
    )
    with _ensure_api_key_returns():
        result = cli_runner.invoke(cli, ["--plain", "health"])
    assert result.exit_code == 0, result.output
    assert "✓" not in result.output
    assert "✗" not in result.output
    assert "[OK]" in result.output


def test_health_help_shows_options(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(cli, ["health", "--help"])
    assert result.exit_code == 0
    assert "--full" in result.output
    assert "--wallet" in result.output
    assert "--wallet-env" in result.output


def test_health_no_api_key_raises_click_exception(
    cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ensure_api_key raises (no key), the command exits non-zero with the helpful message."""
    # Don't patch VeniceClient — ensure_api_key should fail before construction.
    monkeypatch.delenv("VENICE_API_KEY", raising=False)
    # Mock ensure_api_key to raise the actual ClickException it would raise.
    import click as _click

    with patch(
        "venice_ai.cli.commands.health.ensure_api_key",
        side_effect=_click.ClickException("No API key found."),
    ):
        result = cli_runner.invoke(cli, ["health"])
    assert result.exit_code != 0
    assert "No API key" in result.output
