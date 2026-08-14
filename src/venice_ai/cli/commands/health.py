"""``venice-py health`` — connectivity + balance diagnostic.

Quick check that the SDK can reach Venice, the API key is valid, the model
catalog is accessible, and (optionally) a tiny embedding ping succeeds and
the wallet's prepaid x402 ledger has balance.

Exit code is 0 when every check passes, 1 if any check fails.
"""

from __future__ import annotations

import asyncio
import os
from typing import NamedTuple

import click

from ... import VeniceClient
from ..config import ensure_api_key, get_base_url
from ..utils.console import is_plain_mode
from ..utils.output import OutputManager


class _CheckResult(NamedTuple):
    name: str
    ok: bool
    detail: str


def _print_check(result: _CheckResult) -> None:
    """Print a single check line, respecting plain mode.

    In plain mode emit ASCII markers (``[OK]``/``[FAIL]``) so the output is
    free of Unicode glyphs / colors; otherwise keep the ✓/✗ glyphs.
    """
    if is_plain_mode():
        ok_marker, fail_marker = "[OK]  ", "[FAIL]"
    else:
        ok_marker, fail_marker = "  ✓ ", "  ✗ "
    if result.ok:
        click.echo(f"{ok_marker} {result.name:<24} {result.detail}")
    else:
        click.echo(f"{fail_marker} {result.name:<24} {result.detail}", err=True)


@click.command("health")
@click.option(
    "--full",
    is_flag=True,
    help="Include a tiny embedding call (small cost — typically <$0.001).",
)
@click.option(
    "--wallet",
    is_flag=True,
    help="Also check x402 prepaid balance (requires the [x402] extra and a wallet env var).",
)
@click.option(
    "--wallet-env",
    default="VENICE_X402_TEST_PRIVATE_KEY",
    show_default=True,
    help="Env var holding the wallet private key when --wallet is set.",
)
@click.pass_context
def health_command(
    ctx: click.Context,
    full: bool,
    wallet: bool,
    wallet_env: str,
) -> None:
    """Diagnose Venice API connectivity, model catalog, and balance.

    Runs a sequence of small checks against the Venice API and reports
    each one. Exits 0 if every check passes, 1 otherwise.

    Default checks:
      \b
      - API key resolves from env / config
      - Models catalog is reachable
      - Billing balance is readable

    Optional:
      \b
      --full     Adds a small embedding call to verify the embedding endpoint.
      --wallet   Adds an x402 prepaid-ledger balance read using the wallet
                 private key in `--wallet-env`.

    Examples::

        venice-py health
        venice-py health --full
        venice-py health --wallet --wallet-env MY_WALLET_KEY
    """
    api_key = ensure_api_key()
    asyncio.run(_run_checks(api_key, full=full, wallet=wallet, wallet_env=wallet_env))


async def _run_checks(api_key: str, *, full: bool, wallet: bool, wallet_env: str) -> None:
    """Run the health check sequence; print results; raise SystemExit on failure."""
    OutputManager.info("venice-py health")
    results: list[_CheckResult] = []

    # 1. API key presence — already validated by ensure_api_key, but record it.
    masked = (api_key[:4] + "…" + api_key[-4:]) if len(api_key) >= 12 else "[redacted]"
    results.append(_CheckResult("api_key", True, f"present ({masked})"))
    _print_check(results[-1])

    async with VeniceClient(api_key=api_key, base_url=get_base_url()) as client:
        # 2. Models catalog
        try:
            catalog = await client.models.list(type="text")
            count = len(catalog.data) if catalog.data else 0
            results.append(_CheckResult("models.list", True, f"{count} text models reachable"))
        except Exception as exc:  # pragma: no cover — network-dependent path
            results.append(_CheckResult("models.list", False, f"{type(exc).__name__}: {exc}"))
        _print_check(results[-1])

        # 3. Billing balance
        try:
            balance = await client.billing.get_balance()
            usd = balance.balances.usd if balance.balances else None
            diem = balance.balances.diem if balance.balances else None
            detail = (
                f"USD ${usd}" + (f" / DIEM {diem}" if diem is not None else "")
                if usd is not None
                else "no USD balance reported"
            )
            results.append(_CheckResult("billing.get_balance", True, detail))
        except Exception as exc:  # pragma: no cover — network-dependent
            results.append(
                _CheckResult("billing.get_balance", False, f"{type(exc).__name__}: {exc}")
            )
        _print_check(results[-1])

        # 4. Optional embedding ping
        if full:
            try:
                model_id = await client.models.resolve_embedding()
                resp = await client.embeddings.create(
                    model=model_id,
                    input="ping",
                )
                tokens = resp.usage.total_tokens if resp.usage else "?"
                results.append(
                    _CheckResult("embeddings.create", True, f"{model_id} ({tokens} tokens)")
                )
            except Exception as exc:
                results.append(
                    _CheckResult("embeddings.create", False, f"{type(exc).__name__}: {exc}")
                )
            _print_check(results[-1])

        # 5. Optional wallet balance
        if wallet:
            wallet_key = os.environ.get(wallet_env)
            if not wallet_key:
                results.append(_CheckResult("x402.balance", False, f"env var {wallet_env} not set"))
                _print_check(results[-1])
            else:
                try:
                    from ...auth.x402 import X402Auth
                except ImportError as exc:
                    results.append(
                        _CheckResult(
                            "x402.balance",
                            False,
                            f"[x402] extra not installed: {exc}",
                        )
                    )
                    _print_check(results[-1])
                else:
                    try:
                        auth = X402Auth(private_key=wallet_key)
                        bal = await client.x402.balance(auth=auth)
                        usd_balance = bal.data.balanceUsd
                        results.append(
                            _CheckResult(
                                "x402.balance",
                                True,
                                f"{auth.wallet_address[:10]}… ${usd_balance}",
                            )
                        )
                    except Exception as exc:
                        results.append(
                            _CheckResult("x402.balance", False, f"{type(exc).__name__}: {exc}")
                        )
                    _print_check(results[-1])

    failed = [r for r in results if not r.ok]
    click.echo()
    if failed:
        OutputManager.error(f"{len(failed)} of {len(results)} checks failed.")
        raise SystemExit(1)
    OutputManager.success(f"All {len(results)} checks passed.")
