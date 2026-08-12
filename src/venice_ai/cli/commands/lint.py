"""``venice lint <path>`` — flag v1 / OpenAI-style / non-idiomatic Venice patterns.

Wraps :mod:`venice_ai.cli.utils.lint_rules` as a Click subcommand. Reports
findings in flake8-compatible ``path:line:col: CODE message`` format and
exits 1 when findings are present (or 0 when only informational findings
are present without ``--strict``).
"""

from __future__ import annotations

from pathlib import Path

import click

from ..utils.lint_rules import INFORMATIONAL_CODES, Finding, lint_path
from ..utils.output import OutputManager


@click.command("lint")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--code",
    "codes",
    multiple=True,
    help="Filter to specific rule codes (V100, V200, ...). Repeatable.",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Treat informational findings (e.g., V401) as errors.",
)
@click.pass_context
def lint_command(
    ctx: click.Context,
    path: Path,
    codes: tuple[str, ...],
    strict: bool,
) -> None:
    """Lint Python code for v1 / OpenAI-style / non-idiomatic Venice patterns.

    Walks PATH (a Python file or a directory tree) and reports patterns that
    won't work or work sub-optimally on Venice SDK v2 — e.g., AsyncVeniceClient
    imports, hardcoded model IDs, max_tokens= kwargs, or PaymentRequiredError
    .payment_instructions accesses.

    Exits 0 if clean (or only informational findings without --strict); exits
    1 if any error-level finding is present.

    Examples::

        venice lint src/

        venice lint --strict path/to/file.py

        venice lint --code V100 --code V200 src/
    """
    findings: list[Finding] = lint_path(path, codes=codes if codes else None)

    for finding in findings:
        click.echo(str(finding))

    by_code: dict[str, int] = {}
    for finding in findings:
        by_code[finding.code] = by_code.get(finding.code, 0) + 1

    if findings:
        files = len({f.path for f in findings})
        summary = f"{len(findings)} finding(s) across {files} file(s):"
        # Determine exit code: any error-level finding (or any finding under --strict).
        has_error_level = any(f.code not in INFORMATIONAL_CODES for f in findings)
        if strict or has_error_level:
            OutputManager.error(summary)
        else:
            OutputManager.warning(summary)
        for code, count in sorted(by_code.items()):
            tag = " [info]" if code in INFORMATIONAL_CODES else ""
            click.echo(f"  {code}: {count}{tag}", err=True)

        if strict or has_error_level:
            ctx.exit(1)
        return

    OutputManager.success("No findings.")
