"""
OutputManager — unified plain/rich output for Venice AI CLI.

Encapsulates the ``if plain:`` branching that pervades CLI commands so that
each call-site becomes a single method call instead of a forked code block.
"""

from __future__ import annotations

import click

from .console import console, is_plain_mode


class OutputManager:
    """Encapsulates plain/rich output branching for CLI commands.

    Instantiate once per command invocation and call its methods instead of
    writing inline ``if plain:`` / ``else:`` blocks.
    """

    # ------------------------------------------------------------------
    # Simple messages
    # ------------------------------------------------------------------

    @staticmethod
    def info(message: str) -> None:
        """Print an informational message."""
        if is_plain_mode():
            click.echo(f"INFO: {message}")
        else:
            console.print(f"[blue]ℹ️  {message}[/blue]")

    @staticmethod
    def success(message: str) -> None:
        """Print a success message."""
        if is_plain_mode():
            click.echo(f"SUCCESS: {message}")
        else:
            console.print(f"[bold green]✅ {message}[/bold green]")

    @staticmethod
    def warning(message: str) -> None:
        """Print a warning message."""
        if is_plain_mode():
            click.echo(f"WARNING: {message}")
        else:
            console.print(f"[yellow]⚠️  {message}[/yellow]")

    @staticmethod
    def error(message: str) -> None:
        """Print an error message."""
        if is_plain_mode():
            click.echo(f"ERROR: {message}", err=True)
        else:
            console.print(f"[red]❌ Error:[/red] {message}")

    @staticmethod
    def echo(message: str) -> None:
        """Print plain text (no decoration in either mode)."""
        if is_plain_mode():
            click.echo(message)
        else:
            console.print(message)

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------

    @staticmethod
    def table(
        headers: list[str],
        rows: list[list[str]],
        *,
        title: str | None = None,
        col_styles: list[str] | None = None,
    ) -> None:
        """Render a table.

        Parameters
        ----------
        headers:
            Column header strings.
        rows:
            List of rows, each a list of cell strings.
        title:
            Optional table title.
        col_styles:
            Optional per-column Rich styles (ignored in plain mode).
        """
        if is_plain_mode():
            if title:
                click.echo(f"\n{title}")
                click.echo("=" * len(title))
            # Calculate column widths
            widths = [len(h) for h in headers]
            for row in rows:
                for i, cell in enumerate(row):
                    if i < len(widths):
                        widths[i] = max(widths[i], len(cell))
            fmt = "  ".join(f"{{:<{w}}}" for w in widths)
            click.echo(fmt.format(*headers))
            click.echo("-" * (sum(widths) + 2 * (len(widths) - 1)))
            for row in rows:
                # Pad row to match header count
                padded = row + [""] * (len(headers) - len(row))
                click.echo(fmt.format(*padded[: len(headers)]))
            click.echo(f"\nTotal: {len(rows)}")
        else:
            from rich.table import Table

            tbl = Table(title=title, show_lines=False)
            styles = col_styles or []
            for i, header in enumerate(headers):
                style = styles[i] if i < len(styles) else ""
                tbl.add_column(header, style=style or None)
            for row in rows:
                tbl.add_row(*row)
            console.print(tbl)
            console.print(f"\n[dim]Total: {len(rows)}[/dim]")

    @staticmethod
    def panel(content: str, *, title: str = "", style: str = "cyan") -> None:
        """Render content inside a panel."""
        if is_plain_mode():
            if title:
                click.echo(f"=== {title} ===")
            click.echo(content)
        else:
            from rich.panel import Panel

            console.print(
                Panel(
                    content,
                    title=f"[bold]{title}[/bold]" if title else None,
                    border_style=style,
                )
            )

    @staticmethod
    def progress(message: str, pct: float | None = None) -> None:
        """Print a progress message."""
        if is_plain_mode():
            if pct is not None:
                click.echo(f"[{pct:.0f}%] {message}")
            else:
                click.echo(message)
        else:
            if pct is not None:
                console.print(f"  [dim]⏳ {pct:.0f}% {message}[/dim]")
            else:
                console.print(f"  [dim]⏳ {message}[/dim]")
