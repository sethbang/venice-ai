"""``venice models capabilities <id>`` — typed capability view.

Wraps :meth:`venice_ai.resources.models.Models.get_capabilities`. The
SDK returns one of :class:`ChatCapabilities` / :class:`ImageCapabilities` /
:class:`VideoCapabilities` / :class:`InpaintCapabilities` /
:class:`GenericCapabilities`; we render whichever shape came back.
"""

from __future__ import annotations

import asyncio
import json

import click

from ...utils.console import console


@click.command("capabilities")
@click.argument("model_id")
@click.option("--json", "output_json", is_flag=True, help="Output JSON for scripting.")
@click.pass_context
def capabilities(ctx: click.Context, model_id: str, output_json: bool) -> None:
    """Show the typed capability view for a model.

    Returns capability flags appropriate to the model's type (chat,
    image, video, inpaint, or generic).

    Examples:

      venice models capabilities llama-3.3-70b
      venice models capabilities flux-dev --json
    """
    asyncio.run(_capabilities_async(ctx, model_id=model_id, output_json=output_json))


async def _capabilities_async(ctx: click.Context, *, model_id: str, output_json: bool) -> None:
    from venice_ai import VeniceClient

    from ...config import get_client_kwargs

    async with VeniceClient(**get_client_kwargs()) as client:
        try:
            caps = await client.models.get_capabilities(model_id=model_id)
        except ValueError as exc:
            if output_json:
                click.echo(json.dumps({"error": str(exc)}))
            else:
                console.print(f"[red]Model not found:[/red] {exc}")
            ctx.exit(1)
            return

    if hasattr(caps, "model_dump"):
        payload = caps.model_dump()
    else:
        payload = dict(getattr(caps, "__dict__", {}))

    if output_json:
        click.echo(json.dumps(payload, default=str))
        return

    _render_capabilities(model_id, payload)


def _render_capabilities(model_id: str, payload: dict) -> None:
    """Render the capability dict as a Rich panel + key/value table."""
    from rich.panel import Panel
    from rich.table import Table

    cap_type = payload.get("type", "unknown")

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")

    for key, value in payload.items():
        if key == "type":
            continue
        if isinstance(value, bool):
            mark = "[green]✓[/green]" if value else "[dim]✗[/dim]"
            table.add_row(key, mark)
        elif isinstance(value, list):
            table.add_row(key, ", ".join(str(v) for v in value) if value else "[dim](none)[/dim]")
        elif value is None:
            table.add_row(key, "[dim](unset)[/dim]")
        else:
            table.add_row(key, str(value))

    panel = Panel(
        table,
        title=f"{model_id} — type=[yellow]{cap_type}[/yellow]",
        border_style="cyan",
    )
    console.print(panel)


__all__ = ["capabilities"]
