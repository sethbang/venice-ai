"""``venice models get <id>`` — fetch a single ModelResponse.

Wraps :meth:`venice_ai.resources.models.Models.get`.
"""

from __future__ import annotations

import asyncio
import json

import click

from ...utils.console import console


@click.command("get")
@click.argument("model_id")
@click.option("--json", "output_json", is_flag=True, help="Output JSON for scripting.")
@click.pass_context
def get(ctx: click.Context, model_id: str, output_json: bool) -> None:
    """Fetch a single model entry by ID.

    Examples:

      venice models get llama-3.3-70b
      venice models get qwen3-235b --json
    """
    asyncio.run(_get_async(ctx, model_id=model_id, output_json=output_json))


async def _get_async(ctx: click.Context, *, model_id: str, output_json: bool) -> None:
    from venice_ai import VeniceClient

    from ...config import get_client_kwargs
    from .formatters import ModelFormatter

    async with VeniceClient(**get_client_kwargs()) as client:
        try:
            model = await client.models.get(model_id=model_id)
        except ValueError as exc:
            if output_json:
                click.echo(json.dumps({"error": str(exc)}))
            else:
                console.print(f"[red]Model not found:[/red] {exc}")
            ctx.exit(1)
            return

    if output_json:
        if hasattr(model, "model_dump"):
            click.echo(json.dumps(model.model_dump(by_alias=True), default=str))
        else:
            click.echo(json.dumps({"id": getattr(model, "id", None)}, default=str))
        return

    panel = ModelFormatter.format_verbose_model(model, currency="both")
    console.print(panel)


__all__ = ["get"]
